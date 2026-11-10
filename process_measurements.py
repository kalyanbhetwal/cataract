#!/usr/bin/env python3
"""Extract a camera order, optionally resample with calibration, and pair it with SLM phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from acquire_slm_camera import TWO_PI, make_preview_u8, phase_to_u8, process_neuws_frame, resize_nearest, zernike_basis


def source_path(root: Path, relative: str) -> Path:
    """Read either Windows or POSIX relative paths from an acquisition manifest."""
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path lies outside the acquisition: {relative}")
    return path


def validate_roi(roi: tuple[int, int, int, int], shape: tuple[int, ...]) -> None:
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D raw camera frame, got {shape}")
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("ROI needs nonnegative x/y and positive width/height")
    if x + width > shape[1] or y + height > shape[0]:
        raise ValueError(f"ROI {roi} lies outside camera frame {shape[1]}x{shape[0]}")


def crop_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    validate_roi(roi, frame.shape)
    x, y, width, height = roi
    return frame[y:y + height, x:x + width].copy()


def select_roi(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Let the operator select a camera crop without assuming an optical mapping."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
    except ImportError as exc:
        raise RuntimeError("Interactive selection requires matplotlib; install requirements.txt") from exc
    if matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
        raise RuntimeError("No interactive plotting backend; supply --roi X Y WIDTH HEIGHT instead")
    preview = make_preview_u8(frame)[0]
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(preview, cmap="gray", vmin=0, vmax=255)
    axis.set_xlabel("Raw camera x [pixels]")
    axis.set_ylabel("Raw camera y [pixels]")
    axis.set_title("Drag around the steered image; Enter accepts, Escape cancels")
    selected = None
    rectangle = RectangleSelector(
        axis, lambda _start, _end: None, interactive=True,
        button=[1], minspanx=1, minspany=1, spancoords="pixels",
    )

    def on_key(event):
        nonlocal selected
        if event.key == "escape":
            plt.close(figure)
        elif event.key == "enter":
            x0, x1, y0, y1 = rectangle.extents
            x = max(0, int(math.floor(x0)))
            y = max(0, int(math.floor(y0)))
            right = min(frame.shape[1], int(math.ceil(x1)))
            bottom = min(frame.shape[0], int(math.ceil(y1)))
            candidate = (x, y, right - x, bottom - y)
            try:
                validate_roi(candidate, frame.shape)
            except ValueError:
                return
            selected = candidate
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    if selected is None:
        raise RuntimeError("ROI selection cancelled; no processed dataset was created")
    return selected


@lru_cache(maxsize=4)
def _basis(modes: int) -> np.ndarray:
    return zernike_basis(256, modes)


def reconstruct_full_phase(config: dict, coefficients: np.ndarray, modulation: np.ndarray | None = None) -> np.ndarray:
    """Rebuild the display from saved modulation or legacy generating coefficients."""
    if modulation is None:
        phase_256 = np.tensordot(coefficients, _basis(len(coefficients)), axes=(0, 0))
        modulation = phase_256[56:200]
    modulation = resize_nearest(modulation, (config["slm_height"], config["slm_width"]))
    slope = config["fringe_slope_rad_per_pixel"]
    if config["fringe_orientation"] == "horizontal":
        fringe = np.mod(slope * np.arange(config["slm_height"], dtype=np.float64), TWO_PI)[:, None]
    elif config["fringe_orientation"] == "vertical":
        fringe = np.mod(slope * np.arange(config["slm_width"], dtype=np.float64), TWO_PI)
    else:
        raise ValueError("Unknown fringe orientation in source configuration")
    phase = np.mod(modulation + fringe, TWO_PI).astype(np.float32)
    np.minimum(phase, np.nextafter(np.float32(TWO_PI), np.float32(0)), out=phase)
    return phase


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_pair(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            # All payload arrays are numeric or Unicode; no object/pickle arrays.
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def process_run(
    input_dir: Path,
    output_dir: Path,
    roi: tuple[int, int, int, int] | None = None,
    exclude_indices: set[int] | None = None,
    include_full_phase: bool = False,
    export_mat: bool = False,
    calibration_file: Path | None = None,
    camera_center: tuple[float, float] | None = None,
    preserve_camera_orientation: bool = False,
    downsample_factor: float | tuple[float, float] | None = None,
    crop_size: int = 256,
    crop_center: tuple[float, float] | None = None,
) -> dict[str, Any]:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Choose a separate output directory outside the source acquisition")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {output_dir}")
    if preserve_camera_orientation and calibration_file is not None:
        raise ValueError("--preserve-camera-orientation allows crop-only or scale-and-crop processing; omit --calibration-file")
    if downsample_factor is not None:
        if calibration_file is not None:
            raise ValueError("Use either --downsample-factor or --calibration-file, not both")
        factors_xy = np.asarray(downsample_factor, dtype=float)
        if factors_xy.ndim == 0:
            factors_xy = np.repeat(factors_xy, 2)
        if factors_xy.shape != (2,) or not np.isfinite(factors_xy).all() or np.any(factors_xy <= 0):
            raise ValueError("Downsample factor must be a positive finite scalar or (camera x, camera y) pair")
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError("Crop size must be a positive integer")
    elif crop_center is not None:
        raise ValueError("--crop-center requires --downsample-factor")
    source = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    config, records = source["configuration"], source["frames"]
    if not records:
        raise ValueError("The source manifest contains no camera measurements")
    exclusions = exclude_indices or set()
    unknown = exclusions - {r["pattern_index"] for r in records}
    if unknown:
        raise ValueError(f"Excluded indices are not in this run: {sorted(unknown)}")
    retained = [r for r in records if r["pattern_index"] not in exclusions]
    if not retained:
        raise ValueError("No measurements remain after exclusions")
    if camera_center is not None and calibration_file is None:
        raise ValueError("--camera-center requires --calibration-file")
    # Respect current manifest paths, including any selected later recaptures.
    # Fail before creating output if a referenced replacement was not copied.
    for record in retained:
        for key in ("raw_path", "modulation_path"):
            path = source_path(input_dir, record[key])
            if not path.is_file():
                raise FileNotFoundError(f"Missing source pattern {record['pattern_index']} {key}: {path}")
    savemat = None
    if export_mat:
        try:
            from scipy.io import savemat
        except ImportError as exc:
            raise RuntimeError("MAT export requires scipy; install requirements.txt") from exc

    reference_record = source.get("fringe_reference") or retained[0]
    reference = np.load(source_path(input_dir, reference_record["raw_path"]), allow_pickle=False)
    # Preserve original reference arrays and their acquisition records, including
    # earlier steering attempts. A PNG is insufficient for quantitative checks.
    references = list(source.get("fringe_references") or [])
    if source.get("fringe_reference") and source["fringe_reference"] not in references:
        references.append(source["fringe_reference"])
    reference_paths = set()
    for record in references:
        for field in ("raw_path", "preview_path", "full_pattern_path"):
            if record.get(field):
                relative = record[field].replace("\\", "/")
                if not source_path(input_dir, relative).is_file():
                    raise FileNotFoundError(f"Missing acquisition reference {field}: {relative}")
                reference_paths.add(relative)
    if roi is None:
        roi = select_roi(reference)
    validate_roi(roi, reference.shape)
    scale_crop = None
    scaled_reference = None
    if downsample_factor is not None:
        center = crop_center if crop_center is not None else (roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("Crop center must contain two finite camera coordinates")
        scaled_reference = process_neuws_frame(reference, downsample_factor, crop_size, *center)
        scaled_shape = [max(1, int(round(n / factor))) for n, factor in zip(reference.shape, factors_xy[::-1])]
        scale_crop = {
            "downsample_factor": downsample_factor,
            "downsample_factors_camera_xy": factors_xy.tolist(),
            "raw_camera_shape": list(reference.shape),
            "scaled_full_frame_shape": scaled_shape,
            "effective_camera_pixels_per_output_pixel_xy": [reference.shape[1] / scaled_shape[1], reference.shape[0] / scaled_shape[0]],
            "crop_size": crop_size,
            "crop_center_raw_xy": list(center),
            "crop_center_source": "explicit_override" if crop_center is not None else "selected_camera_roi_center",
            "interpolation": "Center-aligned bilinear resize of full camera frame, then square crop; integer inputs rounded to original dtype",
            "orientation": "Original camera row/column order; no rotation, reflection, or transpose",
            "intensity_processing": "Camera count units; no background subtraction or intensity normalization",
            "roi_use": "ROI selects the default center and unscaled diagnostic crop; final crop support follows the scale and crop size",
            "reference_image_path": "reference_processed.npy",
        }
    sampler = None
    if calibration_file is not None:
        from calibrated_sampling import FourierSampler
        modulation = np.load(source_path(input_dir, retained[0]["modulation_path"]), allow_pickle=False)
        sampler = FourierSampler(calibration_file, config, reference.shape, modulation.shape, roi, camera_center)
    source_quality = {}
    if (input_dir / "quality_report.json").is_file():
        source_quality = json.loads((input_dir / "quality_report.json").read_text(encoding="utf-8"))
    source_flagged = set(source_quality.get("low_signal_candidate_indices", []))
    floor = float(np.percentile(reference, 1))
    reference_crop = crop_frame(reference, roi)
    reference_signal = float(reference_crop.mean() - floor)
    reference_contrast = float(np.percentile(reference_crop, 99.9) - np.median(reference_crop))

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_dir / "manifest.json", output_dir / "source_manifest.json")
    for relative in sorted(reference_paths):
        destination = source_path(output_dir, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path(input_dir, relative), destination)
    for directory in ("pairs", "previews", "phase_previews"):
        (output_dir / directory).mkdir(exist_ok=True)
    if scale_crop is not None:
        (output_dir / "camera_previews").mkdir(exist_ok=True)
        np.save(output_dir / "reference_processed.npy", scaled_reference, allow_pickle=False)
        Image.fromarray(make_preview_u8(scaled_reference)[0]).save(output_dir / "reference_processed.png")
    if sampler is not None:
        (output_dir / "camera_previews").mkdir(exist_ok=True)
        write_json(output_dir / "calibration.json", sampler.calibration)
        Image.fromarray(sampler.valid.astype(np.uint8) * 255).save(output_dir / "image_valid_mask.png")
        from calibrated_sampling import masked_preview
        Image.fromarray(masked_preview(sampler.resample(reference), sampler.valid)).save(output_dir / "reference_calibrated.png")
    if source_quality:
        write_json(output_dir / "source_quality_report.json", source_quality)
    if export_mat:
        (output_dir / "neuws_mat").mkdir(exist_ok=True)
    image = Image.fromarray(make_preview_u8(reference)[0]).convert("RGB")
    x, y, width, height = roi
    ImageDraw.Draw(image).rectangle((x, y, x + width - 1, y + height - 1), outline=(255, 120, 30), width=5)
    image.save(output_dir / "roi_preview.png")
    Image.fromarray(make_preview_u8(reference_crop)[0]).save(output_dir / "reference_crop.png")
    roi_info = {"x": x, "y": y, "width": width, "height": height, "units": "raw camera pixels", "origin": "upper left"}
    write_json(output_dir / "roi.json", roi_info)
    manifest: dict[str, Any] = {
        "source_directory": str(input_dir),
        "source_manifest_sha256": hashlib.sha256((input_dir / "manifest.json").read_bytes()).hexdigest(),
        "source_manifest_path": "source_manifest.json",
        "source_configuration": config,
        "fringe_reference": source.get("fringe_reference"),
        "fringe_references": references,
        "reference_files": sorted(reference_paths),
        "roi": roi_info,
        "image_coordinate_system": "raw_camera_axes" if sampler is None else "slm_fourier_model_axes",
        "preserve_camera_orientation": sampler is None,
        "scale_and_crop": scale_crop,
        "image_processing": (
            "Bilinear scale of full camera frame followed by fixed square crop; camera orientation and count units retained; no rotation, reflection, or transpose" if scale_crop is not None else
            "Camera crop only; original values and dtype; no rotation, reflection, transpose, resizing, filtering, or intensity normalization" if sampler is None else
            "Fixed calibrated Fourier grid; Gaussian anti-alias filter and bilinear interpolation; NaN outside image_valid_mask; original crop also retained"
        ),
        "calibrated_sampling": sampler.info if sampler is not None else None,
        "phase_processing": "Original SLM modulation is paired by source pattern index; camera ROI is not applied to SLM phase",
        "phase_pattern_source": source.get("phase_pattern_source"),
        "include_full_phase": include_full_phase,
        "export_mat": export_mat,
        "excluded_source_pattern_indices": sorted(exclusions),
        "quality_reference": "fringe_only" if source.get("fringe_reference") else "first_retained_measurement",
        "observed_floor_counts": floor,
        "source_low_signal_candidate_indices": sorted(source_flagged),
        "quality_note": "Relative signal uses the selected ROI and an observed floor, not a calibrated dark frame. Flags do not remove pairs.",
        "frames_completed": 0,
        "frames": [],
    }
    write_json(output_dir / "manifest.json", manifest)

    for output_index, record in enumerate(retained, 1):
        source_index = record["pattern_index"]
        raw = np.load(source_path(input_dir, record["raw_path"]), allow_pickle=False)
        cropped = crop_frame(raw, roi)
        modulation = np.load(source_path(input_dir, record["modulation_path"]), allow_pickle=False)
        coefficient_values = record["zernike_coefficients_rad"]
        coefficients = np.asarray(coefficient_values if coefficient_values is not None else [], dtype=np.float64)
        if modulation.ndim != 2 or not np.all(np.isfinite(modulation)):
            raise ValueError(f"Invalid modulation phase for source frame {source_index}")
        arrays = {
            "image": cropped,
            "phase_modulation_rad": modulation,
            "zernike_coefficients_rad": coefficients,
            "zernike_coefficients_available": np.asarray(coefficient_values is not None),
            "source_pattern_index": np.asarray(source_index, dtype=np.int64),
            "camera_roi_xywh": np.asarray(roi, dtype=np.int64),
            "slm_shape": np.asarray([config["slm_height"], config["slm_width"]], dtype=np.int64),
            "fringe_slope_rad_per_pixel": np.asarray(config["fringe_slope_rad_per_pixel"]),
            "fringe_orientation": np.asarray(config["fringe_orientation"]),
        }
        output_image = cropped
        if scale_crop is not None:
            if list(raw.shape) != scale_crop["raw_camera_shape"]:
                raise ValueError("Camera dimensions changed; cannot apply one fixed scale and crop")
            output_image = process_neuws_frame(raw, downsample_factor, crop_size, *scale_crop["crop_center_raw_xy"])
            arrays.update({"image": output_image, "image_camera_crop": cropped})
        if sampler is not None:
            output_image = sampler.resample(raw)
            arrays.update({
                "image": output_image,
                "image_camera_crop": cropped,
                "image_valid_mask": sampler.valid,
                "phase_model_rad": sampler.model_phase(modulation),
                "pupil_mask": sampler.pupil_mask,
                "camera_pixels_per_fft_bin_matrix": sampler.matrix,
                "camera_center_xy": sampler.center,
                "model_grid_origin_xy": np.asarray([sampler.size // 2, sampler.size // 2], dtype=np.int64),
            })
        phase_source = None
        if include_full_phase:
            full_path = record.get("full_pattern_path")
            if full_path and source_path(input_dir, full_path).is_file():
                phase = np.load(source_path(input_dir, full_path), allow_pickle=False)
                phase_source = "saved_full_phase"
            else:
                if coefficient_values is None:
                    phase = reconstruct_full_phase(config, coefficients, modulation=modulation)
                    phase_source = "reconstructed_from_saved_modulation"
                else:
                    phase = reconstruct_full_phase(config, coefficients)
                    phase_source = "reconstructed_from_coefficients"
            expected_shape = (config["slm_height"], config["slm_width"])
            if phase.shape != expected_shape or not np.all(np.isfinite(phase)):
                raise ValueError(f"Invalid full SLM phase for source frame {source_index}")
            arrays["phase_slm_rad"] = phase

        filename = f"frame_{output_index:04d}"
        pair_path = output_dir / "pairs" / f"{filename}.npz"
        write_pair(pair_path, arrays)
        if sampler is None:
            Image.fromarray(make_preview_u8(output_image)[0]).save(output_dir / "previews" / f"{filename}.png")
            if scale_crop is not None:
                Image.fromarray(make_preview_u8(cropped)[0]).save(output_dir / "camera_previews" / f"{filename}.png")
        else:
            Image.fromarray(masked_preview(output_image, sampler.valid)).save(output_dir / "previews" / f"{filename}.png")
            Image.fromarray(make_preview_u8(cropped)[0]).save(output_dir / "camera_previews" / f"{filename}.png")
        Image.fromarray(phase_to_u8(modulation)).save(output_dir / "phase_previews" / f"{filename}.png")
        if savemat is not None:
            image_payload = {"imsdata": output_image}
            phase_payload = {"proj_sim": modulation}
            if sampler is not None:
                image_payload["valid_mask"] = sampler.valid
                phase_payload = {"proj_sim": arrays["phase_model_rad"], "pupil_mask": sampler.pupil_mask}
            savemat(output_dir / "neuws_mat" / f"SLM_raw{output_index}.mat", image_payload, do_compression=True)
            savemat(output_dir / "neuws_mat" / f"SLM_sim{output_index}.mat", phase_payload, do_compression=True)
        relative_signal = (float(cropped.mean()) - floor) / reference_signal if reference_signal > 0 else None
        quality = "not_evaluated" if relative_signal is None else (
            "low_signal" if relative_signal < 0.01 else ("partial_signal" if relative_signal < 0.5 else "ok")
        )
        contrast = float(np.percentile(cropped, 99.9) - np.median(cropped))
        relative_contrast = contrast / reference_contrast if reference_contrast > 0 else None
        flags = []
        if source_index in source_flagged:
            flags.append("source_low_signal_candidate")
        if relative_contrast is not None and relative_contrast < 0.2:
            flags.append("low_contrast_to_reference")
        manifest["frames"].append({
            "output_index": output_index,
            "source_pattern_index": source_index,
            "phase_pattern_source": record.get("phase_pattern_source"),
            "source_frame_id": record.get("frame_id"),
            "source_display_time_ns": record.get("display_time_ns"),
            "source_capture_time_ns": record.get("capture_time_ns"),
            "source_camera_timestamp_ns": record.get("camera_timestamp_ns"),
            "source_capture_start_time_ns": record.get("capture_start_time_ns"),
            "source_capture_end_time_ns": record.get("capture_end_time_ns"),
            "source_averaging": record.get("averaging"),
            "pair_path": str(pair_path.relative_to(output_dir)),
            "preview_path": f"previews/{filename}.png",
            "phase_preview_path": f"phase_previews/{filename}.png",
            "image_shape": list(output_image.shape),
            "image_dtype": str(output_image.dtype),
            "image_mean": float(np.nanmean(output_image)),
            "camera_crop_shape": list(cropped.shape),
            "camera_crop_mean": float(cropped.mean()),
            "relative_signal_to_reference": relative_signal,
            "quality": quality,
            "quality_flags": flags,
            "contrast_relative_to_reference": relative_contrast,
            "source_record_quality_flags": record.get("quality_flags"),
            "full_phase_source": phase_source,
            "source_raw_path": record["raw_path"],
            "source_modulation_path": record["modulation_path"],
            "source_full_pattern_path": record.get("full_pattern_path"),
        })
        manifest["frames_completed"] = len(manifest["frames"])
        write_json(output_dir / "manifest.json", manifest)
        print(f"[{output_index:04d}/{len(retained):04d}] source {source_index:04d}: image {output_image.shape}, {quality}, flags={flags}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Omit to draw a crop interactively on the fringe reference")
    parser.add_argument("--exclude-indices", nargs="*", type=int, default=[], help="Explicit source pattern indices to omit; image and phase are always removed together")
    parser.add_argument("--include-full-phase", action="store_true", help="Include the complete displayed SLM phase in each compressed pair; otherwise retain the small modulation and ramp parameters")
    parser.add_argument("--export-mat", action="store_true", help="Also export imsdata/proj_sim MAT pairs; calibrated exports include validity/pupil masks")
    parser.add_argument("--calibration-file", type=Path, help="Approved hardware calibration JSON; resample to its model grid with an explicit validity mask")
    parser.add_argument("--preserve-camera-orientation", action="store_true", help="Allow crop-only or scale-and-crop processing; reject --calibration-file and its orientation transform")
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--downsample-factor", type=float, help="Scalar size divisor: resize full camera frame, then crop; preserves camera orientation")
    scaling.add_argument("--downsample-xy", nargs=2, type=float, metavar=("X", "Y"), help="Separate size divisors in camera x/y axes; scale then crop, preserving row/column order")
    parser.add_argument("--crop-size", type=int, default=256, help="Square output size after --downsample-factor (default: 256)")
    parser.add_argument("--crop-center", nargs=2, type=float, metavar=("X", "Y"), help="Center in raw camera pixels for --downsample-factor; defaults to selected ROI center")
    parser.add_argument("--camera-center", nargs=2, type=float, metavar=("X", "Y"), help="Override the calibrated carrier position as model-grid center; requires --calibration-file")
    args = parser.parse_args(argv)
    try:
        result = process_run(
            args.input_dir, args.output_dir,
            roi=tuple(args.roi) if args.roi else None,
            exclude_indices=set(args.exclude_indices),
            include_full_phase=args.include_full_phase, export_mat=args.export_mat,
            calibration_file=args.calibration_file,
            camera_center=tuple(args.camera_center) if args.camera_center else None,
            preserve_camera_orientation=args.preserve_camera_orientation,
            downsample_factor=tuple(args.downsample_xy) if args.downsample_xy is not None else args.downsample_factor,
            crop_size=args.crop_size,
            crop_center=tuple(args.crop_center) if args.crop_center else None,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved {result['frames_completed']} paired samples to {args.output_dir.resolve()}")
    if result["calibrated_sampling"] is not None:
        sampling = result["calibrated_sampling"]
        print(f"Supported model pixels: {sampling['valid_pixels']} ({sampling['valid_fraction']:.1%}); use image_valid_mask and ignore NaN pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
