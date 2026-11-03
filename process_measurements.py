#!/usr/bin/env python3
"""Crop one camera region and export correctly paired images and SLM phases."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from acquire_slm_camera import TWO_PI, make_preview_u8, phase_to_u8, resize_nearest, zernike_basis


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


def reconstruct_full_phase(config: dict, coefficients: np.ndarray) -> np.ndarray:
    """Rebuild a missing full phase from coefficients, never from camera coordinates."""
    phase_256 = np.tensordot(coefficients, _basis(len(coefficients)), axes=(0, 0))
    modulation = resize_nearest(phase_256[56:200], (config["slm_height"], config["slm_width"]))
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
) -> dict[str, Any]:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Choose a separate output directory outside the source acquisition")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {output_dir}")
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
    savemat = None
    if export_mat:
        try:
            from scipy.io import savemat
        except ImportError as exc:
            raise RuntimeError("MAT export requires scipy; install requirements.txt") from exc

    reference_record = source.get("fringe_reference") or retained[0]
    reference = np.load(source_path(input_dir, reference_record["raw_path"]), allow_pickle=False)
    if roi is None:
        roi = select_roi(reference)
    validate_roi(roi, reference.shape)
    floor = float(np.percentile(reference, 1))
    reference_crop = crop_frame(reference, roi)
    reference_signal = float(reference_crop.mean() - floor)

    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("pairs", "previews", "phase_previews"):
        (output_dir / directory).mkdir(exist_ok=True)
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
        "source_configuration": config,
        "roi": roi_info,
        "image_processing": "Camera crop only; original values and dtype; no resizing or intensity normalization",
        "phase_processing": "Original SLM modulation is paired by source pattern index; camera ROI is not applied to SLM phase",
        "include_full_phase": include_full_phase,
        "export_mat": export_mat,
        "excluded_source_pattern_indices": sorted(exclusions),
        "quality_reference": "fringe_only" if source.get("fringe_reference") else "first_retained_measurement",
        "observed_floor_counts": floor,
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
        coefficients = np.asarray(record["zernike_coefficients_rad"], dtype=np.float64)
        if modulation.ndim != 2 or not np.all(np.isfinite(modulation)):
            raise ValueError(f"Invalid modulation phase for source frame {source_index}")
        arrays = {
            "image": cropped,
            "phase_modulation_rad": modulation,
            "zernike_coefficients_rad": coefficients,
            "source_pattern_index": np.asarray(source_index, dtype=np.int64),
            "camera_roi_xywh": np.asarray(roi, dtype=np.int64),
            "slm_shape": np.asarray([config["slm_height"], config["slm_width"]], dtype=np.int64),
            "fringe_slope_rad_per_pixel": np.asarray(config["fringe_slope_rad_per_pixel"]),
            "fringe_orientation": np.asarray(config["fringe_orientation"]),
        }
        phase_source = None
        if include_full_phase:
            full_path = record.get("full_pattern_path")
            if full_path and source_path(input_dir, full_path).is_file():
                phase = np.load(source_path(input_dir, full_path), allow_pickle=False)
                phase_source = "saved_full_phase"
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
        Image.fromarray(make_preview_u8(cropped)[0]).save(output_dir / "previews" / f"{filename}.png")
        Image.fromarray(phase_to_u8(modulation)).save(output_dir / "phase_previews" / f"{filename}.png")
        if savemat is not None:
            savemat(output_dir / "neuws_mat" / f"SLM_raw{output_index}.mat", {"imsdata": cropped}, do_compression=True)
            savemat(output_dir / "neuws_mat" / f"SLM_sim{output_index}.mat", {"proj_sim": modulation}, do_compression=True)
        relative_signal = (float(cropped.mean()) - floor) / reference_signal if reference_signal > 0 else None
        quality = "not_evaluated" if relative_signal is None else (
            "low_signal" if relative_signal < 0.01 else ("partial_signal" if relative_signal < 0.5 else "ok")
        )
        manifest["frames"].append({
            "output_index": output_index,
            "source_pattern_index": source_index,
            "source_frame_id": record.get("frame_id"),
            "pair_path": str(pair_path.relative_to(output_dir)),
            "preview_path": f"previews/{filename}.png",
            "phase_preview_path": f"phase_previews/{filename}.png",
            "image_shape": list(cropped.shape),
            "image_dtype": str(cropped.dtype),
            "image_mean": float(cropped.mean()),
            "relative_signal_to_reference": relative_signal,
            "quality": quality,
            "full_phase_source": phase_source,
            "source_raw_path": record["raw_path"],
            "source_modulation_path": record["modulation_path"],
            "source_full_pattern_path": record.get("full_pattern_path"),
        })
        manifest["frames_completed"] = len(manifest["frames"])
        write_json(output_dir / "manifest.json", manifest)
        print(f"[{output_index:04d}/{len(retained):04d}] source {source_index:04d}: image {cropped.shape}, {quality}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Omit to draw a crop interactively on the fringe reference")
    parser.add_argument("--exclude-indices", nargs="*", type=int, default=[], help="Explicit source pattern indices to omit; image and phase are always removed together")
    parser.add_argument("--include-full-phase", action="store_true", help="Include the complete displayed SLM phase in each compressed pair; otherwise retain the small modulation and ramp parameters")
    parser.add_argument("--export-mat", action="store_true", help="Also export cropped imsdata and modulation proj_sim MAT pairs")
    args = parser.parse_args(argv)
    try:
        result = process_run(
            args.input_dir, args.output_dir,
            roi=tuple(args.roi) if args.roi else None,
            exclude_indices=set(args.exclude_indices),
            include_full_phase=args.include_full_phase, export_mat=args.export_mat,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Saved {result['frames_completed']} paired samples to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
