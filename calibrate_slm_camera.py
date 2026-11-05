#!/usr/bin/env python3
"""Measure camera displacement versus SLM ramp frequency using a small aperture."""

from __future__ import annotations

import itertools
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from acquire_slm_camera import (
    TWO_PI, HoloeyeSLM, PlayerOneCamera, PySpinCamera, SimulatedSLM,
    build_parser as acquisition_parser, config_from_args, make_preview_u8,
    open_fringe_preview, phase_to_u8,
)
from process_measurements import select_roi, source_path, validate_roi, write_json


def ramp_phase(shape, frequency):
    """frequency = (fx, fy) in cycles per physical SLM pixel."""
    fx, fy = np.asarray(frequency, dtype=float)
    if not np.all(np.isfinite([fx, fy])) or max(abs(fx), abs(fy)) >= 0.5:
        raise ValueError("Each ramp frequency must be finite and strictly between -0.5 and 0.5")
    y, x = np.ogrid[:shape[0], :shape[1]]
    phase = np.mod(TWO_PI * (fx * x + fy * y), TWO_PI).astype(np.float32)
    np.minimum(phase, np.nextafter(np.float32(TWO_PI), np.float32(0)), out=phase)
    return phase


def scan_frequencies(carrier, step, steps):
    if not math.isfinite(step) or step <= 0 or steps < 1:
        raise ValueError("Scan step must be finite and positive; steps must be at least 1")
    carrier = np.asarray(carrier, dtype=float)
    frequencies = [carrier.copy()]
    for distance in range(1, steps + 1):
        for axis in (0, 1):
            for sign in (1, -1):
                frequency = carrier.copy()
                frequency[axis] += sign * distance * step
                frequencies.append(frequency)
    frequencies.append(carrier.copy())  # Return to center to check drift.
    if np.max(np.abs(frequencies)) >= 0.5:
        raise ValueError("Scan crosses the sampled-frequency boundary; reduce the step")
    return frequencies


def measure_spot(frame, roi, centroid_radius=24, saturation_level=65535):
    """Find a compact bright aperture inside a selected/search camera region."""
    validate_roi(roi, frame.shape)
    x, y, width, height = roi
    values = np.asarray(frame[y:y + height, x:x + width], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite camera values in spot search region")
    background = float(np.median(values))
    signal = np.maximum(values - background, 0)
    smoothed = gaussian_filter(signal, 1.5)
    peak_y, peak_x = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    yy, xx = np.indices(values.shape)
    center_x, center_y = float(peak_x), float(peak_y)
    # Recenter the integration window to avoid bias toward an integer peak pixel.
    for _ in range(4):
        aperture = (xx - center_x)**2 + (yy - center_y)**2 <= centroid_radius**2
        weights = np.where(aperture, signal, 0)
        flux = float(weights.sum())
        if flux <= 0:
            raise ValueError("No aperture signal in the selected camera region")
        center_x = float((weights * xx).sum() / flux)
        center_y = float((weights * yy).sum() / flux)
    near_saturation = float(np.mean(values[aperture] >= 0.999 * saturation_level))
    edge_distance = min(peak_x, peak_y, width - 1 - peak_x, height - 1 - peak_y)
    return {
        "center_xy": [center_x + x, center_y + y], "flux": flux,
        "background": background, "peak": float(values[aperture].max()),
        "near_saturation_fraction": near_saturation,
        "search_edge_distance_px": int(edge_distance),
    }


def search_roi(center, radius, shape):
    x = max(0, int(math.floor(center[0] - radius)))
    y = max(0, int(math.floor(center[1] - radius)))
    right = min(shape[1], int(math.ceil(center[0] + radius + 1)))
    bottom = min(shape[0], int(math.ceil(center[1] + radius + 1)))
    return x, y, right - x, bottom - y


def fit_mapping(frequencies, centers, tolerance=2.0):
    """Robust affine fit: camera_xy = origin + B @ (fx, fy)."""
    frequencies = np.asarray(frequencies, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if len(frequencies) < 5 or frequencies.shape != centers.shape or frequencies.shape[1] != 2:
        raise ValueError("At least five 2-D ramp/position observations are required")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(centers)):
        raise ValueError("Calibration observations must be finite")
    frequency_origin = frequencies[0]
    design = np.column_stack((frequencies - frequency_origin, np.ones(len(frequencies))))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Ramp observations must span both SLM frequency axes")
    best = None
    for subset in itertools.combinations(range(len(design)), 3):
        selected = design[list(subset)]
        if np.linalg.matrix_rank(selected) < 3:
            continue
        parameters = np.linalg.solve(selected, centers[list(subset)])
        residuals = np.linalg.norm(design @ parameters - centers, axis=1)
        inliers = residuals <= tolerance
        score = (int(inliers.sum()), -float(np.median(residuals)))
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[1].sum() < 5:
        raise ValueError("Too few consistent aperture positions; inspect the saved previews")
    inliers = best[1]
    for axis in (0, 1):
        if len(np.unique(frequencies[inliers, axis])) < 3:
            raise ValueError("Consistent observations must include both signed offsets on each axis")
    parameters = np.linalg.lstsq(design[inliers], centers[inliers], rcond=None)[0]
    predicted = design @ parameters
    residuals = np.linalg.norm(predicted - centers, axis=1)
    if np.min(np.ptp(predicted[inliers], axis=0)) < 4 * tolerance:
        raise ValueError("Aperture motion is too small to resolve; check the selected order or increase the scan step")
    matrix = parameters[:2].T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= 1e-6 or singular_values[0] / singular_values[-1] > 100:
        raise ValueError("No reliable two-axis steering response; check first-order selection")
    return {
        "camera_pixels_per_cycle_per_slm_pixel": matrix.tolist(),
        "camera_origin_at_zero_ramp_xy": (parameters[2] - matrix @ frequency_origin).tolist(),
        "inliers": inliers.tolist(), "predicted_centers_xy": predicted.tolist(),
        "residuals_px": residuals.tolist(),
        "rms_residual_px": float(np.sqrt(np.mean(residuals[inliers]**2))),
        "maximum_inlier_residual_px": float(residuals[inliers].max()),
        "matrix_condition_number": float(singular_values[0] / singular_values[-1]),
    }


def model_sampling(matrix, slm_shape, grid_size, pupil_rows):
    """Convert physical-SLM ramp response into the chosen FFT-grid sampling."""
    height, width = slm_shape
    if min(height, width, grid_size, pupil_rows) <= 0 or pupil_rows > grid_size:
        raise ValueError("Invalid SLM/model-grid/pupil dimensions")
    # A ramp shifts an N-bin FFT by N * physical pixels per model sample * f.
    fft_bins_per_frequency = np.array([width, grid_size * height / pupil_rows])
    camera_per_fft_bin = np.asarray(matrix) / fft_bins_per_frequency[None, :]
    return {
        "grid_size": grid_size, "pupil_rows": pupil_rows,
        "fft_bins_per_cycle_per_slm_pixel_xy": fft_bins_per_frequency.tolist(),
        "camera_pixels_per_fft_bin_matrix": camera_per_fft_bin.tolist(),
        "fft_bins_per_camera_pixel_matrix": np.linalg.inv(camera_per_fft_bin).tolist(),
        "camera_pixels_per_fft_bin_slm_xy": np.linalg.norm(camera_per_fft_bin, axis=0).tolist(),
        "note": "Use the matrix for rotation/reflection or unequal scaling. These are FFT-sampling factors, not a spatial pixel-to-pixel SLM image registration.",
    }


def save_array(path, array):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(f"Could not write {path}: {exc}. Check disk space and the output drive.") from exc


class SimulatedCalibrationCamera:
    """Synthetic aperture images with a known optical response; never hardware data."""

    shape = (720, 960)
    angle = np.deg2rad(8)
    matrix = 1920 * 3.57 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.array([450., 150.])

    def __init__(self, _config):
        self.rng = np.random.default_rng(14)
        self.frame_id = 0

    def open(self):
        print("[dry-run] SYNTHETIC calibration camera; no hardware calibration will be produced")

    def capture(self, phase):
        frequency = np.array([
            np.angle(np.exp(1j * np.diff(phase[0].astype(float))).mean()) / TWO_PI,
            np.angle(np.exp(1j * np.diff(phase[:, 0].astype(float))).mean()) / TWO_PI,
        ])
        center = self.origin + self.matrix @ frequency
        yy, xx = np.indices(self.shape)
        values = 100 + self.rng.normal(0, 1, self.shape)
        values += 40000 * np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * 8**2))
        values += 12000 * np.exp(-((xx - self.origin[0])**2 + (yy - self.origin[1])**2) / (2 * 7**2))
        self.frame_id += 1
        return np.clip(np.rint(values), 0, 65535).astype(np.uint16), {"frame_id": self.frame_id, "synthetic": True}

    def close(self):
        pass


def prepare_output(output):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def capture_calibration(config, args, output):
    carrier = np.zeros(2)
    axis = 1 if config.fringe_orientation == "horizontal" else 0
    carrier[axis] = math.remainder(config.fringe_slope_rad_per_pixel, TWO_PI) / TWO_PI
    frequencies = scan_frequencies(carrier, args.step_cycles_per_pixel, args.steps)
    slm = SimulatedSLM(config) if config.dry_run else HoloeyeSLM(config)
    camera = SimulatedCalibrationCamera(config) if config.dry_run else (
        PlayerOneCamera(config) if config.camera_backend == "playerone" else PySpinCamera(config)
    )
    for name in ("raw", "medians", "previews", "phase_previews"):
        (output / name).mkdir(exist_ok=True)
    manifest = {
        "kind": "slm_camera_frequency_calibration", "synthetic": config.dry_run,
        "configuration": asdict(config), "carrier_cycles_per_pixel_xy": carrier.tolist(),
        "step_cycles_per_pixel": args.step_cycles_per_pixel,
        "tracking_roi_xywh": list(args.roi) if args.roi else None, "points": [],
    }
    write_json(output / "manifest.json", manifest)
    try:
        slm.open()
        camera.open()
        for index, frequency in enumerate(frequencies):
            phase = ramp_phase((config.slm_height, config.slm_width), frequency)
            slm.show(phase)
            display_ns = time.time_ns()
            if not config.dry_run:
                time.sleep(config.settle_ms / 1000)
            point = {"point_index": index, "frequency_cycles_per_slm_pixel_xy": frequency.tolist(), "captures": []}
            manifest["points"].append(point)
            frames = []
            for repeat in range(args.repeats):
                frame, metadata = camera.capture(phase)
                capture_ns = time.time_ns()
                if index == 0 and repeat == 0:
                    estimated = len(frequencies) * (
                        args.repeats * frame.nbytes + frame.size * 6 + phase.size
                    ) + 8 * 1024**2
                    free = shutil.disk_usage(output).free
                    if free < estimated:
                        raise RuntimeError(
                            f"Calibration needs approximately {estimated / 1024**2:.0f} MiB "
                            f"of output space; only {free / 1024**2:.0f} MiB is free. "
                            "Choose an output folder on a drive with sufficient space."
                        )
                path = output / "raw" / f"point_{index:03d}_repeat_{repeat + 1:02d}.npy"
                save_array(path, frame)
                point["captures"].append({
                    "raw_path": path.relative_to(output).as_posix(),
                    "display_time_ns": display_ns, "capture_time_ns": capture_ns, **metadata,
                })
                frames.append(frame)
                write_json(output / "manifest.json", manifest)
            median = np.median(frames, axis=0).astype(np.float32)
            median_path = output / "medians" / f"point_{index:03d}.npy"
            save_array(median_path, median)
            point["median_path"] = median_path.relative_to(output).as_posix()
            Image.fromarray(make_preview_u8(median)[0]).save(output / "previews" / f"point_{index:03d}.png")
            Image.fromarray(phase_to_u8(phase)).save(output / "phase_previews" / f"point_{index:03d}.png")
            if index == 0 and manifest["tracking_roi_xywh"] is None:
                if config.dry_run:
                    center = camera.origin + camera.matrix @ carrier
                    manifest["tracking_roi_xywh"] = list(search_roi(center, 45, median.shape))
                else:
                    print("Select ONLY the steered aperture image, excluding the static zero order; leave at least 24 pixels around its center.")
                    manifest["tracking_roi_xywh"] = list(select_roi(median))
            write_json(output / "manifest.json", manifest)
            if index == 0:
                baseline = measure_spot(median, tuple(manifest["tracking_roi_xywh"]), args.centroid_radius, args.saturation_level)
                if baseline["near_saturation_fraction"] > 0.01:
                    raise ValueError("Aperture reference is near saturation; lower --exposure-ms and repeat in a new output folder")
            print(f"[{index + 1}/{len(frequencies)}] captured ramp {frequency.tolist()}", flush=True)
    finally:
        try:
            camera.close()
        finally:
            slm.close()
    return manifest


def analyze_calibration(source, manifest, args, output):
    points = manifest["points"]
    if not points or not points[0].get("median_path"):
        raise ValueError("No complete aperture reference in the capture manifest")
    first = np.load(source_path(source, points[0]["median_path"]), allow_pickle=False)
    roi = tuple(args.roi or manifest.get("tracking_roi_xywh") or select_roi(first))
    baseline = measure_spot(first, roi, args.centroid_radius, args.saturation_level)
    if baseline["near_saturation_fraction"] > 0.01:
        raise ValueError("Aperture reference is near saturation; lower --exposure-ms and repeat calibration")
    region = search_roi(baseline["center_xy"], args.tracking_radius, first.shape)
    observations = []
    for point in points:
        detections = []
        for capture in point["captures"]:
            frame = np.load(source_path(source, capture["raw_path"]), allow_pickle=False)
            try:
                detection = measure_spot(frame, roi if point["point_index"] == 0 else region, args.centroid_radius, args.saturation_level)
                reason = None
                if detection["flux"] < args.minimum_flux_fraction * baseline["flux"]:
                    reason = "low_signal"
                elif detection["near_saturation_fraction"] > 0.01:
                    reason = "near_saturation"
                elif detection["search_edge_distance_px"] < args.centroid_radius:
                    reason = "spot_near_search_edge"
                detections.append({**detection, "accepted": reason is None, "rejection_reason": reason})
            except ValueError as exc:
                detections.append({"accepted": False, "rejection_reason": str(exc)})
        good = [d for d in detections if d["accepted"]]
        observations.append({
            "point_index": point["point_index"],
            "frequency_cycles_per_slm_pixel_xy": point["frequency_cycles_per_slm_pixel_xy"],
            "detections": detections,
            "center_xy": np.median([d["center_xy"] for d in good], axis=0).tolist() if good else None,
        })
    valid = [o for o in observations if o["center_xy"] is not None]
    write_json(output / "detections.json", {
        "tracking_roi_xywh": list(roi), "search_roi_xywh": list(region),
        "baseline": baseline, "observations": observations,
    })
    fitted = fit_mapping(
        [o["frequency_cycles_per_slm_pixel_xy"] for o in valid],
        [o["center_xy"] for o in valid], args.max_residual_px,
    )
    config = manifest["configuration"]
    slm_shape = (config["slm_height"], config["slm_width"])
    pupil_rows = args.pupil_rows or round(args.grid_size * slm_shape[0] / slm_shape[1])
    sampling = model_sampling(fitted["camera_pixels_per_cycle_per_slm_pixel"], slm_shape, args.grid_size, pupil_rows)
    reasons = []
    if fitted["maximum_inlier_residual_px"] > args.max_residual_px:
        reasons.append("residual exceeds requested tolerance")
    drift = None
    if observations[0]["center_xy"] is not None and observations[-1]["center_xy"] is not None:
        drift = float(np.linalg.norm(np.subtract(observations[-1]["center_xy"], observations[0]["center_xy"])))
        if drift > args.max_residual_px:
            reasons.append("aperture position drifted between the initial and final center ramps")
    else:
        reasons.append("missing initial/final center spot; drift cannot be checked")
    result = {
        "kind": "camera_pixels_vs_slm_ramp_frequency_affine",
        "synthetic": manifest["synthetic"], "source_capture_directory": str(source),
        "camera_shape": list(first.shape), "slm_shape": list(slm_shape),
        "wavelength_nm": config["wavelength_nm"],
        "carrier_cycles_per_slm_pixel_xy": manifest["carrier_cycles_per_pixel_xy"],
        "tracking_roi_xywh": list(roi), "observations": observations,
        "fitted_point_indices": [o["point_index"] for o in valid],
        "fit": fitted, "model_sampling": sampling,
        "aspect_matched_pupil_rows": round(args.grid_size * slm_shape[0] / slm_shape[1]),
        "center_return_drift_px": drift, "quality_pass": not reasons,
        "quality_failures": reasons, "operator_approved": None, "usable_for_hardware": False,
        "scope": "Fourier sampling and ramp-to-camera steering only; does not measure SLM phase LUT, illumination footprint, or camera-to-SLM image-plane registration.",
    }
    write_json(output / "calibration.json", result)
    render_diagnostics(first, result, output)
    return result


def render_diagnostics(first, result, output):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(15, 5), layout="constrained")
    FigureCanvasAgg(figure)
    overview_axis, image_axis, residual_axis = figure.subplots(1, 3)
    low, high = np.percentile(first, [1, 99.99])
    if high <= low:
        high = max(float(first.max()), float(low) + 1)
    for axis in (overview_axis, image_axis):
        axis.imshow(first, cmap="gray", vmin=low, vmax=high)
        axis.set_xlabel("Camera x [pixels]")
        axis.set_ylabel("Camera y [pixels]")
    overview_axis.set_title("Full camera reference")
    image_axis.set_title("Zoom: measured circles / fitted crosses")
    valid = [o for o in result["observations"] if o["center_xy"] is not None]
    labels = []
    for observation, prediction, inlier in zip(valid, result["fit"]["predicted_centers_xy"], result["fit"]["inliers"]):
        x, y = observation["center_xy"]
        image_axis.plot(x, y, "o", ms=5, mfc="none", mec="tab:orange" if inlier else "tab:red")
        image_axis.plot(*prediction, "+", color="tab:cyan", ms=6)
        matching = next((label for label in labels if np.linalg.norm(np.subtract(label[:2], [x, y])) < 1), None)
        if matching is None:
            labels.append([x, y, [observation["point_index"]]])
        else:
            matching[2].append(observation["point_index"])
    for x, y, indices in labels:
        image_axis.annotate("/".join(map(str, indices)), (x, y), xytext=(6, 6), textcoords="offset points", color="tab:orange", fontsize=9)
    positions = np.asarray([o["center_xy"] for o in valid])
    left, top = positions.min(axis=0) - 30
    right, bottom = positions.max(axis=0) + 30
    image_axis.set_xlim(left, right)
    image_axis.set_ylim(bottom, top)
    overview_axis.plot([left, right, right, left, left], [top, top, bottom, bottom, top], color="tab:orange", linewidth=1)
    residual_axis.bar(result["fitted_point_indices"], result["fit"]["residuals_px"], color=["tab:blue" if i else "tab:red" for i in result["fit"]["inliers"]])
    residual_axis.set_xlabel("Calibration point index")
    residual_axis.set_ylabel("Measured minus fitted position [pixels]")
    residual_axis.set_title(f"Inlier RMS error: {result['fit']['rms_residual_px']:.3f} pixels")
    figure.suptitle("SYNTHETIC EXAMPLE — NOT A HARDWARE CALIBRATION" if result["synthetic"] else "SLM/camera calibration review")
    figure.savefig(output / "calibration_review.png", dpi=150)


def main(argv=None):
    # Calibration already captures and averages its own --repeats per ramp.
    parser = acquisition_parser(include_averaging=False)
    parser.description = __doc__
    parser.set_defaults(
        slm_width=1920, slm_height=1200, exposure_ms=10.0, settle_ms=1000.0,
        discard_frames=2, fringe_orientation="horizontal",
        fringe_slope_rad_per_pixel=TWO_PI / 20, output_format="npy",
        no_neuws_processing=True, capture_fringe_first=False, confirm_fringe_preview=False,
    )
    parser.add_argument("--step-cycles-per-pixel", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=2, help="Scan each axis at +/- 1..steps increments")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Initial aperture search rectangle; omit to draw it on the first preview")
    parser.add_argument("--tracking-radius", type=int, default=120)
    parser.add_argument("--centroid-radius", type=int, default=24)
    parser.add_argument("--saturation-level", type=float, default=65535)
    parser.add_argument("--minimum-flux-fraction", type=float, default=0.2)
    parser.add_argument("--max-residual-px", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--pupil-rows", type=int, default=None, help="Use 144 for existing acquisitions; 160 for an aspect-matched full 1920x1200 pupil")
    parser.add_argument("--analyze-only", type=Path, help="Refit an existing calibration capture without opening hardware")
    parser.add_argument("--skip-review", action="store_true", help="Save an unreviewed fit without a Y/N prompt")
    args = parser.parse_args(argv)
    output = (args.output_dir or Path("data") / f"calibration_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    result = None
    output_prepared = False
    try:
        if args.repeats < 1 or args.tracking_radius <= args.centroid_radius or args.centroid_radius < 2:
            raise ValueError("Use repeats >= 1 and tracking radius > centroid radius >= 2")
        if args.max_residual_px <= 0 or not 0 < args.minimum_flux_fraction <= 1:
            raise ValueError("Use positive residual tolerance and a flux fraction in (0, 1]")
        prepare_output(output)
        output_prepared = True
        if args.analyze_only:
            source = args.analyze_only.resolve()
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        else:
            config = config_from_args(args)
            print("Calibration requires a small aperture at the object plane and removal of the aberrating sample.")
            manifest = capture_calibration(config, args, output)
            source = output
        result = analyze_calibration(source, manifest, args, output)
        if not result["synthetic"] and not args.skip_review and result["quality_pass"]:
            open_fringe_preview(output / "calibration_review.png")
            answer = ""
            while answer not in {"Y", "YES", "N", "NO"}:
                try:
                    answer = input("Do the marked positions follow the same steered aperture? Accept calibration [Y/N]: ").strip().upper()
                except (EOFError, OSError):
                    answer = "N"
            result["operator_approved"] = answer in {"Y", "YES"}
        result["usable_for_hardware"] = bool(result["quality_pass"] and result["operator_approved"] and not result["synthetic"])
        write_json(output / "calibration.json", result)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if output_prepared:
            try:
                write_json(output / "calibration_failure.json", {"error": str(exc), "usable_for_hardware": False})
            except OSError:
                pass
        parser.exit(2, f"error: {exc}\n")
    print(f"Calibration output: {output}")
    print(f"Camera pixels per FFT bin along SLM x/y: {result['model_sampling']['camera_pixels_per_fft_bin_slm_xy']}")
    print(f"Fit RMS error: {result['fit']['rms_residual_px']:.3f} camera pixels")
    print(f"Usable for hardware: {result['usable_for_hardware']}")
    if result["synthetic"]:
        print("SIMULATION ONLY: run the hardware command to measure your optical system.")
    return 0 if result["quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
