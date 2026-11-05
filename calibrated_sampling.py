"""Resample camera intensities onto the calibrated, centered Fourier grid."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class FourierSampler:
    """One fixed map for every frame; unsupported output pixels remain missing."""

    def __init__(self, path, config, camera_shape, modulation_shape, roi, center_xy=None):
        self.path = Path(path).resolve()
        payload = self.path.read_bytes()
        self.calibration = calibration = json.loads(payload)
        if calibration.get("kind") != "camera_pixels_vs_slm_ramp_frequency_affine":
            raise ValueError("Unsupported calibration kind")
        if calibration.get("synthetic") is not False or any(
            calibration.get(key) is not True
            for key in ("quality_pass", "operator_approved", "usable_for_hardware")
        ):
            raise ValueError("Use a passing, operator-approved hardware calibration")
        slm_shape = [config["slm_height"], config["slm_width"]]
        if calibration["slm_shape"] != slm_shape or calibration["camera_shape"] != list(camera_shape):
            raise ValueError("Calibration SLM/camera dimensions do not match this acquisition")
        if not math.isclose(calibration["wavelength_nm"], config["wavelength_nm"], rel_tol=1e-9):
            raise ValueError("Calibration wavelength does not match this acquisition")
        sampling = calibration["model_sampling"]
        self.size, self.pupil_rows = sampling["grid_size"], sampling["pupil_rows"]
        if (
            not isinstance(self.size, int) or not isinstance(self.pupil_rows, int)
            or not 0 < self.pupil_rows <= self.size
            or tuple(modulation_shape) != (self.pupil_rows, self.size)
        ):
            raise ValueError("Calibration grid/pupil dimensions do not match the saved phase modulation")
        self.camera_shape = tuple(camera_shape)
        self.matrix = np.asarray(sampling["camera_pixels_per_fft_bin_matrix"], dtype=float)
        frequency_matrix = np.asarray(calibration["fit"]["camera_pixels_per_cycle_per_slm_pixel"], dtype=float)
        if any(m.shape != (2, 2) or not np.all(np.isfinite(m)) for m in (self.matrix, frequency_matrix)):
            raise ValueError("Calibration matrices must be finite 2 x 2 arrays")
        if np.linalg.cond(self.matrix) > 100:
            raise ValueError("Calibration matrix is singular or poorly conditioned")
        denominators = [slm_shape[1], self.size * slm_shape[0] / self.pupil_rows]
        if not np.allclose(self.matrix, frequency_matrix / denominators, rtol=1e-9, atol=1e-9):
            raise ValueError("Calibration matrix disagrees with its recorded FFT-grid interpretation")
        carrier = np.zeros(2)
        if config["fringe_orientation"] not in {"horizontal", "vertical"}:
            raise ValueError("Unknown acquisition fringe orientation")
        axis = 1 if config["fringe_orientation"] == "horizontal" else 0
        carrier[axis] = math.remainder(config["fringe_slope_rad_per_pixel"], 2 * math.pi) / (2 * math.pi)
        if not np.allclose(carrier, calibration["carrier_cycles_per_slm_pixel_xy"], atol=1e-10, rtol=0):
            raise ValueError("Acquisition carrier differs from the calibrated carrier; verify/recalibrate the steering")
        predicted = np.asarray(calibration["fit"]["camera_origin_at_zero_ramp_xy"], dtype=float) + frequency_matrix @ carrier
        self.center = np.asarray(predicted if center_xy is None else center_xy, dtype=float)
        if self.center.shape != (2,) or not np.all(np.isfinite(self.center)):
            raise ValueError("Camera center must contain two finite coordinates")
        # Pixel (N//2, N//2) is the centered FFT origin, with x=column, y=row.
        yy, xx = np.indices((self.size, self.size), dtype=float)
        self.camera_xy = self.center[:, None, None] + np.einsum(
            "ij,jhw->ihw", self.matrix, np.array([xx - self.size // 2, yy - self.size // 2])
        )
        # Approximate each output footprint with a Gaussian in sensor axes.
        # Record this separately from intensity normalization: raw count levels
        # are retained; there is no per-frame scaling or Jacobian flux multiplier.
        self.sigma_xy = 0.5 * np.sqrt(np.maximum(np.sum(self.matrix**2, axis=1) - 1, 0))
        margin = np.ceil(4 * self.sigma_xy).astype(int) + 1
        x, y, width, height = roi
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > camera_shape[1] or y + height > camera_shape[0]:
            raise ValueError("Calibration ROI lies outside the camera frame")
        self.valid = np.all(
            (self.camera_xy >= np.array([x, y])[:, None, None] + margin[:, None, None])
            & (self.camera_xy <= np.array([x + width - 1, y + height - 1])[:, None, None] - margin[:, None, None]),
            axis=0,
        )
        if not self.valid.any():
            raise ValueError("No calibrated samples fit inside the selected ROI with filtering margins")
        self.pupil_mask = np.zeros((self.size, self.size), dtype=bool)
        self.pupil_start = (self.size - self.pupil_rows) // 2
        self.pupil_mask[self.pupil_start:self.pupil_start + self.pupil_rows] = True
        self.info = {
            "source_file": str(self.path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "grid_size": self.size, "pupil_rows": self.pupil_rows,
            "camera_pixels_per_fft_bin_matrix": self.matrix.tolist(),
            "camera_center_xy": self.center.tolist(),
            "camera_center_source": "calibrated_carrier_position" if center_xy is None else "explicit_override",
            "mapping": "camera_xy = camera_center_xy + C @ [column - N//2, row - N//2]",
            "antialias_gaussian_sigma_camera_xy": self.sigma_xy.tolist(),
            "roi_filter_margin_camera_xy": margin.tolist(),
            "interpolation": "bilinear after Gaussian prefilter, truncate=4; float32 camera count levels",
            "intensity_processing": "No background subtraction, per-frame normalization, or flux Jacobian multiplier",
            "valid_pixels": int(self.valid.sum()),
            "valid_fraction": float(self.valid.mean()),
            "missing_pixels": "NaN in image; false in image_valid_mask. Never use these as observed zero intensity.",
            "scope": "Fourier sampling/orientation; camera crop is not applied to the SLM phase",
        }

    def resample(self, frame):
        if frame.shape != self.camera_shape or not np.all(np.isfinite(frame)):
            raise ValueError("Invalid camera frame shape or non-finite raw intensities")
        filtered = gaussian_filter(
            np.asarray(frame, dtype=np.float32), self.sigma_xy[::-1], mode="nearest", truncate=4,
        )
        image = map_coordinates(filtered, self.camera_xy[::-1], order=1, mode="constant", cval=0, prefilter=False)
        image[~self.valid] = np.nan
        return image

    def model_phase(self, modulation):
        if modulation.shape != (self.pupil_rows, self.size):
            raise ValueError("Phase modulation dimensions changed within the acquisition")
        phase = np.zeros((self.size, self.size), dtype=modulation.dtype)
        phase[self.pupil_start:self.pupil_start + self.pupil_rows] = modulation
        return phase


def masked_preview(image, mask):
    """RGBA viewing aid: missing measurements are transparent, not dark data."""
    values = image[mask]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = values.min(), values.max()
    gray = np.zeros(image.shape, dtype=np.uint8)
    if high > low:
        gray[mask] = np.clip((values - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
    return np.dstack((gray, gray, gray, mask.astype(np.uint8) * 255))
