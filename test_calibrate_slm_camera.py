import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from calibrate_slm_camera import (
    SimulatedCalibrationCamera, fit_mapping, main, measure_spot, model_sampling,
    ramp_phase, scan_frequencies,
)
from acquire_slm_camera import TWO_PI, make_preview_u8


class CalibrationTests(unittest.TestCase):
    def test_ramp_frequencies_follow_physical_slm_axes(self):
        phase = ramp_phase((60, 96), (0.0125, -0.025))
        self.assertEqual(phase.shape, (60, 96))
        self.assertGreaterEqual(float(phase.min()), 0)
        self.assertLess(float(phase.max()), TWO_PI)
        fx = np.angle(np.exp(1j * np.diff(phase[0].astype(float))).mean()) / TWO_PI
        fy = np.angle(np.exp(1j * np.diff(phase[:, 0].astype(float))).mean()) / TWO_PI
        np.testing.assert_allclose([fx, fy], [0.0125, -0.025], atol=1e-8)

    def test_two_axis_fit_recovers_rotation_scale_and_rejects_outlier(self):
        frequencies = np.asarray(scan_frequencies([0, 0.05], 0.0025, 2))
        matrix = np.array([[6800., 500.], [-400., 7300.]])
        origin = np.array([400., 150.])
        centers = frequencies @ matrix.T + origin
        centers[3] += [80, -90]
        fitted = fit_mapping(frequencies, centers)
        np.testing.assert_allclose(fitted["camera_pixels_per_cycle_per_slm_pixel"], matrix, atol=1e-8)
        np.testing.assert_allclose(fitted["camera_origin_at_zero_ramp_xy"], origin, atol=1e-8)
        self.assertFalse(fitted["inliers"][3])
        self.assertEqual(sum(fitted["inliers"]), len(frequencies) - 1)

    def test_static_zero_order_cannot_pass_as_measurable_steering(self):
        frequencies = np.asarray(scan_frequencies([0, 0.05], 0.0025, 2))
        rng = np.random.default_rng(4)
        centers = [400, 300] + rng.normal(0, 0.005, frequencies.shape)
        with self.assertRaises(ValueError):
            fit_mapping(frequencies, centers)

    def test_model_sampling_accounts_for_legacy_and_aspect_matched_pupils(self):
        matrix = np.eye(2) * 1920 * 3.57
        matched = model_sampling(matrix, (1200, 1920), 256, 160)
        legacy = model_sampling(matrix, (1200, 1920), 256, 144)
        np.testing.assert_allclose(matched["camera_pixels_per_fft_bin_slm_xy"], [3.57, 3.57])
        np.testing.assert_allclose(legacy["camera_pixels_per_fft_bin_slm_xy"], [3.57, 3.213])

    def test_sampling_conversion_matches_an_actual_discrete_fourier_shift(self):
        grid_size, pupil_rows = 256, 144
        sampling = model_sampling(np.eye(2), (1200, 1920), grid_size, pupil_rows)
        d = np.asarray(sampling["fft_bins_per_cycle_per_slm_pixel_xy"])
        frequency = np.array([11., 13.]) / d
        coordinate = np.arange(grid_size) - grid_size // 2
        yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
        mask = np.zeros((grid_size, grid_size))
        mask[56:200] = 1
        phase = TWO_PI * (frequency[0] * xx * 1920 / grid_size + frequency[1] * yy * 1200 / pupil_rows)
        intensity = np.abs(np.fft.fftshift(np.fft.fft2(mask * np.exp(1j * phase))))**2
        peak_y, peak_x = np.unravel_index(intensity.argmax(), intensity.shape)
        self.assertEqual((peak_x - 128, peak_y - 128), (11, 13))

    def test_aperture_centroid_and_sparse_preview(self):
        yy, xx = np.indices((120, 180))
        image = 100 + 10000 * np.exp(-((xx - 82.3)**2 + (yy - 61.7)**2) / 50)
        measured = measure_spot(image, (40, 20, 90, 85), centroid_radius=20)
        np.testing.assert_allclose(measured["center_xy"], [82.3, 61.7], atol=0.01)
        sparse = np.zeros((100, 100), dtype=np.uint16)
        sparse[40:43, 50:53] = 1000
        preview, low, high = make_preview_u8(sparse)
        self.assertGreater(preview.max(), 0)
        self.assertEqual((low, high), (0, 1000))

    def test_dry_run_is_identified_as_synthetic_and_can_be_refitted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            arguments = [
                "--dry-run", "--steps", "1", "--repeats", "2", "--pupil-rows", "160",
                "--output-dir", str(output),
            ]
            with patch("sys.stderr", new_callable=io.StringIO), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(arguments), 0)
            result = json.loads((output / "calibration.json").read_text())
            np.testing.assert_allclose(
                result["fit"]["camera_pixels_per_cycle_per_slm_pixel"],
                SimulatedCalibrationCamera.matrix, atol=1,
            )
            self.assertTrue(result["synthetic"])
            self.assertTrue(result["quality_pass"])
            self.assertFalse(result["usable_for_hardware"])
            manifest = json.loads((output / "manifest.json").read_text())
            dark_path = output / manifest["points"][1]["captures"][0]["raw_path"]
            np.save(dark_path, np.full(SimulatedCalibrationCamera.shape, 100, dtype=np.uint16), allow_pickle=False)
            refit = Path(directory) / "refit"
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main([
                    "--analyze-only", str(output), "--output-dir", str(refit), "--pupil-rows", "160",
                ]), 0)
            refitted = json.loads((refit / "calibration.json").read_text())
            self.assertFalse(refitted["observations"][1]["detections"][0]["accepted"])
            self.assertTrue(refitted["quality_pass"])
            self.assertFalse(refitted["usable_for_hardware"])

    def test_existing_output_is_not_modified_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "keep"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("existing calibration", encoding="utf-8")
            with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                main(["--dry-run", "--output-dir", str(output)])
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), "existing calibration")

    def test_low_disk_space_stops_scan_and_closes_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture"
            slm, camera = Mock(), Mock()
            camera.capture.return_value = (np.zeros((12, 16), dtype=np.uint16), {"frame_id": 1})
            with (
                patch("calibrate_slm_camera.SimulatedSLM", return_value=slm),
                patch("calibrate_slm_camera.SimulatedCalibrationCamera", return_value=camera),
                patch("calibrate_slm_camera.shutil.disk_usage", return_value=Mock(free=0)),
                patch("sys.stderr", new_callable=io.StringIO),
                patch("sys.stdout", new_callable=io.StringIO),
                self.assertRaises(SystemExit) as stopped,
            ):
                main(["--dry-run", "--output-dir", str(output)])
            self.assertEqual(stopped.exception.code, 2)
            camera.capture.assert_called_once()
            camera.close.assert_called_once()
            slm.close.assert_called_once()
            self.assertFalse((output / "calibration.json").exists())
            failure = json.loads((output / "calibration_failure.json").read_text())
            self.assertIn("free", failure["error"])
            self.assertFalse(failure["usable_for_hardware"])


if __name__ == "__main__":
    unittest.main()
