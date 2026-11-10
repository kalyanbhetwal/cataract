import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat

from acquire_slm_camera import NeuWSPatternGenerator, build_parser, config_from_args
from process_measurements import crop_frame, process_run, source_path
from calibrated_sampling import FourierSampler, masked_preview


def make_source(root):
    with patch("sys.stderr", new_callable=io.StringIO):
        config = config_from_args(build_parser().parse_args([
            "--dry-run", "--generated-patterns", "--num-patterns", "3", "--slm-width", "16", "--slm-height", "12",
            "--frames-per-pattern", "1",
        ]))
    config = replace(config, fringe_orientation="horizontal", fringe_slope_rad_per_pixel=0.3)
    for directory in ("raw", "patterns", "reference"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    reference = np.full((12, 16), 10, dtype=np.uint16)
    reference[3:10, 4:10] = 100
    np.save(root / "reference" / "fringe_only.npy", reference, allow_pickle=False)
    generator = NeuWSPatternGenerator(config)
    records, expected = [], {}
    for index in range(1, 4):
        phase, modulation, coefficients = generator.next()
        frame = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16) + index * 100
        if index == 2:
            frame = np.full((12, 16), 10, dtype=np.uint16)
        np.save(root / "raw" / f"frame_{index:04d}.npy", frame, allow_pickle=False)
        np.save(root / "patterns" / f"modulation_{index:04d}.npy", modulation, allow_pickle=False)
        if index != 3:
            np.save(root / "patterns" / f"phase_{index:04d}.npy", phase, allow_pickle=False)
        records.append({
            "pattern_index": index, "frame_id": 100 + index,
            "display_time_ns": 1700000000000000000 + index,
            "capture_time_ns": 1700000001000000000 + index,
            "camera_timestamp_ns": 900000000000 + index,
            "raw_path": f"raw\\frame_{index:04d}.npy",
            "modulation_path": f"patterns\\modulation_{index:04d}.npy",
            "full_pattern_path": f"patterns\\phase_{index:04d}.npy" if index != 3 else None,
            "zernike_coefficients_rad": coefficients.tolist(),
        })
        expected[index] = {"image": frame, "modulation": modulation, "phase": phase}
    manifest = {
        "configuration": asdict(config), "frames": records,
        "fringe_reference": {"raw_path": "reference\\fringe_only.npy"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return reference, expected


def make_calibration(path, config, camera_shape=(60, 64)):
    matrix = np.array([[0., -2.], [-3., 0.]])
    denominators = [config["slm_width"], 256 * config["slm_height"] / 144]
    b = matrix * denominators
    carrier = np.array([0., config["fringe_slope_rad_per_pixel"] / (2 * np.pi)])
    center = np.array([32., 30.])
    result = {
        "kind": "camera_pixels_vs_slm_ramp_frequency_affine", "synthetic": False,
        "quality_pass": True, "operator_approved": True, "usable_for_hardware": True,
        "slm_shape": [config["slm_height"], config["slm_width"]],
        "camera_shape": list(camera_shape), "wavelength_nm": config["wavelength_nm"],
        "carrier_cycles_per_slm_pixel_xy": carrier.tolist(),
        "fit": {"camera_pixels_per_cycle_per_slm_pixel": b.tolist(), "camera_origin_at_zero_ramp_xy": (center - b @ carrier).tolist()},
        "model_sampling": {"grid_size": 256, "pupil_rows": 144, "camera_pixels_per_fft_bin_matrix": matrix.tolist()},
    }
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


class ProcessingTests(unittest.TestCase):
    def test_averaged_counts_and_burst_provenance_survive_crop_and_mat_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "processed"
            reference, expected = make_source(source)
            manifest = json.loads((source / "manifest.json").read_text())
            manifest["configuration"].update(frames_per_pattern=10, save_burst_frames=False)
            record = manifest["frames"][0]
            record.update(capture_start_time_ns=100, capture_end_time_ns=200,
                          averaging={"frame_count": 10, "method": "arithmetic_mean",
                                     "raw_frames_saved": False,
                                     "frames": [{"frame_id": i, "camera_timestamp_ns": i*10,
                                                 "capture_time_ns": 100+i, "raw_path": None}
                                                for i in range(10)]})
            mean = expected[1]["image"].astype(np.float32) + .5
            np.save(source_path(source, record["raw_path"]), mean)
            np.save(source / "reference/fringe_only.npy", reference.astype(np.float32) + .25)
            (source / "manifest.json").write_text(json.dumps(manifest))
            result = process_run(source, output, roi=(4, 3, 6, 7), exclude_indices={2, 3},
                                 preserve_camera_orientation=True, export_mat=True)
            with np.load(output / result["frames"][0]["pair_path"]) as pair:
                np.testing.assert_array_equal(pair["image"], mean[3:10, 4:10])
                self.assertEqual(pair["image"].dtype, np.float32)
            np.testing.assert_array_equal(loadmat(output / "neuws_mat/SLM_raw1.mat")["imsdata"],
                                          mean[3:10, 4:10])
            np.testing.assert_array_equal(np.load(output / "reference/fringe_only.npy"),
                                          reference.astype(np.float32) + .25)
            self.assertEqual(result["frames"][0]["source_averaging"], record["averaging"])
            self.assertEqual(result["frames"][0]["source_capture_start_time_ns"], 100)
            self.assertEqual(result["frames"][0]["source_capture_end_time_ns"], 200)

    def test_camera_roi_preserves_values_and_rejects_out_of_bounds(self):
        frame = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
        np.testing.assert_array_equal(crop_frame(frame, (4, 3, 6, 7)), frame[3:10, 4:10])
        self.assertEqual(crop_frame(frame, (4, 3, 6, 7)).dtype, np.uint16)
        for roi in ((-1, 0, 3, 3), (0, 0, 0, 3), (12, 10, 8, 8)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                crop_frame(frame, roi)

    def test_filtered_image_phase_pairs_keep_source_identity_and_full_slm_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            _, expected = make_source(source)
            original_manifest = (source / "manifest.json").read_bytes()
            result = process_run(
                source, output, roi=(4, 3, 6, 7), exclude_indices={2},
                include_full_phase=True, export_mat=True,
                preserve_camera_orientation=True,
            )
            self.assertEqual(result["frames_completed"], 2)
            self.assertEqual([r["source_pattern_index"] for r in result["frames"]], [1, 3])
            for output_index, source_index in enumerate((1, 3), 1):
                record = result["frames"][output_index - 1]
                self.assertEqual(record["source_display_time_ns"], 1700000000000000000 + source_index)
                self.assertEqual(record["source_capture_time_ns"], 1700000001000000000 + source_index)
                self.assertEqual(record["source_camera_timestamp_ns"], 900000000000 + source_index)
                with np.load(output / record["pair_path"], allow_pickle=False) as pair:
                    np.testing.assert_array_equal(pair["image"], expected[source_index]["image"][3:10, 4:10])
                    np.testing.assert_array_equal(pair["phase_modulation_rad"], expected[source_index]["modulation"])
                    np.testing.assert_array_equal(pair["phase_slm_rad"], expected[source_index]["phase"])
                    self.assertEqual(pair["phase_slm_rad"].shape, (12, 16))
                    self.assertEqual(pair["image"].shape, (7, 6))
                    self.assertEqual(int(pair["source_pattern_index"]), source_index)
                    self.assertEqual(str(pair["fringe_orientation"]), "horizontal")
                mat_image = loadmat(output / "neuws_mat" / f"SLM_raw{output_index}.mat")["imsdata"]
                mat_phase = loadmat(output / "neuws_mat" / f"SLM_sim{output_index}.mat")["proj_sim"]
                np.testing.assert_array_equal(mat_image, expected[source_index]["image"][3:10, 4:10])
                np.testing.assert_array_equal(mat_phase, expected[source_index]["modulation"])
            self.assertEqual(result["frames"][0]["full_phase_source"], "saved_full_phase")
            self.assertEqual(result["frames"][1]["full_phase_source"], "reconstructed_from_coefficients")
            self.assertEqual((source / "manifest.json").read_bytes(), original_manifest)
            self.assertEqual((output / "source_manifest.json").read_bytes(), original_manifest)
            self.assertEqual((output / "reference/fringe_only.npy").read_bytes(), (source / "reference/fringe_only.npy").read_bytes())
            self.assertEqual(result["image_coordinate_system"], "raw_camera_axes")
            self.assertTrue(result["preserve_camera_orientation"])
            self.assertTrue((output / "roi_preview.png").is_file())
            self.assertEqual(json.loads((output / "roi.json").read_text())["width"], 6)

    def test_preserve_camera_orientation_rejects_calibrated_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / "source")
            with self.assertRaisesRegex(ValueError, "crop-only"):
                process_run(root / "source", root / "output", roi=(4, 3, 6, 7),
                            calibration_file=root / "calibration.json", preserve_camera_orientation=True)
            self.assertFalse((root / "output").exists())

    def test_scale_then_crop_keeps_camera_axes_and_reference_phase_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            _, saved = make_source(source)
            yy, xx = np.indices((12, 16))
            frame = (100 + 40 * yy + 4 * xx).astype(np.uint16)
            np.save(source / "reference/fringe_only.npy", frame)
            for index in range(1, 4):
                np.save(source / f"raw/frame_{index:04d}.npy", frame + index * 1000)
            result = process_run(source, output, roi=(2, 0, 12, 12),
                                 preserve_camera_orientation=True, downsample_factor=2,
                                 crop_size=4, export_mat=True)
            # Independently compute an asymmetric ramp at the expected sample
            # locations: rows advance downward, columns rightward, at 2 pixels.
            expected = (100 + 40 * np.array([2.5, 4.5, 6.5, 8.5])[:, None]
                        + 4 * np.array([4.5, 6.5, 8.5, 10.5])[None, :]).astype(np.uint16)
            np.testing.assert_array_equal(np.load(output / "reference_processed.npy"), expected)
            for index in range(1, 4):
                with np.load(output / f"pairs/frame_{index:04d}.npz", allow_pickle=False) as pair:
                    np.testing.assert_array_equal(pair["image"], expected + index * 1000)
                    np.testing.assert_array_equal(pair["image_camera_crop"], (frame + index * 1000)[:, 2:14])
                    np.testing.assert_array_equal(pair["phase_modulation_rad"], saved[index]["modulation"])
                    self.assertEqual(pair["image"].dtype, np.uint16)
                np.testing.assert_array_equal(loadmat(output / f"neuws_mat/SLM_raw{index}.mat")["imsdata"], expected + index * 1000)
                np.testing.assert_array_equal(loadmat(output / f"neuws_mat/SLM_sim{index}.mat")["proj_sim"], saved[index]["modulation"])
            self.assertEqual(result["scale_and_crop"]["crop_center_raw_xy"], [8., 6.])
            self.assertEqual(result["scale_and_crop"]["effective_camera_pixels_per_output_pixel_xy"], [2., 2.])
            self.assertEqual(result["image_coordinate_system"], "raw_camera_axes")
            self.assertIsNone(result["calibrated_sampling"])

    def test_separate_camera_axis_scales_do_not_swap_or_flip_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_source(source)
            yy, xx = np.indices((12, 16))
            frame = (100 + 40 * yy + 4 * xx).astype(np.uint16)
            np.save(source / "reference/fringe_only.npy", frame)
            for index in range(1, 4):
                np.save(source / f"raw/frame_{index:04d}.npy", frame)
            result = process_run(source, output, roi=(0, 0, 16, 12),
                                 downsample_factor=(2., 3.), crop_size=4,
                                 preserve_camera_orientation=True)
            expected = (100 + 40 * np.array([1., 4., 7., 10.])[:, None]
                        + 4 * np.array([4.5, 6.5, 8.5, 10.5])[None, :]).astype(np.uint16)
            with np.load(output / "pairs/frame_0001.npz", allow_pickle=False) as pair:
                np.testing.assert_array_equal(pair["image"], expected)
            np.testing.assert_array_equal(np.load(output / "reference_processed.npy"), expected)
            self.assertEqual(result["scale_and_crop"]["downsample_factors_camera_xy"], [2., 3.])

    def test_scale_and_crop_rejects_invalid_parameters_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_source(source)
            cases = [
                {"downsample_factor": float("nan")}, {"downsample_factor": 0},
                {"downsample_factor": (2., 0.)}, {"downsample_factor": (2., 3., 4.)},
                {"downsample_factor": 2, "crop_size": 100},
                {"downsample_factor": 2, "crop_size": 0},
                {"downsample_factor": 2, "crop_size": 4, "crop_center": (0, 0)},
                {"downsample_factor": 2, "calibration_file": root / "calibration.json"},
                {"crop_center": (8, 6)},
            ]
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    process_run(source, output, roi=(0, 0, 16, 12), **kwargs)
                self.assertFalse(output.exists())

    def test_dark_frames_are_flagged_and_kept_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / "source")
            result = process_run(root / "source", root / "output", roi=(4, 3, 6, 7))
            self.assertEqual(result["frames_completed"], 3)
            self.assertEqual(result["frames"][1]["quality"], "low_signal")
            with np.load(root / "output" / result["frames"][1]["pair_path"], allow_pickle=False) as pair:
                self.assertTrue(np.all(pair["image"] == 10))
                self.assertNotIn("phase_slm_rad", pair.files)
                self.assertEqual(int(pair["source_pattern_index"]), 2)

    def test_interactive_selection_is_used_when_roi_is_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, _ = make_source(root / "source")
            with patch("process_measurements.select_roi", return_value=(4, 3, 6, 7)) as selector:
                result = process_run(root / "source", root / "output")
            selector.assert_called_once()
            np.testing.assert_array_equal(selector.call_args.args[0], reference)
            self.assertEqual(result["frames"][0]["image_shape"], [7, 6])

    def test_source_and_existing_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / "source")
            with self.assertRaisesRegex(ValueError, "separate output directory"):
                process_run(root / "source", root / "source" / "output", roi=(4, 3, 6, 7))
            output = root / "output"
            output.mkdir()
            (output / "keep.txt").write_text("preserve me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                process_run(root / "source", output, roi=(4, 3, 6, 7))
            self.assertEqual((output / "keep.txt").read_text(), "preserve me")
            with self.assertRaisesRegex(ValueError, "outside the acquisition"):
                source_path(root / "source", "..\\keep.npy")

    def test_calibrated_axes_counts_and_missing_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / "source")
            config = json.loads((root / "source/manifest.json").read_text())["configuration"]
            path = root / "calibration.json"
            make_calibration(path, config)
            sampler = FourierSampler(path, config, (60, 64), (144, 256), (0, 0, 64, 60))
            yy, xx = np.indices((60, 64))
            frame = (100 + 2 * xx + 5 * yy).astype(np.uint16)
            image = sampler.resample(frame)
            self.assertEqual(image.shape, (256, 256))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(sampler.valid.sum(), 0)
            self.assertLess(sampler.valid.mean(), 1)
            np.testing.assert_allclose(image[sampler.valid], (100 + 2 * sampler.camera_xy[0] + 5 * sampler.camera_xy[1])[sampler.valid], atol=1e-3)
            self.assertAlmostEqual(float(image[128, 129] - image[128, 128]), -15)
            self.assertAlmostEqual(float(image[129, 128] - image[128, 128]), -4)
            self.assertTrue(np.isnan(image[~sampler.valid]).all())
            preview = masked_preview(image, sampler.valid)
            np.testing.assert_array_equal(preview[:, :, 3] > 0, sampler.valid)
            modulation = np.arange(144 * 256, dtype=np.float32).reshape(144, 256)
            model = sampler.model_phase(modulation)
            np.testing.assert_array_equal(model[56:200], modulation)
            self.assertFalse(sampler.pupil_mask[:56].any())
            self.assertTrue(sampler.pupil_mask[56:200].all())

    def test_calibration_rejects_incompatible_or_unapproved_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / "source")
            config = json.loads((root / "source/manifest.json").read_text())["configuration"]
            path = root / "calibration.json"
            original = make_calibration(path, config)
            for key, value in (("synthetic", True), ("usable_for_hardware", False), ("operator_approved", False), ("quality_pass", False), ("wavelength_nm", 999), ("camera_shape", [20, 20]), ("slm_shape", [100, 100])):
                bad = dict(original, **{key: value})
                path.write_text(json.dumps(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    FourierSampler(path, config, (60, 64), (144, 256), (0, 0, 64, 60))
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "pupil"):
                FourierSampler(path, config, (60, 64), (160, 256), (0, 0, 64, 60))
            wrong_carrier = dict(config, fringe_slope_rad_per_pixel=0.4)
            with self.assertRaisesRegex(ValueError, "carrier"):
                FourierSampler(path, wrong_carrier, (60, 64), (144, 256), (0, 0, 64, 60))

    def test_calibrated_export_uses_selected_recapture_and_retains_quality_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_source(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            config = manifest["configuration"]
            yy, xx = np.indices((60, 64))
            frame = (100 + 2 * xx + 5 * yy).astype(np.uint16)
            np.save(source / "reference/fringe_only.npy", frame)
            for i in range(1, 4):
                np.save(source / f"raw/frame_{i:04d}.npy", frame + i)
            (source / "recaptures").mkdir()
            replacement = frame + 500
            np.save(source / "recaptures/replacement.npy", replacement)
            manifest["frames"][1]["raw_path"] = "recaptures\\replacement.npy"
            manifest_path.write_text(json.dumps(manifest))
            original_manifest = manifest_path.read_bytes()
            (source / "quality_report.json").write_text(json.dumps({"low_signal_candidate_indices": [3]}))
            calibration_path = root / "calibration.json"
            make_calibration(calibration_path, config)
            result = process_run(source, output, roi=(0, 0, 64, 60), calibration_file=calibration_path, export_mat=True)
            self.assertEqual(result["frames_completed"], 3)
            self.assertIn("source_low_signal_candidate", result["frames"][2]["quality_flags"])
            sampler = FourierSampler(calibration_path, config, frame.shape, (144, 256), (0, 0, 64, 60))
            with np.load(output / "pairs/frame_0002.npz", allow_pickle=False) as pair:
                np.testing.assert_array_equal(pair["image_camera_crop"], replacement)
                np.testing.assert_allclose(pair["image"], sampler.resample(replacement), equal_nan=True)
                np.testing.assert_array_equal(pair["phase_model_rad"][56:200], pair["phase_modulation_rad"])
                np.testing.assert_array_equal(pair["image_valid_mask"], sampler.valid)
                self.assertEqual(int(pair["source_pattern_index"]), 2)
            mat = loadmat(output / "neuws_mat/SLM_raw2.mat")
            np.testing.assert_array_equal(mat["valid_mask"], sampler.valid)
            self.assertTrue(np.isnan(mat["imsdata"][~sampler.valid]).all())
            self.assertEqual(loadmat(output / "neuws_mat/SLM_sim2.mat")["proj_sim"].shape, (256, 256))
            self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_missing_selected_recapture_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            make_source(source)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["frames"][1]["raw_path"] = "recaptures\\missing.npy"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileNotFoundError, "Missing source pattern 2"):
                process_run(source, output, roi=(4, 3, 6, 7))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
