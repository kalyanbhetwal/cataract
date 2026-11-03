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


def make_source(root):
    with patch("sys.stderr", new_callable=io.StringIO):
        config = config_from_args(build_parser().parse_args([
            "--dry-run", "--num-patterns", "3", "--slm-width", "16", "--slm-height", "12",
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


class ProcessingTests(unittest.TestCase):
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
            )
            self.assertEqual(result["frames_completed"], 2)
            self.assertEqual([r["source_pattern_index"] for r in result["frames"]], [1, 3])
            for output_index, source_index in enumerate((1, 3), 1):
                record = result["frames"][output_index - 1]
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
            self.assertTrue((output / "roi_preview.png").is_file())
            self.assertEqual(json.loads((output / "roi.json").read_text())["width"], 6)

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


if __name__ == "__main__":
    unittest.main()
