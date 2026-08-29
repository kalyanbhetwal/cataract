import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from acquire_slm_camera import (
    ExperimentConfig,
    HoloeyeSLM,
    NeuWSPatternGenerator,
    noll_indices,
    phase_to_u8,
    process_neuws_frame,
    resize_nearest,
    zernike_basis,
)


def test_config(seed=7):
    return ExperimentConfig(
        num_patterns=2,
        seed=seed,
        zernike_modes=15,
        coefficient_std_rad=5.0,
        fringe_slope_rad_per_pixel=4.0 * math.pi / 3.0,
        slm_height=108,
        slm_width=192,
        settle_ms=0.0,
        exposure_ms=100.0,
        gain_db=0.0,
        gamma=1.0,
        camera_timeout_ms=3000,
        discard_frames=1,
        pixel_format="Mono16",
        camera_serial=None,
        slm_preselect=None,
        heds_examples_dir=None,
        heds_api_major=4,
        heds_api_minor=2,
        magnification=3.57,
        crop_size=32,
        neuws_processing=True,
        output_format="npy",
        save_full_patterns=False,
        require_16_bit=True,
        dry_run=True,
    )


class PatternTests(unittest.TestCase):
    def test_first_fifteen_indices(self):
        self.assertEqual(len(noll_indices(15)), 15)
        self.assertEqual(noll_indices(6), [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)])

    def test_basis_shape_and_outside_disk(self):
        basis = zernike_basis(32, 15)
        self.assertEqual(basis.shape, (15, 32, 32))
        np.testing.assert_array_equal(basis[:, 0, 0], 0.0)

    def test_pattern_is_reproducible_and_wrapped(self):
        first = NeuWSPatternGenerator(test_config(seed=11)).next()
        second = NeuWSPatternGenerator(test_config(seed=11)).next()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[0].shape, (108, 192))
        self.assertEqual(first[1].shape, (144, 256))
        self.assertGreaterEqual(float(first[0].min()), 0.0)
        self.assertLess(float(first[0].max()), 2.0 * math.pi)

    def test_quantization_uses_full_byte_period(self):
        phases = np.array([0.0, math.pi, 2.0 * math.pi - 1e-7])
        np.testing.assert_array_equal(phase_to_u8(phases), [0, 128, 255])

    def test_nearest_neighbor_preserves_blocks(self):
        source = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        np.testing.assert_array_equal(resize_nearest(source, (4, 4)), expected)

    def test_processing_returns_center_crop_and_dtype(self):
        frame = np.arange(120 * 160, dtype=np.uint16).reshape(120, 160)
        processed = process_neuws_frame(frame, magnification=2.0, crop_size=32)
        self.assertEqual(processed.shape, (32, 32))
        self.assertEqual(processed.dtype, np.uint16)

    def test_explicit_heds_examples_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            examples_dir = Path(directory)
            (examples_dir / "HEDS").mkdir()
            config = replace(test_config(), heds_examples_dir=directory)
            slm = HoloeyeSLM(config)
            original_path = list(sys.path)
            try:
                detected = slm._prepare_heds_import_path()
                self.assertEqual(detected, examples_dir)
                self.assertEqual(sys.path[0], str(examples_dir.resolve()))
            finally:
                sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
