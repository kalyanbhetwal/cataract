import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from camera_smoke_test import make_preview_u8

from acquire_slm_camera import (
    ExperimentConfig,
    HoloeyeSLM,
    NeuWSPatternGenerator,
    PlayerOneCamera,
    build_parser,
    config_from_args,
    main,
    noll_indices,
    open_fringe_preview,
    phase_to_u8,
    process_neuws_frame,
    resize_nearest,
    run,
    zernike_basis,
)


def test_config(seed=7):
    return ExperimentConfig(
        num_patterns=2,
        seed=seed,
        zernike_modes=15,
        coefficient_std_rad=5.0,
        fringe_slope_rad_per_pixel=4.0 * math.pi / 3.0,
        fringe_orientation="vertical",
        slm_height=108,
        slm_width=192,
        wavelength_nm=532.0,
        settle_ms=0.0,
        exposure_ms=100.0,
        gain=0.0,
        gamma=1.0,
        camera_timeout_ms=3000,
        discard_frames=1,
        pixel_format="auto",
        camera_backend="playerone",
        camera_serial=None,
        playerone_sdk_dir=None,
        slm_preselect=None,
        heds_examples_dir=None,
        heds_api_major=4,
        heds_api_minor=2,
        magnification=3.57,
        crop_size=32,
        crop_center_x=None,
        crop_center_y=None,
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

    def test_vertical_fringe_lines_vary_along_x(self):
        generator = NeuWSPatternGenerator(
            replace(
                test_config(),
                fringe_orientation="vertical",
                fringe_slope_rad_per_pixel=math.pi / 2.0,
            )
        )
        np.testing.assert_array_equal(generator._fringe[0], generator._fringe[-1])
        self.assertNotEqual(generator._fringe[0, 0], generator._fringe[0, 1])

    def test_horizontal_fringe_lines_vary_along_y(self):
        generator = NeuWSPatternGenerator(
            replace(
                test_config(),
                fringe_orientation="horizontal",
                fringe_slope_rad_per_pixel=math.pi / 2.0,
            )
        )
        np.testing.assert_array_equal(generator._fringe[:, 0], generator._fringe[:, -1])
        self.assertNotEqual(generator._fringe[0, 0], generator._fringe[1, 0])

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

    def test_processing_crops_around_steered_order_coordinate(self):
        frame = np.zeros((100, 100), dtype=np.uint16)
        frame[25, 75] = 65535
        processed = process_neuws_frame(
            frame,
            magnification=1.0,
            crop_size=20,
            crop_center_x=75,
            crop_center_y=25,
        )
        self.assertEqual(processed.shape, (20, 20))
        self.assertEqual(int(processed[10, 10]), 65535)

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

    def test_explicit_playerone_sdk_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk_dir = Path(directory)
            (sdk_dir / "python").mkdir()
            (sdk_dir / "lib" / "x64").mkdir(parents=True)
            (sdk_dir / "python" / "pyPOACamera.py").touch()
            (sdk_dir / "lib" / "x64" / "PlayerOneCamera.dll").touch()
            config = replace(test_config(), playerone_sdk_dir=directory)
            camera = PlayerOneCamera(config)
            python_dir, dll_dir = camera._locate_sdk()
            self.assertEqual(python_dir, (sdk_dir / "python").resolve())
            self.assertEqual(dll_dir, (sdk_dir / "lib" / "x64").resolve())

    def test_playerone_snap_capture_returns_two_dimensional_raw16(self):
        class Errors(Enum):
            POA_OK = 0

        class States(Enum):
            STATE_OPENED = 1

        fake_module = SimpleNamespace(
            POAErrors=Errors,
            POACameraState=States,
            StartExposure=lambda camera_id, single: Errors.POA_OK,
            GetCameraState=lambda camera_id: (Errors.POA_OK, States.STATE_OPENED),
            ImageReady=lambda camera_id: (Errors.POA_OK, True),
            GetImage=lambda camera_id, timeout: (
                Errors.POA_OK,
                np.arange(12, dtype=np.uint16).reshape(3, 4, 1),
            ),
        )
        camera = PlayerOneCamera(replace(test_config(), discard_frames=0))
        camera.module = fake_module
        camera.camera_id = 7
        camera.properties = SimpleNamespace(cameraModelName=b"TestCam", SN=b"SN123")
        frame, metadata = camera.capture()
        self.assertEqual(frame.shape, (3, 4))
        self.assertEqual(frame.dtype, np.uint16)
        self.assertEqual(metadata["pixel_format"], "RAW16")
        self.assertEqual(metadata["camera_serial"], "SN123")

    def test_default_camera_backend_is_pyspin(self):
        parser = build_parser()
        config = config_from_args(parser.parse_args(["--dry-run"]))
        self.assertEqual(config.camera_backend, "pyspin")
        self.assertEqual(config.fringe_orientation, "vertical")
        self.assertFalse(config.capture_fringe_first)
        self.assertFalse(config.confirm_fringe_preview)
        self.assertIsNone(config.crop_center_x)
        self.assertIsNone(config.crop_center_y)
        self.assertEqual(config.discard_frames, 1)
        self.assertEqual(config.pixel_format, "auto")

    def test_playerone_default_discards_no_snap_frames(self):
        parser = build_parser()
        config = config_from_args(
            parser.parse_args(["--dry-run", "--camera-backend", "playerone"])
        )
        self.assertEqual(config.discard_frames, 0)

    def test_signed_fringe_period_option_sets_ramp_slope(self):
        parser = build_parser()
        for period in (20, -20, 2):
            with self.subTest(period=period):
                config = config_from_args(
                    parser.parse_args(["--fringe-period-pixels", str(period)])
                )
                self.assertAlmostEqual(config.fringe_slope_rad_per_pixel, 2 * math.pi / period)

    def test_fringe_spacing_rejects_invalid_or_conflicting_values(self):
        parser = build_parser()
        for period in ("0", "1", "-1", "nan", "inf", "not-a-number"):
            with self.subTest(period=period), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    parser.parse_args([f"--fringe-period-pixels={period}"])
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            parser.parse_args(
                ["--fringe-period-pixels", "20", "--fringe-slope-rad-per-pixel", "1"]
            )
        with self.assertRaisesRegex(ValueError, "Fringe slope must be finite"):
            config_from_args(parser.parse_args(["--fringe-slope-rad-per-pixel", "nan"]))

    def test_camera_png_preview_scaling(self):
        frame = np.arange(100, dtype=np.uint16).reshape(10, 10)
        preview, low, high = make_preview_u8(frame)
        self.assertEqual(preview.shape, frame.shape)
        self.assertEqual(preview.dtype, np.uint8)
        self.assertAlmostEqual(low, 0.99)
        self.assertAlmostEqual(high, 98.01)
        self.assertEqual(int(preview[0, 0]), 0)
        self.assertEqual(int(preview[-1, -1]), 255)


class AcquisitionSequenceTests(unittest.TestCase):
    def test_preview_confirmation_enables_fringe_reference(self):
        parser = build_parser()
        config = config_from_args(parser.parse_args(["--confirm-fringe-preview"]))
        self.assertTrue(config.capture_fringe_first)
        self.assertTrue(config.confirm_fringe_preview)

    def test_fringe_reference_then_random_phase_preserves_full_camera_frames(self):
        parser = build_parser()
        config = config_from_args(
            parser.parse_args(
                [
                    "--dry-run", "--capture-fringe-first", "--num-patterns", "2",
                    "--fringe-orientation", "horizontal", "--settle-ms", "300",
                    "--no-neuws-processing", "--output-format", "npy",
                    "--save-full-patterns",
                ]
            )
        )
        config = replace(config, slm_height=108, slm_width=192)
        # Two separated signals stand in for both orders in one camera frame.
        full_frame = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
        full_frame[6, 8] = 60000
        full_frame[2, 8] = 30000
        slm, camera, events = Mock(), Mock(), Mock()
        camera.capture.side_effect = [
            (full_frame.copy(), {"frame_id": number}) for number in range(1, 4)
        ]
        events.attach_mock(slm.show, "display")
        events.attach_mock(camera.capture, "capture")
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with (
                patch("acquire_slm_camera.SimulatedSLM", return_value=slm),
                patch("acquire_slm_camera.SimulatedCamera", return_value=camera),
                patch("acquire_slm_camera.time.sleep") as settle,
            ):
                events.attach_mock(settle, "settle")
                run(config, output_dir)

            self.assertEqual(
                [call[0] for call in events.mock_calls],
                ["display", "settle", "capture"] * 3,
            )
            self.assertEqual([call.args[0] for call in settle.call_args_list], [0.3] * 3)
            expected_generator = NeuWSPatternGenerator(config)
            reference_phase = slm.show.call_args_list[0].args[0]
            np.testing.assert_array_equal(reference_phase, expected_generator.fringe_only())
            np.testing.assert_array_equal(reference_phase[:, 0], reference_phase[:, -1])
            self.assertNotEqual(reference_phase[0, 0], reference_phase[1, 0])

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["frames_completed"], 2)
            self.assertEqual(len(manifest["frames"]), 2)
            self.assertTrue(manifest["configuration"]["capture_fringe_first"])
            self.assertFalse((output_dir / "processed").exists())
            reference = manifest["fringe_reference"]
            self.assertEqual(reference["frame_id"], 1)
            self.assertTrue((output_dir / reference["preview_path"]).is_file())
            np.testing.assert_array_equal(
                np.load(output_dir / reference["raw_path"]), full_frame
            )
            np.testing.assert_array_equal(
                np.load(output_dir / reference["full_pattern_path"]), reference_phase
            )
            for index, record in enumerate(manifest["frames"]):
                expected_phase, _, expected_coefficients = expected_generator.next()
                np.testing.assert_array_equal(
                    slm.show.call_args_list[index + 1].args[0], expected_phase
                )
                np.testing.assert_array_equal(
                    record["zernike_coefficients_rad"], expected_coefficients
                )
                np.testing.assert_array_equal(
                    np.load(output_dir / record["raw_path"]), full_frame
                )
                self.assertEqual(record["pattern_index"], index + 1)
                self.assertEqual(record["frame_id"], index + 2)
                self.assertIsNone(record["processed_path"])
                self.assertTrue((output_dir / record["preview_path"]).is_file())
            slm.open.assert_called_once()
            camera.open.assert_called_once()
            camera.close.assert_called_once()
            slm.close.assert_called_once()

    def test_fringe_reference_survives_later_capture_failure(self):
        config = replace(test_config(), capture_fringe_first=True, neuws_processing=False)
        slm, camera = Mock(), Mock()
        camera.capture.side_effect = [
            (np.zeros((12, 16), dtype=np.uint16), {"frame_id": 1}),
            RuntimeError("capture failed"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with (
                patch("acquire_slm_camera.SimulatedSLM", return_value=slm),
                patch("acquire_slm_camera.SimulatedCamera", return_value=camera),
                self.assertRaisesRegex(RuntimeError, "capture failed"),
            ):
                run(config, output_dir)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["frames_completed"], 0)
            self.assertEqual(manifest["fringe_reference"]["frame_id"], 1)
            self.assertTrue(
                (output_dir / manifest["fringe_reference"]["raw_path"]).is_file()
            )
            camera.close.assert_called_once()
            slm.close.assert_called_once()

    def test_preview_feedback_gates_random_phase_and_keeps_fringe_displayed(self):
        cases = [
            ("approve_after_invalid_answers", ["", "maybe", " y "], True, "approved"),
            ("decline", ["n"], False, "declined"),
            ("end_of_input", [EOFError()], False, "no_input"),
            ("unavailable_input", [OSError("no terminal")], False, "no_input"),
        ]
        for label, answers, expected_complete, expected_status in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                # Review alone must request a reference, even for direct Python callers.
                config = replace(
                    test_config(), num_patterns=1, confirm_fringe_preview=True,
                    neuws_processing=False,
                )
                output_dir = Path(directory) / "run"
                slm, camera = Mock(), Mock()
                camera.capture.side_effect = [
                    (np.zeros((12, 16), dtype=np.uint16), {"frame_id": number})
                    for number in (1, 2)
                ]
                answer_iterator = iter(answers)

                def feedback(_prompt):
                    # No random phase may be shown or captured before an explicit Y.
                    self.assertEqual(slm.show.call_count, 1)
                    self.assertEqual(camera.capture.call_count, 1)
                    slm.close.assert_not_called()
                    camera.close.assert_not_called()
                    manifest = json.loads((output_dir / "manifest.json").read_text())
                    self.assertEqual(manifest["frames_completed"], 0)
                    self.assertEqual(
                        manifest["fringe_reference"]["preview_feedback"]["status"],
                        "pending",
                    )
                    self.assertTrue(
                        (output_dir / manifest["fringe_reference"]["preview_path"]).is_file()
                    )
                    response = next(answer_iterator)
                    if isinstance(response, Exception):
                        raise response
                    return response

                with (
                    patch("acquire_slm_camera.SimulatedSLM", return_value=slm),
                    patch("acquire_slm_camera.SimulatedCamera", return_value=camera),
                    patch("acquire_slm_camera.open_fringe_preview") as open_preview,
                    patch("builtins.input", side_effect=feedback) as prompt,
                ):
                    completed = run(config, output_dir)

                self.assertEqual(completed, expected_complete)
                self.assertEqual(prompt.call_count, len(answers))
                open_preview.assert_called_once_with(output_dir / "reference" / "fringe_only.png")
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(
                    manifest["fringe_reference"]["preview_feedback"]["status"], expected_status
                )
                self.assertEqual(manifest["frames_completed"], int(expected_complete))
                self.assertEqual(slm.show.call_count, 1 + int(expected_complete))
                self.assertEqual(camera.capture.call_count, 1 + int(expected_complete))
                camera.close.assert_called_once()
                slm.close.assert_called_once()

    def test_cancelled_run_is_reported_as_stopped(self):
        with (
            patch("acquire_slm_camera.run", return_value=False),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            result = main(["--dry-run", "--confirm-fringe-preview"])
        self.assertEqual(result, 0)
        self.assertIn("Acquisition stopped before random phase capture:", output.getvalue())
        self.assertNotIn("Acquisition complete:", output.getvalue())

    def test_steering_adjustments_preserve_previews_and_use_accepted_fringe(self):
        config = replace(
            test_config(), capture_fringe_first=True, confirm_fringe_preview=True,
            neuws_processing=False, save_full_patterns=True,
            fringe_orientation="horizontal", settle_ms=300,
        )
        period_config = replace(config, fringe_slope_rad_per_pixel=2 * math.pi / 20)
        reversed_config = replace(period_config, fringe_slope_rad_per_pixel=-2 * math.pi / 20)
        vertical_config = replace(reversed_config, fringe_orientation="vertical")
        accepted_config = replace(vertical_config, fringe_orientation="horizontal")
        expected_configs = [
            config, period_config, reversed_config, vertical_config, accepted_config,
            accepted_config,
        ]
        answers = ["P", "P 0", "P 1", "P nan", "P twenty", "P 20", "R", "V", "H", "C", "Y"]
        expected_counts = [1] * 6 + [2, 3, 4, 5, 6]
        feedback_iterator = iter(zip(expected_counts, answers))
        total_captures = len(expected_configs) + config.num_patterns
        slm, camera = Mock(), Mock()
        camera.capture.side_effect = [
            (np.full((12, 16), number, dtype=np.uint16), {"frame_id": number})
            for number in range(1, total_captures + 1)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"

            def feedback(_prompt):
                reference_count, answer = next(feedback_iterator)
                self.assertEqual(slm.show.call_count, reference_count)
                self.assertEqual(camera.capture.call_count, reference_count)
                slm.close.assert_not_called()
                camera.close.assert_not_called()
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["frames_completed"], 0)
                self.assertEqual(len(manifest["fringe_references"]), reference_count)
                self.assertEqual(
                    manifest["fringe_reference"]["preview_feedback"]["status"], "pending"
                )
                return answer

            with (
                patch("acquire_slm_camera.SimulatedSLM", return_value=slm),
                patch("acquire_slm_camera.SimulatedCamera", return_value=camera),
                patch("acquire_slm_camera.time.sleep") as settle,
                patch("acquire_slm_camera.open_fringe_preview") as open_preview,
                patch("builtins.input", side_effect=feedback) as prompt,
            ):
                self.assertTrue(run(config, output_dir))

            self.assertEqual(prompt.call_count, len(answers))
            self.assertEqual(open_preview.call_count, len(expected_configs))
            self.assertEqual(
                [call.args[0] for call in settle.call_args_list], [0.3] * total_captures
            )
            manifest = json.loads((output_dir / "manifest.json").read_text())
            references = manifest["fringe_references"]
            self.assertEqual(len(references), len(expected_configs))
            self.assertEqual(manifest["fringe_reference"], references[-1])
            self.assertEqual(
                [r["preview_feedback"]["status"] for r in references],
                ["adjusted"] * 4 + ["recaptured", "approved"],
            )
            self.assertEqual(len({r["raw_path"] for r in references}), len(expected_configs))
            for index, (reference, reference_config) in enumerate(zip(references, expected_configs)):
                expected_phase = NeuWSPatternGenerator(reference_config).fringe_only()
                np.testing.assert_array_equal(slm.show.call_args_list[index].args[0], expected_phase)
                np.testing.assert_array_equal(
                    np.load(output_dir / reference["full_pattern_path"]), expected_phase
                )
                np.testing.assert_array_equal(
                    np.load(output_dir / reference["raw_path"]),
                    np.full((12, 16), index + 1, dtype=np.uint16),
                )
                self.assertTrue((output_dir / reference["preview_path"]).is_file())
                self.assertEqual(reference["fringe_orientation"], reference_config.fringe_orientation)
                self.assertAlmostEqual(
                    reference["fringe_slope_rad_per_pixel"], reference_config.fringe_slope_rad_per_pixel
                )
            self.assertEqual(manifest["frames_completed"], config.num_patterns)
            self.assertAlmostEqual(
                manifest["configuration"]["fringe_slope_rad_per_pixel"],
                accepted_config.fringe_slope_rad_per_pixel,
            )
            self.assertEqual(manifest["configuration"]["fringe_orientation"], "horizontal")
            expected_generator = NeuWSPatternGenerator(accepted_config)
            for index, record in enumerate(manifest["frames"]):
                expected_phase, _, coefficients = expected_generator.next()
                np.testing.assert_array_equal(
                    slm.show.call_args_list[len(expected_configs) + index].args[0], expected_phase
                )
                np.testing.assert_array_equal(record["zernike_coefficients_rad"], coefficients)
            camera.close.assert_called_once()
            slm.close.assert_called_once()

    def test_decline_or_missing_input_after_adjustment_stops_before_random_phase(self):
        for response, status in (("N", "declined"), (EOFError(), "no_input")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                config = replace(test_config(), confirm_fringe_preview=True, neuws_processing=False)
                slm, camera = Mock(), Mock()
                camera.capture.return_value = (np.zeros((12, 16), dtype=np.uint16), {"frame_id": 1})
                output_dir = Path(directory) / "run"
                with (
                    patch("acquire_slm_camera.SimulatedSLM", return_value=slm),
                    patch("acquire_slm_camera.SimulatedCamera", return_value=camera),
                    patch("acquire_slm_camera.open_fringe_preview"),
                    patch("builtins.input", side_effect=["P -16", response]),
                ):
                    self.assertFalse(run(config, output_dir))
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["frames_completed"], 0)
                self.assertEqual(len(manifest["fringe_references"]), 2)
                self.assertEqual(manifest["fringe_reference"]["preview_feedback"]["status"], status)
                self.assertEqual(slm.show.call_count, 2)
                self.assertEqual(camera.capture.call_count, 2)
                self.assertFalse(any((output_dir / "raw").iterdir()))
                camera.close.assert_called_once()
                slm.close.assert_called_once()

    def test_windows_preview_opens_saved_image_path(self):
        preview_path = Path("data") / "run with spaces" / "reference" / "fringe_only.png"
        with (
            patch("acquire_slm_camera.sys.platform", "win32"),
            patch("acquire_slm_camera.os.startfile", create=True) as open_image,
            patch("acquire_slm_camera.subprocess.Popen") as launch_process,
        ):
            open_fringe_preview(preview_path)
        open_image.assert_called_once_with(str(preview_path.resolve()))
        launch_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
