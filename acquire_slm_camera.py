#!/usr/bin/env python3
"""Display NeuWS-style phase patterns and acquire one camera frame per pattern.

The pattern recipe follows pages 4-5 of sciadv.adg4671_sm.pdf. Hardware mode
defaults to the HOLOEYE SLM Display SDK and FLIR/Point Grey Spinnaker (PySpin).
An optional Player One backend is available for later use. Use --dry-run to
validate pattern generation and output without either device.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class ExperimentConfig:
    num_patterns: int
    seed: int
    zernike_modes: int
    coefficient_std_rad: float
    fringe_slope_rad_per_pixel: float
    slm_height: int
    slm_width: int
    wavelength_nm: float
    settle_ms: float
    exposure_ms: float
    gain: float
    gamma: float
    camera_timeout_ms: int
    discard_frames: int
    pixel_format: str
    camera_backend: str
    camera_serial: Optional[str]
    playerone_sdk_dir: Optional[str]
    slm_preselect: Optional[str]
    heds_examples_dir: Optional[str]
    heds_api_major: int
    heds_api_minor: int
    magnification: float
    crop_size: int
    neuws_processing: bool
    output_format: str
    save_full_patterns: bool
    require_16_bit: bool
    dry_run: bool


def noll_indices(count: int) -> list[tuple[int, int]]:
    """Return the first ``count`` (n, m) Zernike pairs in Noll-like order."""
    if count < 1:
        raise ValueError("Zernike mode count must be positive")
    result: list[tuple[int, int]] = []
    radial_order = 0
    while len(result) < count:
        for azimuthal_order in range(-radial_order, radial_order + 1, 2):
            result.append((radial_order, azimuthal_order))
            if len(result) == count:
                return result
        radial_order += 1
    return result


def _zernike_radial(n: int, m_abs: int, radius: np.ndarray) -> np.ndarray:
    radial = np.zeros_like(radius, dtype=np.float64)
    for k in range((n - m_abs) // 2 + 1):
        coefficient = ((-1) ** k * math.factorial(n - k)) / (
            math.factorial(k)
            * math.factorial((n + m_abs) // 2 - k)
            * math.factorial((n - m_abs) // 2 - k)
        )
        radial += coefficient * radius ** (n - 2 * k)
    return radial


def zernike_basis(size: int = 256, modes: int = 15) -> np.ndarray:
    """Create RMS-normalized Zernike modes on a unit disk."""
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x_grid, y_grid)
    angle = np.arctan2(y_grid, x_grid)
    aperture = radius <= 1.0

    basis = np.zeros((modes, size, size), dtype=np.float64)
    for index, (n, m) in enumerate(noll_indices(modes)):
        radial = _zernike_radial(n, abs(m), radius)
        if m < 0:
            mode = math.sqrt(2.0 * (n + 1)) * radial * np.sin(abs(m) * angle)
        elif m > 0:
            mode = math.sqrt(2.0 * (n + 1)) * radial * np.cos(m * angle)
        else:
            mode = math.sqrt(n + 1) * radial
        basis[index, aperture] = mode[aperture]
    return basis


def resize_nearest(array: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize matching the paper's upsampling step."""
    output_height, output_width = output_shape
    row_indices = np.minimum(
        (np.arange(output_height) * array.shape[0] / output_height).astype(int),
        array.shape[0] - 1,
    )
    column_indices = np.minimum(
        (np.arange(output_width) * array.shape[1] / output_width).astype(int),
        array.shape[1] - 1,
    )
    return array[row_indices[:, None], column_indices[None, :]]


def resize_bilinear(array: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D image with center-aligned bilinear interpolation."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D camera frame, got shape {array.shape}")
    output_height, output_width = output_shape
    source_height, source_width = array.shape
    source_y = np.clip(
        (np.arange(output_height) + 0.5) * source_height / output_height - 0.5,
        0,
        source_height - 1,
    )
    source_x = np.clip(
        (np.arange(output_width) + 0.5) * source_width / output_width - 0.5,
        0,
        source_width - 1,
    )
    y0 = np.floor(source_y).astype(int)
    x0 = np.floor(source_x).astype(int)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = source_y - y0
    wx = source_x - x0

    top = (1.0 - wx) * array[y0[:, None], x0] + wx * array[y0[:, None], x1]
    bottom = (1.0 - wx) * array[y1[:, None], x0] + wx * array[y1[:, None], x1]
    resized = (1.0 - wy[:, None]) * top + wy[:, None] * bottom
    if np.issubdtype(array.dtype, np.integer):
        limits = np.iinfo(array.dtype)
        resized = np.clip(np.rint(resized), limits.min, limits.max).astype(array.dtype)
    return resized


def process_neuws_frame(frame: np.ndarray, magnification: float, crop_size: int) -> np.ndarray:
    """Apply the paper's magnification correction and centered 256-pixel crop."""
    if magnification <= 0:
        raise ValueError("Magnification must be positive")
    scaled_shape = (
        max(1, int(round(frame.shape[0] / magnification))),
        max(1, int(round(frame.shape[1] / magnification))),
    )
    scaled = resize_bilinear(frame, scaled_shape)
    if min(scaled.shape) < crop_size:
        raise ValueError(
            f"Scaled frame {scaled.shape} is smaller than requested {crop_size}x{crop_size} crop"
        )
    row_start = (scaled.shape[0] - crop_size) // 2
    column_start = (scaled.shape[1] - crop_size) // 2
    return scaled[
        row_start : row_start + crop_size,
        column_start : column_start + crop_size,
    ].copy()


def make_preview_u8(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale the 1st-99th percentile range into a viewable 8-bit preview."""
    preview_source = frame[:, :, 0] if frame.ndim == 3 and frame.shape[2] == 1 else frame
    if preview_source.ndim not in (2, 3):
        raise ValueError(f"Cannot create a PNG preview from shape {frame.shape}")
    if preview_source.ndim == 3 and preview_source.shape[2] not in (3, 4):
        raise ValueError(f"Cannot create a PNG preview from shape {frame.shape}")

    values = np.asarray(preview_source, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot create a PNG preview from an image without finite values")
    low, high = (float(value) for value in np.percentile(finite_values, (1.0, 99.0)))
    if high <= low:
        preview = np.zeros(preview_source.shape, dtype=np.uint8)
    else:
        preview = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(
            np.uint8
        )
    return preview, low, high


class NeuWSPatternGenerator:
    """Generate deterministic paper-style random Zernike phase patterns."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._basis = zernike_basis(256, config.zernike_modes)
        self._crop_start = (256 - 144) // 2
        x = np.arange(config.slm_width, dtype=np.float64)
        fringe_row = np.mod(config.fringe_slope_rad_per_pixel * x, TWO_PI)
        self._fringe = np.broadcast_to(
            fringe_row, (config.slm_height, config.slm_width)
        )

    def next(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coefficients = self._rng.normal(
            0.0,
            self.config.coefficient_std_rad,
            self.config.zernike_modes,
        )
        phase_256 = np.tensordot(coefficients, self._basis, axes=(0, 0))
        phase_144 = phase_256[
            self._crop_start : self._crop_start + 144,
            :,
        ]
        modulation = resize_nearest(
            phase_144,
            (self.config.slm_height, self.config.slm_width),
        )
        displayed_phase = np.mod(modulation + self._fringe, TWO_PI).astype(np.float32)
        # float32 can round a value just below 2*pi back up to 2*pi.
        maximum_phase = np.nextafter(np.float32(TWO_PI), np.float32(0.0))
        np.minimum(displayed_phase, maximum_phase, out=displayed_phase)
        return displayed_phase, phase_144.astype(np.float32), coefficients


def phase_to_u8(phase_rad: np.ndarray) -> np.ndarray:
    """Quantize wrapped radians to one 8-bit 2-pi phase period."""
    wrapped = np.mod(phase_rad, TWO_PI)
    return np.floor(wrapped * (256.0 / TWO_PI)).astype(np.uint8)


class SimulatedSLM:
    def __init__(self, _: ExperimentConfig):
        self.last_phase: Optional[np.ndarray] = None

    def open(self) -> None:
        print("[dry-run] Simulated SLM opened")

    def show(self, phase_rad: np.ndarray) -> None:
        self.last_phase = phase_rad

    def close(self) -> None:
        print("[dry-run] Simulated SLM closed")


class HoloeyeSLM:
    """Compatibility adapter for HOLOEYE Display SDK 4.x and legacy 3.x."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backend: Optional[str] = None
        self.module: Any = None
        self.slm: Any = None
        self.sdk_initialized = False

    def _check_error(self, error: Any, operation: str) -> None:
        if error is None:
            return
        try:
            code = int(error)
        except (TypeError, ValueError):
            code = 0 if not error else -1
        if code != 0:
            description = str(error)
            if (
                self.module is not None
                and hasattr(self.module, "SDK")
                and hasattr(self.module.SDK, "ErrorString")
            ):
                try:
                    description = str(self.module.SDK.ErrorString(error))
                except Exception:
                    pass
            raise RuntimeError(
                f"HOLOEYE {operation} failed with code {error}: {description}"
            )

    def _prepare_heds_import_path(self) -> Optional[Path]:
        """Locate the SDK v4 Python Convenience API's examples/HEDS folder."""
        candidates: list[Path] = []
        if self.config.heds_examples_dir:
            explicit = Path(self.config.heds_examples_dir).expanduser()
            if not (explicit / "HEDS").is_dir():
                raise RuntimeError(
                    f"--heds-examples-dir must contain the HEDS folder: {explicit}"
                )
            candidates.append(explicit)
        else:
            environment_path = os.environ.get("HEDS_EXAMPLES_DIR")
            if environment_path:
                candidates.append(Path(environment_path).expanduser())

            # The manual also permits copying HEDS beside the user script.
            candidates.append(Path(__file__).resolve().parent)

            program_files = os.environ.get("ProgramFiles")
            if program_files:
                installation_root = Path(program_files) / "HOLOEYE Photonics"
                if installation_root.is_dir():
                    candidates.extend(
                        sorted(
                            installation_root.glob(
                                "SLM Display SDK (Python) v*/examples"
                            ),
                            reverse=True,
                        )
                    )

        for candidate in candidates:
            if (candidate / "HEDS").is_dir():
                candidate_text = str(candidate.resolve())
                if candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)
                return candidate
        return None

    def open(self) -> None:
        detected_examples_dir = self._prepare_heds_import_path()
        heds_import_error: Optional[Exception] = None
        try:
            import HEDS  # type: ignore
        except (ImportError, AssertionError) as exc:
            heds_import_error = exc
            HEDS = None
        if HEDS is not None:
            self.module = HEDS
            self.backend = "HEDS-4.x"
            error = HEDS.SDK.Init(
                self.config.heds_api_major,
                self.config.heds_api_minor,
            )
            self._check_error(error, "SDK initialization")
            self.sdk_initialized = True
            if self.config.slm_preselect:
                self.slm = HEDS.SLM.Init(self.config.slm_preselect)
            else:
                self.slm = HEDS.SLM.Init()
            self._check_error(self.slm.errorCode(), "SLM initialization")
            actual_size = (int(self.slm.height_px()), int(self.slm.width_px()))
            expected_size = (self.config.slm_height, self.config.slm_width)
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Connected SLM is {actual_size[1]}x{actual_size[0]}, but the "
                    f"requested pattern is {expected_size[1]}x{expected_size[0]}"
                )
            self._check_error(
                self.slm.setWavelength(self.config.wavelength_nm),
                "wavelength configuration",
            )
            if detected_examples_dir:
                print(f"Loaded HEDS from {detected_examples_dir}")
        else:
            try:
                from holoeye import slmdisplaysdk  # type: ignore
            except ImportError as exc:
                detail = (
                    f" Original HEDS import error: {heds_import_error}."
                    if heds_import_error
                    else ""
                )
                raise RuntimeError(
                    "HOLOEYE SDK Python module not found. For SDK v4, copy the "
                    "examples/HEDS folder beside this script or pass "
                    f"--heds-examples-dir with the SDK examples directory.{detail}"
                ) from exc
            self.module = slmdisplaysdk
            self.slm = slmdisplaysdk.SLMDisplay()
            self._check_error(self.slm.open(), "SLM open")
            self.backend = "slmdisplaysdk-3.x"
        print(f"HOLOEYE SLM opened through {self.backend}")

    def show(self, phase_rad: np.ndarray) -> None:
        if self.backend == "HEDS-4.x":
            error = self.slm.showPhaseData(
                np.asarray(phase_rad, dtype=np.float32),
                phase_unit=TWO_PI,
            )
        elif self.backend == "slmdisplaysdk-3.x":
            error = self.slm.showPhasevalues(np.asarray(phase_rad, dtype=np.float32))
        else:
            raise RuntimeError("SLM has not been opened")
        self._check_error(error, "phase display")

    def close(self) -> None:
        try:
            if self.slm is not None and self.backend == "HEDS-4.x":
                self._check_error(self.slm.window().close(), "SLM close")
            elif self.slm is not None:
                self.slm.close()
        finally:
            self.slm = None
            if (
                self.backend == "HEDS-4.x"
                and self.module is not None
                and self.sdk_initialized
                and hasattr(self.module.SDK, "Close")
            ):
                self.module.SDK.Close()
                self.sdk_initialized = False


class SimulatedCamera:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed + 1)
        self._frame_id = 0

    def open(self) -> None:
        print("[dry-run] Simulated 1384x1036 16-bit camera opened")

    def capture(self, phase_rad: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._frame_id += 1
        if phase_rad is None:
            phase_small = np.zeros((1036, 1384), dtype=np.float32)
        else:
            phase_small = resize_nearest(phase_rad, (1036, 1384))
        signal = 0.50 + 0.22 * np.cos(phase_small) + 0.08 * np.cos(2.0 * phase_small)
        noise = self._rng.normal(0.0, 0.01, signal.shape)
        frame = np.clip(np.rint((signal + noise) * 65535.0), 0, 65535).astype(np.uint16)
        return frame, {
            "frame_id": self._frame_id,
            "camera_timestamp_ns": time.time_ns(),
            "pixel_format": "Mono16-simulated",
        }

    def close(self) -> None:
        print("[dry-run] Simulated camera closed")


class PlayerOneCamera:
    """Player One Camera SDK adapter using the vendor's pyPOACamera wrapper."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.module: Any = None
        self.camera_id: Optional[int] = None
        self.properties: Any = None
        self.opened = False
        self.exposing = False
        self.frame_id = 0

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return str(value)

    def _locate_sdk(self) -> tuple[Path, Path]:
        candidates: list[Path] = []
        if self.config.playerone_sdk_dir:
            candidates.append(Path(self.config.playerone_sdk_dir).expanduser())
        else:
            environment_path = os.environ.get("PLAYERONE_SDK_DIR")
            if environment_path:
                candidates.append(Path(environment_path).expanduser())
            script_dir = Path(__file__).resolve().parent
            candidates.extend(
                sorted(script_dir.glob("PlayerOne_Camera_SDK*"), reverse=True)
            )

        architecture = "x64" if sys.maxsize > 2**32 else "x86"
        checked: list[str] = []
        for candidate in candidates:
            candidate = candidate.resolve()
            python_dir = (
                candidate
                if (candidate / "pyPOACamera.py").is_file()
                else candidate / "python"
            )
            if not (python_dir / "pyPOACamera.py").is_file():
                checked.append(str(candidate))
                continue
            sdk_root = python_dir.parent
            dll_candidates = (
                python_dir / "PlayerOneCamera.dll",
                sdk_root / "lib" / architecture / "PlayerOneCamera.dll",
            )
            for dll_path in dll_candidates:
                if dll_path.is_file():
                    return python_dir, dll_path.parent
            checked.append(
                f"{candidate} (Python wrapper found, PlayerOneCamera.dll missing)"
            )

        detail = f" Checked: {checked}." if checked else ""
        raise RuntimeError(
            "Player One SDK not found. Pass --playerone-sdk-dir with the SDK root "
            "or set PLAYERONE_SDK_DIR; it must contain python/pyPOACamera.py and "
            f"lib/{architecture}/PlayerOneCamera.dll.{detail}"
        )

    def _check_error(self, error: Any, operation: str) -> None:
        if error == self.module.POAErrors.POA_OK:
            return
        try:
            description = self.module.GetErrorString(error)
            description = self._decode(description)
        except Exception:
            description = str(error)
        raise RuntimeError(f"Player One {operation} failed: {description} ({error})")

    def _import_sdk(self, python_dir: Path, dll_dir: Path) -> Any:
        python_text = str(python_dir)
        if python_text not in sys.path:
            sys.path.insert(0, python_text)
        original_directory = Path.cwd()
        try:
            # The vendor wrapper loads './PlayerOneCamera.dll', so its import must
            # occur with the DLL directory as the current working directory.
            os.chdir(dll_dir)
            return importlib.import_module("pyPOACamera")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Could not load the Player One Python SDK. Use 64-bit Python with "
                "lib/x64/PlayerOneCamera.dll (or 32-bit Python with lib/x86), and "
                "install the Player One camera driver."
            ) from exc
        finally:
            os.chdir(original_directory)

    def open(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("The bundled Player One Camera SDK is Windows-only")
        python_dir, dll_dir = self._locate_sdk()
        self.module = self._import_sdk(python_dir, dll_dir)
        count = int(self.module.GetCameraCount())
        if count == 0:
            raise RuntimeError("No Player One camera was detected")

        available_serials: list[str] = []
        selected = None
        for index in range(count):
            error, properties = self.module.GetCameraProperties(index)
            self._check_error(error, f"camera {index} properties")
            serial = self._decode(properties.SN)
            available_serials.append(serial)
            if self.config.camera_serial is None or serial == self.config.camera_serial:
                selected = properties
                break
        if selected is None:
            raise RuntimeError(
                f"Camera serial {self.config.camera_serial!r} not found; available: "
                f"{available_serials}"
            )

        self.properties = selected
        self.camera_id = int(selected.cameraID)
        self._check_error(self.module.OpenCamera(self.camera_id), "open")
        self.opened = True
        self._check_error(self.module.InitCamera(self.camera_id), "initialization")
        self._check_error(self.module.SetImageStartPos(self.camera_id, 0, 0), "ROI origin")
        self._check_error(
            self.module.SetImageSize(self.camera_id, selected.maxWidth, selected.maxHeight),
            "full-frame ROI",
        )
        self._check_error(self.module.SetImageBin(self.camera_id, 1), "1x binning")

        format_name = (
            "RAW16"
            if self.config.pixel_format.lower() == "auto"
            else self.config.pixel_format
        )
        enum_name = format_name.upper()
        if not enum_name.startswith("POA_"):
            enum_name = f"POA_{enum_name}"
        try:
            image_format = getattr(self.module.POAImgFormat, enum_name)
        except AttributeError as exc:
            raise RuntimeError(
                "Player One --pixel-format must be RAW8, RAW16, RGB24, MONO8, or auto"
            ) from exc
        if image_format not in selected.imgFormats:
            supported = [item.name.removeprefix("POA_") for item in selected.imgFormats]
            raise RuntimeError(f"Camera does not support {format_name}; supported: {supported}")
        self._check_error(
            self.module.SetImageFormat(self.camera_id, image_format),
            "image format",
        )
        self._check_error(
            self.module.SetExp(
                self.camera_id,
                int(round(self.config.exposure_ms * 1000.0)),
                False,
            ),
            "exposure",
        )
        self._check_error(
            self.module.SetGain(self.camera_id, int(round(self.config.gain)), False),
            "gain",
        )

        model = self._decode(selected.cameraModelName)
        serial = self._decode(selected.SN)
        print(
            f"Player One camera {model} ({serial}) opened; {selected.maxWidth}x"
            f"{selected.maxHeight} {image_format.name.removeprefix('POA_')}"
        )

    def _capture_once(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.camera_id is None:
            raise RuntimeError("Player One camera has not been opened")
        self._check_error(
            self.module.StartExposure(self.camera_id, True),
            "snap exposure start",
        )
        self.exposing = True
        deadline = time.monotonic() + self.config.camera_timeout_ms / 1000.0
        while True:
            error, state = self.module.GetCameraState(self.camera_id)
            self._check_error(error, "camera-state query")
            if state == self.module.POACameraState.STATE_OPENED:
                self.exposing = False
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Player One exposure timed out after {self.config.camera_timeout_ms} ms"
                )
            time.sleep(0.001)

        error, ready = self.module.ImageReady(self.camera_id)
        self._check_error(error, "image-ready query")
        if not ready:
            raise RuntimeError("Player One snap exposure ended but no image is ready")
        error, image = self.module.GetImage(self.camera_id, self.config.camera_timeout_ms)
        self._check_error(error, "image retrieval")
        frame = np.asarray(image).copy()
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        self.frame_id += 1
        return frame, {
            "frame_id": self.frame_id,
            "camera_timestamp_ns": time.time_ns(),
            "pixel_format": (
                "RAW16" if self.config.pixel_format.lower() == "auto" else self.config.pixel_format
            ),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "camera_model": self._decode(self.properties.cameraModelName),
            "camera_serial": self._decode(self.properties.SN),
        }

    def capture(
        self,
        phase_rad: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del phase_rad
        result: tuple[np.ndarray, dict[str, Any]]
        for _ in range(self.config.discard_frames + 1):
            result = self._capture_once()
        frame, metadata = result
        if self.config.require_16_bit and frame.dtype.itemsize != 2:
            raise RuntimeError(
                f"Expected a 16-bit frame but Player One returned dtype {frame.dtype}; "
                "use --pixel-format RAW16 or pass --allow-non-16-bit"
            )
        return frame, metadata

    def close(self) -> None:
        if self.module is None or self.camera_id is None:
            return
        try:
            if self.exposing:
                self.module.StopExposure(self.camera_id)
                self.exposing = False
        finally:
            if self.opened:
                self.module.CloseCamera(self.camera_id)
                self.opened = False


class PySpinCamera:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.pyspin: Any = None
        self.system: Any = None
        self.camera_list: Any = None
        self.camera: Any = None
        self.initialized = False
        self.acquiring = False

    def _set_enum(self, nodemap: Any, name: str, entry_name: str, required: bool = True) -> bool:
        ps = self.pyspin
        node = ps.CEnumerationPtr(nodemap.GetNode(name))
        if not ps.IsAvailable(node) or not ps.IsWritable(node):
            if required:
                raise RuntimeError(f"Camera node {name} is unavailable or read-only")
            return False
        entry = node.GetEntryByName(entry_name)
        if not ps.IsAvailable(entry) or not ps.IsReadable(entry):
            if required:
                raise RuntimeError(f"Camera {name} does not support {entry_name}")
            return False
        node.SetIntValue(entry.GetValue())
        return True

    def _set_float(self, nodemap: Any, name: str, value: float, required: bool = True) -> float:
        ps = self.pyspin
        node = ps.CFloatPtr(nodemap.GetNode(name))
        if not ps.IsAvailable(node) or not ps.IsWritable(node):
            if required:
                raise RuntimeError(f"Camera node {name} is unavailable or read-only")
            return float("nan")
        applied = min(max(value, node.GetMin()), node.GetMax())
        node.SetValue(applied)
        if not math.isclose(applied, value, rel_tol=0.0, abs_tol=1e-6):
            print(f"Warning: requested {name}={value}, camera applied {applied}", file=sys.stderr)
        return applied

    def _set_bool(self, nodemap: Any, name: str, value: bool, required: bool = False) -> bool:
        ps = self.pyspin
        node = ps.CBooleanPtr(nodemap.GetNode(name))
        if not ps.IsAvailable(node) or not ps.IsWritable(node):
            if required:
                raise RuntimeError(f"Camera node {name} is unavailable or read-only")
            return False
        node.SetValue(value)
        return True

    def _serial(self, camera: Any) -> str:
        node = self.pyspin.CStringPtr(camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
        return node.GetValue() if self.pyspin.IsReadable(node) else "unknown"

    def open(self) -> None:
        try:
            import PySpin  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PySpin not found. Install the FLIR/Teledyne Spinnaker SDK and its "
                "matching Python package before using hardware mode."
            ) from exc

        self.pyspin = PySpin
        self.system = PySpin.System.GetInstance()
        self.camera_list = self.system.GetCameras()
        if self.camera_list.GetSize() == 0:
            raise RuntimeError("No Spinnaker camera was detected")

        selected = None
        available_serials = []
        for index in range(self.camera_list.GetSize()):
            candidate = self.camera_list.GetByIndex(index)
            serial = self._serial(candidate)
            available_serials.append(serial)
            if self.config.camera_serial is None or serial == self.config.camera_serial:
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                f"Camera serial {self.config.camera_serial!r} not found; available: "
                f"{available_serials}"
            )

        self.camera = selected
        self.camera.Init()
        self.initialized = True
        nodemap = self.camera.GetNodeMap()
        self._set_enum(nodemap, "AcquisitionMode", "Continuous")
        self._set_enum(nodemap, "TriggerMode", "Off")
        self._set_enum(nodemap, "ExposureAuto", "Off")
        self._set_enum(nodemap, "ExposureMode", "Timed", required=False)
        self._set_float(nodemap, "ExposureTime", self.config.exposure_ms * 1000.0)
        self._set_enum(nodemap, "GainAuto", "Off")
        self._set_float(nodemap, "Gain", float(self.config.gain))
        if self._set_bool(nodemap, "GammaEnable", True, required=False):
            self._set_float(nodemap, "Gamma", self.config.gamma, required=False)
        else:
            print(
                "Warning: camera gamma control is unavailable; leaving it unchanged",
                file=sys.stderr,
            )
        pixel_format = (
            "Mono16"
            if self.config.pixel_format.lower() == "auto"
            else self.config.pixel_format
        )
        self._set_enum(nodemap, "PixelFormat", pixel_format)
        self._set_enum(
            self.camera.GetTLStreamNodeMap(),
            "StreamBufferHandlingMode",
            "NewestOnly",
            required=False,
        )
        self.camera.BeginAcquisition()
        self.acquiring = True
        print(f"Camera {self._serial(self.camera)} opened; pixel format {pixel_format}")

    def _get_image(self) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.camera.GetNextImage(self.config.camera_timeout_ms)
        try:
            if image.IsIncomplete():
                raise RuntimeError(f"Incomplete camera image; status={image.GetImageStatus()}")
            frame = image.GetNDArray().copy()
            metadata = {
                "frame_id": int(image.GetFrameID()),
                "camera_timestamp_ns": int(image.GetTimeStamp()),
                "pixel_format": str(image.GetPixelFormatName()),
                "width": int(image.GetWidth()),
                "height": int(image.GetHeight()),
            }
            return frame, metadata
        finally:
            image.Release()

    def capture(self, phase_rad: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[str, Any]]:
        del phase_rad
        for _ in range(self.config.discard_frames):
            self._get_image()
        frame, metadata = self._get_image()
        if self.config.require_16_bit and frame.dtype.itemsize != 2:
            raise RuntimeError(
                f"Expected a 16-bit frame but camera returned dtype {frame.dtype}; "
                "select an unpacked 16-bit --pixel-format or pass --allow-non-16-bit"
            )
        return frame, metadata

    def close(self) -> None:
        try:
            if self.camera is not None and self.acquiring:
                self.camera.EndAcquisition()
                self.acquiring = False
            if self.camera is not None and self.initialized:
                self.camera.DeInit()
                self.initialized = False
        finally:
            self.camera = None
            if self.camera_list is not None:
                self.camera_list.Clear()
                self.camera_list = None
            if self.system is not None:
                self.system.ReleaseInstance()
                self.system = None


class OutputWriter:
    def __init__(self, output_dir: Path, config: ExperimentConfig):
        self.output_dir = output_dir
        if output_dir.exists():
            if not output_dir.is_dir():
                raise FileExistsError(f"Output path is not a directory: {output_dir}")
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. "
                    "Choose a new directory to avoid overwriting data."
                )
        if config.output_format == "mat":
            try:
                from scipy.io import savemat  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "--output-format mat requires scipy. Install dependencies with "
                    "python -m pip install -r requirements.txt"
                ) from exc
            self._savemat = savemat
        else:
            self._savemat = None
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PNG preview output requires Pillow. Install dependencies with "
                "python -m pip install -r requirements.txt"
            ) from exc
        self._image_class = Image

        self.raw_dir = output_dir / "raw"
        self.preview_dir = output_dir / "previews"
        self.processed_dir = output_dir / "processed"
        self.pattern_dir = output_dir / "patterns"
        self.neuws_dir = output_dir / "neuws_mat"
        for directory in (self.raw_dir, self.preview_dir, self.pattern_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if config.neuws_processing:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        if config.output_format == "mat":
            self.neuws_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.records: list[dict[str, Any]] = []

    def save(
        self,
        index: int,
        frame: np.ndarray,
        modulation: np.ndarray,
        displayed_phase: np.ndarray,
        coefficients: np.ndarray,
        camera_metadata: dict[str, Any],
        display_time_ns: int,
        capture_time_ns: int,
    ) -> None:
        number = index + 1
        raw_path = self.raw_dir / f"frame_{number:04d}.npy"
        preview_path = self.preview_dir / f"frame_{number:04d}.png"
        modulation_path = self.pattern_dir / f"phase_modulation_{number:04d}.npy"
        np.save(raw_path, frame, allow_pickle=False)
        preview, preview_low, preview_high = make_preview_u8(frame)
        self._image_class.fromarray(preview).save(preview_path)
        np.save(modulation_path, modulation, allow_pickle=False)

        full_pattern_path: Optional[Path] = None
        if self.config.save_full_patterns:
            full_pattern_path = self.pattern_dir / f"phase_slm_{number:04d}.npy"
            np.save(full_pattern_path, displayed_phase, allow_pickle=False)

        processed = None
        processed_path: Optional[Path] = None
        if self.config.neuws_processing:
            processed = process_neuws_frame(
                frame,
                self.config.magnification,
                self.config.crop_size,
            )
            processed_path = self.processed_dir / f"frame_{number:04d}.npy"
            np.save(processed_path, processed, allow_pickle=False)

        if self._savemat is not None:
            measurement_for_mat = processed if processed is not None else frame
            self._savemat(
                self.neuws_dir / f"SLM_raw{number}.mat",
                {"imsdata": measurement_for_mat},
                do_compression=True,
            )
            self._savemat(
                self.neuws_dir / f"SLM_sim{number}.mat",
                {"proj_sim": modulation},
                do_compression=True,
            )

        record = {
            "pattern_index": number,
            "raw_path": str(raw_path.relative_to(self.output_dir)),
            "preview_path": str(preview_path.relative_to(self.output_dir)),
            "preview_scale_1st_percentile": preview_low,
            "preview_scale_99th_percentile": preview_high,
            "processed_path": (
                str(processed_path.relative_to(self.output_dir)) if processed_path else None
            ),
            "modulation_path": str(modulation_path.relative_to(self.output_dir)),
            "full_pattern_path": (
                str(full_pattern_path.relative_to(self.output_dir)) if full_pattern_path else None
            ),
            "zernike_coefficients_rad": coefficients.tolist(),
            "display_time_ns": display_time_ns,
            "capture_time_ns": capture_time_ns,
            "frame_dtype": str(frame.dtype),
            "frame_shape": list(frame.shape),
            **camera_metadata,
        }
        self.records.append(record)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference": "sciadv.adg4671_sm.pdf, pages 3-5",
            "configuration": asdict(self.config),
            "frames_completed": len(self.records),
            "frames": self.records,
        }
        temporary_path = self.output_dir / "manifest.json.tmp"
        final_path = self.output_dir / "manifest.json"
        temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, final_path)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data") / f"neuws_acquisition_{stamp}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def sdk_version(value: str) -> tuple[int, int]:
    try:
        major_text, minor_text = value.split(".", maxsplit=1)
        major, minor = int(major_text), int(minor_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must look like 4.2") from exc
    if major < 1 or minor < 0:
        raise argparse.ArgumentTypeError("must look like 4.2")
    return major, minor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display NeuWS random-Zernike SLM patterns and record one camera frame each."
    )
    parser.add_argument("--num-patterns", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use simulated SLM and camera")
    parser.add_argument("--zernike-modes", type=positive_int, default=15)
    parser.add_argument("--coefficient-std-rad", type=float, default=5.0)
    parser.add_argument(
        "--fringe-slope-rad-per-pixel",
        type=float,
        default=4.0 * math.pi / 3.0,
        help="Paper value: 4*pi/3",
    )
    parser.add_argument("--slm-width", type=positive_int, default=1920)
    parser.add_argument("--slm-height", type=positive_int, default=1080)
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=532.0,
        help="Laser wavelength used for HOLOEYE phase calibration (default: 532)",
    )
    parser.add_argument("--slm-preselect", default=None, help="Optional HEDS 4.x selection string")
    parser.add_argument(
        "--heds-examples-dir",
        default=None,
        metavar="PATH",
        help="SDK v4 examples directory containing the HEDS folder",
    )
    parser.add_argument(
        "--heds-api-version",
        type=sdk_version,
        default=(4, 2),
        metavar="MAJOR.MINOR",
        help="Installed HEDS SDK API version (default: 4.2)",
    )
    parser.add_argument("--settle-ms", type=float, default=150.0)
    parser.add_argument("--exposure-ms", type=float, default=100.0)
    parser.add_argument(
        "--gain-db",
        type=nonnegative_float,
        dest="gain",
        default=0.0,
        help="Manual Spinnaker camera gain in dB (default: 0)",
    )
    parser.add_argument(
        "--gain",
        type=nonnegative_float,
        dest="gain",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--camera-backend",
        choices=("playerone", "pyspin"),
        default="pyspin",
        help="Camera SDK to use in hardware mode (default: pyspin)",
    )
    parser.add_argument(
        "--playerone-sdk-dir",
        default=None,
        metavar="PATH",
        help="Player One SDK root, or its python directory",
    )
    parser.add_argument(
        "--pixel-format",
        default="auto",
        help="auto selects RAW16 for Player One and Mono16 for PySpin",
    )
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--camera-timeout-ms", type=positive_int, default=3000)
    parser.add_argument(
        "--discard-frames",
        type=nonnegative_int,
        default=None,
        help="Frames to discard after each SLM update (default: PySpin 1, Player One 0)",
    )
    parser.add_argument("--magnification", type=float, default=3.57)
    parser.add_argument("--crop-size", type=positive_int, default=256)
    parser.add_argument(
        "--no-neuws-processing",
        action="store_true",
        help="Do not create the 3.57x downscaled center crop",
    )
    parser.add_argument("--output-format", choices=("npy", "mat"), default="mat")
    parser.add_argument("--save-full-patterns", action="store_true")
    parser.add_argument("--allow-non-16-bit", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.slm_height != 1080 or args.slm_width != 1920:
        print(
            "Warning: the reference uses a 1080x1920 LETO-3; nonstandard dimensions were requested",
            file=sys.stderr,
        )
    if (
        args.settle_ms < 0
        or args.exposure_ms <= 0
        or args.coefficient_std_rad < 0
        or args.wavelength_nm <= 0
    ):
        raise ValueError(
            "Settle time and coefficient std must be nonnegative; exposure and "
            "wavelength must be positive"
        )
    discard_frames = (
        args.discard_frames
        if args.discard_frames is not None
        else (1 if args.camera_backend == "pyspin" else 0)
    )
    return ExperimentConfig(
        num_patterns=args.num_patterns,
        seed=args.seed,
        zernike_modes=args.zernike_modes,
        coefficient_std_rad=args.coefficient_std_rad,
        fringe_slope_rad_per_pixel=args.fringe_slope_rad_per_pixel,
        slm_height=args.slm_height,
        slm_width=args.slm_width,
        wavelength_nm=args.wavelength_nm,
        settle_ms=args.settle_ms,
        exposure_ms=args.exposure_ms,
        gain=args.gain,
        gamma=args.gamma,
        camera_timeout_ms=args.camera_timeout_ms,
        discard_frames=discard_frames,
        pixel_format=args.pixel_format,
        camera_backend=args.camera_backend,
        camera_serial=args.camera_serial,
        playerone_sdk_dir=args.playerone_sdk_dir,
        slm_preselect=args.slm_preselect,
        heds_examples_dir=args.heds_examples_dir,
        heds_api_major=args.heds_api_version[0],
        heds_api_minor=args.heds_api_version[1],
        magnification=args.magnification,
        crop_size=args.crop_size,
        neuws_processing=not args.no_neuws_processing,
        output_format=args.output_format,
        save_full_patterns=args.save_full_patterns,
        require_16_bit=not args.allow_non_16_bit,
        dry_run=args.dry_run,
    )


def run(config: ExperimentConfig, output_dir: Path) -> None:
    writer = OutputWriter(output_dir, config)
    generator = NeuWSPatternGenerator(config)
    slm = SimulatedSLM(config) if config.dry_run else HoloeyeSLM(config)
    if config.dry_run:
        camera = SimulatedCamera(config)
    elif config.camera_backend == "playerone":
        camera = PlayerOneCamera(config)
    else:
        camera = PySpinCamera(config)
    try:
        slm.open()
        camera.open()
        for index in range(config.num_patterns):
            displayed_phase, modulation, coefficients = generator.next()
            slm.show(displayed_phase)
            display_time_ns = time.time_ns()
            time.sleep(config.settle_ms / 1000.0)
            frame, camera_metadata = camera.capture(displayed_phase)
            capture_time_ns = time.time_ns()
            writer.save(
                index,
                frame,
                modulation,
                displayed_phase,
                coefficients,
                camera_metadata,
                display_time_ns,
                capture_time_ns,
            )
            print(
                f"[{index + 1:04d}/{config.num_patterns:04d}] "
                f"saved {frame.shape} {frame.dtype} frame"
            )
    except KeyboardInterrupt:
        print("Acquisition interrupted; completed frames are preserved", file=sys.stderr)
        raise
    finally:
        try:
            camera.close()
        finally:
            slm.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        output_dir = args.output_dir or default_output_dir()
        run(config, output_dir)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Acquisition complete: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
