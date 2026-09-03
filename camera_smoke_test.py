#!/usr/bin/env python3
"""Open a Spinnaker camera and save one frame as NPY plus a PNG preview."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from acquire_slm_camera import PySpinCamera, make_preview_u8


@dataclass(frozen=True)
class CameraConfig:
    camera_serial: Optional[str]
    exposure_ms: float
    gain: float
    gamma: float
    pixel_format: str
    camera_timeout_ms: int
    discard_frames: int
    require_16_bit: bool


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data") / f"camera_test_{stamp}.npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one frame from the Grasshopper3/Spinnaker camera."
    )
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--exposure-ms", type=float, default=100.0)
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--pixel-format", default="Mono16")
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--discard-frames", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-non-16-bit", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.exposure_ms <= 0 or args.gain_db < 0:
        parser.error("exposure must be positive and gain must be nonnegative")
    if args.timeout_ms <= 0 or args.discard_frames < 0:
        parser.error("timeout must be positive and discarded frames must be nonnegative")

    output_path = args.output or default_output_path()
    if output_path.suffix.lower() != ".npy":
        output_path = output_path.with_suffix(".npy")
    png_path = output_path.with_suffix(".png")
    existing_outputs = [path for path in (output_path, png_path) if path.exists()]
    if existing_outputs:
        parser.error(f"output already exists: {existing_outputs[0]}")

    config = CameraConfig(
        camera_serial=args.camera_serial,
        exposure_ms=args.exposure_ms,
        gain=args.gain_db,
        gamma=args.gamma,
        pixel_format=args.pixel_format,
        camera_timeout_ms=args.timeout_ms,
        discard_frames=args.discard_frames,
        require_16_bit=not args.allow_non_16_bit,
    )
    camera = PySpinCamera(config)  # type: ignore[arg-type]
    try:
        camera.open()
        frame, metadata = camera.capture()
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        camera.close()

    try:
        from PIL import Image
    except ImportError:
        parser.exit(
            2,
            "error: PNG output requires Pillow; run python -m pip install Pillow\n",
        )
    preview, preview_low, preview_high = make_preview_u8(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, frame, allow_pickle=False)
    Image.fromarray(preview).save(png_path)
    summary = {
        **metadata,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": int(frame.min()),
        "maximum": int(frame.max()),
        "mean": float(frame.mean()),
        "preview_scale_1st_percentile": preview_low,
        "preview_scale_99th_percentile": preview_high,
        "raw_npy": str(output_path.resolve()),
        "preview_png": str(png_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
