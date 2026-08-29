#!/usr/bin/env python3
"""Open a Spinnaker camera, capture one frame, and save it as a NumPy array."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from acquire_slm_camera import PySpinCamera


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
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, frame, allow_pickle=False)
    summary = {
        **metadata,
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": int(frame.min()),
        "maximum": int(frame.max()),
        "mean": float(frame.mean()),
        "saved_to": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
