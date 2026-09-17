#!/usr/bin/env python3
"""Interactive 4D scene-flow viewer over a saved inference directory.

Reads the ``predictions.npz`` written by ``scripts/infer_4d.py`` and opens a
Viser viewer with per-query motion trails, a moving point cloud, and (optional)
camera frustums.

Usage:
    python scripts/infer_4d.py --input path/to/images --output outputs/demo
    python scripts/view_4d.py --input outputs/demo --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from uniquery4r.vis.motion_viewer import MotionViewer

_REQUIRED_KEYS = ("images", "global_points", "frame_points")


def parse_args():
    parser = argparse.ArgumentParser(description="UniQuery4R interactive scene-flow viewer")
    parser.add_argument("--input", type=str, required=True,
                        help="Inference output dir (containing predictions.npz) or an .npz path")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stride", type=int, default=8,
                        help="Pixel stride for sampling scene-flow query tracks")
    parser.add_argument("--max-tracks", type=int, default=20000,
                        help="Maximum number of sampled scene-flow tracks")
    parser.add_argument("--conf-thresh", type=float, default=0.0,
                        help="Percentile threshold on motion confidence for track filtering (0 = keep all)")
    parser.add_argument("--point-size", type=float, default=0.002,
                        help="Point size of the scene-flow tracking dots")
    parser.add_argument("--show-cameras", action="store_true",
                        help="Show camera frustums (hidden by default)")
    return parser.parse_args()


def resolve_npz(input_path: str) -> Path:
    path = Path(input_path)
    if path.is_dir():
        path = path / "predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/infer_4d.py first (without --no-npz)"
        )
    return path


def main():
    args = parse_args()
    npz_path = resolve_npz(args.input)
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)
    missing = [key for key in _REQUIRED_KEYS if key not in data.files]
    if missing:
        raise KeyError(f"{npz_path} lacks {missing}; regenerate it with scripts/infer_4d.py")

    viewer = MotionViewer(
        images=data["images"],
        global_points=data["global_points"],
        frame_points=data["frame_points"],
        delta_conf=data["delta_conf"] if "delta_conf" in data.files else None,
        extrinsic=data["extrinsic"] if "extrinsic" in data.files else None,
        intrinsic=data["intrinsic"] if "intrinsic" in data.files else None,
        port=args.port,
        stride=args.stride,
        max_tracks=args.max_tracks,
        conf_thresh=args.conf_thresh,
        point_size=args.point_size,
        show_camera=args.show_cameras,
    )
    print(f"Viewing {npz_path.name} ({viewer.num_frames} frames); Ctrl-C to quit.")
    viewer.run()


if __name__ == "__main__":
    main()
