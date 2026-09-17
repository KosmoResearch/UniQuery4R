#!/usr/bin/env python3
"""CLI scene reconstruction: an image sequence -> 4D geometry, cameras and motion.

Runs UniQuery4R (``uniquery4r.models.UniQuery4R``) over a stack of images and
exports per-view depth / global points, camera extrinsics / intrinsics, and
cross-view motion. Per-view geometry is read off the self-pairs ``(v, v)``
inside the model; here we just unproject / export:

    scene.ply / scene.glb    fused colored point cloud (+ camera frustums)
    cameras.ply              camera frustum meshes
    predictions.npz          raw tensors + scene-flow tracks for the interactive viewer

The saved ``predictions.npz`` is self-contained; view it offline with
``python scripts/view_4d.py --input <output dir>`` (``--no-npz`` skips it).

NOTE: without a GT scale the outputs are only defined up to a global scale.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from uniquery4r.models.uniquery4r import UniQuery4R
from uniquery4r.utils.geometry import unproject_depth_map_to_point_map
from uniquery4r.utils.hub import resolve_checkpoint
from uniquery4r.utils.load_fn import collect_image_paths, load_and_preprocess_images
from uniquery4r.vis.visual_util import cameras_to_ply, predictions_to_glb, predictions_to_ply


def parse_args():
    parser = argparse.ArgumentParser(description="UniQuery4R standalone inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Local state_dict path (ckpt.pth) or Hugging Face repo id")
    parser.add_argument("--input", type=str, default=str(_REPO_ROOT / "samples" / "rollerblade"),
                        help="Image file, dir, or glob (default: bundled 20-frame DAVIS sample)")
    parser.add_argument("--output", type=str, default="outputs/uniquery4r_infer")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=504)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=30000)
    parser.add_argument("--conf-thresh", type=float, default=20.0)
    parser.add_argument("--points-from", type=str, default="depth", choices=["depth", "warp3d"],
                        help="Export points unprojected from predicted depth, or the warp3d global points")
    parser.add_argument("--no-npz", action="store_true",
                        help="Skip predictions.npz (scripts/view_4d.py reads this file)")
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--no-strict", action="store_true", help="Load checkpoint non-strictly")
    parser.add_argument("--motion-source", type=str, default="warp3d", choices=["warp3d", "delta"],
                        help="Scene-flow source baked into predictions.npz for the viewer: the absolute "
                             "warp3d head, or frame-0 + warp3d_delta")
    return parser.parse_args()


def tensor_to_numpy(predictions: dict) -> dict:
    out = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            out[key] = value
        else:
            out[key] = value
    return out


def _to_hwc(x: np.ndarray) -> np.ndarray:
    """[S,C,H,W] -> [S,H,W,C]."""
    return np.transpose(x, (0, 2, 3, 1))


def assemble_frame_points(pred_np: dict, global_points: np.ndarray, motion_source: str = "warp3d"):
    """Compose the per-frame world position of the frame-0 query points.

    The model decodes, per cross pair ``(0, t)``, both the absolute world point
    ``warp3d`` of the frame-0 query at frame ``t`` and its scene flow
    ``warp3d_delta = delta_direction * delta_magnitude``. Either reconstructs the
    moved positions (they are independent heads):

        * ``motion_source="warp3d"`` : moved[t] = warp3d(0, t)           (absolute)
        * ``motion_source="delta"``  : moved[t] = global_points[0] + warp3d_delta(0, t)

    Frame 0 is always the reference geometry ``global_points[0]``.

    Returns:
        frame_points: (N, H, W, 3) absolute per-frame positions of the frame-0 points.
        delta_conf:   (N, H, W) per-query confidence (index 0 zeros), or None.
    """
    pair_idx = pred_np.get("pair_idx")
    if pair_idx is None:
        return None, None

    num_frames, h, w, _ = global_points.shape
    src = global_points[0]  # (H, W, 3)

    if motion_source == "warp3d" and pred_np.get("warp3d") is not None:
        moved_pair = _to_hwc(pred_np["warp3d"])           # (P, H, W, 3)
        conf_pair = pred_np.get("warp3d_confidence")
    else:
        direction = pred_np.get("warp3d_delta_direction")  # (P, 3, H, W)
        magnitude = pred_np.get("warp3d_delta_magnitude")  # (P, 1, H, W)
        if direction is None or magnitude is None:
            return None, None
        moved_pair = src[None] + _to_hwc(direction) * _to_hwc(magnitude)  # (P, H, W, 3)
        conf_pair = pred_np.get("warp3d_delta_confidence")
    conf_pair = _to_hwc(conf_pair)[..., 0] if conf_pair is not None else None  # (P, H, W)

    frame_points = np.repeat(src[None], num_frames, axis=0).astype(np.float32)  # frame 0 = src
    delta_conf = np.zeros((num_frames, h, w), dtype=np.float32) if conf_pair is not None else None
    for p, (s, t) in enumerate(pair_idx):
        if s == 0 and t != 0 and t < num_frames:
            frame_points[t] = moved_pair[p]
            if delta_conf is not None:
                delta_conf[t] = conf_pair[p]
    return frame_points, delta_conf


@torch.inference_mode()
def main():
    args = parse_args()
    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.input}")
    if args.max_frames and args.max_frames > 0 and len(image_paths) > args.max_frames:
        print(f"Truncating {len(image_paths)} -> {args.max_frames} frames")
        image_paths = image_paths[: args.max_frames]

    os.makedirs(args.output, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference is not supported")
    device = torch.device("cuda")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    model = UniQuery4R(patch_size=args.patch_size, chunk_size=args.chunk_size).eval()
    model.load_checkpoint(checkpoint_path, strict=not args.no_strict)
    motion_output_names = model.motion_output_names
    print(f"Motion outputs ({len(motion_output_names)}): {', '.join(motion_output_names)}")
    model = model.to(device)

    print(f"Loading {len(image_paths)} images ...")
    images = load_and_preprocess_images(
        image_paths, max_size=args.max_size, patch_size=args.patch_size,
    ).to(device)  # [N, 3, H, W]
    print(f"Preprocessed images: {tuple(images.shape)}")

    predictions = model(images)

    pred_np = tensor_to_numpy(predictions)

    del predictions, images, model
    torch.cuda.empty_cache()

    depth = _to_hwc(pred_np["depth"])            # [S,H,W,1]
    conf = _to_hwc(pred_np["confidence"])         # [S,H,W,1]
    global_points = _to_hwc(pred_np["global_points"])        # [S,H,W,3]
    global_conf = _to_hwc(pred_np["global_confidence"])      # [S,H,W,1]
    extrinsic = pred_np["extrinsic"]              # [S,3,4]
    intrinsic = pred_np["intrinsic"]              # [S,3,3]

    if args.points_from == "depth":
        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        points_conf = conf
    else:
        world_points = global_points
        points_conf = global_conf

    pred_np["depth"] = depth
    pred_np["confidence"] = points_conf
    pred_np["world_points_from_depth"] = world_points

    frame_points, delta_conf = assemble_frame_points(
        pred_np, global_points, motion_source=args.motion_source,
    )

    if not args.no_npz:
        npz_path = os.path.join(args.output, "predictions.npz")
        npz_data = dict(
            depth=depth,
            confidence=conf,
            global_points=global_points,
            global_confidence=global_conf,
            world_points=world_points,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            images=pred_np["images"],
        )
        if frame_points is not None:
            npz_data["frame_points"] = frame_points
            if delta_conf is not None:
                npz_data["delta_conf"] = delta_conf
        else:
            print("No warp3d / warp3d_delta in predictions; "
                  "predictions.npz will not include the viewer scene-flow tracks.")
        np.savez_compressed(npz_path, **npz_data)
        print(f"Saved {npz_path}")

    if not args.no_ply:
        ply_path = os.path.join(args.output, "scene.ply")
        predictions_to_ply(
            pred_np, ply_path, conf_thresh=args.conf_thresh,
            conf_key="confidence", points_key="world_points_from_depth",
        )
        print(f"Saved {ply_path}")
        cameras_ply_path = os.path.join(args.output, "cameras.ply")
        cameras_to_ply(pred_np, cameras_ply_path)
        print(f"Saved {cameras_ply_path}")

    if not args.no_glb:
        glb_path = os.path.join(args.output, "scene.glb")
        scene = predictions_to_glb(
            pred_np, conf_thresh=args.conf_thresh,
            conf_key="confidence", points_key="world_points_from_depth",
        )
        scene.export(glb_path)
        print(f"Saved {glb_path}")

    print("Done.")
    print(
        f"  frames={depth.shape[0]} "
        f"depth={depth.shape} extrinsic={extrinsic.shape} intrinsic={intrinsic.shape}"
    )
    if not args.no_npz:
        print(f"  interactive viewer: python scripts/view_4d.py --input {args.output}")


if __name__ == "__main__":
    main()
