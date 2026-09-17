#!/usr/bin/env python3
"""CLI motion mask inference: per-frame binary moving/static masks.

For each adjacent frame pair ``(i, i-1)`` the model runs on just those two
frames (frame ``i`` as source) and pixels whose ``warp3d_delta_magnitude``
exceeds the threshold are marked as moving. Masks are defined in the source
frame's coordinate system, so frame 0 has no mask (all zeros).

Usage:
    python scripts/infer_motion_mask.py \\
        --checkpoint path/to/ckpt.pth \\
        --input /path/to/images/ \\
        --output outputs/motion_mask
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from uniquery4r.models.uniquery4r import UniQuery4R
from uniquery4r.utils.hub import resolve_checkpoint
from uniquery4r.utils.load_fn import collect_image_paths, load_and_preprocess_images
from uniquery4r.utils.motion_mask import compute_per_frame_motion_masks
from uniquery4r.vis.visual_util import blend_motion_overlay, save_gif


def parse_args():
    parser = argparse.ArgumentParser(description="UniQuery4R motion mask inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Local state_dict path (ckpt.pth) or Hugging Face repo id")
    parser.add_argument("--input", type=str, default=str(_REPO_ROOT / "samples" / "rollerblade"),
                        help="Image file, dir, or glob (default: bundled 20-frame DAVIS sample)")
    parser.add_argument("--output", type=str, default="outputs/uniquery4r_motion_mask")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=504)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=30000)
    parser.add_argument("--no-strict", action="store_true", help="Load checkpoint non-strictly")
    parser.add_argument("--motion-mask-thresh", type=float, default=0.02,
                        help="Magnitude threshold for motion mask (default: 0.02)")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay visualization")
    parser.add_argument("--overlay-alpha", type=float, default=0.55,
                        help="Alpha blending for overlay (default: 0.55)")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF export")
    parser.add_argument("--gif-fps", type=float, default=4.0, help="GIF frame rate (default: 4.0)")
    parser.add_argument("--dynamic-color", type=int, nargs=3, default=[255, 50, 50],
                        metavar=("R", "G", "B"), help="Overlay color for moving pixels")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.input}")
    if args.max_frames and args.max_frames > 0 and len(image_paths) > args.max_frames:
        print(f"Truncating {len(image_paths)} -> {args.max_frames} frames")
        image_paths = image_paths[: args.max_frames]

    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 frames for motion mask computation")

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
    all_images = load_and_preprocess_images(
        image_paths, max_size=args.max_size, patch_size=args.patch_size,
    ).to(device)  # [N, 3, H, W]
    n, c, h, w = all_images.shape
    print(f"Preprocessed images: {(n, c, h, w)}")

    print(f"Running {n - 1} adjacent pair forward passes ...")
    masks = compute_per_frame_motion_masks(
        model, all_images, threshold=args.motion_mask_thresh,
    )

    del all_images, model
    torch.cuda.empty_cache()

    moving_ratio = masks[1:].mean()  # exclude frame 0
    print(f"\nMotion masks (thresh={args.motion_mask_thresh}): {masks.shape}, "
          f"moving ratio (excl. frame 0) = {moving_ratio:.4f}")

    # ── Save binary masks ────────────────────────────────────────────────
    masks_dir = os.path.join(args.output, "motion_masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i in range(n):
        Image.fromarray(masks[i] * 255).save(os.path.join(masks_dir, f"mask_{i:04d}.png"))

    npz_path = os.path.join(args.output, "motion_masks.npz")
    np.savez_compressed(npz_path, masks=masks)
    print(f"Saved {npz_path}")
    print(f"Saved mask images to {masks_dir}/")

    # ── Overlay visualization ────────────────────────────────────────────
    overlay_frames: list[np.ndarray] = []
    if not args.no_overlay:
        overlay_dir = os.path.join(args.output, "motion_overlay")
        os.makedirs(overlay_dir, exist_ok=True)

        for i in range(n):
            # Reload original image at the processed resolution
            img = Image.open(image_paths[i]).resize((w, h))
            overlay = blend_motion_overlay(
                np.array(img), masks[i], alpha=args.overlay_alpha, color=args.dynamic_color,
            )
            cv2.imwrite(
                os.path.join(overlay_dir, f"overlay_{i:04d}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            overlay_frames.append(overlay)

        print(f"Saved overlay images to {overlay_dir}/")

    # ── GIF ──────────────────────────────────────────────────────────────
    if not args.no_gif and overlay_frames:
        gif_path = os.path.join(args.output, "motion_mask_overlay.gif")
        save_gif(overlay_frames, gif_path, fps=args.gif_fps)
        print(f"Saved GIF: {gif_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = dict(
        total_frames=n,
        motion_outputs=list(motion_output_names),
        motion_mask_threshold=args.motion_mask_thresh,
        moving_ratio_excl_frame0=round(float(moving_ratio), 6),
    )
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
