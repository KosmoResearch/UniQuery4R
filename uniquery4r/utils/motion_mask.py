"""Per-frame motion masks from the model's scene-flow magnitude."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


@torch.inference_mode()
def compute_per_frame_motion_masks(
    model,
    images: torch.Tensor,
    threshold: float = 0.02,
    frame_indices: Optional[Sequence[int]] = None,
    log_every: int = 10,
) -> np.ndarray:
    """Binary motion mask per frame from adjacent-pair ``warp3d_delta`` magnitude.

    For each pair ``(i, i-1)`` the model runs on just those two frames with
    frame ``i`` as the source; pixels whose scene-flow magnitude exceeds
    ``threshold`` are marked as moving. The mask is therefore defined in the
    source (frame ``i``) coordinate system and frame 0 has no mask (all zeros).

    Args:
        model: ``UniQuery4R`` instance on the target device.
        images: preprocessed ``[N, 3, H, W]`` float tensor on the same device.
        threshold: scene-flow magnitude threshold in world units per frame.
        frame_indices: optional subset of frames ``1..N-1`` to compute.
    Returns:
        ``uint8`` masks of shape ``[N, H, W]`` (1 = moving).
    """
    n, _, h, w = images.shape
    masks = np.zeros((n, h, w), dtype=np.uint8)
    indices = range(1, n) if frame_indices is None else frame_indices

    for i in indices:
        # Reverse order [i, i-1]: frame i is the source (index 0), so
        # warp3d_delta is expressed in frame i's coordinate system.
        pair_images = images[[i, i - 1]].unsqueeze(0)  # [1, 2, 3, H, W]
        predictions = model(pair_images)

        # For N=2, pair_idx = [(0,1), (0,0), (1,1)]; index [0, 0] is pair (0, 1)
        # with shape [B, P, C, H, W] -> [H, W].
        magnitude = predictions["warp3d_delta_magnitude"][0, 0, 0]
        masks[i] = (magnitude.cpu().numpy() > threshold).astype(np.uint8)

        if log_every and (i % log_every == 0 or i == 1):
            print(f"  frame {i} (src->tgt = {i}->{i - 1}): moving ratio = {masks[i].mean():.4f}")

    return masks
