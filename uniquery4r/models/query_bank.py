"""Dense query bank: the full-image UV grid as query points.

At inference the bank emits every pixel of the input grid as a normalized
``(u, v)`` query; sparse queries bypass it by passing ``query_uv`` directly to
the model (see ``UniQuery4R.forward``).
"""

import torch
import torch.nn as nn


class BaseQuery(nn.Module):
    def __init__(self, uv=None, xyz=None, batch_uv=None, full_uv=False, width=None, height=None):
        super().__init__()

        self.uv = uv
        self.xyz = xyz
        self.batch_uv = batch_uv

        self.full_uv = full_uv
        self.width = width
        self.height = height

    @property
    def query_nums(self):
        return self.uv.shape[-2]


class QueryBank(nn.Module):
    """Emit a normalized full-image UV grid as query points (inference-only)."""

    def __init__(self, offset=0):
        super().__init__()

        self.offset = offset

    def create_uv_grid(self, width, height, dtype=None, device=None):
        u = torch.arange(width, dtype=dtype, device=device)
        v = torch.arange(height, dtype=dtype, device=device)

        uu, vv = torch.meshgrid(u, v, indexing="xy")
        uv_grid = torch.stack((uu, vv), dim=-1) + self.offset

        return uv_grid

    @torch.no_grad()
    def forward(self, image):
        # Full-image grid at the input resolution; ``image`` is [..., H, W].
        height = int(image.shape[-2])
        width = int(image.shape[-1])

        uv_grid = self.create_uv_grid(width, height, dtype=image.dtype, device=image.device)

        uv_grid[..., 0] /= (width - 1)
        uv_grid[..., 1] /= (height - 1)
        uv_grid = torch.clamp(uv_grid, min=0.0, max=1.0)

        return BaseQuery(uv=uv_grid.reshape(-1, 2), full_uv=True, width=width, height=height)
