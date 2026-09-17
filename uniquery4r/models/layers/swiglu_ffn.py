# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, Optional

import torch.nn.functional as F
from torch import Tensor, nn


def _fsdp_enabled() -> bool:
    # accelerate sets this when launching with distributed_type: FSDP
    return os.environ.get("ACCELERATE_USE_FSDP", "false").lower() in ("true", "1", "yes")


def _round_swiglu_hidden_dim(hidden_features: int) -> int:
    return (int(hidden_features * 2 / 3) + 7) // 8 * 8


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class _SwiGLUFFNFusedSafe(SwiGLUFFN):
    """FSDP-safe SwiGLU FFN using standard nn.Linear (no xformers fused op)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = _round_swiglu_hidden_dim(hidden_features)
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
            bias=bias,
        )


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


if XFORMERS_AVAILABLE and not _fsdp_enabled():

    class SwiGLUFFNFused(SwiGLU):
        def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = None,
            drop: float = 0.0,
            bias: bool = True,
        ) -> None:
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            hidden_features = _round_swiglu_hidden_dim(hidden_features)
            super().__init__(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                bias=bias,
            )

else:
    SwiGLUFFNFused = _SwiGLUFFNFusedSafe
