"""Pair-wise cross-attention readout for query decoding.

For each ``(src, tgt)`` pair the aggregator builds multi-scale feature pyramids
from the encoder tokens, samples the source pyramids at the query UVs
(``src_hidden``), and lets those query tokens cross-attend the full target
feature map (``cross_readout``).

The pyramid construction and the cross-attention memory are query-independent:
``encode`` runs once per image set and its outputs are reused across query
chunks, so only the cheap sampling + cross-attention (``decode``) re-runs per
chunk. ``padding_mode="border"`` matches the trained model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.conv_blocks import ResNet


def _build_refinenet(cfg, input_channel):
    """Build the interp refine net (always a ``ResNet`` for this model)."""
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    kwargs["input_channel"] = input_channel
    return ResNet(**kwargs)


class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention + FFN residual block (nn.MultiheadAttention backend)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        qn = self.q_norm(q)
        kn = self.k_norm(mem)
        attn_out, _ = self.cross(qn, kn, kn, need_weights=False)
        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class PairCrossAttnAggregator(nn.Module):
    """Pair-wise cross-attention readout used by UniQuery4R.

    For each ``(src, tgt)`` pair: build multi-scale pyramid feature maps, sample
    the src pyramids at the query UVs (``src_hidden``), then let those query
    tokens cross-attend the tgt view's last-scale spatial tokens
    (``cross_readout``). Returns ``(cross_readout, src_hidden)``, each of dim
    ``embed_dims[-1]``.
    """

    _PADDING_MODE = "border"

    def __init__(
        self,
        patch_size=14,
        in_chans=[3072, 3072, 3072, 3072],
        embed_dims=[256, 256, 256, 256],
        upsample_scales=[4, 2, 1, 1],
        mode="bilinear",
        interp_refinenet_cfg=None,
        intermediate_layer_idx=None,
        cross_attn_dim: int = 256,
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 4,
        cross_ff_ratio: int = 4,
        cross_dropout: float = 0.0,
        align_corners=True,
    ) -> None:
        super().__init__()

        assert cross_attn_dim % cross_attn_heads == 0, (
            f"cross_attn_dim ({cross_attn_dim}) must be divisible by "
            f"cross_attn_heads ({cross_attn_heads})"
        )

        self.patch_size = patch_size
        self.mode = mode
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.upsample_scales = upsample_scales
        self.intermediate_layer_idx = intermediate_layer_idx
        self.align_corners = align_corners

        self.cross_attn_dim = int(cross_attn_dim)
        self.cross_attn_heads = int(cross_attn_heads)
        self.cross_attn_layers = int(cross_attn_layers)

        self.q_projs = nn.ModuleList()
        for ch0, ch1, s in zip(self.in_chans, self.embed_dims, self.upsample_scales):
            self.q_projs.append(nn.Linear(ch0, ch1 * s * s))

        self.interp_refinenet = nn.ModuleList()
        for i in range(len(self.in_chans)):
            self.interp_refinenet.append(_build_refinenet(interp_refinenet_cfg, self.embed_dims[i]))

        self.h_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for i in range(len(self.embed_dims) - 1):
            hidden_dim = self.embed_dims[i + 1]
            self.h_projs.append(nn.Linear(self.embed_dims[i], hidden_dim))
            self.gates.append(
                nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            )
            self.ffns.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            )

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in self.embed_dims])
        self.interp_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.embed_dims])
        self.fuse_norms = nn.ModuleList([nn.LayerNorm(d) for d in self.embed_dims[1:]])

        in_dim = int(self.embed_dims[-1])
        dim_ff = int(self.cross_attn_dim * cross_ff_ratio)

        self.query_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, self.cross_attn_dim), nn.GELU(),
        )
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, self.cross_attn_dim), nn.GELU(),
        )
        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnBlock(
                    d_model=self.cross_attn_dim,
                    n_heads=self.cross_attn_heads,
                    dim_ff=dim_ff,
                    dropout=float(cross_dropout),
                )
                for _ in range(self.cross_attn_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.cross_attn_dim), nn.Linear(self.cross_attn_dim, in_dim), nn.GELU(),
        )

    def _sample_and_fuse(self, layer_feats, uv_grid):
        """grid_sample per pyramid scale then gated hierarchical fusion."""
        sampled_feats_list = []
        for feats in layer_feats:
            input_dtype = feats.dtype
            sampled = F.grid_sample(
                feats.float(),
                uv_grid.float(),
                mode=self.mode,
                align_corners=self.align_corners,
                padding_mode=self._PADDING_MODE,
            )
            sampled = sampled.to(input_dtype)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # (B_pair, Q, embed_dim)
            sampled_feats_list.append(sampled)

        hidden = sampled_feats_list[0]
        for k in range(1, len(sampled_feats_list)):
            hidden = self.h_projs[k - 1](hidden)
            gate_weights = self.gates[k - 1](torch.cat([hidden, sampled_feats_list[k]], dim=-1))
            hidden = gate_weights * hidden + (1 - gate_weights) * sampled_feats_list[k]
            hidden = self.fuse_norms[k - 1](hidden)
            hidden = self.ffns[k - 1](hidden)

        return hidden

    def encode(self, x, pair_idx, patch_h, patch_w):
        """Query-independent stage: build src pyramids and cross-attn memory tokens.

        Run this once per image set; the returned tensors are reused across query
        chunks (they do not depend on the query UVs).

        Args:
            x: list of per-layer patch tokens, each (B, N, L, C) with L=patch_h*patch_w.
            pair_idx: list of (src, tgt) view index pairs.
            patch_h, patch_w: token grid size (input H/W // patch_size).

        Returns:
            src_pyramids: list of (B*P, C, H_i, W_i) per pyramid scale.
            memory_tokens: (B*P, H*W, D) projected tgt last-scale spatial tokens.
        """
        num_pair = len(pair_idx)

        if self.intermediate_layer_idx is not None:
            x = [x[idx] for idx in self.intermediate_layer_idx]
        assert len(x) == len(self.in_chans)

        B, N = x[0].shape[:2]

        # Build src pyramids (all scales); the tgt view is only needed at the last
        # scale as cross-attention memory (memory_from_encoder_last=False).
        src_pyramids = []
        mem_map = None
        last_idx = len(x) - 1
        for i, feats in enumerate(x):
            feats = self.q_projs[i](feats)  # (B, N, L, embed_dim * s^2)
            feats = feats.transpose(-1, -2).view(B * N, -1, patch_h, patch_w)
            feats = F.pixel_shuffle(feats, self.upsample_scales[i])  # (B*N, C, H, W)

            spatial_h = patch_h * self.upsample_scales[i]
            spatial_w = patch_w * self.upsample_scales[i]

            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
            feats = self.norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            feats = self.interp_refinenet[i](feats)
            feats = feats.permute(0, 2, 3, 1).contiguous().view(B * N, -1, feats.shape[1])
            feats = self.interp_norms[i](feats)
            feats = feats.permute(0, 2, 1).contiguous().view(B * N, -1, spatial_h, spatial_w)

            feats_bnchw = feats.view(B, N, -1, spatial_h, spatial_w)

            src_pyramid = torch.stack([feats_bnchw[:, src] for src, _ in pair_idx], dim=1)
            src_pyramids.append(src_pyramid.view(B * num_pair, *src_pyramid.shape[2:]))

            if i == last_idx:
                tgt_pyramid = torch.stack([feats_bnchw[:, tgt] for _, tgt in pair_idx], dim=1)
                mem_map = tgt_pyramid.view(B * num_pair, *tgt_pyramid.shape[2:])  # [B*P, C, H, W]

        mem_map = mem_map.flatten(2).transpose(1, 2).contiguous()  # [B*P, H*W, C_emb]
        memory_tokens = self.memory_proj(mem_map)  # [B*P, H*W, D]

        return src_pyramids, memory_tokens

    def decode(self, src_pyramids, memory_tokens, query):
        """Query-dependent stage: sample src pyramids at query UVs + cross-attn.

        Cheap enough to run per query chunk, reusing ``encode`` outputs.

        Returns:
            (cross_readout, src_hidden), each of shape (B*P, Q, embed_dims[-1]).
        """
        b_pair = src_pyramids[0].shape[0]

        # Inference queries are a shared [Q, 2] UV grid (see QueryBank), broadcast
        # to every (src, tgt) pair.
        uv_grid = query.uv
        assert uv_grid.ndim == 2, f"Expected uv_grid [Q, 2], got {tuple(uv_grid.shape)}"
        uv_grid = uv_grid * 2 - 1
        uv_grid = uv_grid.unsqueeze(0).unsqueeze(2).expand(b_pair, -1, -1, -1)

        src_hidden = self._sample_and_fuse(src_pyramids, uv_grid)

        query_tokens = self.query_proj(src_hidden)  # [B*P, Q, D]
        for blk in self.cross_blocks:
            query_tokens = blk(query_tokens, memory_tokens)
        cross_readout = self.readout(query_tokens)  # [B*P, Q, C]

        return cross_readout, src_hidden

    def forward(self, x, query, pair_idx, patch_h, patch_w):
        src_pyramids, memory_tokens = self.encode(x, pair_idx, patch_h, patch_w)
        return self.decode(src_pyramids, memory_tokens, query)
