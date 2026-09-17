"""UniQuery4R: unified 4D scene reconstruction from a single query.

A variable-length multi-view clip is jointly encoded once; at decoding time a
continuous source-pixel query ``q = (u, v, src, tgt)`` selects the source and
target views. Each query jointly predicts the target correspondence
(``warp2d``), the target-time 3D position (``warp3d``), the scene flow
(``warp3d_delta``, parameterized as direction x magnitude) and the source
depth, while camera parameters are estimated per view.

Stages:
    Stage 1  fuse_encoder            -> multi-scale patch features (multi-view ViT)
    Stage 2  query_bank              -> dense UV query grid (or caller-supplied UVs)
    Stage 3  camera_head             -> pose_enc -> extrinsic / intrinsic
    Stage 4  pair cross-attn + heads -> per-view depth / points + cross-view motion

Per-view geometry is read off the self-pairs ``(v, v)`` and cross-view motion is
kept per pair:
    pair_idx = [(0, i) for i in 1..N-1] + [(i, i) for i in 0..N-1]
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from uniquery4r.models.encoder import DinoV2
from uniquery4r.models.heads.camera_head import CameraHead
from uniquery4r.models.heads.mlp_head import Head
from uniquery4r.models.pair_aggregator import PairCrossAttnAggregator
from uniquery4r.models.query_bank import BaseQuery, QueryBank
from uniquery4r.utils.geometry import normalize_cameras_to_first
from uniquery4r.utils.pose_enc import pose_encoding_to_extri_intri


# Backbone config for the DA3 ViT-g encoder used by the 4D model.
_VITG = dict(embed_dim=1536, out_layers=[19, 27, 33, 39], alt_start=13)
# ``query_pair_decoder`` final-layer width: 3+1 (warp3d) + 3+1+1 (scene flow)
# + 2+4 (warp2d) = 15 outputs.
_MOTION_HEAD_OUTPUT_DIM = 15


class UniQuery4R(nn.Module):
    """UniQuery4R inference network."""

    def __init__(
        self,
        patch_size: int = 14,
        chunk_size: Optional[int] = 30000,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.chunk_size = chunk_size

        cfg = _VITG
        dim_in = cfg["embed_dim"] * 2  # cat_token=True -> 3072

        self.fuse_encoder = DinoV2(
            name="vitg",
            out_layers=cfg["out_layers"],
            alt_start=cfg["alt_start"],
            qknorm_start=cfg["alt_start"],
            rope_start=cfg["alt_start"],
            cat_token=True,
            num_register_tokens=0,
            register_attention_block_indices=[],
            patch_size=patch_size,
        )

        self.query_bank = QueryBank()

        self.camera_head = CameraHead(
            dim_in=dim_in,
            trunk_depth=4,
            pose_encoding_type="absT_quaR_FoV",
            num_heads=16,
            mlp_ratio=4,
            init_values=0.01,
            trans_act="linear",
            quat_act="linear",
            fl_act="relu",
            num_iterations=4,
        )

        self.query_pair_feats_aggregator = PairCrossAttnAggregator(
            patch_size=patch_size,
            intermediate_layer_idx=[0, 1, 2, 3],
            in_chans=[dim_in, dim_in, dim_in, dim_in],
            embed_dims=[256, 256, 256, 256],
            upsample_scales=[4, 2, 1, 1],
            mode="bilinear",
            interp_refinenet_cfg=dict(
                type="ResNet",
                layer_num=1,
                use_bn=False,
            ),
            cross_attn_dim=256,
            cross_attn_heads=8,
            cross_attn_layers=4,
            cross_ff_ratio=4,
            cross_dropout=0.0,
            align_corners=True,
        )

        self.query_pair_decoder = self._build_query_pair_decoder()

        self.query_pair_decoder2 = Head(
            in_chan=256,
            hidden_dim=64,
            names=dict(pair_depth=1, pair_confidence=1),
            acts=dict(pair_depth="exp", pair_confidence=""),
        )

    @staticmethod
    def _build_query_pair_decoder() -> Head:
        names = dict(
            warp3d=3,
            warp3d_confidence=1,
            warp3d_delta_direction=3,
            warp3d_delta_magnitude=1,
            warp3d_delta_confidence=1,
            warp2d=2,
            warp2d_confidence=4,
        )
        acts = dict(
            warp3d="inv_log",
            warp3d_confidence="",
            warp3d_delta_direction="norm",
            warp3d_delta_magnitude="softplus",
            warp3d_delta_confidence="",
            warp2d="",
            warp2d_confidence="",
        )
        return Head(
            in_chan=256,
            hidden_dim=64,
            names=names,
            acts=acts,
        )

    def _validate_motion_head_from_state_dict(self, state: dict) -> None:
        weight_key = "query_pair_decoder.mlp.4.weight"
        weight = state.get(weight_key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise KeyError(
                f"Checkpoint does not contain a valid {weight_key!r}; "
                "cannot determine the UniQuery4R motion-head output dimension"
            )

        output_dim = int(weight.shape[0])
        if output_dim != _MOTION_HEAD_OUTPUT_DIM:
            raise ValueError(
                f"Unsupported motion-head output dimension {output_dim}; "
                f"expected {_MOTION_HEAD_OUTPUT_DIM}"
            )

    @property
    def motion_output_names(self):
        return tuple(self.query_pair_decoder.names)

    @staticmethod
    def _build_pair_idx(num_views: int):
        """Cross-pairs (0, i) for every other view plus one self-pair (i, i) per view."""
        cross = [(0, i) for i in range(1, num_views)]
        self_pairs = [(i, i) for i in range(num_views)]
        return cross + self_pairs

    def pair_forward(self, src_pyramids, memory_tokens, query):
        """Query-dependent pair decode + two MLP decoders (return_src_hidden path).

        ``src_pyramids`` / ``memory_tokens`` come from
        ``query_pair_feats_aggregator.encode`` and are reused across query chunks,
        so the query-independent feature extraction is not recomputed per chunk.
        """
        feats, single_feats = self.query_pair_feats_aggregator.decode(
            src_pyramids, memory_tokens, query,
        )
        results = self.query_pair_decoder(x=feats)
        single_results = self.query_pair_decoder2(x=single_feats)
        results.update(single_results)
        return results

    def forward(
        self,
        images: torch.Tensor,
        query_uv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            images: [N,3,H,W] or [B,N,3,H,W], values in [0, 1].
            query_uv: optional sparse query UVs in ``[0, 1]``, shape ``[Q, 2]``
                (u, v). When provided, pair outputs stay ``[B, P, Q, C]`` instead
                of being reshaped to a dense ``H×W`` grid (WorldTrack eval path).
        Returns:
            dict with per-view depth / confidence / global_points / global_confidence,
            pose_enc / extrinsic / intrinsic, cross-view motion (warp*), pair_idx,
            and the input images. Sparse mode also sets ``sparse_query=True``.
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"Expected images [B,N,3,H,W], got {tuple(images.shape)}")

        b, n, _, h, w = images.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"H/W must be divisible by patch_size={self.patch_size}, got {(h, w)}"
            )

        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        pair_idx = self._build_pair_idx(n)
        num_pair = len(pair_idx)
        sparse = query_uv is not None

        amp_dtype = torch.float16
        device_type = images.device.type

        pose_enc = None
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=images.is_cuda):
            # Stage 1: dense feature extraction
            patch_features, patch_start_idx = self.fuse_encoder(images)
            if not isinstance(patch_features, (list, tuple)):
                patch_features = [patch_features]
            patch_tokens = [f[:, :, patch_start_idx:] for f in patch_features]

            # Stage 2: dense full-grid query, or caller-provided sparse UVs.
            if sparse:
                uv = query_uv
                if not isinstance(uv, torch.Tensor):
                    uv = torch.as_tensor(uv)
                if uv.ndim != 2 or uv.shape[-1] != 2:
                    raise ValueError(f"query_uv must be [Q, 2], got {tuple(uv.shape)}")
                uv = uv.to(device=images.device, dtype=images.dtype)
                query = BaseQuery(uv=uv, full_uv=False, width=w, height=h)
            else:
                query = self.query_bank(images)
            query_nums = query.query_nums

            # Stage 5: camera tokens -> pose encoding
            camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            pose_enc = self.camera_head(camera_tokens)

            # Stage 7: pair-wise cross-view matching.
            # The pyramid features / cross-attn memory are query-independent, so build
            # them once and only sample + cross-attend per query chunk.
            src_pyramids, memory_tokens = self.query_pair_feats_aggregator.encode(
                patch_tokens, pair_idx, patch_h, patch_w,
            )
            if self.chunk_size is not None and self.chunk_size > 1 and query_nums > self.chunk_size:
                chunk_list = []
                for q0 in range(0, query_nums, self.chunk_size):
                    q1 = min(q0 + self.chunk_size, query_nums)
                    chunk_query = BaseQuery(uv=query.uv[q0:q1])
                    chunk_list.append(self.pair_forward(src_pyramids, memory_tokens, chunk_query))
                pair_results = {}
                for key in chunk_list[0].keys():
                    pair_results[key] = torch.cat([res[key] for res in chunk_list], dim=1)
            else:
                pair_results = self.pair_forward(src_pyramids, memory_tokens, query)

        predictions: Dict[str, torch.Tensor] = {}
        self_ids = [pair_idx.index((v, v)) for v in range(n)]

        if sparse:
            # Keep query dimension explicit: [B, P, Q, C].
            pair_results = {
                k: v.float().view(b, num_pair, query_nums, v.shape[-1])
                for k, v in pair_results.items()
            }
            # Sparse per-view geometry: [B, N, Q, C] (no H×W layout).
            predictions["depth"] = pair_results["pair_depth"][:, self_ids]
            predictions["confidence"] = pair_results["pair_confidence"][:, self_ids]
            predictions["global_points"] = pair_results["warp3d"][:, self_ids]
            predictions["global_confidence"] = pair_results["warp3d_confidence"][:, self_ids]
            # Cross-view motion: [B, P, Q, C].
            for key in self.motion_output_names:
                predictions[key] = pair_results[key]
            predictions["sparse_query"] = True
            predictions["query_uv"] = query.uv.detach()
        else:
            # Dense: [B, P, Q, C] with Q = H*W -> [B, P, H, W, C].
            pair_results = {
                k: v.float().view(b, num_pair, h, w, v.shape[-1])
                for k, v in pair_results.items()
            }
            # Per-view geometry from the self-pairs (v, v): [B, N, C, H, W].
            predictions["depth"] = pair_results["pair_depth"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
            predictions["confidence"] = pair_results["pair_confidence"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
            predictions["global_points"] = pair_results["warp3d"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
            predictions["global_confidence"] = (
                pair_results["warp3d_confidence"][:, self_ids].permute(0, 1, 4, 2, 3).contiguous()
            )

            # Cross-view motion (kept per pair): [B, P, C, H, W].
            # ``warp3d`` is the absolute predicted world point of the src-view query at
            # the tgt frame; ``warp3d_delta`` is its (independently decoded) scene flow.
            for key in self.motion_output_names:
                predictions[key] = pair_results[key].permute(0, 1, 4, 2, 3).contiguous()

        predictions["pair_idx"] = pair_idx

        # Decode cameras in fp32 OUTSIDE autocast (quaternion->matrix + normalization
        # are precision sensitive; fp16 makes poses drift).
        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]
        predictions["pose_enc"] = pose_enc
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding=pose_enc.float(),
            image_size_hw=(h, w),
            build_intrinsics=True,
        )
        extrinsics = normalize_cameras_to_first(extrinsics)
        predictions["extrinsic"] = extrinsics[..., :3, :]
        predictions["intrinsic"] = intrinsics

        predictions["images"] = images
        return predictions

    def load_checkpoint(
        self,
        checkpoint,
        strict: bool = True,
        map_location: str = "cpu",
        weights_only: bool = True,
    ) -> None:
        """Load a state_dict; unwraps common wrappers and strips ``model.`` prefix.

        Checkpoints are loaded with ``weights_only=True`` by default; if the file
        contains non-tensor objects (plain state_dicts should not), loading falls
        back to full deserialization with a warning — only use checkpoints whose
        origin you trust. Pass ``weights_only=False`` to skip the safe path.
        """
        try:
            state = torch.load(
                os.fspath(checkpoint), map_location=map_location, weights_only=weights_only
            )
        except Exception as exc:  # e.g. UnpicklingError from non-tensor objects
            if not weights_only:
                raise
            print(
                "[UniQuery4R] safe (weights_only) load failed, retrying with full "
                f"deserialization — make sure the checkpoint is trusted. Reason: {exc}"
            )
            state = torch.load(
                os.fspath(checkpoint), map_location=map_location, weights_only=False
            )
        if isinstance(state, dict):
            for key in ("state_dict", "model", "module"):
                inner = state.get(key)
                if isinstance(inner, dict) and inner and all(
                    isinstance(k, str) for k in list(inner.keys())[:8]
                ):
                    state = inner
                    break
        if isinstance(state, dict) and state and all(
            isinstance(k, str) and k.startswith("model.") for k in state.keys()
        ):
            state = {k[len("model."):]: v for k, v in state.items()}
        if not isinstance(state, dict):
            raise TypeError(
                f"Checkpoint must resolve to a state_dict, got {type(state).__name__}"
            )
        self._validate_motion_head_from_state_dict(state)
        missing, unexpected = self.load_state_dict(state, strict=strict)
        if missing:
            suffix = "..." if len(missing) > 8 else ""
            print(f"[UniQuery4R] missing keys ({len(missing)}): {missing[:8]}{suffix}")
        if unexpected:
            suffix = "..." if len(unexpected) > 8 else ""
            print(f"[UniQuery4R] unexpected keys ({len(unexpected)}): {unexpected[:8]}{suffix}")

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Optional[str] = None,
        *,
        strict: bool = True,
        device: Optional[str] = None,
        **model_kwargs,
    ) -> "UniQuery4R":
        """Build a model from a local checkpoint path or a Hugging Face repo id.

        Args:
            checkpoint: local ``.pth`` path or HF repo id (e.g.
                ``"Kosmo-Research/UniQuery4R"``). Defaults to
                ``uniquery4r.utils.hub.DEFAULT_HF_REPO``.
            device: torch device string; ``None`` keeps the model on CPU.
        """
        from uniquery4r.utils.hub import DEFAULT_HF_REPO, resolve_checkpoint

        model = cls(**model_kwargs).eval()
        model.load_checkpoint(resolve_checkpoint(checkpoint or DEFAULT_HF_REPO), strict=strict)
        if device is not None:
            model = model.to(device)
        return model
