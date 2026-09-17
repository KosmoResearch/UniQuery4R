"""Checkpoint resolution: local file path or Hugging Face Hub repo.

``--checkpoint`` / ``from_pretrained`` accept either

* a local ``.pth`` path, or
* a Hugging Face repo id such as ``"Kosmo-Research/UniQuery4R"`` (optionally with an
  explicit file name: ``"hf:Kosmo-Research/UniQuery4R:uniquery4r.pth"``), which is
  downloaded through ``huggingface_hub`` and cached locally.
"""

from __future__ import annotations

import os

DEFAULT_HF_REPO = "Kosmo-Research/UniQuery4R"
DEFAULT_CHECKPOINT_FILENAME = "uniquery4r.pth"


def looks_like_repo_id(value: str) -> bool:
    """Heuristic: ``org/name`` (optionally ``hf:``-prefixed) that is not a path."""
    stripped = value.removeprefix("hf:")
    stripped = stripped.split(":", 1)[0]
    if os.path.exists(value) or stripped.startswith((".", "/", "~")):
        return False
    parts = stripped.split("/")
    return len(parts) == 2 and all(parts)


def resolve_checkpoint(
    checkpoint: str,
    filename: str = DEFAULT_CHECKPOINT_FILENAME,
) -> str:
    """Return a local checkpoint path, downloading it from the HF Hub if needed."""
    if os.path.exists(checkpoint):
        return checkpoint

    if not looks_like_repo_id(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint!r} is neither an existing local "
            f"path nor a Hugging Face repo id ('org/name')."
        )

    repo_id = checkpoint.removeprefix("hf:")
    if ":" in repo_id:
        repo_id, filename = repo_id.split(":", 1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Downloading checkpoints from the Hugging Face Hub requires "
            "`huggingface_hub`; install it with `pip install huggingface_hub`."
        ) from exc

    print(f"[hub] downloading {filename!r} from {repo_id!r} ...")
    return hf_hub_download(repo_id=repo_id, filename=filename)
