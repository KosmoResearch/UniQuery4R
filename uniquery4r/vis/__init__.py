"""Visualization / export helpers for UniQuery4R."""

from uniquery4r.vis.visual_util import (
    blend_motion_overlay,
    cameras_to_ply,
    extract_colored_point_cloud,
    predictions_to_glb,
    predictions_to_ply,
    save_gif,
)

__all__ = [
    "blend_motion_overlay",
    "cameras_to_ply",
    "extract_colored_point_cloud",
    "predictions_to_glb",
    "predictions_to_ply",
    "save_gif",
]

# The interactive viewer pulls in optional deps (viser, matplotlib). Keep it
# lazily importable so the export-only path works without them installed.
try:
    from uniquery4r.vis.motion_viewer import MotionViewer  # noqa: F401

    __all__.append("MotionViewer")
except ImportError:
    pass
