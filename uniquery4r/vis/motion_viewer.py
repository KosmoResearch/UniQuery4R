"""
Interactive scene-flow (warp3d) viewer using Viser.

``frame_points[t]`` is the world position at frame ``t`` of the points queried
in frame 0. It comes straight from the model's ``warp3d`` head (absolute
predicted point) or equivalently from ``global_points[0] + warp3d_delta(0->t)``;
frame 0 is the reference geometry ``global_points[0]``.

Motion is drawn as a per-query trail (line segments chaining
``frame_points[t-1] -> frame_points[t]``) with a moving point cloud at
``frame_points[t]``. Playback sweeps ``t``.

Expected inputs (numpy, world frame normalized to the first camera):
    * ``images``       (S, 3, H, W) float in [0, 1]
    * ``global_points``(S, H, W, 3) per-view world points (self-pair warp3d)
    * ``frame_points`` (S, H, W, 3) absolute per-frame positions of the frame-0 points
                        (index 0 = ``global_points[0]``)
    * ``delta_conf``   (S, H, W) optional per-query confidence (index 0 unused)
    * ``extrinsic``    (S, 3, 4) world-to-camera (optional, for frustums)
    * ``intrinsic``    (S, 3, 3) in image-pixel units (optional)
"""

import threading
import time
from typing import Optional

import numpy as np
import matplotlib.cm as cm

import viser
import viser.transforms as tf


class MotionViewer:
    """Interactive warp3d scene-flow viewer (Viser)."""

    def __init__(
        self,
        images: np.ndarray,
        global_points: np.ndarray,
        frame_points: np.ndarray,
        delta_conf: Optional[np.ndarray] = None,
        extrinsic: Optional[np.ndarray] = None,
        intrinsic: Optional[np.ndarray] = None,
        port: int = 8080,
        stride: int = 8,
        max_tracks: int = 20000,
        conf_thresh: float = 0.0,
        point_size: float = 0.002,
        bg_point_size: float = 0.005,
        show_camera: bool = False,
    ):
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large")

        self.images = images
        self.global_points = global_points
        self.frame_points = frame_points
        self.delta_conf = delta_conf
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.num_frames = int(global_points.shape[0])
        self.point_size = point_size
        self.bg_point_size = bg_point_size
        self.show_camera = show_camera and extrinsic is not None

        self._build_tracks(stride=max(1, int(stride)), max_tracks=int(max_tracks), conf_thresh=float(conf_thresh))
        self._build_background()
        self._build_cameras()

        self.seg_handles = []
        self.point_handles = []
        self._setup_gui()
        self._build_scene()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _build_tracks(self, stride: int, max_tracks: int, conf_thresh: float):
        """Sample frame-0 query points and gather their per-frame moved positions."""
        S, H, W, _ = self.global_points.shape

        # moved[t] = absolute frame-0-point position at frame t; shape (S, H*W, 3)
        moved = self.frame_points.reshape(S, -1, 3)

        # Sample a coarse pixel grid to keep the track/segment count manageable.
        grid = np.zeros((H, W), dtype=bool)
        grid[::stride, ::stride] = True
        mask = grid.reshape(-1)

        # Keep only finite tracks.
        finite = np.isfinite(moved).all(axis=(0, 2))
        mask &= finite

        # Confidence filter (percentile threshold over sampled queries).
        if self.delta_conf is not None and conf_thresh > 0:
            # Use the max confidence across target frames as the per-query score.
            conf = np.nan_to_num(self.delta_conf.reshape(S, -1), nan=-np.inf)
            score = conf[1:].max(axis=0) if S > 1 else conf[0]
            valid_scores = score[mask]
            if valid_scores.size:
                thr = np.percentile(valid_scores[np.isfinite(valid_scores)], conf_thresh) \
                    if np.isfinite(valid_scores).any() else -np.inf
                mask &= score >= thr

        idx = np.flatnonzero(mask)
        if idx.size > max_tracks:
            rng = np.random.default_rng(seed=0)
            idx = np.sort(rng.choice(idx, size=max_tracks, replace=False))

        self.moved = moved[:, idx, :]                          # (S, Q, 3)
        # Rainbow palette: one hue per track, spread over the sampled queries.
        self.track_colors = cm.get_cmap("hsv")(np.linspace(0, 1, idx.size, endpoint=False))[:, :3]
        self.track_colors_u8 = (self.track_colors * 255).astype(np.uint8)
        self.num_queries = idx.size

    def _build_background(self):
        """Per-frame scene cloud; only the displayed frame is shown, so moving
        content appears once at its position for that timestep."""
        self.bg_clouds = []
        for i in range(self.num_frames):
            p = self.global_points[i].reshape(-1, 3)
            c = np.transpose(self.images[i], (1, 2, 0)).reshape(-1, 3)
            finite = np.isfinite(p).all(axis=1)
            self.bg_clouds.append((p[finite][::16], np.clip(c[finite][::16], 0.0, 1.0)))

    def _build_cameras(self):
        """Precompute cam2world for frustum display."""
        self.cam_c2w = None
        if self.extrinsic is None:
            return
        ext = np.asarray(self.extrinsic, dtype=np.float64)
        R = ext[:, :3, :3]
        t = ext[:, :3, 3]
        Rinv = np.transpose(R, (0, 2, 1))
        c2w = np.zeros((ext.shape[0], 4, 4))
        c2w[:, 3, 3] = 1.0
        c2w[:, :3, :3] = Rinv
        c2w[:, :3, 3] = -np.einsum("sij,sj->si", Rinv, t)
        self.cam_c2w = c2w
        num = self.num_frames
        self.cam_colors = cm.get_cmap("viridis")(np.linspace(0, 1, max(num, 1)))

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _setup_gui(self):
        s = self.server
        with s.gui.add_folder("Playback"):
            self.gui_timestep = s.gui.add_slider(
                "Target Frame", min=0, max=self.num_frames - 1, step=1, initial_value=self.num_frames - 1
            )
            self.gui_playing = s.gui.add_checkbox("Playing", False)
            self.gui_fps = s.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=6)
            self.gui_cumulative = s.gui.add_checkbox(
                "Cumulative Trail", True,
                hint="On: keep the whole trail up to the current frame. Off: only the current step.",
            )

        with s.gui.add_folder("Display"):
            self.gui_show_trail = s.gui.add_checkbox("Show Trail", True)
            self.gui_show_points = s.gui.add_checkbox("Show Moving Points", True)
            self.gui_show_bg = s.gui.add_checkbox(
                "Show Background", True,
                hint="Show the current frame's scene geometry (switches with the timestep).",
            )
            self.gui_show_cams = s.gui.add_checkbox("Show Cameras", self.show_camera)
            self.gui_point_size = s.gui.add_number("Point Size", self.point_size, min=1e-4, max=1.0, step=1e-4)

        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_visibility()

        for _g in (self.gui_show_trail, self.gui_show_points,
                   self.gui_show_bg, self.gui_show_cams, self.gui_cumulative):
            _g.on_update(lambda _e: self._update_visibility())

        @self.gui_point_size.on_update
        def _(_) -> None:
            for h in self.point_handles:
                h.point_size = self.gui_point_size.value

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self):
        s = self.server
        s.scene.add_frame("/motion", show_axes=False)

        # Per-frame scene clouds (only the current timestep's is visible).
        self.bg_handles = []
        for i, (pts, cols) in enumerate(self.bg_clouds):
            if len(pts) == 0:
                self.bg_handles.append(None)
                continue
            self.bg_handles.append(
                s.scene.add_point_cloud(
                    f"/motion/background/{i}", points=pts, colors=cols,
                    point_size=self.bg_point_size, point_shape="circle",
                )
            )

        # Per-frame moving points and incremental trail segments.
        for t in range(self.num_frames):
            self.point_handles.append(
                s.scene.add_point_cloud(
                    f"/motion/points/{t}", points=self.moved[t], colors=self.track_colors_u8,
                    point_size=self.point_size, point_shape="circle",
                )
            )
            if t >= 1:
                seg = np.stack([self.moved[t - 1], self.moved[t]], axis=1)  # (Q, 2, 3)
                # add_line_segments wants a color per endpoint: (Q, 2, 3).
                seg_colors = np.repeat(self.track_colors_u8[:, None, :], 2, axis=1)
                self.seg_handles.append(
                    s.scene.add_line_segments(
                        f"/motion/trail/{t}", points=seg, colors=seg_colors, line_width=2.0,
                    )
                )
            else:
                self.seg_handles.append(None)

        self._build_camera_frustums()
        self._update_visibility()

    def _build_camera_frustums(self):
        self.cam_handles = []
        if self.cam_c2w is None:
            return
        for i in range(self.num_frames):
            R = self.cam_c2w[i, :3, :3]
            t = self.cam_c2w[i, :3, 3]
            q = tf.SO3.from_matrix(R).wxyz
            if self.intrinsic is not None:
                fx = float(self.intrinsic[i, 0, 0])
                cx = float(self.intrinsic[i, 0, 2])
                cy = float(self.intrinsic[i, 1, 2])
                fov = 2 * np.arctan(cx / fx)
                aspect = cx / max(cy, 1e-6)
            else:
                fov, aspect = np.deg2rad(60.0), 1.0
            color = tuple(int(255 * x) for x in self.cam_colors[i][:3])
            self.cam_handles.append(
                self.server.scene.add_camera_frustum(
                    f"/motion/cameras/{i}", fov=fov, aspect=aspect, wxyz=q, position=t,
                    scale=0.02, color=color,
                )
            )

    def _update_visibility(self):
        t = int(self.gui_timestep.value)
        cumulative = self.gui_cumulative.value
        show_trail = self.gui_show_trail.value
        show_points = self.gui_show_points.value

        with self.server.atomic():
            for s_idx in range(1, self.num_frames):
                seg = self.seg_handles[s_idx]
                if seg is not None:
                    seg.visible = show_trail and (s_idx <= t if cumulative else s_idx == t)
            for i, h in enumerate(self.point_handles):
                h.visible = show_points and (i == t)
            for i, h in enumerate(getattr(self, "bg_handles", [])):
                if h is not None:
                    h.visible = self.gui_show_bg.value and (i == t)
            for h in getattr(self, "cam_handles", []):
                h.visible = self.gui_show_cams.value
        self.server.flush()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(self, background_mode: bool = False):
        def _loop():
            while True:
                if self.gui_playing.value and self.num_frames > 1:
                    self.gui_timestep.value = (self.gui_timestep.value + 1) % self.num_frames
                time.sleep(1.0 / max(1, self.gui_fps.value))

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        print(f"[MotionViewer] serving on port {self.server.get_port()} "
              f"({self.num_queries} tracks x {self.num_frames} frames)")
        if not background_mode:
            while True:
                time.sleep(10.0)
