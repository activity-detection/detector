from __future__ import annotations

import numpy as np
import torch
from ultralytics.engine.results import Results

from src.detector import logger
from src.detector.scene.config import SceneConfig
from src.detector.scene.detectors import detect_crowd, detect_forbidden_zones
from src.detector.scene.geometry import bbox_centroid, foot_point
from src.detector.vectors import ActionVector


class SceneAnalyzer:
    """Per-frame scene analysis: forbidden zones + crowd clustering.

    Consumes YOLO pose Results already produced by the main Detector — no
    extra inference. Stateless beyond the parsed config and pre-compiled
    polygon arrays.
    """

    def __init__(self, cfg: SceneConfig, fps: float):
        self.cfg = cfg
        self.fps = fps
        self._forbidden_polys: list[np.ndarray] = [
            np.array(z.points, dtype=np.int32)
            for z in cfg.zones
            if z.policy == "forbidden"
        ]
        self._frame_size_warned = False

    def process(self, pose_result: Results) -> ActionVector:
        vector = ActionVector()

        if pose_result.boxes is None or pose_result.boxes.id is None:
            return vector

        self._maybe_warn_frame_size(pose_result)

        boxes_xyxy = self._to_numpy(pose_result.boxes.xyxy)
        keypoints_xy = self._extract_keypoints(pose_result)

        centroids = [bbox_centroid(b) for b in boxes_xyxy]

        if self._forbidden_polys:
            foot_points = [
                foot_point(
                    keypoints_xy[i] if i < len(keypoints_xy) else None,
                    boxes_xyxy[i],
                )
                for i in range(len(boxes_xyxy))
            ]
            n = detect_forbidden_zones(foot_points, self._forbidden_polys)
            if n > 0:
                vector.update({"forbidden_zone": n})

        if self.cfg.crowd is not None and centroids:
            n = detect_crowd(centroids, self.cfg.crowd)
            if n > 0:
                vector.update({"crowd": n})

        return vector

    def _maybe_warn_frame_size(self, pose_result: Results) -> None:
        if self._frame_size_warned:
            return
        try:
            h, w = pose_result.orig_shape
        except Exception:
            return
        cw, ch = self.cfg.frame_size
        if (cw, ch) != (w, h):
            logger.warning(
                "Scene config frame_size (%dx%d) does not match source frame "
                "(%dx%d); polygon coordinates will not auto-scale.",
                cw, ch, w, h,
            )
        self._frame_size_warned = True

    @staticmethod
    def _to_numpy(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return np.asarray(t)

    @staticmethod
    def _extract_keypoints(pose_result: Results) -> np.ndarray:
        if pose_result.keypoints is None:
            return np.array([])
        xy = pose_result.keypoints.xy
        if isinstance(xy, torch.Tensor):
            return xy.cpu().numpy()
        return np.asarray(xy)
