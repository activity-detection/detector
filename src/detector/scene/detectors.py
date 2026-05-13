from __future__ import annotations

import numpy as np

from src.detector.scene.config import CrowdCfg
from src.detector.scene.geometry import point_in_polygon


def detect_forbidden_zones(
    foot_points: list[tuple[float, float]],
    zone_polygons: list[np.ndarray],
) -> int:
    if not zone_polygons or not foot_points:
        return 0
    count = 0
    for fp in foot_points:
        for poly in zone_polygons:
            if point_in_polygon(fp, poly):
                count += 1
                break
    return count


def detect_crowd(centroids: list[tuple[float, float]], cfg: CrowdCfg) -> int:
    """Size of the largest cluster within radius_px of some centroid.

    O(n²) pairwise. For each person, count how many neighbours (including
    self) sit within radius_px. Returns the max if it meets min_people,
    else 0.
    """
    n = len(centroids)
    if n < cfg.min_people:
        return 0

    pts = np.asarray(centroids, dtype=np.float32)
    r2 = float(cfg.radius_px) ** 2

    max_cluster = 0
    for i in range(n):
        diffs = pts - pts[i]
        d2 = np.sum(diffs * diffs, axis=1)
        cluster_size = int(np.sum(d2 <= r2))
        if cluster_size > max_cluster:
            max_cluster = cluster_size

    return max_cluster if max_cluster >= cfg.min_people else 0
