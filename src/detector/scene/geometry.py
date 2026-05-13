from __future__ import annotations

import cv2
import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def foot_point(keypoints: np.ndarray | None, bbox_xyxy: np.ndarray) -> tuple[float, float]:
    """Ground-contact point of a person.

    Average of left/right ankle keypoints (COCO indices 15, 16) when both are
    valid; otherwise the bottom-center of the bounding box. YOLO emits (0, 0)
    for occluded/missing keypoints.
    """
    if keypoints is not None and len(keypoints) > 16:
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]
        l_valid = bool(l_ankle[0] > 0 and l_ankle[1] > 0)
        r_valid = bool(r_ankle[0] > 0 and r_ankle[1] > 0)
        if l_valid and r_valid:
            return (float((l_ankle[0] + r_ankle[0]) / 2),
                    float((l_ankle[1] + r_ankle[1]) / 2))
        if l_valid:
            return float(l_ankle[0]), float(l_ankle[1])
        if r_valid:
            return float(r_ankle[0]), float(r_ankle[1])

    x1, _y1, x2, y2 = bbox_xyxy
    return float((x1 + x2) / 2), float(y2)


def bbox_centroid(bbox_xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return float((x1 + x2) / 2), float((y1 + y2) / 2)
