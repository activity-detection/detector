"""
Inkrementalna ekstrakcja 10D scene-level features per klatka dla detekcji bójek.

Identyczna semantyka jak research-notebooks/scripts/fights_hockey/03_compute_features.py,
ale on-line: utrzymuje stan poprzedniej klatki (per-track kps i bbox) i policzy
velocities między bieżącą a poprzednią klatką.

Wektor cech (10D):
    f0 = n_persons
    f1 = max_kp_velocity      (najszybszy keypoint w klatce, dowolna osoba)
    f2 = mean_kp_velocity     (średnia po osobach i keypointach)
    f3 = max_arm_velocity     (kp 7,8,9,10 — łokcie/nadgarstki = punche)
    f4 = max_leg_velocity     (kp 13,14,15,16 — kolana/kostki = kopnięcia)
    f5 = max_bbox_velocity    (najszybciej poruszający się bbox)
    f6 = min_pair_dist        (najbliższa para bbox center)
    f7 = mean_pair_dist
    f8 = n_close_pairs        (par z dystansem < close_threshold)
    f9 = max_torso_tilt       (najbardziej pochylony tułów, deg od pionu)
"""

from __future__ import annotations

import numpy as np

KP_ARMS = (7, 8, 9, 10)
KP_LEGS = (13, 14, 15, 16)


class SceneFeatureExtractor:
    def __init__(self, close_threshold: float = 100.0):
        self.close_threshold = close_threshold
        self._prev: dict[int, dict] = {}

    def reset(self) -> None:
        self._prev = {}

    def update(self, persons: list[dict]) -> np.ndarray:
        """
        persons: lista dictów, jeden per osoba w bieżącej klatce. Każdy dict:
            'track_id': int
            'bbox':     (cx, cy, w, h)
            'kps':      np.ndarray shape (17, 2) — piksele
        Zwraca: feature vector shape (10,).
        """
        n = len(persons)
        out = np.zeros(10, dtype=np.float32)
        out[0] = n

        if n == 0:
            self._prev = {}
            return out

        per_person_kp_speeds: list[np.ndarray] = []
        per_person_bbox_speed: list[float] = []
        per_person_arm_max: list[float] = []
        per_person_leg_max: list[float] = []

        for p in persons:
            tid = p["track_id"]
            kps = p["kps"]
            bbox = p["bbox"]
            prev = self._prev.get(tid)
            if prev is None:
                continue
            d_kp = np.linalg.norm(kps - prev["kps"], axis=1)  # (17,)
            per_person_kp_speeds.append(d_kp)
            d_bbox = float(np.hypot(bbox[0] - prev["bbox"][0], bbox[1] - prev["bbox"][1]))
            per_person_bbox_speed.append(d_bbox)
            per_person_arm_max.append(float(np.nanmax(d_kp[list(KP_ARMS)])))
            per_person_leg_max.append(float(np.nanmax(d_kp[list(KP_LEGS)])))

        if per_person_kp_speeds:
            stacked = np.stack(per_person_kp_speeds)  # (P, 17)
            out[1] = float(np.nanmax(stacked))
            out[2] = float(np.nanmean(stacked))
            out[3] = float(max(per_person_arm_max))
            out[4] = float(max(per_person_leg_max))
            out[5] = float(max(per_person_bbox_speed))

        if n >= 2:
            cx = np.array([p["bbox"][0] for p in persons], dtype=np.float32)
            cy = np.array([p["bbox"][1] for p in persons], dtype=np.float32)
            dists: list[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(float(np.hypot(cx[i] - cx[j], cy[i] - cy[j])))
            out[6] = min(dists)
            out[7] = float(np.mean(dists))
            out[8] = sum(1 for d in dists if d < self.close_threshold)

        out[9] = self._max_torso_tilt(persons)

        self._prev = {p["track_id"]: {"kps": p["kps"].copy(), "bbox": p["bbox"]} for p in persons}
        return out

    @staticmethod
    def _max_torso_tilt(persons: list[dict]) -> float:
        max_angle = 0.0
        for p in persons:
            kps = p["kps"]
            sx = (kps[5, 0] + kps[6, 0]) / 2
            sy = (kps[5, 1] + kps[6, 1]) / 2
            hx = (kps[11, 0] + kps[12, 0]) / 2
            hy = (kps[11, 1] + kps[12, 1]) / 2
            dx = abs(sx - hx)
            dy = abs(sy - hy) + 1e-6
            angle = float(np.degrees(np.arctan2(dx, dy)))
            if angle > max_angle:
                max_angle = angle
        return max_angle
