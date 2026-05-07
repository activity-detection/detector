# pyright: reportUnknownMemberType=false

import numpy as np
import torch
import sys
from collections import defaultdict, deque
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import cast

from src.detector.config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from src.detector.action import ActionVector
from src.detector.lstm import MultiClassLSTM
from src.detector.scene_lstm import FightLSTM
from src.detector.scene_features import SceneFeatureExtractor
from src.detector import logger


class Detector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        window = Config.LSTM_WINDOW
        self.track_history: defaultdict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=window))
        self.lying_history: defaultdict[int, deque[float]] = defaultdict(lambda: deque(maxlen=Config.LYING_SUSTAIN_FRAMES))
        self.scene_history: deque[np.ndarray] = deque(maxlen=Config.SCENE_WINDOW)
        self.crowd_history: deque[int] = deque(maxlen=Config.CROWD_SUSTAIN_FRAMES)

        self.scene_features = SceneFeatureExtractor()

        self.pose_model = YOLO(Config.POSE_MODEL_PATH)
        self.base_model = YOLO(Config.BASE_MODEL_PATH)

        self.fps = Config.FRAME_RATE

        self.action_model = MultiClassLSTM()
        self._load_or_die(self.action_model, "MultiClassLSTM", Config.LSTM_MODEL_PATH)

        self.fight_model = FightLSTM()
        if Config.RAW_FIGHT_LSTM_MODEL_PATH:
            self._load_or_die(self.fight_model, "FightLSTM", Config.FIGHT_LSTM_MODEL_PATH)
        else:
            logger.warning("FIGHT_LSTM_MODEL_PATH not configured — fight detection disabled")

    @staticmethod
    def _load_or_die(model, name: str, path: str):
        try:
            model.load_model()
        except FileNotFoundError:
            logger.critical(f"Cannot start system. {name} model file is missing: {path}")
            sys.exit(1)
        except RuntimeError as e:
            logger.critical(f"{name} model file exists, but architecture mismatch: {e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to load {name} due to an unexpected error: {e}", exc_info=True)
            sys.exit(1)

    def process_batch(self, frames: list[np.ndarray]) -> list[ActionVector]:
        vectors_base = self.detect_objects(frames)
        vectors_pose = self.detect_people_actions(frames)

        vector_list = [x + y for x, y in zip(vectors_base, vectors_pose)]

        return vector_list

    def detect_objects(self, frames: list[np.ndarray]) -> list[ActionVector]:
        results = self.base_model.track(frames, verbose=False, half=True)
        vector_list: list[ActionVector] = []
        for result in results:
            classes: list[str] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id in BASE_YOLO_MAPPING:
                        field_name = BASE_YOLO_MAPPING[class_id]
                        classes.append(field_name)
            vector = ActionVector(classes)
            vector.base_yolo_result = result
            vector_list.append(vector)

        return vector_list

    def detect_people_actions(self, frames: list[np.ndarray]) -> list[ActionVector]:
        results = self.pose_model.track(frames, persist=True, verbose=False, half=True)
        vector_list: list[ActionVector] = []
        for result in results:
            vector = ActionVector()
            vector.pose_results = result

            track_ids, keypoints, bboxes = self._extract_tracking_data(result)
            person_count = len(track_ids)
            vector.update({'person': person_count})

            lstm_labels: list[str] = []
            scene_persons: list[dict] = []

            for person_idx, track_id in enumerate(track_ids):
                kps = keypoints[person_idx]
                bbox = bboxes[person_idx]

                flat_kps = self.normalize(kps)
                self.track_history[track_id].append(flat_kps)
                if len(self.track_history[track_id]) == Config.LSTM_WINDOW:
                    label = self._predict_action(track_id)
                    if label != 'normal':
                        lstm_labels.append(label)

                aspect = self._bbox_aspect(bbox)
                self.lying_history[track_id].append(aspect)
                if self._is_lying_still(self.lying_history[track_id]):
                    lstm_labels.append('lying_still')

                scene_persons.append({"track_id": int(track_id), "kps": kps, "bbox": tuple(bbox)})

            vector.update(lstm_labels)

            scene_vec = self.scene_features.update(scene_persons)
            self.scene_history.append(scene_vec)
            if len(self.scene_history) == Config.SCENE_WINDOW and self.fight_model.is_loaded:
                seq = np.stack(list(self.scene_history))
                if self.fight_model.predict(seq) == 1:
                    vector.update(['fight'])

            self.crowd_history.append(person_count)
            if self._is_crowded():
                vector.update(['crowd'])

            vector_list.append(vector)
        return vector_list

    def _extract_tracking_data(self, result: Results) -> tuple[list[int], np.ndarray, np.ndarray]:
        track_ids: list[int] = []
        keypoints: np.ndarray = np.array([])
        bboxes: np.ndarray = np.array([])

        if result.boxes is not None and result.boxes.id is not None:
            if isinstance(result.boxes.id, torch.Tensor):
                track_ids = cast(list[int], result.boxes.id.int().cpu().tolist())
            else:
                track_ids = result.boxes.id.astype(int).tolist()

            if isinstance(result.boxes.xywh, torch.Tensor):
                bboxes = result.boxes.xywh.cpu().numpy()
            else:
                bboxes = result.boxes.xywh

        if result.keypoints is not None and hasattr(result.keypoints, 'xy'):
            if isinstance(result.keypoints.xy, torch.Tensor):
                keypoints = result.keypoints.xy.cpu().numpy()
            else:
                keypoints = result.keypoints.xy

        return track_ids, keypoints, bboxes

    def _predict_action(self, track_id: int) -> str:
        sequence = np.array(self.track_history[track_id])
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        action_id = int(self.action_model.predict(input_tensor))

        return LSTM_MAPPING[action_id]

    @staticmethod
    def _bbox_aspect(bbox: np.ndarray) -> float:
        w = float(bbox[2])
        h = float(bbox[3])
        if h <= 0:
            return 0.0
        return w / h

    @staticmethod
    def _is_lying_still(history: deque[float]) -> bool:
        if len(history) < Config.LYING_SUSTAIN_FRAMES:
            return False
        return all(a >= Config.LYING_ASPECT_THRESHOLD for a in history)

    def _is_crowded(self) -> bool:
        if len(self.crowd_history) < Config.CROWD_SUSTAIN_FRAMES:
            return False
        return all(n >= Config.CROWD_PERSON_THRESHOLD for n in self.crowd_history)

    @staticmethod
    def normalize(kps: np.ndarray) -> np.ndarray:
        l_sh, r_sh = kps[5], kps[6]
        l_hip, r_hip = kps[11], kps[12]

        hc_x = (l_hip[0] + r_hip[0]) / 2
        hc_y = (l_hip[1] + r_hip[1]) / 2

        sc_y = (l_sh[1] + r_sh[1]) / 2

        torso_h = np.abs(hc_y - sc_y)
        if torso_h < 1.0:
            torso_h = 1.0

        kps_norm = kps.copy().astype(np.float32)
        kps_norm[:, 0] = (kps[:, 0] - hc_x) / torso_h
        kps_norm[:, 1] = (kps[:, 1] - hc_y) / torso_h

        flat_kps = kps_norm[:, :2].flatten()
        return flat_kps
