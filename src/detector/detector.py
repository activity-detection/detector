# pyright: reportUnknownMemberType=false

import numpy as np
import torch
import sys
from collections import defaultdict, deque
from ultralytics.engine.results import Results
from ultralytics import YOLO
from typing import cast

from src.detector.config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING, WINDOW_SIZE
from src.detector.action import ActionVector
from src.detector.lstm import MultiClassLSTM
from src.detector import logger


POSE_CONF = 0.3
POSE_IMGSZ = 640
STALE_TRACK_FRAMES = 5  # >tyle klatek bez detekcji -> wyzeruj historię tracku


class Detector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.track_history: defaultdict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
        self.track_last_seen: dict[int, int] = {}
        self.frame_counter: int = 0

        self.pose_model = YOLO(Config.POSE_MODEL_PATH)
        self.base_model = YOLO(Config.BASE_MODEL_PATH)

        self.fps = Config.FRAME_RATE

        self.action_model = MultiClassLSTM()
        try:
            self.action_model.load_model()
        except FileNotFoundError as e:
            logger.critical(f"Cannot start system. LSTM model file is missing: {Config.LSTM_MODEL_PATH}")
            sys.exit(1)
        except RuntimeError as e:
            logger.critical(f"LSTM model file exists, but architecture mismatch: {e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to load LSTM model due to an unexpected error: {e}", exc_info=True)
            sys.exit(1)
    
    def process_batch(self, frames: list[np.ndarray]) -> list[ActionVector]:
        vectors_base = self.detect_objects(frames)
        vectors_pose = self.detect_people_actions(frames)

        vector_list = [x + y for x, y in zip(vectors_base, vectors_pose)]

        return vector_list
    
    def detect_objects(self, frames: list[np.ndarray]) -> list[ActionVector]: # detects and count objects on frame using yolo
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
    
    def detect_people_actions(self, frames: list[np.ndarray]) -> list[ActionVector]: # detects people on frame using yolo pose and detects actions using lstm
        results = self.pose_model.track(
            frames,
            persist=True,
            verbose=False,
            conf=POSE_CONF,
            imgsz=POSE_IMGSZ,
        )
        vector_list: list[ActionVector] = []
        for result in results:
            self.frame_counter += 1
            vector = ActionVector()
            vector.pose_results = result
            if result.boxes is not None and result.boxes.id is not None:
                track_ids, keypoints = self._extract_tracking_data(result)
                person_count = len(track_ids)
                vector.update({'person': person_count})

                lstm_list: list[str] = []
                for person_idx, track_id in enumerate(track_ids):
                    kps = keypoints[person_idx]
                    flat_kps = self.normalize(kps)
                    self._update_track_history(track_id, flat_kps)

                    if len(self.track_history[track_id]) == WINDOW_SIZE:
                        action_name = self._predict_action(track_id)
                        lstm_list.append(action_name)

                vector.update(lstm_list)
            self._evict_stale_tracks()
            vector_list.append(vector)
        return vector_list

    def _update_track_history(self, track_id: int, flat_kps: np.ndarray) -> None:
        """Append klatki z forward-fill na krótkie luki (<=STALE_TRACK_FRAMES)
        i resetem dla długich. Bez tego okno LSTM stitchuje nieciągłe segmenty
        czasu (np. tracking zgubił osobę na 2s i wrócił -> pierwsze 28 klatek
        z przed luki + 2 nowe = wektor "skleja" ruch sprzed/po)."""
        last_seen = self.track_last_seen.get(track_id)
        if last_seen is not None:
            gap = self.frame_counter - last_seen - 1
            if gap > STALE_TRACK_FRAMES:
                self.track_history[track_id].clear()
            elif gap > 0 and len(self.track_history[track_id]) > 0:
                last_kps = self.track_history[track_id][-1]
                for _ in range(gap):
                    self.track_history[track_id].append(last_kps)
        self.track_history[track_id].append(flat_kps)
        self.track_last_seen[track_id] = self.frame_counter

    def _evict_stale_tracks(self) -> None:
        threshold = STALE_TRACK_FRAMES * 2
        stale = [tid for tid, ls in self.track_last_seen.items()
                 if self.frame_counter - ls > threshold]
        for tid in stale:
            self.track_last_seen.pop(tid, None)
            self.track_history.pop(tid, None)

    def _extract_tracking_data(self, result: Results) -> tuple[list[int], np.ndarray]:
        track_ids: list[int] = []
        keypoints: np.ndarray = np.array([]) 

        if result.boxes is not None and result.boxes.id is not None:
            if isinstance(result.boxes.id, torch.Tensor):
                track_ids = cast(list[int], result.boxes.id.int().cpu().tolist())
            else:
                track_ids = result.boxes.id.astype(int).tolist()

        if result.keypoints is not None and hasattr(result.keypoints, 'xy'):
            if isinstance(result.keypoints.xy, torch.Tensor):
                keypoints = result.keypoints.xy.cpu().numpy()
            else:
                keypoints = result.keypoints.xy

        return track_ids, keypoints
    
    def _predict_action(self, track_id: int) -> str:
        sequence = np.array(self.track_history[track_id])
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device) 

        action_id = int(self.action_model.predict(input_tensor))

        return LSTM_MAPPING[action_id]
    
    @staticmethod
    def normalize(kps: np.ndarray) -> np.ndarray: # keypoints normalization relative to torso
        l_sh, r_sh = kps[5], kps[6]
        l_hip, r_hip = kps[11], kps[12]

        hc_x = (l_hip[0] + r_hip[0]) / 2
        hc_y = (l_hip[1] + r_hip[1]) / 2
        
        sc_y = (l_sh[1] + r_sh[1]) / 2
        
        torso_h = np.abs(hc_y - sc_y)
        if torso_h < 1.0: torso_h = 1.0 

        kps_norm = kps.copy().astype(np.float32)
        kps_norm[:, 0] = (kps[:, 0] - hc_x) / torso_h
        kps_norm[:, 1] = (kps[:, 1] - hc_y) / torso_h

        flat_kps = kps_norm[:, :2].flatten()
        return flat_kps