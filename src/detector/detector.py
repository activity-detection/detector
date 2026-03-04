import torch
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

from .lstm import MultiClassLSTM
from .config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from .action import ActionVector

class Detector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.track_history = defaultdict(lambda: deque(maxlen=60)) # maxlen eq lstm input

        self.pose_model = YOLO(Config.POSE_MODEL_PATH)
        self.yolo_base = YOLO(Config.BASE_MODEL_PATH)

        self.fps = Config.FRAME_RATE
        
        self._load_lstm_model()

    def _load_lstm_model(self):  # loads lstm TODO borys przesuń to do lstm.py
        self.detection_model = MultiClassLSTM()

        lstm_path = Config.LSTM_MODEL_PATH
        
        try:
            checkpoint = torch.load(lstm_path, map_location=self.device)
            self.detection_model.load_state_dict(checkpoint)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            print(f"[INFO] LSTM Model correctly loaded from {lstm_path}")
        except FileNotFoundError:
            print(f"[ERROR] File not found: {lstm_path}")
            self.detection_model = None
        except Exception as e:
            print(f"[ERROR] Error during loading LSTM model: {e}")
            self.detection_model = None

    def predict_action(self, sequence_tensor): # predicts action using lstm, TODO to też przesuń
        if self.detection_model is None:
            return "LSTM is not loaded!"

        sequence_tensor = sequence_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.detection_model(sequence_tensor)
            
            _, predicted_idx = torch.max(outputs, 1)
            class_id = predicted_idx.item()
            
        return class_id
    
    def process_batch(self, frames):
        vectors_base = self.detect_base_yolo(frames)
        vectors_pose = self.detect_yolo_pose(frames)

        vector_list = [x + y for x, y in zip(vectors_base, vectors_pose)]

        return vector_list

    
    def detect_base_yolo(self, frames): # detects and count objects on frame using yolo
        results = self.yolo_base.track(frames, verbose=False, half=True)
        vector_list = []
        for result in results:
            classes = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in BASE_YOLO_MAPPING:
                    field_name = BASE_YOLO_MAPPING[class_id]
                    classes.append(field_name)
            vector = ActionVector(classes)
            vector.base_yolo_result = result
            vector_list.append(vector)
                
        return vector_list
    
    def detect_yolo_pose(self, frames): # detects people on frame using yolo pose and detects actions using lstm
        results = self.pose_model.track(frames, persist=True, verbose=False, half=True)
        vector_list = []
        for result in results:
            vector = ActionVector()
            vector.pose_results = result
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                person = {'person' : len(track_ids)}
                vector.update(person)

                keypoints = result.keypoints.xy.cpu().numpy()
                lstm_list = []
                for person_idx, track_id in enumerate(track_ids):
                    kps = keypoints[person_idx]
        
                    flat_kps = self.normalize(kps)

                    self.track_history[track_id].append(flat_kps)

                    action_id = 0

                    if len(self.track_history[track_id]) == 60: # TODO Borys popraw to bo mi się nie podoba. Daj to do innej funkcji czy coś

                        sequence = np.array(self.track_history[track_id])
                        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device) 

                        action_id = self.predict_action(input_tensor)
                        lstm_list.append(LSTM_MAPPING[action_id])
                vector.update(lstm_list)
            vector_list.append(vector)
        return vector_list
    
    @staticmethod
    def normalize(kps): # keypoints normalization relative to torso
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