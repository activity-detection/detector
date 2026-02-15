import torch
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

from .lstm import MultiClassLSTM, CLASS_NAMES
from .config import Config
from .action import ActionVector

class Detector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.track_history = defaultdict(lambda: deque(maxlen=60))
        
        self.CLASS_NAMES = CLASS_NAMES

        self.BASE_YOLO_MAPPING = {
            1: 'count_bicycle',
            2: 'count_car',
            14: 'count_bird',
            15: 'count_cat',
            16: 'count_dog',
            26: 'count_handbag',
            28: 'count_suitcase',
            43: 'count_knife'
        }

        self.pose_model = YOLO(Config.POSE_MODEL_PATH)
        self.yolo_base = YOLO(Config.BASE_MODEL_PATH)

        self.fps = Config.FRAME_RATE
        
        self._load_lstm_model()

    def _load_lstm_model(self):
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

    def predict_action(self, sequence_tensor):
        """
        Wykonuje predykcję akcji na podstawie sekwencji 30 klatek.
        
        Args:
            sequence_tensor (torch.Tensor): Tensor o kształcie (1, 30, 34)
                                            Batch=1, Seq=30, Features=34
        Returns:
            str: Nazwa wykrytej akcji (np. "Boksowanie")
        """

        if self.detection_model is None:
            return "LSTM się nie załadował!"

        sequence_tensor = sequence_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.detection_model(sequence_tensor)
            
            _, predicted_idx = torch.max(outputs, 1)
            class_id = predicted_idx.item()
            
        # Mapping ID on names
        return class_id, self.CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
    
    def process_batch(self, frames):
        vector_base = self.detect_base_yolo(frames)
        vector_pose = self.detect_yolo_pose(frames)

        vector_list = [x + y for x, y in zip(vector_base, vector_pose)]

        return vector_list

    
    def detect_base_yolo(self, frames) -> ActionVector:
        results = self.yolo_base.track(frames, verbose=False, half=True)
        vector_list = []
        for result in results:
            counts = {field: 0 for field in self.BASE_YOLO_MAPPING.values()}
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                
                if class_id in self.BASE_YOLO_MAPPING:
                    field_name = self.BASE_YOLO_MAPPING[class_id]
                    counts[field_name] += 1
            vector = ActionVector(**counts)
            vector_list.append(vector)
                
        return vector_list
    
    def detect_yolo_pose(self, frames):
        results = self.pose_model.track(frames, persist=True, verbose=False, half=True)
        vector_list = []
        for result in results:
            vector = ActionVector()
            vector.pose_results = result
            if result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                vector.count_person = len(track_ids)

                keypoints = result.keypoints.xy.cpu().numpy()

                for person_idx, track_id in enumerate(track_ids):
                    kps = keypoints[person_idx]
        
                    flat_kps = self.normalize(kps)

                    self.track_history[track_id].append(flat_kps)

                    action_id = 0
                    action_label = "inne"

                    if len(self.track_history[track_id]) == 60:

                        sequence = np.array(self.track_history[track_id])
                        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device) 

                        action_id, action_label = self.predict_action(input_tensor)
                        if action_label == 'squat':
                            vector.count_squat += 1
                        if action_label == 'jumping_jacks':
                            vector.count_jumping_jacks += 1
            vector_list.append(vector)
        return vector_list
    
    @staticmethod
    def normalize(kps):
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