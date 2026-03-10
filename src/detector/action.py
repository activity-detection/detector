from collections import deque, Counter
import csv
import cv2
import threading
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
import os

from .config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from .anonymizer import Anonymizer

MIN_TRIGGER_COUNT = 10

class ActionState(Enum):
    IDLE = auto()
    RECORDING = auto()

class ActionVector:
    pose_results = None
    base_yolo_result = None

    def __init__(self, data=None):
        self.counter = Counter()
        if data is not None:
            self.counter.update(data)

    def update(self, data):
        self.counter.update(data)

    def __add__(self, other: 'ActionVector') -> 'ActionVector':
        """adds values from two vectors,
        takes results from first vector if present otherwise from second"""
        if not isinstance(other, ActionVector):
            return NotImplemented
        
        new_vector = ActionVector()
        new_vector.update(self.counter)
        new_vector.update(other.counter)
        
        pose_results = self.pose_results if self.pose_results is not None else other.pose_results
        new_vector.pose_results = pose_results
        base_yolo_result = self.base_yolo_result if self.base_yolo_result is not None else other.base_yolo_result
        new_vector.base_yolo_result = base_yolo_result
        return new_vector

    def __ge__(self, other: 'ActionVector') -> bool: # greater or equals
        return all(
            self.counter[f] >= other.counter[f]
            for f in other.counter
        )
    

    def __str__(self) -> str:
        active_counts = [
            f"{f}: {self.counter[f]}"
            for f in self.counter
            if self.counter[f] > 0
        ]
        
        if not active_counts:
            return "ActionVector(empty)"
        
        return f"ActionVector({', '.join(active_counts)})"

class ActionClass:
    def __init__(self, name: str, required_vector: ActionVector, pre_buffer_seconds=2.0, cooldown_seconds=2.0, max_duration=10.0):
        self.name = name
        self.required_vector = required_vector
        buffer_len = int(pre_buffer_seconds * Config.FRAME_RATE)
        self.pre_buffer = deque(maxlen=buffer_len)
        self.post_buffer = []
        
        self.state = ActionState.IDLE
        self.inactive_frames = 0
        self.idling = 0
        self.trigger_count = 0
        
        self.max_inactive_frames = int(cooldown_seconds * Config.FRAME_RATE)
        self.max_duration_frames = int(max_duration * Config.FRAME_RATE)

    def check(self, frame, current_vector: ActionVector):
        triggered = current_vector >= self.required_vector #TODO Dodać metodykę dla triggerowania
        frame_vector = {'frame' : frame, 'vector' : current_vector}

        if self.state is ActionState.IDLE:
            if triggered:
                self.state = ActionState.RECORDING
                self.post_buffer.append(frame_vector)

                self.trigger_count += 1
                self.idling = 0
            else:
                self.pre_buffer.append(frame_vector)

        elif self.state is ActionState.RECORDING:
            self.post_buffer.append(frame_vector)
            if triggered:
                self.trigger_count += 1
                self.idling = 0
            else:
                self.idling += 1
                if self.idling >= self.max_inactive_frames:
                    self.stop()
            if len(self.post_buffer) >= self.max_duration_frames:
                self.stop()
    
    def stop(self, info=True):
        if self.state is not ActionState.RECORDING:
            if info:
                print(f"[{self.name}] Próba zakończenia nagrywania w stanie: {self.state}!")
            return

        self.state = ActionState.IDLE

        full_clip = list(self.pre_buffer) + self.post_buffer
        if self.trigger_count >= MIN_TRIGGER_COUNT:
            save_thread = threading.Thread(
                target=self._save_clip_task, 
                args=(full_clip, self.name)
            )
            save_thread.start()

        self.trigger_count = 0
        self.pre_buffer.clear()
        self.post_buffer = []
        self.idling = 0

    @staticmethod
    def _save_clip_task(frame_vectors, action_name): # TODO wysyłanie do backend Bartka
        anonymizer = Anonymizer()
        frames = anonymizer.anonymize_clip(frame_vectors)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{action_name}_{timestamp}.avi"
        path = os.path.join(Config.CLIP_FOLDER, filename)
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        
        out = cv2.VideoWriter(path, fourcc, Config.FRAME_RATE, (width, height), True)
        try:
            for frame in frames:
                out.write(frame)
        finally:
            out.release()

    def __str__(self) -> str:
        status = "RECORDING" if self.state is ActionState.RECORDING else "IDLE"
        return f"ActionClass(name='{self.name}', status={status}, cooldown={self.inactive_frames}/{self.max_inactive_frames})"

def load_action_classes(path: str) -> list[ActionClass]: # loads action classes from csv file
    action_classes = []
    csv_path = Path(path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.pop("action_name", "Unknown")
            pre_seconds = float(row.pop('pre_seconds', '2.0')) # seconds before action
            post_seconds = float(row.pop('post_seconds', '2.0'))
            vector_kwargs = {key : int(value) for key, value in row.items()
                             if key in BASE_YOLO_MAPPING.values()
                             or key in LSTM_MAPPING.values()}
            
            vector = ActionVector(vector_kwargs)
            action_classes.append(ActionClass(name=name, pre_buffer_seconds=pre_seconds, cooldown_seconds=post_seconds, required_vector=vector))
            
    return action_classes