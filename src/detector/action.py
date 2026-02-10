from collections import deque
from dataclasses import dataclass, fields
import csv
import cv2
import threading
from datetime import datetime
from pathlib import Path
from .config import Config
import os

MIN_TRIGGER_COUNT = 10

@dataclass(slots=True)
class ActionVector:
    count_person: int = 0
    count_bicycle: int = 0
    count_car: int = 0
    count_bird: int = 0
    count_cat: int = 0
    count_dog: int = 0
    count_handbag: int = 0
    count_suitcase: int = 0
    count_knife: int = 0

    count_squat: int = 0
    count_jumping_jacks: int = 0

    def __add__(self, other: 'ActionVector') -> 'ActionVector':
        if not isinstance(other, ActionVector):
            return NotImplemented
        
        new_values = {
            f.name: getattr(self, f.name) + getattr(other, f.name)
            for f in fields(self)
        }
        return ActionVector(**new_values)

    def __ge__(self, other: 'ActionVector') -> bool:
        return all(
            getattr(self, f.name) >= getattr(other, f.name)
            for f in fields(self)
        )
    
    def __str__(self) -> str:
        active_counts = [
            f"{f.name.replace('count_', '')}: {getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) > 0
        ]
        
        if not active_counts:
            return "ActionVector(empty)"
        
        return f"ActionVector({', '.join(active_counts)})"

class ActionClass:
    def __init__(self, name: str, required_vector: ActionVector, pre_buffer_seconds=2.0, cooldown_seconds=2.0):
        self.name = name
        self.required_vector = required_vector
        buffer_len = int(pre_buffer_seconds * Config.FRAME_RATE)
        self.pre_buffer = deque(maxlen=buffer_len)
        self.post_buffer = [] 
        
        self.is_recording = False
        self.inactive_frames = 0
        self.trigger_count = 0
        
        self.max_inactive_frames = int(cooldown_seconds * Config.FRAME_RATE)

    def next_frame(self, frame, current_vector: ActionVector):
        trigger_active = current_vector >= self.required_vector
        if trigger_active:
            self.trigger_count += 1

        if self.is_recording:
            self.post_buffer.append(frame)
            
            if trigger_active:
                self.inactive_frames = 0
            else:
                self.inactive_frames += 1
                
            if self.inactive_frames >= self.max_inactive_frames:
                self._stop_recording()
        
        else:
            self.pre_buffer.append(frame)
            if trigger_active:
                self._start_recording(frame)

    def _start_recording(self, frame):
        self.is_recording = True
        self.inactive_frames = 0
        self.post_buffer = []

    def _stop_recording(self):
        self.is_recording = False
        if self.trigger_count >= MIN_TRIGGER_COUNT:
            full_clip = list(self.pre_buffer) + self.post_buffer
            
            save_thread = threading.Thread(
                target=self._save_clip_task, 
                args=(full_clip, self.name)
            )
            save_thread.start()
        self.pre_buffer.clear()
        self.post_buffer = []

    @staticmethod
    def _save_clip_task(frames, action_name):
        if not frames:
            return

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
        # print(f"Saved: {filename}")

    def __str__(self) -> str:
        status = "RECORDING" if self.is_recording else "IDLE"
        return f"ActionClass(name='{self.name}', status={status}, cooldown={self.inactive_frames}/{self.max_inactive_frames})"

def load_action_classes(path: str) -> list[ActionClass]:
    action_classes = []
    csv_path = Path(path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.pop("action_name", "Unknown")
            pre_seconds = float(row.pop('pre_seconds', '2.0'))
            post_seconds = float(row.pop('post_seconds', '2.0'))
            vector_kwargs = {}
            for key, value in row.items():
                field_name = f"count_{key}"
                if field_name in ActionVector.__slots__:
                    vector_kwargs[field_name] = int(value)
            
            vector = ActionVector(**vector_kwargs)
            action_classes.append(ActionClass(name=name, pre_buffer_seconds=pre_seconds, cooldown_seconds=post_seconds, required_vector=vector))
            
    return action_classes