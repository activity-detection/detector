from collections import deque, Counter
import csv
import cv2
import threading
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
import os
from dataclasses import dataclass

from .config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from .anonymizer import Anonymizer

MIN_TRIGGER_COUNT = 10

class State(Enum):
    IDLE = auto()
    ACTIVE = auto()

class Command(Enum):
    BEGIN = auto()
    END = auto()
    CONTINUE = auto()
    AWAIT = auto()

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
    
class Recorder:
    def __init__(self, buffer_len, lag_len, max_duration):
        buffer_frames = int(buffer_len * Config.FRAME_RATE)
        self.buffer = deque(maxlen=buffer_frames)
        self.recording = []
        self.action_stack = deque()
        self.state = State.IDLE
        self.action_classes = []

        self.config = ActionConfig(
            max_inactive_frames = int(lag_len * Config.FRAME_RATE),
            max_duration_frames= int(max_duration * Config.FRAME_RATE),
            check_count = 20,
            start_conf = 60,
            end_conf = 30
        )

    def load_action_classes(self, path: str): # loads action classes from csv file
        csv_path = Path(path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.pop("action_name", "Unknown")
                vector_kwargs = {key : int(value) for key, value in row.items()
                                if key in BASE_YOLO_MAPPING.values()
                                or key in LSTM_MAPPING.values()}
                
                vector = ActionVector(vector_kwargs)
                action_class = ActionClass(name, vector, self.config)
                self.action_classes.append(action_class)

    def check_frame(self, frame, reference_vector):
        command_list = []
        for action_class in self.action_classes:
            command = action_class.check(reference_vector)
            command_list.append(command)
            if command is Command.BEGIN:
                self._add(action_class)
            elif command is Command.END:
                self._remove(action_class)

        if Command.CONTINUE in command_list or Command.BEGIN in command_list:
            self.recording.append({'frame' : frame, 'vector' : reference_vector})
        else:
            self.buffer.append({'frame' : frame, 'vector' : reference_vector})

    def _add(self, action_class):
        if not self.recording:
            action_class.offset = 0
            self.action_stack.append(action_class)
            self.recording.extend(self.buffer)
            self.buffer.clear()
        else:
            buffer_len = self.buffer.maxlen
            if buffer_len > len(self.recording):
                buffer_len = len(self.recording)
            action_class.offset = len(self.recording) - buffer_len
            self.action_stack.append(action_class)

    def _remove(self, action_class):
        if not self.action_stack:
            print("Action stack jest pusty!")
            pass #TODO
        elif action_class.offset is None:
            print("Action class offset jest None!")
            pass #TODO
        elif self.action_stack.index(action_class) == 0:
            self._save_clip(action_class.name)
            if len(self.action_stack) > 1:
                second_action = self.action_stack[1]
                second_offset = second_action.offset
                self.recording = self.recording[second_offset:]
                for action in self.action_stack[1:]:
                    action.offset -= second_offset
            else:
                buffer_len = self.buffer.maxlen
                if buffer_len > len(self.recording):
                    buffer_len = len(self.recording)
                self.buffer.extend(self.recording[-buffer_len:])
                self.recording = []
        else:
            self._save_clip()
        
        self.action_stack.remove(action_class)

    def _save_clip(self, name):
        save_thread = threading.Thread(
            target=self._save_clip_task, 
            args=(self.recording, name)
        )
        save_thread.start()

    @staticmethod
    def _save_clip_task(frame_vectors, action_name):
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

@dataclass
class ActionConfig:
    max_inactive_frames: int
    max_duration_frames: int
    check_count: int
    start_conf: float
    end_conf: float

class ActionClass:
    def __init__(self, name, action_vector: ActionVector, config: ActionConfig):
        self.name = name
        self.action_vector = action_vector
        
        self.state = State.IDLE
        self.idling = 0
        self.frame_count = 0
        self.trigger_history = deque(maxlen=config.check_count)
        self.config = config
        self.triggered = False

        self.offset = None

    def check(self, reference_vector: ActionVector):
        ge = reference_vector >= self.action_vector
        self.trigger_history.append(ge)
        if len(self.trigger_history) == self.trigger_history.maxlen:
            ge_count = sum(self.trigger_history)
            percentage = round(ge_count / self.trigger_history.maxlen, 2) * 100
            if percentage >= self.config.start_conf:
                self.triggered = True
            elif percentage <= self.config.end_conf:
                self.triggered = False

        if self.state is State.IDLE:
            if self.triggered:
                self.state = State.ACTIVE
                self.frame_count += 1
                self.idling = 0
                return Command.BEGIN
            else:  
                return Command.AWAIT

        elif self.state is State.ACTIVE:
            self.frame_count += 1
            if self.frame_count >= self.config.max_duration_frames:
                self._stop()
                return Command.END
            elif self.triggered:                
                self.idling = 0
                return Command.CONTINUE
            else:
                self.idling += 1
                if self.idling >= self.config.max_inactive_frames:
                    self._stop()
                    return Command.END
                else:
                    return Command.CONTINUE
    
    def _stop(self, info=True):
        if self.state is not State.ACTIVE:
            if info:
                print(f"[{self.name}] Próba zakończenia nagrywania w stanie: {self.state}!") #TODO
            return

        self.state = State.IDLE
        self.frame_count = 0
        self.idling = 0
        # self.offset = None TODO moze jakis reset offsetu, none obecnie psuje

    def __str__(self) -> str:
        status = "ACTIVE" if self.state is State.ACTIVE else "IDLE"
        return f"ActionClass(name='{self.name}', status={status}, cooldown={self.inactive_frames}/{self.max_inactive_frames})"