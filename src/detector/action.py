from collections import deque, Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
import numpy as np
import threading
import requests
import json
import math
import os
import csv
import cv2

from .config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from .anonymizer import Anonymizer

class State(Enum):
    IDLE = auto()
    ACTIVE = auto()

class Command(Enum):
    BEGIN = auto()
    END = auto()
    CONTINUE = auto()
    AWAIT = auto()

class ActionVector: # TODO looking into this
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

    def __ge__(self, other: 'ActionVector') -> bool:
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
    
@dataclass
class RecorderAction:
    action: ActionClass
    offset: int
    beginning: int
    end: int = -1
    
class Recorder:
    def __init__(self, buffer_len, lag_len, max_duration, send=True):
        buffer_frames = int(buffer_len * Config.FRAME_RATE)
        self.buffer = deque(maxlen=buffer_frames)
        self.recording = []
        self.action_classes = []
        self.action_stack = deque()
        self.state = State.IDLE
        self.send = send

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
                                or key in LSTM_MAPPING.values()
                                or key == "person"} # osoby nie ma w configu
                
                vector = ActionVector(vector_kwargs)
                action_class = ActionClass(name, vector, self.config)
                self.action_classes.append(action_class)

    def check_frame(self, frame: np.ndarray, reference_vector: ActionVector):
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

    def _add(self, action_class: ActionClass):
        if not self.recording:
            self.recording.extend(self.buffer)
            beginning = len(self.recording)
            action_recording = RecorderAction(action_class, 0, beginning)
            self.action_stack.append(action_recording)
            self.buffer.clear()
        else:
            buffer_len = self.buffer.maxlen
            if buffer_len >= len(self.recording):
                offset = 0
            else:
                offset = len(self.recording) - buffer_len
            beginning = len(self.recording) - offset
            action_recording = RecorderAction(action_class, offset, beginning)
            self.action_stack.append(action_recording)

    def _remove(self, action_class: ActionClass):
        if not self.action_stack:
            print("Action stack jest pusty!")
            return #TODO
        
        recorder_action = None
        target_index = -1
        
        for i, rec in enumerate(self.action_stack):
            if rec.action == action_class:
                recorder_action = rec
                target_index = i
                break
                
        if recorder_action is None:
            print(f"Nie znaleziono RecorderAction dla {action_class.name} na stosie!")
            return
            
        if recorder_action.offset is None:
            print("Action recording offset jest None!")
            return
        
        recorder_action.end = len(self.recording) - 1 - recorder_action.offset - action_class.idling_final
        action_class.idling_final = 0
        self._save_clip(recorder_action)

        if target_index == 0:
            if len(self.action_stack) > 1:
                second_action = self.action_stack[1]
                second_offset = second_action.offset
                self.recording = self.recording[second_offset:]
                for action in list(self.action_stack)[1:]:
                    action.offset -= second_offset
            else:
                buffer_len = self.buffer.maxlen
                if buffer_len > len(self.recording):
                    buffer_len = len(self.recording)
                self.buffer.extend(self.recording[-buffer_len:])
                self.recording = []
        
        self.action_stack.remove(recorder_action)

    def _save_clip(self, recorder_action: RecorderAction):
        recording = self.recording.copy()

        save_thread = threading.Thread(
            target=self._save_clip_task, 
            args=(recording, recorder_action)
        )
        save_thread.start()
    
    def _save_clip_task(self, frame_vectors: dict, recorder_action: RecorderAction):
        action_name = recorder_action.action.name

        anonymizer = Anonymizer()
        frames = anonymizer.anonymize_clip(frame_vectors)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{action_name}_{timestamp}.mp4"
        path = os.path.join(Config.CLIP_FOLDER, filename)
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(path, fourcc, Config.FRAME_RATE, (width, height), True)
        try:
            for frame in frames:
                out.write(frame)
        finally:
            out.release()

            if self.send:
                reference_vector = recorder_action.action.action_vector.counter
                reference_detections = []
                for detection, count in reference_vector.items():
                    if count > 0:
                        reference_detections.append(detection)
                timestamper = Timestamper(reference_detections)

                detections = timestamper.timestamp(frame_vectors)

                beggining_sec = int(recorder_action.beginning/Config.FRAME_RATE)
                end_sec = int(recorder_action.end/Config.FRAME_RATE) + 1
                events = self._event_stamp(action_name, beggining_sec, end_sec)

                details_dict = {"events": events, "detections": detections}

                print(json.dumps(details_dict))

                data = {
                    "video-name": filename,
                    "description": "DESCRIPTION",
                    "relative-path": filename
                }

                with open(path, "rb") as video_file:
                    
                    files = {
                        "file": (filename, video_file, "video/quicktime"),
                        "details": (None, json.dumps(details_dict), "application/json")
                    }
                    response = requests.post(Config.DB_URL, data=data, files=files)

                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")

    @staticmethod
    def _event_stamp(name: str, beginning: int , end: int) -> list[dict]:
        events = []
        events.append({
            "label": name,
            "timestamp": {
                "from": f"PT{beginning}S",
                "to": f"PT{end}S"
            }
        })
        return events

class Timestamper:
    def __init__(self, reference_detections: list[str]):
        self.reference_detections = reference_detections

    def timestamp(self, recording: list[dict]) -> list[dict]:
        detections_per_second = self.find_most_common(recording)
        timestamped_list = self._make_timestamped_list(detections_per_second)
        return timestamped_list

    def find_most_common(self, recording: list[dict]) -> list[dict[str, int]]:
        duration = math.ceil(len(recording) / Config.FRAME_RATE)
        detections_per_second = self._init_seconds(duration)

        for index, frame_vector in enumerate(recording):
            curr_vector = frame_vector["vector"]
            curr_second = int(index / Config.FRAME_RATE)

            for detection in self.reference_detections:
                count = curr_vector.counter[detection]
                counter = detections_per_second[curr_second][detection]
                counter[count] += 1

        for detections_second in detections_per_second:
            for detection, counter in detections_second.items():
                most_common_list = counter.most_common(1)
                if not most_common_list:
                    most_common_count = 0
                else:
                    most_common_count = most_common_list[0][0]
                detections_second[detection] = most_common_count

        return detections_per_second

    def _init_seconds(self, duration: int) -> list[dict[str, Counter]]:
        detections_per_second = []
        for _ in range(duration):
            detections_second = {}
            for detection in self.reference_detections:
                detections_second[detection] = Counter()
            detections_per_second.append(detections_second)

        return detections_per_second
    
    def _make_timestamped_list(self, detections_per_second: list[dict[str, int]]) -> list[dict]:
        detections = []
        prev_det_vector = detections_per_second[0]
        time_from = 0
        for second in range(1, len(detections_per_second)):
            curr_det_vector = detections_per_second[second]
            if curr_det_vector != prev_det_vector:
                detection = self._detection_stamp(prev_det_vector, time_from, second)
                detections.append(detection)
                time_from = second
                prev_det_vector = curr_det_vector

        detection = self._detection_stamp(prev_det_vector, time_from, len(detections_per_second))
        detections.append(detection)

        return detections
    
    @staticmethod
    def _detection_stamp(det_vector: dict[str, int], time_from: int, time_to: int) -> dict:
        objects = []
        for detection, count in det_vector.items():
            if detection == "person": # baza chce human
                detection = "human"
            objects.append({
                "name": detection,
                "count": count
            })
        return {
                "objects": objects,
                "timestamp": {
                    "from": f"PT{time_from}S",
                    "to": f"PT{time_to}S"
                }
            }

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

        self.idling_final = 0

    def check(self, reference_vector: ActionVector) -> Command:
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
        self.idling_final = self.idling
        self.idling = 0

    def __str__(self) -> str:
        status = "ACTIVE" if self.state is State.ACTIVE else "IDLE"
        return f"ActionClass(name='{self.name}', status={status}, cooldown={self.idling}/{self.config.max_inactive_frames})"