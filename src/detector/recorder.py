from collections import deque
from dataclasses import dataclass

from pathlib import Path
import threading
import numpy as np
import csv

from src.detector.clip_manager import ClipManager
from src.detector.vectors import FrameVector, ActionVector
from src.detector.config import Config, BASE_YOLO_MAPPING, LSTM_MAPPING
from src.detector.action import ActionClass, ActionConfig
from src.detector.enums import State, Command
from src.detector import logger


class ActionNotInStackError(Exception):
    pass


@dataclass
class RecorderAction:
    action: ActionClass
    offset: int
    beginning: int
    end: int = -1
    

class Recorder:
    def __init__(
            self, 
            buffer_len: int, 
            lag_len: int, 
            max_duration: int, 
            send: bool = True
    ):
        self.buffer_frames = int(buffer_len * Config.FRAME_RATE)
        self.buffer: deque[FrameVector] = deque(maxlen=self.buffer_frames)
        self.recording: list[FrameVector] = []
        self.action_classes: list[ActionClass] = []
        self.action_stack: deque[RecorderAction] = deque()
        self.clip_manager = ClipManager()
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
        command_list: list[Command] = []
        for action_class in self.action_classes:
            command = action_class.check(reference_vector)
            command_list.append(command)
            if command is Command.BEGIN:
                self._add(action_class)
            elif command is Command.END:
                try:
                    recorder_action, index = self._find_action_in_stack(action_class)
                except ActionNotInStackError as e:
                    logger.error(f'Failed to remove action {action_class.name} from Recorder stack: {e}')
                    if action_class.state is not State.IDLE:
                        logger.error(f'An ActionClass outside of Recorder stack was not IDLE. Stopping.')
                        action_class.end()
                else:
                    self._process_clip(recorder_action)
                    self._remove(index)

        if Command.CONTINUE in command_list or Command.BEGIN in command_list:
            self.recording.append(FrameVector(frame, reference_vector))
        else:
            self.buffer.append(FrameVector(frame, reference_vector))

    def _add(self, action_class: ActionClass):
        if not self.recording:
            self.recording.extend(self.buffer)
            beginning = len(self.recording)
            action_recording = RecorderAction(action_class, 0, beginning)
            self.action_stack.append(action_recording)
            self.buffer.clear()
        else:
            buffer_len = self.buffer_frames
            if action_class.awaiting < buffer_len:
                buffer_len = action_class.awaiting
            if buffer_len >= len(self.recording):
                offset = 0
            else:
                offset = len(self.recording) - buffer_len
            beginning = len(self.recording) - offset
            action_recording = RecorderAction(action_class, offset, beginning)
            self.action_stack.append(action_recording)

    def _remove(self, index: int):
        recorder_action = self.action_stack[index]
        
        self._set_action_end(recorder_action)

        self._cleanup_stack(index)
        
        self.action_stack.remove(recorder_action)

    def _find_action_in_stack(self, action_class: ActionClass) -> tuple[RecorderAction, int]:
            for i, rec in enumerate(self.action_stack):
                if rec.action == action_class:
                    return rec, i
                    
            raise ActionNotInStackError(f"Action '{action_class.name}' was not found in recorder stack.")

    def _set_action_end(self, recorder_action: RecorderAction):
        action_class = recorder_action.action
        end_shift = recorder_action.offset + action_class.idling_final
        recorder_action.end = len(self.recording) - end_shift
        action_class.idling_final = 0

    def _cleanup_stack(self, target_index: int):
        if target_index == 0:
            if len(self.action_stack) > 1:
                second_action = self.action_stack[1]
                second_offset = second_action.offset
                self.recording = self.recording[second_offset:]
                for action in list(self.action_stack)[1:]:
                    action.offset -= second_offset
            else:
                buffer_len = self.buffer_frames
                if buffer_len > len(self.recording):
                    buffer_len = len(self.recording)
                self.buffer.extend(self.recording[-buffer_len:])
                self.recording = []

    def _process_clip(self, recorder_action: RecorderAction):
        offset = recorder_action.offset
        clip = self.recording[offset:].copy()

        start_sec = int(recorder_action.beginning / Config.FRAME_RATE)
        end_sec = int(recorder_action.end / Config.FRAME_RATE) + 1
        event_span = (start_sec, end_sec)

        action_name = recorder_action.action.name
        reference_counter = recorder_action.action.action_vector.counter
        
        start_inactive_count = recorder_action.action.awaiting_final
        end_inactive_count = recorder_action.action.idling_final
        inactive_counts = (start_inactive_count, end_inactive_count)
        
        thread = threading.Thread(
            target=self.clip_manager.handle,
            args=(
                clip,
                action_name,
                event_span,
                reference_counter,
                inactive_counts
            ),
            daemon=True
        )
        thread.start()
