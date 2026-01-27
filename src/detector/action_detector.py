import cv2
import numpy as np
from collections import deque
import time
from .config import Config

class ActionDetector:
    def __init__(self, buffer_seconds=5, post_event_seconds=3, fps=30):
        self.fps = fps
        
        self.video_buffer_len = buffer_seconds * fps
        self.video_buffer = deque(maxlen=self.video_buffer_len)
        
        self.post_event_len = post_event_seconds * fps
        self.is_recording_event = False
        self.frames_left_to_record = 0
        self.event_clip = [] 

        self.consecutive_triggers = 0
        self.TRIGGER_THRESHOLD_FRAMES = 5 

    def process(self, frame, result):
        if not self.is_recording_event:
            self.video_buffer.append(frame.copy())

        detected_now = False
        
        if not self.is_recording_event:
            if self._check_hands_up(result):
                self.consecutive_triggers += 1
            else:
                self.consecutive_triggers = 0 

            if self.consecutive_triggers >= self.TRIGGER_THRESHOLD_FRAMES:
                detected_now = True
                self.consecutive_triggers = 0

        if detected_now:
            self._start_event_recording()

        saved_filename = None
        if self.is_recording_event:
            self.event_clip.append(frame.copy())
            self.frames_left_to_record -= 1

            if self.frames_left_to_record <= 0:
                saved_filename = self._save_clip_to_disk()
                self._reset_state()
        
        return saved_filename

    def _check_hands_up(self, result):
        if result.keypoints is None or result.keypoints.data.numel() == 0:
            return False
        
        kpts = result.keypoints.data[0].cpu().numpy()
        
        indices_to_check = [1, 2, 9, 10]
        if np.any(kpts[indices_to_check, 2] < 0.5):
            return False 
            
        l_eye_y = kpts[1, 1]
        r_eye_y = kpts[2, 1]
        l_wrist_y = kpts[9, 1]
        r_wrist_y = kpts[10, 1]
        hands_up = (l_wrist_y < l_eye_y) or (r_wrist_y < r_eye_y)
        
        return hands_up

    def _start_event_recording(self):
        self.is_recording_event = True
        self.frames_left_to_record = self.post_event_len
        self.event_clip = list(self.video_buffer)

    def _reset_state(self):
        self.is_recording_event = False
        self.event_clip = []
        self.video_buffer.clear() 
        self.consecutive_triggers = 0

    def _save_clip_to_disk(self):
        if not self.event_clip: return None
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{Config.CLIP_FOLDER}/{timestamp}.avi"
        h, w = self.event_clip[0].shape[:2]
        
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (w, h))
        for f in self.event_clip:
            out.write(f)
        out.release()
        
        return filename