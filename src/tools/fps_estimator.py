from time import perf_counter
from collections import deque
from src.detector.config import Config
from numpy import mean

class FPS_estimator:
    def __init__(self, buffer_len=200):
        self.last_time = perf_counter()
        self.frame_rate_buffer = deque(maxlen=buffer_len)

    def begin(self):
        self.last_time = perf_counter()

    def end(self):
        end_time = perf_counter() - self.last_time
        fps_curr = Config.BATCH_SIZE / end_time
        self.frame_rate_buffer.append(fps_curr)
    
    def get_fps(self):
        avg_fps = mean(self.frame_rate_buffer)
        return avg_fps
