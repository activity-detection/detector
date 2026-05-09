import numpy as np
import cv2
import glob
import os
from abc import ABC, abstractmethod

class VideoSource(ABC):

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def release(self):
        pass

    def get_frame_rate(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0

    def get_frame_size(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        raise ValueError('Video Capture is closed')


class FPSResampledSource(VideoSource):
    """Resampluje dowolne źródło do stałego target_fps przez nearest-neighbor
    w domenie czasu. Mapowanie: required_src = round(target_idx * source_fps / target_fps).
    Dla downsample drop'uje klatki źródła; dla upsample duplikuje. Match training pipeline'u
    (research-notebooks/utils/fps_resampler.py) — LSTM wytrenowany na 25 fps musi dostawać
    sekwencje resamplowane do 25 fps niezależnie od FPS źródła.
    """

    def __init__(self, base_source: VideoSource, target_fps: float):
        self.base = base_source
        self.target_fps = target_fps
        source_fps = base_source.get_frame_rate()
        if source_fps <= 0:
            source_fps = target_fps
        self.source_fps = source_fps
        self.ratio = source_fps / target_fps
        self.src_idx = -1
        self.target_idx = 0
        self._current_frame: np.ndarray | None = None

    def _advance_source(self) -> bool:
        ret, frame = self.base.get_frame()
        if not ret:
            self._current_frame = None
            return False
        self.src_idx += 1
        self._current_frame = frame
        return True

    def get_frame(self) -> tuple[bool, np.ndarray]:
        if self.src_idx < 0:
            if not self._advance_source():
                return False, None  # type: ignore[return-value]

        while True:
            required_src = int(round(self.target_idx * self.ratio))
            if required_src == self.src_idx:
                self.target_idx += 1
                return True, self._current_frame  # type: ignore[return-value]
            if required_src < self.src_idx:
                self.target_idx += 1
                continue
            if not self._advance_source():
                return False, None  # type: ignore[return-value]

    def release(self):
        self.base.release()

    def get_frame_rate(self):
        return self.target_fps

    def get_frame_size(self):
        return self.base.get_frame_size()

class RTSPSource(VideoSource):
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
    
    def get_frame(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self):
        self.cap.release()

class VideoFileSource(VideoSource):
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)

    def get_frame(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self):
        self.cap.release()

class USBCameraSource(VideoSource):
    def __init__(self, device_index):
        self.cap = cv2.VideoCapture(int(device_index))

    def get_frame(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self):
        self.cap.release()

class ImageFolderSource(VideoSource):
    def __init__(self, folder_path):
        img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
        self.images = []
        for file in glob.glob(folder_path + '/*'):
            _, ext = os.path.splitext(file)
            if ext in img_ext_list:
                self.images.append(file)
        self.images.sort()
        self.current_idx = 0

    def get_frame(self) -> tuple[bool, np.ndarray]:
        if self.current_idx >= len(self.images):
            return False, None
        
        frame = cv2.imread(self.images[self.current_idx])
        self.current_idx += 1
        return (frame is not None), frame

    def release(self):
        pass

class SingleImageSource(VideoSource):
    def __init__(self, file_path):
        self.frame = cv2.imread(file_path)
        self.processed = False

    def get_frame(self) -> tuple[bool, np.ndarray]:
        if self.processed:
            return False, None
        self.processed = True
        return (self.frame is not None), self.frame

    def release(self):
        pass