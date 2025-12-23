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

class RTSPSource(VideoSource):
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
    
    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

class VideoFileSource(VideoSource):
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

class USBCameraSource(VideoSource):
    def __init__(self, device_index):
        self.cap = cv2.VideoCapture(int(device_index))

    def get_frame(self):
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

    def get_frame(self):
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

    def get_frame(self):
        if self.processed:
            return False, None
        self.processed = True
        return (self.frame is not None), self.frame

    def release(self):
        pass