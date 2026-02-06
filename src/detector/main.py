import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

from .config import Config
from .sources import RTSPSource, VideoFileSource, USBCameraSource, ImageFolderSource, SingleImageSource
from .anonymizer import Anonymizer
from .detector import Detector
from .lstm import MultiClassLSTM
from src.tools.fps_estimator import FPS_estimator

import torch
import numpy as np

INPUT_DIM = 34
HIDDEN_DIM = 64
OUTPUT_DIM = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_source():
    if Config.MODE == 'VIDEO': return VideoFileSource(Config.SOURCE_PATH)
    elif Config.MODE == 'USB': return USBCameraSource(Config.SOURCE_PATH)
    elif Config.MODE == 'FOLDER': return ImageFolderSource(Config.SOURCE_PATH)
    elif Config.MODE == 'IMAGE': return SingleImageSource(Config.SOURCE_PATH)
    elif Config.MODE == 'RTSP': return RTSPSource(Config.get_rtsp_url())
    else: raise ValueError(f"Unknown mode: {Config.MODE}")

def main():
    print(f"MODE: {Config.MODE} | SOURCE: {Config.SOURCE_PATH} | BATCH SIZE: {Config.BATCH_SIZE}")

    source = get_source()
    models = {
                "pose": {"model": YOLO(Config.POSE_MODEL_PATH), "type": "pose"},
                "plates": {"model": YOLO(Config.PLATES_MODEL_PATH), "type": "box"},
                "detection": {"model": MultiClassLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM), "type": "detection"}
            }
    anonymizer = Anonymizer()
    
    detector = Detector(models, anonymizer)

    batch_buffer = []
    fps_estimator = FPS_estimator()

    try:
        while True:
            ret, frame = source.get_frame()
            
            if ret:
                batch_buffer.append(frame)
            else:
                break
                
            if len(batch_buffer) >= Config.BATCH_SIZE:
                processed_batch = detector.process_batch_multiperson(batch_buffer)
                
                fps_estimator.end()
                avg_fps = fps_estimator.get_fps()

                if Config.SHOW_VIDEO:
                    for p_frame in processed_batch:
                        if Config.VIDEO_WIDTH and Config.VIDEO_HEIGHT:
                            p_frame = cv2.resize(p_frame, (Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT))
                        cv2.putText(p_frame, f'FPS: {avg_fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow('Anonymized Video', p_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            raise KeyboardInterrupt

                batch_buffer.clear()
                fps_estimator.begin()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        source.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()