import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

from config import Config
from sources import RTSPSource, VideoFileSource, USBCameraSource, ImageFolderSource, SingleImageSource
from anonymizer import Anonymizer
from action_detector import ActionDetector

def get_source():
    if Config.MODE == 'VIDEO': return VideoFileSource(Config.SOURCE_PATH)
    elif Config.MODE == 'USB': return USBCameraSource(Config.SOURCE_PATH)
    elif Config.MODE == 'FOLDER': return ImageFolderSource(Config.SOURCE_PATH)
    elif Config.MODE == 'IMAGE': return SingleImageSource(Config.SOURCE_PATH)
    elif Config.MODE == 'RTSP': return RTSPSource(Config.get_rtsp_url())
    else: raise ValueError(f"Unknown mode: {Config.MODE}")

def initialize_recorder(frame, recorder):
    if Config.RECORD and recorder is None:
        h, w = frame.shape[:2]
        return cv2.VideoWriter(Config.RECORD_NAME, cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
    return recorder

def process_batch(batch_frames, model, anonymizer, detector, recorder):
    results = model(batch_frames, verbose=False)
    
    processed_frames = []

    for frame, result in zip(batch_frames, results):
        
        anonymizer.anonymize(frame, result)
        
        saved_file = detector.process(frame, result)
        
        if saved_file:
            print(f"[INFO] Zdarzenie zapisane do pliku: {saved_file}")

        if Config.RECORD and recorder:
            recorder.write(frame)
            
        processed_frames.append(frame)

    return processed_frames

def main():
    print(f"MODE: {Config.MODE} | SOURCE: {Config.SOURCE_PATH} | BATCH SIZE: {Config.BATCH_SIZE}")

    source = get_source()
    model = YOLO(Config.MODEL_PATH)
    anonymizer = Anonymizer()
    
    detector = ActionDetector(buffer_seconds=2, post_event_seconds=3)
    
    frame_rate_buffer = deque(maxlen=200)
    recorder = None
    
    batch_buffer = [] 

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = source.get_frame()
            
            if not ret or frame is None:
                if len(batch_buffer) > 0:
                    process_batch(batch_buffer, model, anonymizer, detector, recorder)
                print("End of stream.")
                break

            if Config.VIDEO_WIDTH and Config.VIDEO_HEIGHT:
                frame = cv2.resize(frame, (Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT))

            recorder = initialize_recorder(frame, recorder)

            batch_buffer.append(frame)

            if len(batch_buffer) >= Config.BATCH_SIZE:
                
                processed_batch = process_batch(batch_buffer, model, anonymizer, detector, recorder)
                
                t_stop = time.perf_counter()
                batch_time = t_stop - t_start
                if batch_time > 0:
                    fps_curr = len(batch_buffer) / batch_time
                    frame_rate_buffer.append(fps_curr)
                
                avg_fps = np.mean(frame_rate_buffer) if frame_rate_buffer else 0

                if Config.SHOW_VIDEO:
                    for p_frame in processed_batch:
                        cv2.putText(p_frame, f'FPS: {avg_fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow('Anonymized Video', p_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            raise KeyboardInterrupt

                batch_buffer = []

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        source.release()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()