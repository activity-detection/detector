from src.detector.config import Config
from src.detector.sources import RTSPSource, VideoFileSource, USBCameraSource, ImageFolderSource, SingleImageSource
from src.detector.detector import Detector
from src.tools.fps_estimator import FPS_estimator 
from src.detector.action import load_action_classes

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
    Config.FRAME_RATE = source.get_frame_rate()
    detector = Detector()
    action_classes = load_action_classes(Config.ACTION_VECTORS_PATH)
    fps = FPS_estimator()
    batch = []
    try:
        while True:
            ret, frame = source.get_frame()
            if not ret:
                break
            batch.append(frame)
            if len(batch) == Config.BATCH_SIZE:
                vector_list = detector.process_batch(batch)
                for vector, frame in zip(vector_list, batch):
                    for action in action_classes:
                        action.next_frame(frame, vector)
                    print(vector)
                batch.clear()
                fps.end()
                print(f'FPS: {fps.get_fps():.2f}')
                
                fps.begin()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        source.release()

if __name__ == "__main__":
    main()