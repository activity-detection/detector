import numpy as np

from src.detector.config import Config
from src.detector.sources import RTSPSource, VideoFileSource, USBCameraSource, ImageFolderSource, SingleImageSource
from src.detector.detector import Detector
from src.detector.fps_normalizer import FrameRateNormalizer
from src.tools.fps_estimator import FPS_estimator
from src.detector.recorder import Recorder

from src.detector.utils.mylogger import setup_logging

def get_source():
    if Config.APP_MODE == 'VIDEO': return VideoFileSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'USB': return USBCameraSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'FOLDER': return ImageFolderSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'IMAGE': return SingleImageSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'RTSP': return RTSPSource(Config.get_rtsp_url())
    else: raise ValueError(f"Unknown mode: {Config.APP_MODE}")

def main():
    print(f"MODE: {Config.APP_MODE} | SOURCE: {Config.SOURCE_PATH} | BATCH SIZE: {Config.BATCH_SIZE}")

    setup_logging()

    source = get_source()
    source_fps = source.get_frame_rate() or Config.TARGET_FPS
    Config.FRAME_RATE = Config.TARGET_FPS  # post-normalization wszystko @ TARGET_FPS
    print(f"Source FPS: {source_fps:.2f} → TARGET_FPS: {Config.TARGET_FPS}")

    fps_norm = FrameRateNormalizer(source_fps=source_fps, target_fps=Config.TARGET_FPS)
    detector = Detector()
    recorder = Recorder(2, 2, 6)
    recorder.load_action_classes(Config.ACTION_VECTORS_PATH)
    for ac in recorder.action_classes:
        print(ac.action_vector)
    batch: list[np.ndarray] = []
    fps = FPS_estimator()
    fps.begin()
    try:
        while True:
            ret, frame = source.get_frame()
            if not ret:
                break

            n_emit = fps_norm.emit_count()
            fps_norm.advance()
            for _ in range(n_emit):
                batch.append(frame)
                if len(batch) == Config.BATCH_SIZE:
                    vector_list = detector.process_batch(batch)
                    for vector, batched_frame in zip(vector_list, batch):
                        recorder.check_frame(batched_frame, vector)
                        print(vector)
                    batch.clear()
                    fps.end()
                    fps.begin()

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        source.release()

if __name__ == "__main__":
    main()
