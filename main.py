import numpy as np

from src.detector.config import Config, TARGET_FPS
from src.detector.sources import (
    RTSPSource,
    VideoFileSource,
    USBCameraSource,
    ImageFolderSource,
    SingleImageSource,
    FPSResampledSource,
)
from src.detector.detector import Detector
from src.tools.fps_estimator import FPS_estimator
from src.detector.recorder import Recorder

from src.detector.utils.mylogger import setup_logging
from src.detector import logger

TEMPORAL_MODES = {'VIDEO', 'USB', 'RTSP'}

def get_source():
    if Config.APP_MODE == 'VIDEO': return VideoFileSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'USB': return USBCameraSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'FOLDER': return ImageFolderSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'IMAGE': return SingleImageSource(Config.SOURCE_PATH)
    elif Config.APP_MODE == 'RTSP': return RTSPSource(Config.get_rtsp_url())
    else: raise ValueError(f"Unknown mode: {Config.APP_MODE}")

def main():
    logger.info(f"MODE: {Config.APP_MODE} | SOURCE: {Config.SOURCE_PATH} | BATCH SIZE: {Config.BATCH_SIZE}")

    setup_logging()

    source = get_source()
    if Config.APP_MODE in TEMPORAL_MODES:
        source = FPSResampledSource(source, TARGET_FPS)
    Config.FRAME_RATE = source.get_frame_rate()
    detector = Detector()
    recorder = Recorder(2, 2, 6)
    recorder.load_action_classes(Config.ACTION_VECTORS_PATH)
    logger.info("Vectors:")
    for ac in recorder.action_classes:
        logger.info(ac.action_vector)
    batch: list[np.ndarray] = []
    fps = FPS_estimator()
    fps.begin()
    try:
        while True:
            ret, frame = source.get_frame()
            if not ret:
                break
            batch.append(frame)
            if len(batch) == Config.BATCH_SIZE:
                vector_list = detector.process_batch(batch)
                for vector, frame in zip(vector_list, batch):
                    recorder.check_frame(frame, vector)
                    # print(vector)
                batch.clear()
                fps.end()
                fps.begin()

    except KeyboardInterrupt:
        logger.warning("Stopped by user")
    finally:
        source.release()

if __name__ == "__main__":
    main()