from pathlib import Path

import numpy as np
import cv2

from src.detector.anonymizer import Anonymizer
from src.detector.vectors import FrameVector
from src.detector.config import Config
from src.detector import logger


class ClipSaver:
    def __init__(self):
        self.anonymizer = Anonymizer()

    def save(self, clip: list[FrameVector], path: Path) -> None:
        frames = self.anonymizer.anonymize_clip(clip)
        filename = path.name

        try:
            path = self._write_video(frames, path)
            logger.info(f"Successfully saved clip locally at {path.name}")
        except (cv2.error, OSError, IOError) as e:
            logger.error(f"Failed to write video '{filename}' to disk: {e}", exc_info=True)
            raise

    def _write_video(self, frames: list[np.ndarray], path: Path) -> Path:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
        
        out = cv2.VideoWriter(str(path), fourcc, Config.FRAME_RATE, (width, height), True)
        if not out.isOpened():
            raise IOError(f"Failed to open video writer for {path}")
            
        try:
            for frame in frames:
                out.write(frame) # type: ignore
        finally:
            out.release()

        return path
