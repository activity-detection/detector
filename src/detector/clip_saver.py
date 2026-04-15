from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import threading
import requests
import numpy as np
import json
import cv2
import os

from src.detector.timestamper import Timestamper
from src.detector.anonymizer import Anonymizer
from src.detector.vectors import FrameVector
from src.detector.config import Config
from src.detector import logger


class ClipSaver:
    def __init__(self, send: bool = True):
        self.send = send
        self.clip_folder = Path(Config.CLIP_FOLDER or "clips")
        
        try:
            self.clip_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Cannot create clip folder {self.clip_folder}: {e}")
            raise
            
        self.anonymizer = Anonymizer()

    def save_in_background(
        self,
        frame_vectors: list[FrameVector],
        action_name: str,
        beginning_sec: int,
        end_sec: int,
        reference_counter: Counter[str],
    ):
        thread = threading.Thread(
            target=self.save,
            args=(frame_vectors, action_name, beginning_sec, end_sec, reference_counter),
            daemon=True,
        )
        thread.start()

    def save(
        self,
        frame_vectors: list[FrameVector],
        action_name: str,
        beginning_sec: int,
        end_sec: int,
        reference_counter: Counter[str],
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames = self.anonymizer.anonymize_clip(frame_vectors)
        filename = f"{action_name}_{timestamp}.mp4"
        path = self.clip_folder / filename
        try:
            path = self._write_video(frames, path)
            logger.info(f"Successfully saved clip locally at {path.name}")
        except (cv2.error, OSError) as e:
            logger.error(f"Failed to write video '{filename}' to disk: {e}", exc_info=True)
            return 

        if self.send:
            try:
                self._upload_clip(path, frame_vectors, action_name, beginning_sec, end_sec, reference_counter)
                logger.info(f"Successfully uploaded clip '{path.name}' to database.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error while uploading clip '{path.name}': {e}")

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

    def _upload_clip(
        self,
        path: Path,
        frame_vectors: list[FrameVector],
        action_name: str,
        beginning_sec: int,
        end_sec: int,
        reference_counter: Counter[str],
    ):
        reference_detections = [detection for detection, count in reference_counter.items() if count > 0]
        timestamper = Timestamper(reference_detections)
        details = timestamper.timestamp(frame_vectors, action_name, (beginning_sec, end_sec))
        details = asdict(details)

        data = {
            "video-name": path.name,
            "description": "DESCRIPTION",
            "relative-path": path.name,
        }

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Uploading file: {path.name} (Size: {file_size_mb:.2f} MB)")

        with open(path, "rb") as video_file:
            files: dict[str, Any] = {
                "file": (path.name, video_file, "video/mp4"),
                "details": (None, json.dumps(details), "application/json"),
            }
            print(json.dumps(details))
            response = requests.post(Config.DB_URL, data=data, files=files, timeout=15)
            
            response.raise_for_status() 

    