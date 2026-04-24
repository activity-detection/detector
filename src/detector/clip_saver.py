from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import threading
import requests
import json
import time
import os

import numpy as np
import cv2

from src.detector.timestamper import Timestamper
from src.detector.anonymizer import Anonymizer
from src.detector.sequencer import Sequencer, RecState
from src.detector.vectors import FrameVector
from src.detector.config import Config
from src.detector import logger

PAUSE_ON_ERROR = 1.0
UPLOAD_LOOP_PAUSE = 0.5
UPLOAD_WAIT = 1.0


@dataclass
class UploadTask:
    """Represents a clip upload task with dependencies"""
    action_name: str
    filename: str
    frame_vectors: list[FrameVector]
    beginning_sec: int
    end_sec: int
    reference_counter: Counter[str]
    created_at: float | None = None
    retries: int = 5
    next_try_at: float | None = None
    dependency: str | None = None  # Previous recording filename this depends on
    stashed_dependency: str | None = None # Stashed for possible sequence continuity split

    def __post_init__(self):
        self.created_at = time.time()


class ClipSaver:
    def __init__(self, sequencer: Sequencer, send: bool = True):
        self.sequencer = sequencer
        self.send = send
        self.clip_folder = Path(Config.CLIP_FOLDER or "clips")

        try:
            self.clip_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Cannot create clip folder {self.clip_folder}: {e}")
            raise

        self.anonymizer = Anonymizer()

        self.upload_queue: deque[UploadTask] = deque()
        self.upload_lock = threading.Lock()
        self.davy_jones_locker: deque[UploadTask] = deque()
        self.davy_jones_lock = threading.Lock()
        self.worker_thread = None
        self.stop_worker = False

        if self.send:
            self._start_upload_worker()

    def __del__(self):
        """Cleanup worker thread on destruction"""
        self._stop_upload_worker()

    def _start_upload_worker(self):
        """Start the background upload worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_worker = False
            self.worker_thread = threading.Thread(target=self._upload_worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Started upload worker thread")

    def _stop_upload_worker(self):
        """Stop the background upload worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_worker = True
            self.worker_thread.join(timeout=5.0)
            logger.info("Stopped upload worker thread")

    def _upload_worker_loop(self):
        while not self.stop_worker:
            try:
                task_to_process = None

                with self.upload_lock:
                    for task in self.upload_queue:
                        if self._is_task_ready(task):
                            task_to_process = task
                            break 
                
                if task_to_process:
                    self._process_upload_task(task_to_process)
                else:
                    time.sleep(UPLOAD_LOOP_PAUSE)

            except Exception as e:
                logger.error(f"Error in upload worker: {e}", exc_info=True)
                time.sleep(PAUSE_ON_ERROR)

    def _is_task_ready(self, task: UploadTask) -> bool:
        """Check if a task is ready to upload"""
        # Is there any dependency
        if task.dependency is None:
            return True  # No dependency
        
        if task.next_try_at is not None:
            if time.time() < task.next_try_at:
                return False

        # Check if dependency is satisfied
        if self._is_dependency_satisfied(task):
            return True

        return False

    # TODO sprawdzić warunki dla None
    def _is_dependency_satisfied(self, task: UploadTask) -> bool:
        """Check if the dependency recording has been uploaded and has an ID
            Also stash the dependency if needed"""
        action_name = task.action_name
        dependency = task.dependency

        if dependency is not None:
            idx_in_seq, sequence = self.sequencer.get_id_sequence(action_name, dependency)

            if sequence is not None and idx_in_seq is not None:
                recording = sequence[idx_in_seq]
                if recording['state'] == RecState.STASHED:
                    task.stashed_dependency = task.dependency
                    task.dependency = None
                    return True
                return recording['state'] == RecState.SENT and recording['id'] is not None
        return True

    def _process_upload_task(self, task: UploadTask):
        """Process a single upload task"""
        try:
            # Get the previous recording ID if dependency is satisfied
            prev_id = None
            if task.dependency and self._is_dependency_satisfied(task):
                id_in_seq, sequence = self.sequencer.get_id_sequence(task.action_name, task.dependency)
                if id_in_seq is not None and sequence is not None:
                    prev_id = sequence[id_in_seq]['id']

            # Mark as uploading
            self.sequencer.mark_as_state(task.action_name, task.filename, RecState.UPLOADING)

            # Create clip path
            path = self.clip_folder / task.filename

            # Upload the clip
            self.upload_clip(
                path,
                task.frame_vectors,
                task.action_name,
                task.beginning_sec,
                task.end_sec,
                task.reference_counter,
                prev_id
            )

            with self.upload_lock:
                if task in self.upload_queue:
                    self.upload_queue.remove(task)

        except Exception as e:
            logger.error(f"Failed to upload clip '{task.filename}': {e}", exc_info=True)

            task.dependency = task.stashed_dependency
            task.stashed_dependency = None
            task.retries -= 1

            if task.retries <= 0:
                with self.upload_lock:
                    if task in self.upload_queue:
                        self.upload_queue.remove(task)
                
                with self.davy_jones_lock:
                    self.davy_jones_locker.append(task)

            task.next_try_at = time.time() + UPLOAD_WAIT
            self.sequencer.mark_as_state(task.action_name, task.filename, RecState.AWAIT_UPLOAD)

    def _cleanup_finished(self):
        """Periodically clean up sequences"""
        # Run cleanup every ~10 seconds
        if int(time.time()) % 10 != 0:
            return
        
        self.sequencer.cleanup_finished_sequences()

    def save_in_background(
        self,
        frame_vectors: list[FrameVector],
        action_name: str,
        beginning_sec: int,
        end_sec: int,
        reference_counter: Counter[str],
        filename: str
    ):
        thread = threading.Thread(
            target=self.save,
            args=(frame_vectors, action_name, beginning_sec, end_sec, reference_counter, filename),
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
        filename: str
    ):
        frames = self.anonymizer.anonymize_clip(frame_vectors)
        path = self.clip_folder / filename
        try:
            path = self._write_video(frames, path)
            logger.info(f"Successfully saved clip locally at {path.name}")
        except (cv2.error, OSError) as e:
            logger.error(f"Failed to write video '{filename}' to disk: {e}", exc_info=True)
            raise

        self.sequencer.mark_as_state(action_name, filename, RecState.AWAIT_UPLOAD)

        dependency = None
        try:
            _, sequence = self.sequencer.get_id_sequence(action_name, filename)
            if sequence is not None:
                if len(sequence) >= 2:
                    dependency = sequence[-2]['filename']
        except (KeyError, IndexError):
            pass

        if self.send:
            task = UploadTask(
                action_name=action_name,
                filename=filename,
                frame_vectors=frame_vectors,
                beginning_sec=beginning_sec,
                end_sec=end_sec,
                reference_counter=reference_counter,
                dependency=dependency
            )

            with self.upload_lock:
                self.upload_queue.append(task)

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

    def upload_clip(
        self,
        path: Path,
        frame_vectors: list[FrameVector],
        action_name: str,
        beginning_sec: int,
        end_sec: int,
        reference_counter: Counter[str],
        prev_id: str | None = None
    ):
        reference_detections = [detection for detection, count in reference_counter.items() if count > 0]
        timestamper = Timestamper(reference_detections)
        details = timestamper.timestamp(frame_vectors, action_name, (beginning_sec, end_sec))
        details = asdict(details)

        filename = path.name

        data = {
            "video-name": filename,
            "description": "DESCRIPTION",
            "relative-path": filename,
        }

        if prev_id:
            data["continuation-of"] = prev_id

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Uploading file: {filename} (Size: {file_size_mb:.2f} MB)")

        with open(path, "rb") as video_file:
            files: dict[str, Any] = {
                "file": (filename, video_file, "video/mp4"),
                "details": (None, json.dumps(details), "application/json"),
            }
            response = requests.post(Config.DB_URL, data=data, files=files, timeout=15)
            response.raise_for_status()

        res_id = response.text
            
        self.sequencer.mark_as_state(action_name, filename, RecState.SENT, res_id)
