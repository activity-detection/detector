from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import threading
import requests
import numpy as np
import json
import cv2
import os
import time

from src.detector.timestamper import Timestamper
from src.detector.anonymizer import Anonymizer
from src.detector.sequencer import Sequencer, RecState
from src.detector.vectors import FrameVector
from src.detector.config import Config
from src.detector import logger


@dataclass
class UploadTask:
    """Represents a clip upload task with dependencies"""
    action_name: str
    filename: str
    frame_vectors: list[FrameVector]
    beginning_sec: int
    end_sec: int
    reference_counter: Counter[str]
    depends_on_filename: str | None = None  # Previous recording filename this depends on
    created_at: float | None = None
    timeout_at: float | None = None  # When to upload without dependency

    def __post_init__(self):
        self.created_at = time.time()
        # Wait up to 30 seconds for dependency, then upload without it
        self.timeout_at = self.created_at + 30.0


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
        """Main loop for processing upload tasks"""
        while not self.stop_worker:
            try:
                task = self._get_next_upload_task()
                if task:
                    self._process_upload_task(task)
                else:
                    # No tasks ready, sleep briefly
                    time.sleep(0.1)

                # Periodic cleanup of stuck tasks
                self._cleanup_stuck_tasks()

            except Exception as e:
                logger.error(f"Error in upload worker: {e}", exc_info=True)
                time.sleep(1.0)  # Brief pause on error

    def _get_next_upload_task(self) -> UploadTask | None:
        """Get the next task that's ready to upload (dependencies satisfied or timed out)"""
        with self.upload_lock:
            for task in self.upload_queue:
                if self._is_task_ready(task):
                    return self.upload_queue.popleft()
        return None

    def _is_task_ready(self, task: UploadTask) -> bool:
        """Check if a task is ready to upload"""
        if task.depends_on_filename is None:
            return True  # No dependency

        # Check if dependency is satisfied
        if self._is_dependency_satisfied(task.action_name, task.depends_on_filename):
            return True

        # Check if we've timed out waiting for dependency
        return time.time() >= task.timeout_at

    def _is_dependency_satisfied(self, action_name: str, depends_on_filename: str) -> bool:
        """Check if the dependency recording has been uploaded and has an ID"""
        try:
            recordings = self.sequencer.sequences[action_name][-1]['recordings']
            for rec in recordings:
                if rec['filename'] == depends_on_filename:
                    return rec['state'] == RecState.SENT and rec['id'] is not None
        except (KeyError, IndexError):
            pass
        return False

    def _process_upload_task(self, task: UploadTask):
        """Process a single upload task"""
        try:
            # Get the previous recording ID if dependency is satisfied
            prev_id = None
            if task.depends_on_filename and self._is_dependency_satisfied(task.action_name, task.depends_on_filename):
                recordings = self.sequencer.sequences[task.action_name][-1]['recordings']
                for rec in recordings:
                    if rec['filename'] == task.depends_on_filename:
                        prev_id = rec['id']
                        break

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

            logger.info(f"Successfully uploaded clip '{task.filename}' to database.")

        except Exception as e:
            logger.error(f"Failed to upload clip '{task.filename}': {e}", exc_info=True)
            # Mark as failed
            self.sequencer.mark_as_state(task.action_name, task.filename, RecState.FAILED)

    def _cleanup_stuck_tasks(self):
        """Periodically clean up tasks that have been stuck too long"""
        # Run cleanup every ~10 seconds
        if int(time.time()) % 10 != 0:
            return

        stuck_threshold = time.time() - 300.0  # 5 minutes

        with self.upload_lock:
            # Remove tasks that have been stuck for too long
            original_count = len(self.upload_queue)
            self.upload_queue = deque(
                task for task in self.upload_queue
                if task.created_at > stuck_threshold
            )

            removed_count = original_count - len(self.upload_queue)
            if removed_count > 0:
                logger.warning(f"Cleaned up {removed_count} stuck upload tasks")

        # Also check for stuck recordings in the sequencer and mark them as failed
        stuck_recordings = self.sequencer.get_stuck_recordings(max_age_seconds=300.0)
        if stuck_recordings:
            logger.warning(f"Found {len(stuck_recordings)} stuck recordings, marking as failed")
            self.sequencer.cleanup_stuck_recordings(max_age_seconds=300.0, mark_as_failed=True)

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

        depends_on_filename = None
        try:
            recordings = self.sequencer.sequences[action_name][-1]['recordings']
            if len(recordings) >= 2:
                depends_on_filename = recordings[-2]['filename']
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
                depends_on_filename=depends_on_filename
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
            
        
        self.sequencer.mark_as_state(action_name, filename, RecState.SENT, prev_id)
