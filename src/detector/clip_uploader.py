from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import threading
import requests
import json
import time
import os

from src.detector.timestamper import FullStampModel
from src.detector.sequencer import Sequencer, RecState
from src.detector.config import Config
from src.detector import logger

PAUSE_ON_ERROR = 1.0
UPLOAD_LOOP_PAUSE = 0.5
UPLOAD_WAIT = 1.0


@dataclass
class UploadTask:
    """Represents a clip upload task with dependencies"""
    action_name: str
    path: Path
    details: FullStampModel
    created_at: float | None = None
    retries: int = 5
    next_try_at: float | None = None
    dependency: str | None = None  # Previous recording filename this depends on
    stashed_dependency: str | None = None # Stashed for possible sequence continuity split

    def __post_init__(self):
        self.created_at = time.time()


class ClipUploader:
    def __init__(self, sequncer: Sequencer) -> None:
        self.clip_folder = Path(Config.CLIP_FOLDER or "clips")
        self.upload_queue: deque[UploadTask] = deque()
        self.upload_lock = threading.Lock()
        self.davy_jones_locker: deque[UploadTask] = deque()
        self.davy_jones_lock = threading.Lock()
        self.worker_thread = None
        self.stop_worker = False
        self.sequencer = sequncer

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
            self.sequencer.mark_as_state(task.action_name, task.path.name, RecState.UPLOADING)

            # Create clip path
            path = task.path

            # Upload the clip
            self._upload_clip(
                path=path,
                action_name=task.action_name,
                prev_id=prev_id,
                details=task.details
            )

            with self.upload_lock:
                if task in self.upload_queue:
                    self.upload_queue.remove(task)

        except Exception as e:
            logger.error(f"Failed to upload clip '{task.path.name}': {e}", exc_info=True)

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
            self.sequencer.mark_as_state(task.action_name, task.path.name, RecState.AWAIT_UPLOAD)

    def _cleanup_finished(self):
        """Periodically clean up sequences"""
        # Run cleanup every ~10 seconds
        if int(time.time()) % 10 != 0:
            return
        
        self.sequencer.cleanup_finished_sequences()

    def start_upload(
            self,
            path: Path,
            action_name: str,
            details: FullStampModel 
    ) -> None:
        
        task = UploadTask(
            action_name=action_name,
            path=path,
            details=details
        )
        with self.upload_lock:
            self.upload_queue.append(task)

    def _upload_clip(
            self,
            path: Path,
            action_name: str,
            details: FullStampModel,
            prev_id: str | None = None,
    ):

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
