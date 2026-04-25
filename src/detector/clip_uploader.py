from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from enum import Enum, auto
import threading
import requests
import json
import time
import os

from src.detector.timestamper import FullStampModel
from src.detector.config import Config
from src.detector import logger

PAUSE_ON_ERROR = 1.0
UPLOAD_LOOP_PAUSE = 0.5
UPLOAD_WAIT = 1.0


class RecState(Enum):
    AWAIT_UPLOAD = auto()
    UPLOADING = auto()
    SENT = auto()
    FAILED = auto()
    STASHED = auto()


@dataclass
class UploadTask:
    """Represents a clip upload task with dependencies"""
    path: Path
    details: FullStampModel
    state: RecState
    id: str | None = None
    created_at: float | None = None
    retries: int = 5
    next_try_at: float | None = None
    dependency: UploadTask | None = None  # Previous recording filename this depends on
    stashed_dependency: UploadTask | None = None # Stashed for possible sequence continuity split

    def __post_init__(self):
        self.created_at = time.time()


class ClipUploader:
    def __init__(self) -> None:
        self.clip_folder = Path(Config.CLIP_FOLDER or "clips")
        self.upload_queue: deque[UploadTask] = deque()
        self.upload_lock = threading.Lock()
        self.davy_jones_locker: deque[UploadTask] = deque()
        self.davy_jones_lock = threading.Lock()
        self.worker_thread = None
        self.stop_worker = False

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
        
        if task.next_try_at is not None:
            if time.time() < task.next_try_at:
                return False

        # Check if dependency is satisfied
        if self._is_dependency_satisfied(task):
            return True

        return False

    def _is_dependency_satisfied(self, task: UploadTask) -> bool:
        """Check if the dependency recording has been uploaded and has an ID
            Also stash the dependency if needed"""
        dependency = task.dependency

        if dependency is not None:
            if dependency.state == RecState.STASHED:
                    task.stashed_dependency = dependency
                    task.dependency = None
                    return True
            elif dependency.state == RecState.SENT and dependency.id is not None:
                return True
        return True

    def _process_upload_task(self, task: UploadTask):
        """Process a single upload task"""
        try:
            # Get the previous recording ID if dependency is satisfied
            prev_id = None
            if task.dependency is not None and self._is_dependency_satisfied(task):
                prev_id = task.dependency.id

            # Mark as uploading
            task.state = RecState.UPLOADING

            # Upload the clip
            self._upload_clip(
                task=task,
                details=task.details,
                prev_id=prev_id
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
            task.state = RecState.AWAIT_UPLOAD
    
    def start_upload(
            self,
            path: Path,
            details: FullStampModel,
            dependency_filename: str | None,
    ) -> None:
        dependency = None
        if dependency_filename is not None:
            with self.upload_lock:
                for task in self.upload_queue:
                    if task.path.name == dependency_filename:
                        dependency = task
                        break
        
        task = UploadTask(
            path=path,
            details=details,
            state=RecState.AWAIT_UPLOAD,
            dependency=dependency
        )
        with self.upload_lock:
            self.upload_queue.append(task)

    def _upload_clip(
            self,
            task: UploadTask,
            details: FullStampModel,
            prev_id: str | None = None,
    ) -> None:
        path = task.path
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
        
        task.state = RecState.SENT
        task.id = res_id
