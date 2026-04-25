from collections import Counter
from datetime import datetime
from pathlib import Path

from src.detector.clip_uploader import ClipUploader
from src.detector.timestamper import Timestamper
from src.detector.clip_saver import ClipSaver
from src.detector.sequencer import Sequencer, RecState
from src.detector.vectors import FrameVector
from src.detector.config import Config
from src.detector import logger


class ClipManager:
    def __init__(self) -> None:
        self.sequencer = Sequencer()
        self.clip_saver = ClipSaver()
        self.timestamper = Timestamper()
        self.clip_uploader = ClipUploader(self.sequencer)
        self.clip_folder = Path(Config.CLIP_FOLDER or "clips")

        try:
            self.clip_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(f"Cannot create clip folder {self.clip_folder}: {e}")
            raise

    def handle(
            self,
            clip: list[FrameVector],
            action_name: str,
            event_span: tuple[int, int],
            reference_counter: Counter[str],
            inactive_counts: tuple[int, int]
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{action_name}_{timestamp}.mp4"
        path = self.clip_folder / filename

        self._update_sequence(action_name, filename, inactive_counts)

        self.sequencer.mark_as_state(action_name, filename, RecState.SAVING)
        self.clip_saver.save(clip, path)
        self.sequencer.mark_as_state(action_name, filename, RecState.AWAIT_UPLOAD)

        reference_detections = [detection for detection, count in reference_counter.items() if count > 0]
        details = self.timestamper.timestamp(clip, action_name, (event_span[0], event_span[1]), reference_detections)

        self.clip_uploader.start_upload(path, action_name, details)

    def _update_sequence(self, action_name: str, filename: str, counts: tuple[int, int]) -> None:
        gap = Config.SEQUENCE_FRAMES_GAP
        start, end = counts
        is_active = self.sequencer.is_active(action_name)

        if is_active:
            if start <= gap and end <= gap:
                self.sequencer.add(action_name, filename)
            elif start <= gap < end:
                self.sequencer.add_close(action_name, filename)
            elif start > gap:
                self.sequencer.close(action_name)
                if end <= gap:
                    self.sequencer.add(action_name, filename)
        elif end <= gap:
            self.sequencer.add(action_name, filename)
    