from enum import Enum, auto
from typing import TypedDict
import time


class RecState(Enum):
    AWAIT_SAVE = auto()
    SAVING = auto()
    AWAIT_UPLOAD = auto()
    UPLOADING = auto()
    SENT = auto()
    FAILED = auto()  # New state for failed uploads


RecordingModel = TypedDict(
    'RecordingModel',
    {
        'id': str | None,
        'state': RecState,
        'filename': str,
        'created_at': float,  # Track when recording was created
        'last_updated': float  # Track last state change
     })


class SeqState(Enum):
    OPEN = auto()
    CLOSED = auto()

SequenceModel = TypedDict(
    'SequenceModel',
    {
        'state': SeqState,
        'recordings': list[RecordingModel]
    }
)


class Sequencer:
    def __init__(self) -> None:
        self.sequences: dict[str, list[SequenceModel]] = {}

    def add_close(self, action_name: str, filename: str, state: RecState = RecState.AWAIT_SAVE):
        self.add(action_name, filename, state)
        self.close(action_name)

    def add(self, action_name: str, filename: str, state: RecState = RecState.AWAIT_SAVE):
        current_time = time.time()
        if action_name not in self.sequences:
            sequence = self._make_new_sequence(filename, state, current_time)
            self.sequences[action_name] = [sequence]

        else: 
            last_sequence = self.sequences[action_name][-1]
            if last_sequence['state'] is SeqState.CLOSED:
                sequence = self._make_new_sequence(filename, state, current_time)
                self.sequences[action_name].append(sequence)

            else:
                recording = RecordingModel({
                    'id': None,
                    'state': state,
                    'filename': filename,
                    'created_at': current_time,
                    'last_updated': current_time
                })
                last_sequence['recordings'].append(recording)

    def close(self, action_name: str): # TODO drugi param boli, patrz clip_saver
        last_sequence = self.sequences[action_name][-1]
        last_sequence['state'] = SeqState.CLOSED
        
    def is_active(self, action_name: str) -> bool:
        if action_name in self.sequences:
            last_sequence = self.sequences[action_name][-1]
            state = last_sequence['state']
            if state is SeqState.OPEN:
                return True
            else:
                return False
        else:
            return False
    
    @staticmethod
    def _make_new_sequence(filename: str, state: RecState, current_time: float) -> SequenceModel:
        recording = RecordingModel({
            'id': None,
            'state': state,
            'filename': filename,
            'created_at': current_time,
            'last_updated': current_time
        })

        sequence = SequenceModel({
            'state': SeqState.OPEN,
            'recordings': [recording]
        })

        return sequence

    def mark_as_state(self, action_name: str, filename: str, state: RecState, db_id: str | None = None):
        current_time = time.time()
        if action_name in self.sequences:
            last_sequence = self.sequences[action_name][-1]
            for rec in last_sequence['recordings']:
                if rec.get('filename') == filename:
                    rec['state'] = state
                    rec['last_updated'] = current_time
                    if state is RecState.SENT:
                        rec['id'] = db_id
                    break

    def get_stuck_recordings(self, max_age_seconds: float = 600.0) -> list[tuple[str, str]]:
        """
        Get recordings that have been stuck in non-terminal states for too long.
        Returns list of (action_name, filename) tuples.
        """
        current_time = time.time()
        stuck_recordings: list[tuple[str, str]] = []

        for action_name, sequences in self.sequences.items():
            for sequence in sequences:
                for recording in sequence['recordings']:
                    state = recording['state']
                    last_updated = recording.get('last_updated', recording.get('created_at', 0))

                    # Consider recordings stuck if they're in intermediate states for too long
                    if state in [RecState.AWAIT_UPLOAD, RecState.UPLOADING, RecState.SAVING]:
                        if current_time - last_updated > max_age_seconds:
                            stuck_recordings.append((action_name, recording['filename']))

        return stuck_recordings

    def cleanup_stuck_recordings(self, max_age_seconds: float = 600.0, mark_as_failed: bool = True):
        """
        Clean up recordings that have been stuck for too long.
        Optionally mark them as failed instead of just removing them.
        """
        stuck_recordings = self.get_stuck_recordings(max_age_seconds)

        for action_name, filename in stuck_recordings:
            if mark_as_failed:
                self.mark_as_state(action_name, filename, RecState.FAILED)
            # Note: In a real implementation, you might want to remove the files from disk too
