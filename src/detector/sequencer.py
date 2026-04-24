from typing import TypedDict
from enum import Enum, auto
import threading
import time


class RecState(Enum):
    AWAIT_SAVE = auto()
    SAVING = auto()
    AWAIT_UPLOAD = auto()
    UPLOADING = auto()
    SENT = auto()
    FAILED = auto()
    STASHED = auto()

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
        self._lock = threading.Lock()

    def add_close(self, action_name: str, filename: str, state: RecState = RecState.AWAIT_SAVE):
        self.add(action_name, filename, state)
        self.close(action_name)

    def add(self, action_name: str, filename: str, state: RecState = RecState.AWAIT_SAVE):
        with self._lock:
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

    def close(self, action_name: str):
        with self._lock:
            last_sequence = self.sequences[action_name][-1]
            last_sequence['state'] = SeqState.CLOSED
    
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
    
    def is_active(self, action_name: str) -> bool:
        with self._lock:
            if action_name in self.sequences:
                last_sequence = self.sequences[action_name][-1]
                state = last_sequence['state']
                if state is SeqState.OPEN:
                    return True
                else:
                    return False
            else:
                return False

    def mark_as_state(self, action_name: str, filename: str, state: RecState, db_id: str | None = None):
        with self._lock:
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
    
    def get_id_sequence(self, action_name: str, filename: str) -> tuple[int | None, list[RecordingModel] | None]:
        with self._lock:
            sequences = self.sequences[action_name]
            if sequences:
                for sequence in sequences:
                    recordings = sequence['recordings']
                    for i, recording in enumerate(recordings):
                        if filename == recording['filename']:
                            return i, recordings
            return (None, None)     

    def del_sequence(self, action_name: str, my_sequence: list[RecordingModel]):
        with self._lock:
            sequences = self.sequences[action_name]
            if sequences:
                for sequence in sequences:
                    if my_sequence == sequence['recordings']:
                        sequences.remove(sequence)

    def cleanup_finished_sequences(self):
        """
        Deletes all CLOSED sequences that contain only 
        recordings of a terminal state (SENT, FAILED, STASHED).
        """
        terminal_states = {RecState.SENT, RecState.FAILED, RecState.STASHED}

        with self._lock:
            for action_name in list(self.sequences.keys()):
                # Remaining active sequences
                active_sequences: list[SequenceModel] = []
                
                for sequence in self.sequences[action_name]:
                    # Is sequence closed
                    if sequence['state'] != SeqState.CLOSED:
                        active_sequences.append(sequence)
                        continue
                        
                    terminal = False
                    for rec in sequence['recordings']:
                        if rec['state'] in terminal_states:
                            terminal = True
                            break
                    
                    if terminal:
                        continue
                    active_sequences.append(sequence)

                # Aktualizujemy listę sekwencji dla danej akcji
                self.sequences[action_name] = active_sequences
                
                # Remove key (action_name) if no sequences left
                if not self.sequences[action_name]:
                    del self.sequences[action_name]

