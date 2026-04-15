from collections import deque
from dataclasses import dataclass

from src.detector.vectors import ActionVector
from src.detector.enums import State, Command


@dataclass
class ActionConfig:
    max_inactive_frames: int
    max_duration_frames: int
    check_count: int
    start_conf: float
    end_conf: float


class ActionClass:
    
    def __init__(self, name: str, action_vector: ActionVector, config: ActionConfig):
        self.name = name
        self.action_vector = action_vector
        
        self.trigger_history: deque[bool] = deque(maxlen=config.check_count)
        self.state = State.IDLE
        self.idling = 0
        self.awaiting = 0 # TODO figure out a cap
        self.frame_count = 0
        self.triggered = False
        self.config = config

        self.idling_final = 0

    def check(self, reference_vector: ActionVector) -> Command:
        ge = reference_vector >= self.action_vector
        self.trigger_history.append(ge)
        if len(self.trigger_history) == self.config.check_count:
            ge_count = sum(self.trigger_history)
            percentage = round(ge_count / self.config.check_count, 2) * 100
            if percentage >= self.config.start_conf:
                self.triggered = True
            else:
                self.triggered = False

            if self.state in (State.ACTIVE, State.PASSIVE):
                if self.frame_count > self.config.max_duration_frames:
                    return self.end()

                if self.state is State.ACTIVE:
                    self.frame_count += 1

                    if self.triggered:
                        return Command.CONTINUE
                    else:
                        self.idling += 1
                        self.state = State.PASSIVE
                        return Command.CONTINUE
                    
                if self.state is State.PASSIVE:
                    if self.idling > self.config.max_inactive_frames:
                        return self.end()
                    elif self.triggered:
                        self.frame_count += 1
                        self.idling = 0
                        return Command.CONTINUE
                    else:
                        self.frame_count += 1
                        self.idling += 1
                        return Command.CONTINUE
                
            elif self.state is State.IDLE:
                if self.triggered:
                    self.state = State.ACTIVE
                    self.frame_count += 1
                    self.awaiting = 0
                    return Command.BEGIN
                self.awaiting += 1
                return Command.AWAIT
            
        return Command.AWAIT

    def end(self) -> Command:
        self.state = State.IDLE
        self.frame_count = 0
        self.idling_final = self.idling
        self.idling = 0
        self.awaiting += 1
        return Command.END

    def __str__(self) -> str:
        status = "ACTIVE" if self.state is State.ACTIVE else "IDLE"
        return f"ActionClass(name='{self.name}', status={status}, cooldown={self.idling}/{self.config.max_inactive_frames})"
    