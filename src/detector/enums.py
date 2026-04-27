from enum import Enum, auto

class State(Enum):
    ACTIVE = auto()
    PASSIVE = auto()
    IDLE = auto()
    
class Command(Enum):
    BEGIN = auto()
    END = auto()
    CONTINUE = auto()
    AWAIT = auto()
    