from ultralytics.engine.results import Results
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Iterable
import numpy as np


type update = Mapping[str, int] | Iterable[str]

class ActionVector:

    def __init__(self, data: update | None = None):
        self.counter: Counter[str] = Counter()
        self.pose_results: Results | None = None
        self.base_yolo_result: Results | None = None
        if data is not None:
            self.counter.update(data)

    def update(self, data: update):
        self.counter.update(data)

    def __add__(self, other: 'ActionVector') -> 'ActionVector':
        """adds values from two vectors,
        takes results from first vector if present otherwise from second"""
        
        new_vector = ActionVector()
        new_vector.update(self.counter)
        new_vector.update(other.counter)
        
        pose_results = self.pose_results if self.pose_results is not None else other.pose_results
        new_vector.pose_results = pose_results
        base_yolo_result = self.base_yolo_result if self.base_yolo_result is not None else other.base_yolo_result
        new_vector.base_yolo_result = base_yolo_result

        return new_vector

    def __ge__(self, other: 'ActionVector') -> bool:
        return all(
            self.counter[f] >= other.counter[f]
            for f in other.counter
        )
    
    def __str__(self) -> str:
        active_counts = [
            f"{f}: {self.counter[f]}"
            for f in self.counter
            if self.counter[f] > 0
        ]
        
        if not active_counts:
            return "ActionVector(empty)"
        
        return f"ActionVector({', '.join(active_counts)})"
    

@dataclass
class FrameVector:
    frame: np.ndarray
    vector: ActionVector