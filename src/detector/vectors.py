"""Module for managing action counts and vision model results.

Provides data structures for aggregating YOLO and pose estimation model 
results, and linking them to specific video frames.
"""

from ultralytics.engine.results import Results
from collections import Counter
from typing import Mapping, Iterable, TypedDict
import numpy as np


type CounterData = Mapping[str, int] | Iterable[str]


class ActionVector:
    """Stores action counts and associated YOLO/Pose model results.

    The class allows updating counters, adding vectors together, 
    and comparing their contents (greater-than-or-equal operator).

    Attributes:
        counter (Counter[str]): Counter for detections of actions or objects.
        pose_results (Results | None): Optional results from a pose estimation model.
        base_yolo_result (Results | None): Optional results from a base YOLO model.
    """

    def __init__(self, data: CounterData | None = None):
        """Initializes the ActionVector object.

        Args:
            data (CounterData | None, optional): Initial data to populate the 
                counter. Can be a mapping of counts or an iterable of class 
                names. Defaults to None.
        """
        self.counter: Counter[str] = Counter()
        self.pose_results: Results | None = None
        self.base_yolo_result: Results | None = None
        if data is not None:
            self.counter.update(data)

    def update(self, data: CounterData):
        """Updates the detection counter with new data.

        Args:
            data (CounterData): Data to add to the counter (e.g., an iterable of detected 
                objects or a mapping of objects to their counts).
        """
        self.counter.update(data)

    def __add__(self, other: 'ActionVector') -> 'ActionVector':
        """Adds values from two action vectors, creating a new vector.

        Combines the counters from both objects. For model results 
        (`pose_results`, `base_yolo_result`), it prioritizes the results from 
        the current vector; if absent (None), it takes them from the other vector.

        Args:
            other (ActionVector): The second vector to add.

        Returns:
            ActionVector: A new vector object combining both vectors.
        """
        new_vector = ActionVector()
        new_vector.update(self.counter)
        new_vector.update(other.counter)
        
        if self.pose_results is not None:
            pose_results = self.pose_results
        else:
            pose_results = other.pose_results
        
        new_vector.pose_results = pose_results

        if self.base_yolo_result is not None:
            base_yolo_result = self.base_yolo_result
        else:
            base_yolo_result = other.base_yolo_result
        
        new_vector.base_yolo_result = base_yolo_result

        return new_vector

    def __ge__(self, other: 'ActionVector') -> bool:
        """Checks if this vector contains at least the same counts as another.

        Args:
            other (ActionVector): The vector to compare against the current object.

        Returns:
            bool: True if for every key in `other`, the value in the current 
                counter is greater or equal. False otherwise.
        """
        return all(
            self.counter[f] >= other.counter[f]
            for f in other.counter
        )
    
    def __str__(self) -> str:
        """Returns a human-readable string representation of the action vector.

        Only actions with a count greater than 0 are included.

        Returns:
            str: A string in the format 'ActionVector(action: count, ...)' 
                or 'ActionVector(empty)' if the counter has no active elements.
        """
        active_counts = [
            f"{f}: {self.counter[f]}"
            for f in self.counter
            if self.counter[f] > 0
        ]
        
        if not active_counts:
            return "ActionVector(empty)"
        
        return f"ActionVector({', '.join(active_counts)})"
    

class FrameVector(TypedDict):
    """Dictionary combining a single image frame to its corresponding action vector.

    Attributes:
        frame (np.ndarray): NumPy array containing the video frame's pixel data.
        vector (ActionVector): Object grouping counts and model results for this frame.
    """
    frame: np.ndarray
    vector: ActionVector
    