from dataclasses import dataclass
from collections import Counter
from typing import TypedDict
import math

from src.detector.vectors import FrameVector
from src.detector.config import Config


@dataclass
class ObjectModel:
    name: str
    count: int

TimestampModel = TypedDict(
    'TimestampModel', 
    {
        'from': str, 
        'to': str
     })

@dataclass
class DetectionStampModel:
    objects: list[ObjectModel]
    timestamp: TimestampModel

@dataclass
class EventStampModel:
    label: str
    timestamp: TimestampModel

@dataclass
class FullStampModel:
    events: list[EventStampModel]
    detections: list[DetectionStampModel]

class Timestamper:

    def __init__(self):
        pass

    def timestamp(
        self,
        recording: list[FrameVector],
        name: str,
        event_span: tuple[int, int],
        reference_detections: list[str]
    ) -> FullStampModel:
        
        detections_per_second = self.find_most_common(recording, reference_detections)

        timestamped_det_list = self._make_timestamped_list(detections_per_second)
        timestamped_event_list = self._event_stamp(name, event_span[0], event_span[1])

        timestamped_all = FullStampModel(
            events=timestamped_event_list,
            detections=timestamped_det_list
        )
        return timestamped_all

    def find_most_common(self, recording: list[FrameVector], reference_detections: list[str]) -> list[dict[str, int]]:
        duration = math.ceil(len(recording) / Config.FRAME_RATE)
        detections_per_second = self._init_seconds(duration, reference_detections)

        for index, frame_vector in enumerate(recording):
            curr_vector = frame_vector.vector
            curr_second = int(index / Config.FRAME_RATE)

            for detection in reference_detections:
                count = curr_vector.counter[detection]
                counter = detections_per_second[curr_second][detection]
                counter[count] += 1

        det_per_second_max: list[dict[str, int]] = []

        for detections_second in detections_per_second:
            det_seconds_max: dict[str, int] = {}

            for detection, counter in detections_second.items():
                most_common_list = counter.most_common(1)
                if not most_common_list:
                    most_common_count = 0
                else:
                    most_common_count = most_common_list[0][0]
                det_seconds_max[detection] = most_common_count

            det_per_second_max.append(det_seconds_max)

        return det_per_second_max

    def _init_seconds(self, duration: int, reference_detections: list[str]) -> list[dict[str, Counter[int]]]:
        detections_per_second: list[dict[str, Counter[int]]] = []
        for _ in range(duration):
            detections_second: dict[str, Counter[int]] = {}
            for detection in reference_detections:
                detections_second[detection] = Counter()
            detections_per_second.append(detections_second)

        return detections_per_second
    
    def _make_timestamped_list(self, detections_per_second: list[dict[str, int]]) -> list[DetectionStampModel]:
        detections: list[DetectionStampModel] = []
        prev_det_vector = detections_per_second[0]
        time_from = 0
        for second in range(1, len(detections_per_second)):
            curr_det_vector = detections_per_second[second]
            if curr_det_vector != prev_det_vector:
                detection = self._detection_stamp(prev_det_vector, time_from, second)
                detections.append(detection)
                time_from = second
                prev_det_vector = curr_det_vector

        detection = self._detection_stamp(prev_det_vector, time_from, len(detections_per_second))
        if detection: # we ignore empty
            detections.append(detection)

        return detections
    
    @staticmethod
    def _detection_stamp(det_vector: dict[str, int], time_from: int, time_to: int) -> DetectionStampModel | None:
        objects: list[ObjectModel] = []
        for detection, count in det_vector.items():
            if count > 0: # db will not accept a zero count
                if detection == "person": # in db it reads human not person
                    detection = "human"
                objects.append(ObjectModel(
                    name=detection,
                    count= count
                ))
        if objects:
            return DetectionStampModel(
                objects=objects,
                timestamp={
                    "from": f"PT{time_from}S",
                    "to": f"PT{time_to}S"
                }
            )
        else:
            return None # thus we return None if no count is > 0

    @staticmethod
    def _event_stamp(name: str, time_from: int, time_to: int) -> list[EventStampModel]:
        return [EventStampModel(
            label=name,
            timestamp={
                "from": f"PT{time_from}S",
                "to": f"PT{time_to}S"
            }
        )]
    