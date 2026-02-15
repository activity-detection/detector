import cv2
import numpy as np
from ultralytics.engine.results import Results
from ultralytics import YOLO
from .config import Config
from itertools import batched

class Anonymizer:
    RADIUS_MULTIPLIER = 2.0
    MEDIAN_RADIUS = 25
    LEFT_EYE_IDX = 1
    RIGHT_EYE_IDX = 2
    PLATE_BATCH_SIZE = Config.BATCH_SIZE
    FADE_OUT_FRAMES = 15

    def __init__(self):
        self.license_plate_model = YOLO(Config.PLATES_MODEL_PATH)
        
        self.tracked_faces = {} # {id : (face_geo, ttl)}
        self.tracked_plates = {} # {id : (coords, ttl)}

    def anonymize_clip(self, frame_vectors):
        frames = [frame_vector.frame for frame_vector in frame_vectors]
        results = [frame_vector.vector.pose_results for frame_vector in frame_vectors]
        self.anonymize_faces(frames, results)
        self.anonymize_license_plates(frames)
        return frames
        
    def anonymize_faces(self, frames, results: Results):
        for frame, result in zip(frames, results):
            self.anonymize_face(frame, result)

    def anonymize_face(self, frame, result: Results):
        h, w = frame.shape[:2]
        current_faces = []
        if result.keypoints is not None:
            keypoints_data = result.keypoints.data.cpu().numpy()
            keypoints_id = result.boxes.id.data.cpu().numpy() if result.boxes.id is not None else None
            for keypoint in keypoints_data:
                face_geom = self._calculate_face_geometry(keypoint)
                current_faces.append(face_geom)

            if result.boxes.id is not None:
                keypoints_id = result.boxes.id.data.cpu().numpy()
                self.tracked_faces = self._update_tracker(self.tracked_faces, current_faces, keypoints_id)
                for key in self.tracked_faces:
                    face = self.tracked_faces[key][0]
                    self._apply_circular_mask(frame, face['x'], face['y'], face['r'], w, h)
            else:
                for face in current_faces:
                    self._apply_circular_mask(frame, face['x'], face['y'], face['r'], w, h)
        return frame

    def anonymize_license_plates(self, frames):
        for batch in batched(frames, self.PLATE_BATCH_SIZE):
            results = self.license_plate_model.track(batch, verbose=False, stream=True)
            for frame, result in zip(batch, results):
                self.anonymize_license_plate(frame, result)

    def anonymize_license_plate(self, frame, result: Results):
        h, w = frame.shape[:2]
    
        current_plates = []
        ids = []
        for box in result.boxes:
            coords = box.xyxy.cpu().numpy().squeeze().astype(int)
            if box.id is not None:
                ids.append(int(box.id))
                current_plates.append(coords)
            else:
                self._anonymize_box_rectangular(frame, coords, w, h)

        self.tracked_plates = self._update_tracker(self.tracked_plates, current_plates, ids)
        
        for key in self.tracked_plates:
            plate = self.tracked_plates[key][0]
            self._anonymize_box_rectangular(frame, plate, w, h)

        return frame

    def _update_tracker(self, tracker_dict, current_detections, ids):
        for key in tracker_dict:
            item, ttl = tracker_dict[key]
            ttl -= 1
            tracker_dict[key] = (item, ttl)
        for item, id in zip(current_detections, ids):
            if id is not None:
                tracker_dict[id] = (item, self.FADE_OUT_FRAMES)
        tracker_dict = {key:val for key, val in tracker_dict.items() if val[1] != 0}
        return tracker_dict

    def _calculate_face_geometry(self, kpts):
        left_eye = kpts[self.LEFT_EYE_IDX]
        right_eye = kpts[self.RIGHT_EYE_IDX]
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        eye_distance = np.sqrt(dx*dx + dy*dy)
        radius = int(eye_distance * self.RADIUS_MULTIPLIER)
        center_x = int((left_eye[0] + right_eye[0]) * 0.5)
        center_y = int((left_eye[1] + right_eye[1]) * 0.5)
        return {'x' : center_x, 'y' : center_y, 'r' : radius}

    def _apply_circular_mask(self, frame, center_x, center_y, radius, img_w, img_h):
        x1 = max(center_x - radius, 0)
        y1 = max(center_y - radius, 0)
        x2 = min(center_x + radius, img_w)
        y2 = min(center_y + radius, img_h)
        if x2 <= x1 or y2 <= y1: return
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        roi_center = (center_x - x1, center_y - y1)
        cv2.circle(mask, roi_center, radius, 255, -1)
        pixelated_roi = cv2.medianBlur(roi, self.MEDIAN_RADIUS)
        frame[y1:y2, x1:x2] = np.where(mask[..., None] > 0, pixelated_roi, roi)

    def _anonymize_box_rectangular(self, frame, coords, img_w, img_h):
        x1, y1, x2, y2 = coords
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1: return
        roi = frame[y1:y2, x1:x2]
        pixelated_roi = cv2.medianBlur(roi, self.MEDIAN_RADIUS)
        frame[y1:y2, x1:x2] = pixelated_roi