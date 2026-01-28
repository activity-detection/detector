import cv2
import numpy as np
from ultralytics.engine.results import Results
from config import Config

class Anonymizer:
    
    def __init__(self):
        self.BLOCK_SIZE = 15
        self.RADIUS_MULTIPLIER = 2.0
        self.LEFT_EYE_IDX = 1
        self.RIGHT_EYE_IDX = 2
        self.CONFIDENCE_THRESHOLD = Config.CONF_THRESHOLD
        
        # --- PERSISTENCE SETTINGS ---
        self.FADE_OUT_FRAMES = 15 
        self.MATCH_DIST_THRESHOLD = 100 
        
        # Trackers
        self.tracked_faces = []   # {'x', 'y', 'r', 'ttl', 'matched'}
        self.tracked_plates = []  # {'coords': [x1,y1,x2,y2], 'ttl': int, 'matched': bool}

    def anonymize(self, frame, result: Results, type):
        h, w = frame.shape[:2]

        if type == "pose":
            current_faces = []
            if result.boxes is not None and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    if len(keypoints_data) > i:
                        face_geom = self._calculate_face_geometry(keypoints_data[i])
                        if face_geom:
                            current_faces.append(face_geom)

            self._update_face_tracker(current_faces)

            for face in self.tracked_faces:
                self._apply_circular_mask(frame, face['x'], face['y'], face['r'], w, h)

        elif type == "box":
            current_plates = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    # 1. Apply Confidence Threshold
                    conf = box.conf.item() if box.conf is not None else 0
                    if conf < self.CONFIDENCE_THRESHOLD:
                        continue

                    coords = box.xyxy.cpu().numpy().squeeze().astype(int)
                    current_plates.append(coords)

            # 2. Update tracker (persistence logic)
            self._update_plate_tracker(current_plates)

            # 3. Draw all active plates (current + buffered)
            for plate in self.tracked_plates:
                self._anonymize_box_rectangular(frame, plate['coords'], w, h)

        return frame

    def _update_plate_tracker(self, current_detections):
        """
        Logic to match new rectangular detections to stored plates.
        Uses center-to-center distance for matching.
        """
        # Mark all old plates as "unmatched" initially
        for plate in self.tracked_plates:
            plate['matched'] = False

        new_tracked_plates = []

        for curr_coords in current_detections:
            matched = False
            
            # Calculate center of current detection
            curr_cx = (curr_coords[0] + curr_coords[2]) / 2
            curr_cy = (curr_coords[1] + curr_coords[3]) / 2

            # Find the closest plate in history
            best_idx = -1
            min_dist = float('inf')

            for idx, old_plate in enumerate(self.tracked_plates):
                if old_plate['matched']: continue 

                # Calculate center of old plate
                old_coords = old_plate['coords']
                old_cx = (old_coords[0] + old_coords[2]) / 2
                old_cy = (old_coords[1] + old_coords[3]) / 2

                dist = np.sqrt((curr_cx - old_cx)**2 + (curr_cy - old_cy)**2)
                
                if dist < self.MATCH_DIST_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            
            if best_idx != -1:
                # MATCH FOUND: Update position and reset TTL
                self.tracked_plates[best_idx]['coords'] = curr_coords
                self.tracked_plates[best_idx]['ttl'] = self.FADE_OUT_FRAMES
                self.tracked_plates[best_idx]['matched'] = True
                matched = True
            
            if not matched:
                # NEW PLATE
                new_tracked_plates.append({
                    'coords': curr_coords, 
                    'ttl': self.FADE_OUT_FRAMES, 
                    'matched': True
                })

        # Rebuild list
        final_list = []
        
        # 1. Add updated old plates
        for plate in self.tracked_plates:
            if plate['matched']:
                final_list.append(plate)
            else:
                # Not detected -> decrease TTL
                plate['ttl'] -= 1
                if plate['ttl'] > 0:
                    final_list.append(plate)
        
        # 2. Add completely new plates
        final_list.extend(new_tracked_plates)
        
        self.tracked_plates = final_list

    # --- EXISTING HELPERS BELOW (Unchanged) ---
    def _calculate_face_geometry(self, kpts):
        if len(kpts) <= max(self.LEFT_EYE_IDX, self.RIGHT_EYE_IDX):
            return None
        left_eye = kpts[self.LEFT_EYE_IDX]
        right_eye = kpts[self.RIGHT_EYE_IDX]
        if left_eye[2] < self.CONFIDENCE_THRESHOLD or right_eye[2] < self.CONFIDENCE_THRESHOLD:
            return None
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        eye_distance = np.sqrt(dx*dx + dy*dy)
        radius = int(eye_distance * self.RADIUS_MULTIPLIER)
        if radius < 1: return None
        center_x = int((left_eye[0] + right_eye[0]) * 0.5)
        center_y = int((left_eye[1] + right_eye[1]) * 0.5)
        return (center_x, center_y, radius)

    def _update_face_tracker(self, current_detections):
        for face in self.tracked_faces:
            face['matched'] = False
        new_tracked_faces = []
        for (curr_x, curr_y, curr_r) in current_detections:
            matched = False
            best_idx = -1
            min_dist = float('inf')
            for idx, old_face in enumerate(self.tracked_faces):
                if old_face['matched']: continue
                dist = np.sqrt((curr_x - old_face['x'])**2 + (curr_y - old_face['y'])**2)
                if dist < self.MATCH_DIST_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            if best_idx != -1:
                self.tracked_faces[best_idx]['x'] = curr_x
                self.tracked_faces[best_idx]['y'] = curr_y
                self.tracked_faces[best_idx]['r'] = curr_r
                self.tracked_faces[best_idx]['ttl'] = self.FADE_OUT_FRAMES
                self.tracked_faces[best_idx]['matched'] = True
                matched = True
            if not matched:
                new_tracked_faces.append({'x': curr_x, 'y': curr_y, 'r': curr_r, 'ttl': self.FADE_OUT_FRAMES, 'matched': True})
        final_list = []
        for face in self.tracked_faces:
            if face['matched']:
                final_list.append(face)
            else:
                face['ttl'] -= 1
                if face['ttl'] > 0:
                    final_list.append(face)
        final_list.extend(new_tracked_faces)
        self.tracked_faces = final_list

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
        pixelated_roi = self._pixelate(roi, roi_w, roi_h)
        frame[y1:y2, x1:x2] = np.where(mask[..., None] > 0, pixelated_roi, roi)

    def _anonymize_box_rectangular(self, frame, coords, img_w, img_h):
        x1, y1, x2, y2 = coords
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x2 <= x1 or y2 <= y1: return
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        frame[y1:y2, x1:x2] = self._pixelate(roi, roi_w, roi_h)

    def _pixelate(self, image, w, h):
        small_w = max(1, w // self.BLOCK_SIZE)
        small_h = max(1, h // self.BLOCK_SIZE)

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)