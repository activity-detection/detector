import cv2
import numpy as np
from ultralytics.engine.results import Results
from config import Config

class Anonymizer:
    
    def __init__(self):
        self.PIXELATION_SCALE = 0.2
        self.RADIUS_MULTIPLIER = 2.0
        self.LEFT_EYE_IDX = 1
        self.RIGHT_EYE_IDX = 2
        self.CONFIDENCE_THRESHOLD = Config.CONF_THRESHOLD
        
        # --- NEW PERSISTENCE SETTINGS ---
        self.FADE_OUT_FRAMES = 15  # Number of frames the mask persists after face loss (e.g., 15 frames @ 30fps = 0.5s)
        self.MATCH_DIST_THRESHOLD = 100 # Max distance in pixels to consider it the same face
        
        # List of dictionaries: {'x': int, 'y': int, 'r': int, 'ttl': int, 'matched': bool}
        self.tracked_faces = [] 

    def anonymize(self, frame, result: Results, type):
        """
        Main method. Handles persistence logic for 'pose' type.
        """
        if type == "pose":
            # 1. Get current detections from the current frame
            current_faces = []
            
            # Check if results exist and correspond to pose (class 0)
            if result.boxes is not None and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    if len(keypoints_data) > i:
                        # Calculate face geometry (center, radius)
                        face_geom = self._calculate_face_geometry(keypoints_data[i])
                        if face_geom:
                            current_faces.append(face_geom)

            # 2. Update tracker (match new detections with old ones)
            self._update_face_tracker(current_faces)

            # 3. Draw all active masks (both new and persisted ones)
            h, w = frame.shape[:2]
            for face in self.tracked_faces:
                self._apply_circular_mask(frame, face['x'], face['y'], face['r'], w, h)

        elif type == "box":
            # For license plates, keep the old logic (no persistence)
            if result.boxes is None:
                return frame
                
            boxes = result.boxes
            for i, box in enumerate(boxes):
                coords = box.xyxy.cpu().numpy().squeeze().astype(int)
                self._anonymize_box_rectangular(frame, coords, frame.shape[1], frame.shape[0])

        return frame

    def _calculate_face_geometry(self, kpts):
        """
        Returns (center_x, center_y, radius) based on eye keypoints.
        Returns None if detection confidence is low.
        """
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
        """
        Logic to match new detections to stored masks.
        """
        # Mark all old faces as "unmatched" initially
        for face in self.tracked_faces:
            face['matched'] = False

        new_tracked_faces = []

        # Try to match every NEW detection to an OLD one
        for (curr_x, curr_y, curr_r) in current_detections:
            matched = False
            
            # Find the closest face in history
            best_idx = -1
            min_dist = float('inf')

            for idx, old_face in enumerate(self.tracked_faces):
                if old_face['matched']: continue # Already matched to another face in this frame

                dist = np.sqrt((curr_x - old_face['x'])**2 + (curr_y - old_face['y'])**2)
                
                if dist < self.MATCH_DIST_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            
            if best_idx != -1:
                # MATCH FOUND: Update position and reset TTL (Time To Live)
                self.tracked_faces[best_idx]['x'] = curr_x
                self.tracked_faces[best_idx]['y'] = curr_y
                self.tracked_faces[best_idx]['r'] = curr_r
                self.tracked_faces[best_idx]['ttl'] = self.FADE_OUT_FRAMES
                self.tracked_faces[best_idx]['matched'] = True
                matched = True
            
            if not matched:
                # NEW FACE: Add to the list
                new_tracked_faces.append({
                    'x': curr_x, 'y': curr_y, 'r': curr_r, 
                    'ttl': self.FADE_OUT_FRAMES, 'matched': True
                })

        # Rebuild list: keep matched faces and unmatched ones with TTL > 0
        final_list = []
        
        # 1. Add updated old faces
        for face in self.tracked_faces:
            if face['matched']:
                final_list.append(face)
            else:
                # Not detected in this frame -> decrease TTL
                face['ttl'] -= 1
                if face['ttl'] > 0:
                    final_list.append(face)
        
        # 2. Add completely new faces
        final_list.extend(new_tracked_faces)
        
        self.tracked_faces = final_list

    def _apply_circular_mask(self, frame, center_x, center_y, radius, img_w, img_h):
        """
        Physically applies pixelation at a given point.
        """
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
        """
        Rectangular pixelation for bounding boxes (e.g. license plates).
        """
        x1, y1, x2, y2 = coords
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)

        if x2 <= x1 or y2 <= y1: return

        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        frame[y1:y2, x1:x2] = self._pixelate(roi, roi_w, roi_h)

    def _pixelate(self, image, w, h):
        """
        Helper function: Downscale -> Upscale (INTER_NEAREST)
        """
        small_w = max(1, int(w * self.PIXELATION_SCALE))
        small_h = max(1, int(h * self.PIXELATION_SCALE))

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)