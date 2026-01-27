import cv2
import numpy as np
from ultralytics.engine.results import Results
from config import Config

class Anonymizer:
    
    def __init__(self):
        self.PIXELATION_SCALE = 0.2  # 0.1 = pixel size becomes 10x bigger
        self.RADIUS_MULTIPLIER = 2.0 # Controls how large the circle is around the eyes
        self.LEFT_EYE_IDX = 1
        self.RIGHT_EYE_IDX = 2
        self.CONFIDENCE_THRESHOLD = Config.CONF_THRESHOLD

    def anonymize(self, frame, result: Results, type):
        """
        Main method accepting the frame and YOLO results.
        Modifies the frame 'in-place' (overwrites it).
        """
        # If no detections, do nothing
        if result.boxes is None:
            return frame

        # Move data to CPU once to avoid copying inside the loop
        boxes = result.boxes
        # Get keypoints only if they exist (for class 0)
        keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
        
        h, w = frame.shape[:2]

        # Iterate through all detected objects
        for i, box in enumerate(boxes):

            # --- CLASS 0: PERSON (Keypoints Logic) ---
            if type == "pose":
                if keypoints is not None and len(keypoints) > i:
                    self._anonymize_face_circular(frame, keypoints[i], w, h)

            # --- CLASS 1: LICENSE PLATE (Bounding Box Logic) ---
            elif type == "box":
                coords = box.xyxy.cpu().numpy().squeeze().astype(int)
                self._anonymize_box_rectangular(frame, coords, w, h)

        return frame

    def _anonymize_face_circular(self, frame, kpts, img_w, img_h):
        """
        Face anonymization based on eyes. 
        Uses a circular mask inside a small Region of Interest (ROI).
        """
        # Check if we have enough points
        if len(kpts) <= max(self.LEFT_EYE_IDX, self.RIGHT_EYE_IDX):
            return

        left_eye = kpts[self.LEFT_EYE_IDX]
        right_eye = kpts[self.RIGHT_EYE_IDX]

        # Check detection confidence
        if left_eye[2] < self.CONFIDENCE_THRESHOLD or right_eye[2] < self.CONFIDENCE_THRESHOLD:
            return

        # Geometric calculations for center and radius
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        eye_distance = np.sqrt(dx*dx + dy*dy)
        
        radius = int(eye_distance * self.RADIUS_MULTIPLIER)
        if radius < 1: return

        center_x = int((left_eye[0] + right_eye[0]) * 0.5)
        center_y = int((left_eye[1] + right_eye[1]) * 0.5)

        # Calculate ROI (square around face) with boundary protection
        x1 = max(center_x - radius, 0)
        y1 = max(center_y - radius, 0)
        x2 = min(center_x + radius, img_w)
        y2 = min(center_y + radius, img_h)

        if x2 <= x1 or y2 <= y1: return

        # Crop only the face area
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Create a circular mask (only for this small crop)
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        roi_center = (center_x - x1, center_y - y1)
        cv2.circle(mask, roi_center, radius, 255, -1)

        # Pixelate the crop
        pixelated_roi = self._pixelate(roi, roi_w, roi_h)

        # Merge: where mask is white -> pixelated, where black -> original
        # Using np.where for speed
        frame[y1:y2, x1:x2] = np.where(mask[..., None] > 0, pixelated_roi, roi)

    def _anonymize_box_rectangular(self, frame, coords, img_w, img_h):
        """
        Anonymization of the entire rectangle (for license plates).
        No masks, pure rectangle pixelation.
        """
        x1, y1, x2, y2 = coords

        # Boundary protection
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img_w), min(y2, img_h)

        if x2 <= x1 or y2 <= y1: return

        # Crop the rectangle
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Pixelate and paste back directly
        frame[y1:y2, x1:x2] = self._pixelate(roi, roi_w, roi_h)

    def _pixelate(self, image, w, h):
        """
        Helper function: Downscale -> Upscale (INTER_NEAREST)
        """
        # Calculate small size, enforcing minimum 1px
        small_w = max(1, int(w * self.PIXELATION_SCALE))
        small_h = max(1, int(h * self.PIXELATION_SCALE))

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)