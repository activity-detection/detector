import cv2

class SkeletonDrawer:

    _FACE_CONNECTIONS = [
    (0, 1),  # nose -> left eye
    (0, 2),  # nose -> right eye
    (1, 2),  # left eye -> right eye
    (1, 3),  # left eye -> left ear
    (2, 4),  # right eye -> right ear
    ]

    _SKELETON = [
        (5, 7), (7, 9),    # left arm
        (6, 8), (8, 10),   # right arm
        (5, 6),            # shoulders
        (5, 11), (6, 12),  # torso
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    _FULL_SKELETON = _SKELETON + _FACE_CONNECTIONS
    
    def __init__(self, confidence_threshold=None):
        self.confidence_threshold = confidence_threshold

    def draw_points(self, frame, keypoints, keypoints_conf):
        for (x, y), kp_conf in zip(keypoints, keypoints_conf):
            if self.confidence_threshold is None or kp_conf > self.confidence_threshold:   # próg widoczności punktu
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    def draw_skeleton(self, frame, keypoints, keypoints_conf):
        for a, b in self._FULL_SKELETON:
            if (self.confidence_threshold is None or
            (keypoints_conf[a] > self.confidence_threshold and
            keypoints_conf[b] > self.confidence_threshold)):
                pt1 = tuple(keypoints[a].astype(int))
                pt2 = tuple(keypoints[b].astype(int))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    def draw(self, frame, keypoints, keypoints_conf):
        self.draw_points(frame, keypoints, keypoints_conf)
        self.draw_skeleton(frame, keypoints, keypoints_conf)
    