"""Pose estimation with MediaPipe: one frame in, 33 joints x (x, y, z, visibility) out."""
import cv2
import mediapipe as mp
import numpy as np

NUM_JOINTS, RAW_CHANNELS = 33, 4          # MediaPipe joints x (x, y, z, visibility)

_mp_pose = mp.solutions.pose
_mp_draw = mp.solutions.drawing_utils


class PoseExtractor:
    def __init__(self, model_complexity: int = 1, static_image_mode: bool = False):
        # complexity 0 = fastest (edge devices), 1 = balanced, 2 = most accurate
        self.pose = _mp_pose.Pose(static_image_mode=static_image_mode,
                                  model_complexity=model_complexity,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    def __call__(self, frame_bgr):
        """Returns (keypoints (33, 4) or None, raw MediaPipe result).
        MediaPipe gives x, y in [0, 1] of width/height and z on the same scale as x.
        We convert to pixels so x, y and z share one unit (no aspect-ratio distortion)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        if result.pose_landmarks is None:
            return None, result
        kp = np.array([[lm.x * w, lm.y * h, lm.z * w, lm.visibility]
                       for lm in result.pose_landmarks.landmark], dtype=np.float32)
        return kp, result

    def close(self):
        self.pose.close()


def empty_keypoints():
    return np.zeros((NUM_JOINTS, RAW_CHANNELS), dtype=np.float32)


def draw_skeleton(frame, result):
    if result is not None and result.pose_landmarks is not None:
        _mp_draw.draw_landmarks(frame, result.pose_landmarks, _mp_pose.POSE_CONNECTIONS)
    return frame


def world_keypoints(result):
    """MediaPipe world landmarks: (33, 4) = x, y, z in metres (hip-centred) + visibility.
    Better than image coordinates for measuring real joint angles."""
    if result is None or result.pose_world_landmarks is None:
        return None
    return np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in result.pose_world_landmarks.landmark], dtype=np.float32)
