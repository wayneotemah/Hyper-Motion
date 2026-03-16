from __future__ import annotations

from pathlib import Path
from typing import List

import cv2

from ..schema import BODY_KEYPOINT_NAMES, HAND_KEYPOINT_NAMES, PoseFrame, PoseInstance, PoseKeypoint, PoseSequence


MEDIAPIPE_BODY_MAPPING = [
    ("nose", 0),
    ("left_eye", 2),
    ("right_eye", 5),
    ("left_ear", 7),
    ("right_ear", 8),
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
]

MEDIAPIPE_FACE_INDICES = [
    10, 152, 234, 454, 1, 4, 33, 133, 263, 362,
    61, 291, 13, 14, 17, 78, 308, 70, 300, 105,
    334, 107, 336, 159, 386, 145, 374,
]


def _body_keypoints(results) -> List[PoseKeypoint]:
    keypoints = []
    if not results.pose_landmarks:
        return keypoints
    for name, landmark_index in MEDIAPIPE_BODY_MAPPING:
        landmark = results.pose_landmarks.landmark[landmark_index]
        keypoints.append(
            PoseKeypoint(
                name=name,
                group="body",
                x=float(landmark.x),
                y=float(landmark.y),
                confidence=max(0.0, min(float(getattr(landmark, "visibility", 1.0)), 1.0)),
            )
        )
    return keypoints


def _hand_keypoints(landmarks, group: str) -> List[PoseKeypoint]:
    keypoints = []
    if not landmarks:
        return keypoints
    for index, landmark in enumerate(landmarks.landmark):
        keypoints.append(
            PoseKeypoint(
                name=HAND_KEYPOINT_NAMES[index],
                group=group,
                x=float(landmark.x),
                y=float(landmark.y),
                confidence=1.0,
            )
        )
    return keypoints


def _face_keypoints(results) -> List[PoseKeypoint]:
    keypoints = []
    if not results.face_landmarks:
        return keypoints
    for index in MEDIAPIPE_FACE_INDICES:
        landmark = results.face_landmarks.landmark[index]
        keypoints.append(
            PoseKeypoint(
                name="face_%03d" % index,
                group="face",
                x=float(landmark.x),
                y=float(landmark.y),
                confidence=1.0,
            )
        )
    return keypoints


def extract_pose_sequence_from_video(
    video_path: Path,
    mode: str = "body_hands_face",
    min_detection_confidence: float = 0.35,
    min_tracking_confidence: float = 0.35,
) -> PoseSequence:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is not installed. Install requirements-mac.txt to enable local pose extraction.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open video: %s" % video_path)

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = []
    frame_index = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            keypoints = []
            keypoints.extend(_body_keypoints(results))
            if mode in {"body_hands", "body_hands_face", "all"}:
                keypoints.extend(_hand_keypoints(results.left_hand_landmarks, "left_hand"))
                keypoints.extend(_hand_keypoints(results.right_hand_landmarks, "right_hand"))
            if mode in {"body_hands_face", "all"}:
                keypoints.extend(_face_keypoints(results))

            frames.append(
                PoseFrame(
                    frame_index=frame_index,
                    instances=[PoseInstance(instance_id="0", score=1.0, keypoints=keypoints)],
                )
            )
            frame_index += 1

    capture.release()
    sequence = PoseSequence(
        source="mediapipe:%s" % video_path,
        fps=fps,
        frame_width=width,
        frame_height=height,
        frames=frames,
    )
    sequence.validate()
    return sequence
