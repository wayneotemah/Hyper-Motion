from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


BODY_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

HAND_KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

BODY_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

MODE_GROUPS = {
    "body": {"body"},
    "body_only": {"body"},
    "body_hands": {"body", "hand", "left_hand", "right_hand"},
    "body_hands_face": {"body", "hand", "left_hand", "right_hand", "face"},
    "all": {"body", "hand", "left_hand", "right_hand", "face"},
}


def groups_for_mode(mode: str) -> Set[str]:
    return MODE_GROUPS.get(mode, MODE_GROUPS["body_hands_face"])


@dataclass
class PoseKeypoint:
    name: str
    group: str
    x: float
    y: float
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseKeypoint":
        return cls(
            name=str(data["name"]),
            group=str(data["group"]),
            x=float(data["x"]),
            y=float(data["y"]),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "confidence": round(float(self.confidence), 6),
        }


@dataclass
class PoseInstance:
    instance_id: str
    score: float = 1.0
    keypoints: List[PoseKeypoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseInstance":
        return cls(
            instance_id=str(data.get("instance_id", "0")),
            score=float(data.get("score", 1.0)),
            keypoints=[PoseKeypoint.from_dict(item) for item in data.get("keypoints", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "score": round(float(self.score), 6),
            "keypoints": [item.to_dict() for item in self.keypoints],
        }


@dataclass
class PoseFrame:
    frame_index: int
    instances: List[PoseInstance] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseFrame":
        return cls(
            frame_index=int(data["frame_index"]),
            instances=[PoseInstance.from_dict(item) for item in data.get("instances", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "instances": [instance.to_dict() for instance in self.instances],
        }


@dataclass
class PoseSequence:
    format_version: str = "hypermotion_pose/v1"
    source: str = "unknown"
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    frames: List[PoseFrame] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoseSequence":
        return cls(
            format_version=str(data.get("format_version", "hypermotion_pose/v1")),
            source=str(data.get("source", "unknown")),
            fps=float(data.get("fps", 0.0)),
            frame_width=int(data.get("frame_width", 0)),
            frame_height=int(data.get("frame_height", 0)),
            frames=[PoseFrame.from_dict(item) for item in data.get("frames", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "source": self.source,
            "fps": round(float(self.fps), 6),
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "frames": [frame.to_dict() for frame in self.sorted_frames()],
        }

    def sorted_frames(self) -> List[PoseFrame]:
        return sorted(self.frames, key=lambda item: item.frame_index)

    def validate(self) -> None:
        if not self.frames:
            raise ValueError("Pose sequence has no frames.")
        if self.frame_width < 0 or self.frame_height < 0:
            raise ValueError("Pose sequence frame dimensions must be non-negative.")

        previous = -1
        for frame in self.sorted_frames():
            if frame.frame_index < 0:
                raise ValueError("Frame indices must be non-negative.")
            if frame.frame_index < previous:
                raise ValueError("Frame indices must be sorted.")
            previous = frame.frame_index
            for instance in frame.instances:
                for keypoint in instance.keypoints:
                    if not 0.0 <= keypoint.x <= 1.0:
                        raise ValueError("Keypoint x must be normalized to [0, 1].")
                    if not 0.0 <= keypoint.y <= 1.0:
                        raise ValueError("Keypoint y must be normalized to [0, 1].")
                    if not 0.0 <= keypoint.confidence <= 1.0:
                        raise ValueError("Keypoint confidence must be in [0, 1].")

    def max_frame_index(self) -> int:
        return max(frame.frame_index for frame in self.frames)

    def keypoints_for_groups(self, frame: PoseFrame, allowed_groups: Sequence[str]) -> List[PoseKeypoint]:
        allowed = set(allowed_groups)
        collected: List[PoseKeypoint] = []
        for instance in frame.instances:
            for keypoint in instance.keypoints:
                if keypoint.group in allowed:
                    collected.append(keypoint)
        return collected
