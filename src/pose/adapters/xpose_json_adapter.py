from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import BODY_KEYPOINT_NAMES, HAND_KEYPOINT_NAMES, PoseFrame, PoseInstance, PoseKeypoint, PoseSequence


def _infer_group(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("_person") or stem.endswith("_body"):
        return "body"
    if stem.endswith("_face"):
        return "face"
    if stem.endswith("_hand"):
        return "hand"
    return "body"


def _normalize_xy(x: float, y: float, frame_width: Optional[int], frame_height: Optional[int]) -> Tuple[float, float]:
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x, y
    if not frame_width or not frame_height:
        raise ValueError("Frame width and height are required for pixel-space pose JSON.")
    return x / float(frame_width), y / float(frame_height)


def _instance_from_xy_points(
    points: Sequence[Sequence[float]],
    group: str,
    instance_id: str,
    frame_width: Optional[int],
    frame_height: Optional[int],
) -> PoseInstance:
    if group == "body":
        names = BODY_KEYPOINT_NAMES[: len(points)]
    elif group == "face":
        names = ["face_%03d" % index for index in range(len(points))]
    else:
        names = HAND_KEYPOINT_NAMES[: len(points)]

    keypoints = []
    for index, pair in enumerate(points):
        if len(pair) < 2:
            continue
        x_value, y_value = _normalize_xy(float(pair[0]), float(pair[1]), frame_width, frame_height)
        confidence = float(pair[2]) if len(pair) > 2 else 1.0
        name = names[index] if index < len(names) else "%s_%03d" % (group, index)
        keypoints.append(PoseKeypoint(name=name, group=group, x=x_value, y=y_value, confidence=max(0.0, min(confidence, 1.0))))

    return PoseInstance(instance_id=instance_id, score=1.0, keypoints=keypoints)


def _parse_motion_x(
    payload: Dict[str, object],
    source: str,
    frame_width: Optional[int],
    frame_height: Optional[int],
    fps: Optional[float],
) -> PoseSequence:
    frames = []
    for frame_index, annotation in enumerate(payload.get("annotations", [])):
        ann = dict(annotation)
        instances = []
        body = ann.get("body_kpts") or []
        if body:
            instances.append(_instance_from_xy_points(body[:17], "body", "0", frame_width, frame_height))
        left_hand = ann.get("lefthand_kpts") or []
        if left_hand:
            instances.append(_instance_from_xy_points(left_hand[:21], "left_hand", "0", frame_width, frame_height))
        right_hand = ann.get("righthand_kpts") or []
        if right_hand:
            instances.append(_instance_from_xy_points(right_hand[:21], "right_hand", "0", frame_width, frame_height))
        face = ann.get("face_kpts") or []
        if face:
            instances.append(_instance_from_xy_points(face, "face", "0", frame_width, frame_height))
        frames.append(PoseFrame(frame_index=frame_index, instances=instances))

    return PoseSequence(
        source=source,
        fps=float(fps or 0.0),
        frame_width=int(frame_width or 0),
        frame_height=int(frame_height or 0),
        frames=frames,
    )


def _parse_xpose_list(
    payload: List[object],
    source: str,
    frame_width: Optional[int],
    frame_height: Optional[int],
    fps: Optional[float],
    group: str,
) -> PoseSequence:
    frames = []
    for item in payload:
        frame_item = dict(item)
        instances = []
        for index, instance in enumerate(frame_item.get("instances", [])):
            instances.append(
                _instance_from_xy_points(
                    instance.get("keypoints", []),
                    group,
                    str(index),
                    frame_width,
                    frame_height,
                )
            )
        frames.append(PoseFrame(frame_index=int(frame_item.get("frame_id", 0)), instances=instances))

    return PoseSequence(
        source=source,
        fps=float(fps or 0.0),
        frame_width=int(frame_width or 0),
        frame_height=int(frame_height or 0),
        frames=frames,
    )


def load_pose_sequence(
    json_path: Path,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    fps: Optional[float] = None,
    group: Optional[str] = None,
) -> PoseSequence:
    with Path(json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    source = str(json_path)
    if isinstance(payload, dict) and payload.get("format_version") == "hypermotion_pose/v1":
        sequence = PoseSequence.from_dict(payload)
    elif isinstance(payload, dict) and "annotations" in payload:
        sequence = _parse_motion_x(payload, source, frame_width, frame_height, fps)
    elif isinstance(payload, list):
        sequence = _parse_xpose_list(payload, source, frame_width, frame_height, fps, group or _infer_group(Path(json_path)))
    else:
        raise ValueError("Unsupported pose JSON format: %s" % json_path)

    sequence.validate()
    return sequence
