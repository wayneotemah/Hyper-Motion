from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .schema import BODY_EDGES, BODY_KEYPOINT_NAMES, HAND_EDGES, HAND_KEYPOINT_NAMES, PoseFrame, PoseInstance, PoseKeypoint, PoseSequence, groups_for_mode


BODY_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
]


def _scale_point(keypoint: PoseKeypoint, width: int, height: int) -> Tuple[int, int]:
    return int(round(keypoint.x * width)), int(round(keypoint.y * height))


def _draw_line(canvas: np.ndarray, point_a: Tuple[int, int], point_b: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
    cv2.line(canvas, point_a, point_b, color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_body(canvas: np.ndarray, instance: PoseInstance, width: int, height: int, confidence_threshold: float) -> None:
    lookup: Dict[str, PoseKeypoint] = {keypoint.name: keypoint for keypoint in instance.keypoints if keypoint.group == "body"}
    points = [lookup.get(name) for name in BODY_KEYPOINT_NAMES]
    line_width = max(2, min(4, int(min(width, height) / 300)))
    point_radius = max(2, min(4, int(min(width, height) / 300)))

    for index, (start, end) in enumerate(BODY_EDGES):
        point_a = points[start]
        point_b = points[end]
        if not point_a or not point_b:
            continue
        if point_a.confidence < confidence_threshold or point_b.confidence < confidence_threshold:
            continue
        _draw_line(canvas, _scale_point(point_a, width, height), _scale_point(point_b, width, height), BODY_COLORS[index % len(BODY_COLORS)], line_width)

    shoulder_names = ["left_shoulder", "right_shoulder", "nose", "left_hip", "right_hip"]
    if all(lookup.get(name) and lookup[name].confidence >= confidence_threshold for name in shoulder_names):
        left_shoulder = lookup["left_shoulder"]
        right_shoulder = lookup["right_shoulder"]
        mid_shoulder = (
            int(round((left_shoulder.x + right_shoulder.x) * 0.5 * width)),
            int(round((left_shoulder.y + right_shoulder.y) * 0.5 * height)),
        )
        _draw_line(canvas, _scale_point(left_shoulder, width, height), mid_shoulder, (0, 255, 0), line_width)
        _draw_line(canvas, _scale_point(right_shoulder, width, height), mid_shoulder, (255, 0, 85), line_width)
        _draw_line(canvas, mid_shoulder, _scale_point(lookup["nose"], width, height), (255, 170, 0), line_width)
        _draw_line(canvas, mid_shoulder, _scale_point(lookup["left_hip"], width, height), (255, 0, 255), line_width)
        _draw_line(canvas, mid_shoulder, _scale_point(lookup["right_hip"], width, height), (0, 0, 255), line_width)

    for index, keypoint in enumerate(points):
        if keypoint and keypoint.confidence >= confidence_threshold:
            cv2.circle(canvas, _scale_point(keypoint, width, height), point_radius, BODY_COLORS[index % len(BODY_COLORS)], thickness=-1)


def _draw_hand(canvas: np.ndarray, instance: PoseInstance, width: int, height: int, group: str, confidence_threshold: float) -> None:
    lookup: Dict[str, PoseKeypoint] = {
        keypoint.name: keypoint
        for keypoint in instance.keypoints
        if keypoint.group in {group, "hand"}
    }
    points = [lookup.get(name) for name in HAND_KEYPOINT_NAMES]
    line_width = max(1, min(2, int(min(width, height) / 500)))
    point_radius = max(2, min(4, int(min(width, height) / 400)))

    for edge_index, (start, end) in enumerate(HAND_EDGES):
        point_a = points[start]
        point_b = points[end]
        if not point_a or not point_b:
            continue
        if point_a.confidence < confidence_threshold or point_b.confidence < confidence_threshold:
            continue
        hue = edge_index / float(len(HAND_EDGES))
        rgb = cv2.cvtColor(np.uint8([[[int(hue * 179), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        _draw_line(canvas, _scale_point(point_a, width, height), _scale_point(point_b, width, height), tuple(int(channel) for channel in rgb), line_width)

    for keypoint in points:
        if keypoint and keypoint.confidence >= confidence_threshold:
            cv2.circle(canvas, _scale_point(keypoint, width, height), point_radius, (0, 0, 255), thickness=-1)


def _draw_face(canvas: np.ndarray, instance: PoseInstance, width: int, height: int, confidence_threshold: float) -> None:
    point_radius = max(1, min(3, int(min(width, height) / 500)))
    for keypoint in instance.keypoints:
        if keypoint.group != "face" or keypoint.confidence < confidence_threshold:
            continue
        cv2.circle(canvas, _scale_point(keypoint, width, height), point_radius, (255, 255, 255), thickness=-1)


def _draw_frame(canvas: np.ndarray, frame: PoseFrame, width: int, height: int, mode: str, confidence_threshold: float) -> np.ndarray:
    allowed_groups = groups_for_mode(mode)
    for instance in frame.instances:
        groups = {keypoint.group for keypoint in instance.keypoints}
        if "body" in allowed_groups and "body" in groups:
            _draw_body(canvas, instance, width, height, confidence_threshold)
        if allowed_groups.intersection({"hand", "left_hand", "right_hand"}):
            if "hand" in groups:
                _draw_hand(canvas, instance, width, height, "hand", confidence_threshold)
            else:
                if "left_hand" in groups:
                    _draw_hand(canvas, instance, width, height, "left_hand", confidence_threshold)
                if "right_hand" in groups:
                    _draw_hand(canvas, instance, width, height, "right_hand", confidence_threshold)
        if "face" in allowed_groups and "face" in groups:
            _draw_face(canvas, instance, width, height, confidence_threshold)
    return canvas


def _writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def render_pose_video(
    sequence: PoseSequence,
    output_path: Path,
    mode: str = "body_hands_face",
    source_video: Optional[Path] = None,
    overlay_output_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    confidence_threshold: float = 0.2,
) -> Dict[str, int]:
    render_width = int(width or sequence.frame_width or 768)
    render_height = int(height or sequence.frame_height or 512)
    render_fps = float(fps or sequence.fps or 24.0)
    clean_writer = _writer(output_path, render_fps, render_width, render_height)
    overlay_writer = _writer(overlay_output_path, render_fps, render_width, render_height) if overlay_output_path else None

    capture = cv2.VideoCapture(str(source_video)) if source_video else None
    for frame in sequence.sorted_frames():
        clean_canvas = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        _draw_frame(clean_canvas, frame, render_width, render_height, mode, confidence_threshold)
        clean_writer.write(clean_canvas)

        if overlay_writer:
            base_frame = None
            if capture and capture.isOpened():
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    base_frame = cv2.resize(candidate, (render_width, render_height), interpolation=cv2.INTER_AREA)
            if base_frame is None:
                base_frame = np.zeros((render_height, render_width, 3), dtype=np.uint8)
            overlay_skeleton = np.zeros_like(base_frame)
            _draw_frame(overlay_skeleton, frame, render_width, render_height, mode, confidence_threshold)
            blended = cv2.addWeighted(base_frame, 0.7, overlay_skeleton, 0.9, 0.0)
            overlay_writer.write(blended)

    clean_writer.release()
    if overlay_writer:
        overlay_writer.release()
    if capture:
        capture.release()

    return {
        "width": render_width,
        "height": render_height,
        "frame_count": len(sequence.frames),
    }
