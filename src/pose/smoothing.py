from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from .schema import PoseFrame, PoseInstance, PoseKeypoint, PoseSequence


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.value = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.value = value
            self.initialized = True
            return value
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.35, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_value = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, dt: float) -> float:
        if self.last_value is None:
            self.last_value = value
        derivative = (value - self.last_value) / dt
        derivative_hat = self.dx_filter.filter(derivative, self._alpha(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        filtered = self.x_filter.filter(value, self._alpha(dt, cutoff))
        self.last_value = filtered
        return filtered


def smooth_pose_sequence(sequence: PoseSequence, method: str = "one_euro", window_size: int = 5) -> PoseSequence:
    normalized_method = (method or "none").lower()
    if normalized_method in {"none", "off"}:
        return sequence

    fps = sequence.fps or 30.0
    dt = 1.0 / max(fps, 1.0)
    histories: Dict[Tuple[str, str, str], Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=window_size))
    filters: Dict[Tuple[str, str, str], Tuple[OneEuroFilter, OneEuroFilter]] = {}
    last_values: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    smoothed_frames = []

    for frame in sequence.sorted_frames():
        new_instances = []
        for instance in frame.instances:
            new_keypoints = []
            for keypoint in instance.keypoints:
                track_id = (instance.instance_id, keypoint.group, keypoint.name)
                x_value = keypoint.x
                y_value = keypoint.y

                if normalized_method == "moving_average":
                    history = histories[track_id]
                    history.append((x_value, y_value))
                    x_value = sum(point[0] for point in history) / len(history)
                    y_value = sum(point[1] for point in history) / len(history)
                elif normalized_method == "ema":
                    previous = last_values.get(track_id, (x_value, y_value))
                    alpha = 0.4
                    x_value = alpha * x_value + (1.0 - alpha) * previous[0]
                    y_value = alpha * y_value + (1.0 - alpha) * previous[1]
                else:
                    if track_id not in filters:
                        filters[track_id] = (OneEuroFilter(), OneEuroFilter())
                    fx, fy = filters[track_id]
                    x_value = fx.filter(x_value, dt)
                    y_value = fy.filter(y_value, dt)

                last_values[track_id] = (x_value, y_value)
                new_keypoints.append(
                    PoseKeypoint(
                        name=keypoint.name,
                        group=keypoint.group,
                        x=x_value,
                        y=y_value,
                        confidence=keypoint.confidence,
                    )
                )
            new_instances.append(
                PoseInstance(
                    instance_id=instance.instance_id,
                    score=instance.score,
                    keypoints=new_keypoints,
                )
            )
        smoothed_frames.append(PoseFrame(frame_index=frame.frame_index, instances=new_instances))

    return PoseSequence(
        format_version=sequence.format_version,
        source=sequence.source,
        fps=sequence.fps,
        frame_width=sequence.frame_width,
        frame_height=sequence.frame_height,
        frames=smoothed_frames,
    )
