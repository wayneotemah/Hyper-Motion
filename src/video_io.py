from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def ffmpeg_available(binary: str = "ffmpeg") -> bool:
    return shutil.which(binary) is not None


def probe_video(video_path: Path) -> Dict[str, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open video: %s" % video_path)

    metadata = {
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    metadata["duration_seconds"] = round(
        metadata["frame_count"] / metadata["fps"], 3
    ) if metadata["fps"] else 0.0
    capture.release()
    return metadata


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open video: %s" % video_path)

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Could not decode frame %s from %s" % (frame_index, video_path))
    return frame


def count_nonempty_frames(video_path: Path, max_checks: Optional[int] = None) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open video: %s" % video_path)

    nonempty = 0
    checked = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        checked += 1
        if frame is not None and int(frame.mean()) > 0:
            nonempty += 1
        if max_checks and checked >= max_checks:
            break
    capture.release()
    return nonempty


def reencode_h264(input_path: Path, output_path: Path, ffmpeg_binary: str = "ffmpeg") -> Path:
    if not ffmpeg_available(ffmpeg_binary):
        raise RuntimeError("ffmpeg is not available on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path
