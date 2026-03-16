#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_io import crop_signer_frame, save_bgr_image
from src.pose.adapters.xpose_json_adapter import load_pose_sequence
from src.video_io import probe_video, read_frame


def _pose_points(sequence, frame_index: int):
    for frame in sequence.sorted_frames():
        if frame.frame_index != frame_index:
            continue
        return [(keypoint.x, keypoint.y) for instance in frame.instances for keypoint in instance.keypoints]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a signer-friendly reference frame.")
    parser.add_argument("--input", required=True, help="Source video path.")
    parser.add_argument("--output", required=True, help="Output JPG path.")
    parser.add_argument("--frame-number", type=int, default=0, help="Frame index to extract.")
    parser.add_argument("--crop-mode", default="waist_up", choices=["waist_up", "center"], help="Crop strategy.")
    parser.add_argument("--output-width", type=int, default=768)
    parser.add_argument("--output-height", type=int, default=512)
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    parser.add_argument("--pose-json", help="Optional normalized or X-Pose JSON for crop guidance.")
    args = parser.parse_args()

    source_video = Path(args.input).resolve()
    frame = read_frame(source_video, args.frame_number)
    points = []
    if args.pose_json:
        metadata = probe_video(source_video)
        sequence = load_pose_sequence(
            Path(args.pose_json).resolve(),
            frame_width=metadata["width"],
            frame_height=metadata["height"],
            fps=metadata["fps"],
        )
        points = _pose_points(sequence, args.frame_number)

    cropped = crop_signer_frame(
        frame,
        points=points if args.crop_mode == "waist_up" else [],
        output_size=(args.output_width, args.output_height),
        padding_ratio=args.padding_ratio,
    )
    save_bgr_image(cropped, Path(args.output).resolve())
    print(json.dumps({"output": str(Path(args.output).resolve()), "frame_number": args.frame_number}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
