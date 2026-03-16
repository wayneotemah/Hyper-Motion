#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.adapters.xpose_json_adapter import load_pose_sequence
from src.pose.render_pose_video import render_pose_video
from src.pose.smoothing import smooth_pose_sequence
from src.video_io import probe_video, reencode_h264


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a clean skeleton control video from pose JSON.")
    parser.add_argument("--input", required=True, help="Pose JSON path.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--source-video", help="Optional source video for debug overlay.")
    parser.add_argument("--overlay-output", help="Optional overlay output video path.")
    parser.add_argument("--mode", default="body_hands_face", choices=["body", "body_hands", "body_hands_face", "all"])
    parser.add_argument("--smooth", default="one_euro", choices=["none", "ema", "moving_average", "one_euro"])
    parser.add_argument("--frame-width", type=int, default=0)
    parser.add_argument("--frame-height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--group", help="Override X-Pose group when rendering legacy JSON.")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--h264", action="store_true", help="Re-encode the rendered video as H.264.")
    args = parser.parse_args()

    source_metadata = {}
    if args.source_video:
        source_metadata = probe_video(Path(args.source_video).resolve())

    sequence = load_pose_sequence(
        Path(args.input).resolve(),
        frame_width=args.frame_width or int(source_metadata.get("width", 0)),
        frame_height=args.frame_height or int(source_metadata.get("height", 0)),
        fps=args.fps or float(source_metadata.get("fps", 0.0)),
        group=args.group,
    )
    if args.smooth != "none":
        sequence = smooth_pose_sequence(sequence, method=args.smooth)

    output_path = Path(args.output).resolve()
    render_target = output_path if not args.h264 else output_path.with_name(output_path.stem + ".raw.mp4")

    render_summary = render_pose_video(
        sequence=sequence,
        output_path=render_target,
        overlay_output_path=Path(args.overlay_output).resolve() if args.overlay_output else None,
        source_video=Path(args.source_video).resolve() if args.source_video else None,
        mode=args.mode,
        width=args.frame_width or int(source_metadata.get("width", 0)) or sequence.frame_width or 768,
        height=args.frame_height or int(source_metadata.get("height", 0)) or sequence.frame_height or 512,
        fps=args.fps or float(source_metadata.get("fps", 0.0)) or sequence.fps or 24.0,
        confidence_threshold=args.confidence_threshold,
    )

    if args.h264:
        reencode_h264(render_target, output_path)
        render_target.unlink(missing_ok=True)

    render_summary["output"] = str(output_path)
    print(json.dumps(render_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
