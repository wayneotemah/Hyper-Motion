from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.image_io import crop_signer_frame, load_image_size, save_bgr_image
from src.pipeline.experiment_builder import ExperimentContext, build_experiment_context, write_json
from src.pose.adapters.mediapipe_adapter import extract_pose_sequence_from_video
from src.pose.adapters.xpose_json_adapter import load_pose_sequence
from src.pose.render_pose_video import render_pose_video
from src.pose.schema import PoseSequence
from src.pose.smoothing import smooth_pose_sequence
from src.video_io import count_nonempty_frames, probe_video, read_frame, reencode_h264


def _pose_points(sequence: PoseSequence, frame_index: int) -> List[Tuple[float, float]]:
    for frame in sequence.sorted_frames():
        if frame.frame_index != frame_index:
            continue
        points = []
        for instance in frame.instances:
            for keypoint in instance.keypoints:
                if keypoint.group in {"body", "left_hand", "right_hand", "hand", "face"}:
                    points.append((keypoint.x, keypoint.y))
        return points
    return []


def validate_experiment_inputs(context: ExperimentContext, allow_pose_generation: bool = False) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    metadata: Dict[str, object] = {}

    if not context.source_video.exists():
        errors.append("Missing source video: %s" % context.source_video)
    else:
        metadata["source_video"] = probe_video(context.source_video)

    if context.reference_image.exists():
        width, height = load_image_size(context.reference_image)
        metadata["reference_image"] = {"width": width, "height": height}
    else:
        warnings.append("Reference image will be derived from the source video.")

    pose_json_exists = context.pose_json.exists()
    pose_video_exists = context.pose_video.exists()

    if pose_json_exists:
        source_meta = metadata.get("source_video", {})
        pose_sequence = load_pose_sequence(
            context.pose_json,
            frame_width=int(source_meta.get("width", 0)) if source_meta else None,
            frame_height=int(source_meta.get("height", 0)) if source_meta else None,
            fps=float(source_meta.get("fps", 0.0)) if source_meta else None,
        )
        metadata["pose_json"] = {
            "frame_count": len(pose_sequence.frames),
            "fps": pose_sequence.fps,
        }
    elif not allow_pose_generation:
        errors.append("Missing pose JSON: %s" % context.pose_json)
    elif context.config.pose.detector.lower() != "mediapipe":
        errors.append("Pose JSON is missing and no local fallback detector is configured.")
    else:
        warnings.append("Pose JSON will be generated with MediaPipe during local prep.")

    if pose_video_exists:
        pose_video_meta = probe_video(context.pose_video)
        if pose_video_meta["frame_count"] <= 0:
            errors.append("Pose video has zero decodable frames: %s" % context.pose_video)
        if count_nonempty_frames(context.pose_video, max_checks=30) == 0:
            errors.append("Pose video appears empty: %s" % context.pose_video)
        metadata["pose_video"] = pose_video_meta
    elif not pose_json_exists and not allow_pose_generation:
        errors.append("Missing pose video: %s" % context.pose_video)

    return {"errors": errors, "warnings": warnings, "metadata": metadata}


def _load_or_extract_pose_sequence(context: ExperimentContext, source_meta: Dict[str, object]) -> PoseSequence:
    if context.pose_json.exists():
        return load_pose_sequence(
            context.pose_json,
            frame_width=int(source_meta["width"]),
            frame_height=int(source_meta["height"]),
            fps=float(source_meta["fps"]),
        )

    sequence = extract_pose_sequence_from_video(context.source_video, mode=context.config.pose.mode)
    write_json(context.pose_json, sequence.to_dict())
    return sequence


def _ensure_reference_image(context: ExperimentContext, pose_sequence: Optional[PoseSequence]) -> None:
    if context.reference_image.exists():
        return

    frame_index = context.config.framing.reference_frame_index
    frame = read_frame(context.source_video, frame_index)
    pose_points = _pose_points(pose_sequence, frame_index) if pose_sequence else []
    cropped = crop_signer_frame(
        frame,
        points=pose_points,
        output_size=(context.config.framing.output_width, context.config.framing.output_height),
        padding_ratio=context.config.framing.padding_ratio,
    )
    save_bgr_image(cropped, context.reference_image)


def run_local_prep(config_path: Path) -> Dict[str, object]:
    context = build_experiment_context(config_path)
    validation = validate_experiment_inputs(context, allow_pose_generation=True)
    if validation["errors"]:
        raise ValueError("\n".join(validation["errors"]))

    source_meta = validation["metadata"]["source_video"]
    pose_sequence = _load_or_extract_pose_sequence(context, source_meta)
    if context.config.pose.smoothing.lower() not in {"none", "off"}:
        pose_sequence = smooth_pose_sequence(pose_sequence, method=context.config.pose.smoothing)

    write_json(context.derived_pose_json, pose_sequence.to_dict())
    _ensure_reference_image(context, pose_sequence)

    render_pose_video(
        sequence=pose_sequence,
        output_path=context.derived_pose_video,
        overlay_output_path=context.overlay_video,
        source_video=context.source_video,
        mode=context.config.pose.mode,
        width=context.config.framing.output_width,
        height=context.config.framing.output_height,
        fps=pose_sequence.fps or float(source_meta["fps"]),
        confidence_threshold=context.config.pose.min_confidence,
    )

    if context.config.pose.render_h264:
        reencode_h264(context.derived_pose_video, context.derived_pose_h264)

    manifest = {
        "experiment_name": context.config.experiment_name,
        "runtime_mode": context.config.runtime.mode,
        "status": "experiment_package_ready",
        "paths": context.to_dict(),
        "validation": validation,
        "pose": {
            "mode": context.config.pose.mode,
            "smoothing": context.config.pose.smoothing,
            "frame_count": len(pose_sequence.frames),
        },
        "notes": context.config.notes,
    }
    write_json(context.manifest_path, manifest)
    return manifest
