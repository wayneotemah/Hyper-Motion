from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.config import ExperimentConfig, load_experiment_config
from src.paths import ensure_dir, load_dotenv, project_root, resolve_path, storage_roots


@dataclass
class ExperimentContext:
    project_root: Path
    config_path: Path
    config: ExperimentConfig
    experiment_dir: Path
    input_dir: Path
    derived_dir: Path
    output_dir: Path
    logs_dir: Path
    source_video: Path
    reference_image: Path
    pose_json: Path
    pose_video: Path
    derived_pose_json: Path
    derived_pose_video: Path
    derived_pose_h264: Path
    overlay_video: Path
    manifest_path: Path
    package_path: Path
    storage: Dict[str, Path]

    def to_dict(self) -> Dict[str, str]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "input_dir": str(self.input_dir),
            "derived_dir": str(self.derived_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "source_video": str(self.source_video),
            "reference_image": str(self.reference_image),
            "pose_json": str(self.pose_json),
            "pose_video": str(self.pose_video),
            "derived_pose_json": str(self.derived_pose_json),
            "derived_pose_video": str(self.derived_pose_video),
            "derived_pose_h264": str(self.derived_pose_h264),
            "overlay_video": str(self.overlay_video),
            "manifest_path": str(self.manifest_path),
            "package_path": str(self.package_path),
            "model_root": str(self.storage["model_root"]),
            "xpose_root": str(self.storage["xpose_root"]),
        }


def _experiment_dir(config: ExperimentConfig, config_path: Path) -> Path:
    configured = resolve_path(config.paths.experiment_dir, config_path.parent)
    if configured:
        return configured
    if config_path.parent.name == config.experiment_name:
        return config_path.parent
    return project_root() / "experiments" / config.experiment_name


def build_experiment_context(config_path: Path) -> ExperimentContext:
    load_dotenv()
    root = project_root()
    resolved_config_path = Path(config_path).resolve()
    config = load_experiment_config(resolved_config_path)
    experiment_dir = _experiment_dir(config, resolved_config_path)

    input_dir = ensure_dir(experiment_dir / "input")
    derived_dir = ensure_dir(experiment_dir / "derived")
    output_dir = ensure_dir(experiment_dir / "output")
    logs_dir = ensure_dir(experiment_dir / "logs")
    storage = storage_roots(config.storage.__dict__)

    reference_default = input_dir / "ref.jpg"
    pose_json_default = input_dir / "pose.json"
    pose_video_default = input_dir / "pose.mp4"

    return ExperimentContext(
        project_root=root,
        config_path=resolved_config_path,
        config=config,
        experiment_dir=experiment_dir,
        input_dir=input_dir,
        derived_dir=derived_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        source_video=resolve_path(config.paths.source_video, resolved_config_path.parent).resolve(),  # type: ignore[union-attr]
        reference_image=(resolve_path(config.paths.reference_image, resolved_config_path.parent) or reference_default).resolve(),
        pose_json=(resolve_path(config.paths.pose_json, resolved_config_path.parent) or pose_json_default).resolve(),
        pose_video=(resolve_path(config.paths.pose_video, resolved_config_path.parent) or pose_video_default).resolve(),
        derived_pose_json=(derived_dir / "pose.normalized.json").resolve(),
        derived_pose_video=(derived_dir / "pose_control.mp4").resolve(),
        derived_pose_h264=(derived_dir / "pose_control_h264.mp4").resolve(),
        overlay_video=(derived_dir / "pose_overlay.mp4").resolve(),
        manifest_path=(derived_dir / "experiment_manifest.json").resolve(),
        package_path=(storage["outputs_root"] / ("%s_package.zip" % config.experiment_name)).resolve(),
        storage=storage,
    )


def write_json(path: Path, payload: Dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
