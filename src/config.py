from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExperimentPaths:
    source_video: str
    reference_image: Optional[str] = None
    pose_json: Optional[str] = None
    pose_video: Optional[str] = None
    experiment_dir: Optional[str] = None


@dataclass
class PoseConfig:
    source: str = "auto"
    detector: str = "mediapipe"
    mode: str = "body_hands_face"
    smoothing: str = "one_euro"
    render_h264: bool = True
    min_confidence: float = 0.2


@dataclass
class FramingConfig:
    target: str = "waist_up"
    crop_mode: str = "waist_up"
    output_width: int = 768
    output_height: int = 512
    padding_ratio: float = 0.2
    reference_frame_index: int = 0


@dataclass
class RuntimeConfig:
    mode: str = "local_prep_only"
    remote_backend: str = "ssh"
    remote_host: str = "gpu-box"
    remote_workdir: str = "~/hypermotion"


@dataclass
class StorageConfig:
    model_root: str = "assets/model_store"
    xpose_root: str = "assets/xpose"
    outputs_root: str = "outputs"
    logs_root: str = "logs"


@dataclass
class ExperimentConfig:
    experiment_name: str
    paths: ExperimentPaths
    pose: PoseConfig = field(default_factory=PoseConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mapping(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return data or {}


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "experiment_name" not in raw:
        raise ValueError("Config file must define 'experiment_name'.")
    if "paths" not in raw or "source_video" not in raw["paths"]:
        raise ValueError("Config file must define 'paths.source_video'.")

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        paths=ExperimentPaths(**_mapping(raw.get("paths"))),
        pose=PoseConfig(**_mapping(raw.get("pose"))),
        framing=FramingConfig(**_mapping(raw.get("framing"))),
        runtime=RuntimeConfig(**_mapping(raw.get("runtime"))),
        storage=StorageConfig(**_mapping(raw.get("storage"))),
        notes=_mapping(raw.get("notes")),
    )
