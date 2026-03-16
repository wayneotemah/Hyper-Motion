from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from src.pipeline.experiment_builder import ExperimentContext, build_experiment_context, write_json


def build_remote_handoff(config_path: Path, package: bool = False) -> Dict[str, object]:
    context = build_experiment_context(config_path)
    pose_json = context.derived_pose_json if context.derived_pose_json.exists() else context.pose_json
    pose_video = context.derived_pose_h264 if context.derived_pose_h264.exists() else (
        context.derived_pose_video if context.derived_pose_video.exists() else context.pose_video
    )

    required_files: List[str] = [
        str(context.config_path),
        str(context.source_video),
        str(context.reference_image),
        str(pose_json),
        str(pose_video),
    ]

    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'export HYPERMOTION_MODEL_ROOT="${HYPERMOTION_MODEL_ROOT:-%s}"' % context.storage["model_root"],
        'EXPERIMENT_DIR="%s"' % context.experiment_dir,
        'echo "Copy the experiment directory to your CUDA box and update scripts/inference.py with:"',
        'echo "  model_name=${HYPERMOTION_MODEL_ROOT}/HyperMotion"',
        'echo "  control_video=${EXPERIMENT_DIR}/derived/pose_control_h264.mp4"',
        'echo "  ref_image=${EXPERIMENT_DIR}/input/ref.jpg"',
        'echo "  save_path=${EXPERIMENT_DIR}/output"',
        'echo "Then run: python scripts/inference.py"',
    ]
    shell_script_path = context.derived_dir / "remote_run.sh"
    shell_script_path.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    shell_script_path.chmod(0o755)

    archive_path = None
    if package:
        archive_base = str(context.package_path.with_suffix(""))
        context.package_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path = shutil.make_archive(archive_base, "zip", root_dir=context.experiment_dir)

    handoff = {
        "experiment_name": context.config.experiment_name,
        "required_files": required_files,
        "remote_command_template": "python scripts/inference.py",
        "remote_shell_script": str(shell_script_path),
        "archive_path": archive_path,
    }
    write_json(context.derived_dir / "remote_handoff.json", handoff)
    return handoff
