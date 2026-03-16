#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.experiment_builder import build_experiment_context
from src.pipeline.local_pipeline import validate_experiment_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local HyperMotion experiment package.")
    parser.add_argument("--config", required=True, help="Experiment config YAML path.")
    parser.add_argument("--allow-pose-generation", action="store_true", help="Allow MediaPipe fallback if pose JSON is missing.")
    args = parser.parse_args()

    context = build_experiment_context(Path(args.config).resolve())
    result = validate_experiment_inputs(context, allow_pose_generation=args.allow_pose_generation)
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
