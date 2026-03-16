#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.remote_pipeline import build_remote_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a zip package for remote CUDA execution.")
    parser.add_argument("--config", required=True, help="Experiment config YAML path.")
    args = parser.parse_args()

    result = build_remote_handoff(Path(args.config).resolve(), package=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
