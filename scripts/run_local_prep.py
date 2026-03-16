#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.local_pipeline import run_local_prep


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate inputs and prepare a Mac-friendly local experiment package.")
    parser.add_argument("--config", required=True, help="Experiment config YAML path.")
    args = parser.parse_args()

    manifest = run_local_prep(Path(args.config).resolve())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
