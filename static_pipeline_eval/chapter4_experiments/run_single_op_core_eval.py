from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chapter4_experiments.chapter4_config import CHAPTER4_OUTPUT_ROOT, SINGLE_OP_ARTIFACT_ROOT  # noqa: E402
from chapter4_experiments.chapter4_shared import run_single_op_core  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 4 single-op core tables and figures.")
    parser.add_argument("--output-root", type=Path, default=CHAPTER4_OUTPUT_ROOT)
    parser.add_argument("--single-op-artifact-root", type=Path, default=SINGLE_OP_ARTIFACT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_single_op_core(
        args.output_root,
        single_op_artifact_root=args.single_op_artifact_root,
    )
    print(json.dumps(result.outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
