from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chapter4_experiments.chapter4_config import CHAPTER4_OUTPUT_ROOT  # noqa: E402
from chapter4_experiments.chapter4_shared import build_chapter4_draft  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the auto-generated Chapter 4 draft.")
    parser.add_argument("--output-root", type=Path, default=CHAPTER4_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_chapter4_draft(args.output_root)
    print(json.dumps(result.outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
