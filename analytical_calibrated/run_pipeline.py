from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .build_analytical_features import build_full_feature_artifacts
    from .contracts import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR
    from .evaluate_generalization import evaluate_generalization
except ImportError:
    from build_analytical_features import build_full_feature_artifacts
    from contracts import DEFAULT_INPUT_CSV, DEFAULT_OUTPUT_DIR
    from evaluate_generalization import evaluate_generalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full calibrated analytical pipeline: feature export + generalization evaluation.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset_full.csv. Defaults to dataset_all_no_trace/dataset_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for analytical_calibrated artifacts.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Coordinate-descent passes used for both full-data calibration and fold-level evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    feature_manifest = build_full_feature_artifacts(
        input_csv,
        output_dir,
        passes=args.passes,
    )
    generalization = evaluate_generalization(
        input_csv,
        output_dir,
        schemes=["leave_one_case_out", "leave_one_combo_out"],
        passes=args.passes,
    )

    payload = {
        "feature_manifest": feature_manifest,
        "generalization": generalization,
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
