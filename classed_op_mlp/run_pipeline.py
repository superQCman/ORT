from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytical_calibrated.contracts import DEFAULT_INPUT_CSV
from analytical_calibrated.build_analytical_features import build_full_feature_artifacts
from analytical_calibrated.evaluate_generalization import evaluate_generalization

try:
    from .build_classed_dataset import build_classed_dataset_artifacts
    from .compare_against_baseline import build_overall_comparison
    from .train_class_models import train_all_classes
except ImportError:
    from build_classed_dataset import build_classed_dataset_artifacts
    from compare_against_baseline import build_overall_comparison
    from train_class_models import train_all_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full classed single-op MLP pipeline on dataset_all_no_trace.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset_full.csv. Defaults to dataset_all_no_trace/dataset_full.csv.",
    )
    parser.add_argument(
        "--analytical-dir",
        default="",
        help="Optional analytical output dir. Defaults to artifacts/latest/analytical_calibrated.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent / "artifacts" / "latest" / "classed_op_mlp"),
        help="Output directory root for classed_op_mlp artifacts.",
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--hidden-layers", default="128,64")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--npu-device-id", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--disable-log-target", action="store_true")
    parser.add_argument("--disable-onnx-export", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    analytical_dir = Path(args.analytical_dir) if args.analytical_dir else output_dir.parent / "analytical_calibrated"

    analytical_dir.mkdir(parents=True, exist_ok=True)
    analytical_features = build_full_feature_artifacts(input_csv, analytical_dir, passes=args.passes)
    analytical_generalization = evaluate_generalization(
        input_csv,
        analytical_dir,
        schemes=["leave_one_case_out", "leave_one_combo_out"],
        passes=args.passes,
    )

    data_root = output_dir
    dataset_payload = build_classed_dataset_artifacts(
        input_data_dir=input_csv.parent,
        analytical_dir=analytical_dir,
        output_dir=data_root,
    )
    model_root = data_root / "models"
    training_payload = train_all_classes(
        data_root=data_root,
        model_root=model_root,
        hidden_layers=args.hidden_layers,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        seed=args.seed,
        train_device=args.train_device,
        npu_device_id=args.npu_device_id,
        early_stopping_patience=args.early_stopping_patience,
        log_target=not args.disable_log_target,
        export_onnx=not args.disable_onnx_export,
        onnx_opset=args.onnx_opset,
    )
    comparison_payload = build_overall_comparison(
        classed_root=model_root,
        baseline_dir=Path(dataset_payload["baseline_compare_dir"]),
        output_dir=model_root / "comparison",
    )

    payload = {
        "analytical_features": analytical_features,
        "analytical_generalization": analytical_generalization,
        "dataset": dataset_payload,
        "training": training_payload,
        "comparison": comparison_payload,
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
