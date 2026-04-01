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
    from .contracts import (
        DEFAULT_FEATURE_BRANCH,
        FEATURE_BRANCH_NO_ANALYTICAL,
        SUPPORTED_FEATURE_BRANCHES,
        resolve_output_dir,
    )
    from .build_classed_dataset import build_classed_dataset_artifacts
    from .compare_against_baseline import build_overall_comparison
    from .train_class_models import train_all_classes
except ImportError:
    from contracts import (
        DEFAULT_FEATURE_BRANCH,
        FEATURE_BRANCH_NO_ANALYTICAL,
        SUPPORTED_FEATURE_BRANCHES,
        resolve_output_dir,
    )
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
        help="Optional analytical output dir. Defaults to artifacts/latest/analytical_calibrated. Ignored for no_analytical branch.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory root for classed_op_mlp artifacts. Defaults to a branch-specific directory under artifacts/latest/classed_op_mlp.",
    )
    parser.add_argument(
        "--feature-branch",
        choices=list(SUPPORTED_FEATURE_BRANCHES),
        default=DEFAULT_FEATURE_BRANCH,
        help="Feature branch to run. no_analytical removes ana_calib_* to isolate pure classed MLP behavior.",
    )
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--skip-analytical-generalization",
        action="store_true",
        help="For with_analytical runs, skip the slow held-out analytical generalization step and only build ana_calib_* features.",
    )
    parser.add_argument(
        "--reuse-analytical-features",
        action="store_true",
        help="For with_analytical runs, reuse analytical_features_full.csv from --analytical-dir instead of rebuilding it.",
    )
    parser.add_argument(
        "--analytical-schemes",
        nargs="+",
        default=("leave_one_case_out", "leave_one_combo_out"),
        help="Analytical generalization schemes to run when analytical generalization is enabled.",
    )
    parser.add_argument("--hidden-layers", default="128,128,128,128,128")
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
    output_dir = resolve_output_dir(args.output_dir, args.feature_branch)
    analytical_dir = Path(args.analytical_dir) if args.analytical_dir else output_dir.parent / "analytical_calibrated"

    analytical_features: dict[str, object]
    analytical_generalization: dict[str, object]
    if args.feature_branch == FEATURE_BRANCH_NO_ANALYTICAL:
        analytical_features = {
            "skipped": True,
            "reason": "feature_branch=no_analytical excludes analytical proxy features from classed MLP inputs",
        }
        analytical_generalization = {
            "skipped": True,
            "reason": "feature_branch=no_analytical excludes analytical proxy features from classed MLP inputs",
        }
    else:
        analytical_dir.mkdir(parents=True, exist_ok=True)
        analytical_feature_csv = analytical_dir / "analytical_features_full.csv"
        if args.reuse_analytical_features:
            if not analytical_feature_csv.exists():
                raise FileNotFoundError(
                    f"--reuse-analytical-features was set, but {analytical_feature_csv} does not exist",
                )
            analytical_features = {
                "reused": True,
                "feature_csv": str(analytical_feature_csv),
                "output_dir": str(analytical_dir),
            }
        else:
            analytical_features = build_full_feature_artifacts(input_csv, analytical_dir, passes=args.passes)
        if args.skip_analytical_generalization:
            analytical_generalization = {
                "skipped": True,
                "reason": "skip_analytical_generalization=True",
                "requested_schemes": list(args.analytical_schemes),
            }
        else:
            analytical_generalization = evaluate_generalization(
                input_csv,
                analytical_dir,
                schemes=list(args.analytical_schemes),
                passes=args.passes,
            )

    data_root = output_dir
    dataset_payload = build_classed_dataset_artifacts(
        input_data_dir=input_csv.parent,
        analytical_dir=None if args.feature_branch == FEATURE_BRANCH_NO_ANALYTICAL else analytical_dir,
        output_dir=data_root,
        feature_branch=args.feature_branch,
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
        "feature_branch": args.feature_branch,
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
