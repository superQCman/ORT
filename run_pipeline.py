from __future__ import annotations

import argparse
from pathlib import Path

from dataset_builder import build_dataset, resolve_selected_case_ids
from train_mlp import parse_hidden_layers, train_model


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-case single-op tables and train an MLP in one command.",
    )
    parser.add_argument(
        "--output-root",
        default=str(SCRIPT_DIR / "artifacts" / "latest"),
        help="Root directory for dataset/ and model/ outputs.",
    )
    parser.add_argument(
        "--case-pattern",
        default="features_extensible_case_*",
        help="Glob-style pattern applied to feature case directories.",
    )
    parser.add_argument(
        "--selected-cases",
        nargs="*",
        default=None,
        help="Optional manual case list; supports space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--selected-cases-file",
        default="",
        help="Optional text/JSON file containing the manually selected cases.",
    )
    parser.add_argument("--max-files-per-case", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-dialect",
        choices=["auto", "trace", "no_trace"],
        default="auto",
        help="Input dataset dialect: auto, trace, or no_trace.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-column", default="sample_group")
    parser.add_argument(
        "--drop-first-profile-batch",
        default="true",
        help="Whether to exclude the earliest profile batch before computing labels and stability metrics.",
    )
    parser.add_argument(
        "--profile-instability-metric",
        default="last2_range_ratio",
        choices=["last2_range_ratio", "last2_cv"],
    )
    parser.add_argument("--profile-instability-threshold", type=float, default=0.20)
    parser.add_argument("--disable-profile-stability-filter", action="store_true")
    parser.add_argument("--hidden-layers", default="128,64")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--npu-device-id", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--disable-log-target", action="store_true")
    parser.add_argument("--disable-onnx-export", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_case_ids = resolve_selected_case_ids(args.selected_cases, args.selected_cases_file)
    output_root = Path(args.output_root)
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "model"

    build_dataset(
        output_dir=dataset_dir,
        case_pattern=args.case_pattern,
        selected_case_ids=selected_case_ids,
        max_files_per_case=args.max_files_per_case,
        group_column=args.group_column,
        seed=args.seed,
        ratios={
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        feature_dialect=args.feature_dialect,
        drop_first_profile_batch=args.drop_first_profile_batch.strip().lower() not in {"0", "false", "no", "off"},
        profile_instability_metric=args.profile_instability_metric,
        profile_instability_threshold=args.profile_instability_threshold,
        disable_profile_stability_filter=args.disable_profile_stability_filter,
    )
    summary = train_model(
        data_dir=dataset_dir,
        output_dir=model_dir,
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        seed=args.seed,
        log_target=not args.disable_log_target,
        train_device=args.train_device,
        npu_device_id=args.npu_device_id,
        early_stopping_patience=args.early_stopping_patience,
        export_onnx=not args.disable_onnx_export,
        onnx_opset=args.onnx_opset,
    )
    print(f"dataset_dir={dataset_dir}")
    print(f"model_dir={model_dir}")
    print(f"onnx_status={summary['onnx_export']['status']}")
    print(f"test_mae_us={summary['metrics']['test']['mae_us']:.6f}")
    print(f"test_rmse_us={summary['metrics']['test']['rmse_us']:.6f}")
    print(f"test_r2={summary['metrics']['test']['r2']:.6f}")


if __name__ == "__main__":
    main()
