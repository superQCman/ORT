from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_mlp  # noqa: E402
from classed_op_mlp.build_classed_dataset import build_classed_dataset_artifacts  # noqa: E402


DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT / "artifacts" / "latest" / "classed_op_mlp_test_2_analytical_5_200_iter"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "inter_threads_eval"
DEFAULT_MODEL_GROUPS = ("gather", "mixed_balanced")
DEFAULT_COMPARE_SPLITS = ("train", "val", "test")
LOWER_IS_BETTER_METRICS = ("mae_us", "rmse_us", "mape", "median_ape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild classed_op_mlp datasets with the current inter_threads feature logic, "
            "retrain selected groups with the baseline hyperparameters, and compare the new "
            "results against an existing baseline experiment."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        default=str(DEFAULT_BASELINE_ROOT),
        help="Existing classed_op_mlp experiment root used as the baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory for rebuilt datasets, retrained models, and comparisons. "
            "Defaults to artifacts/latest/inter_threads_eval/<baseline-name>."
        ),
    )
    parser.add_argument(
        "--model-group",
        action="append",
        default=[],
        help=(
            "Model group to evaluate. Repeat this flag for multiple groups. "
            "Defaults to gather + mixed_balanced."
        ),
    )
    parser.add_argument(
        "--reuse-rebuilt-dataset",
        action="store_true",
        help="Reuse output-dir/datasets_rebuilt if it already exists instead of rebuilding it.",
    )
    parser.add_argument(
        "--train-device",
        default="auto",
        help="Training device passed through to train_mlp.train_model().",
    )
    parser.add_argument("--npu-device-id", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--disable-onnx-export", action="store_true")
    parser.add_argument("--onnx-opset", type=int, default=17)
    parser.add_argument(
        "--max-iter-override",
        type=int,
        default=0,
        help="Optional override for max_iter. Mainly useful for smoke tests; 0 keeps the baseline value.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_output_dir(args: argparse.Namespace, baseline_root: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return DEFAULT_OUTPUT_ROOT / baseline_root.name


def resolve_model_groups(args: argparse.Namespace, baseline_summary: dict[str, Any]) -> list[str]:
    requested = [str(item).strip() for item in args.model_group if str(item).strip()]
    model_group_order = list(baseline_summary.get("model_group_order", []))
    if not requested:
        requested = list(DEFAULT_MODEL_GROUPS)
    available = set(model_group_order)
    for model_group in requested:
        if model_group not in available:
            raise KeyError(
                f"Unknown model_group={model_group!r}; available groups: {model_group_order}"
            )
    return requested


def rebuild_datasets(
    baseline_summary: dict[str, Any],
    rebuilt_root: Path,
    *,
    reuse_rebuilt_dataset: bool,
) -> dict[str, Any]:
    dataset_summary_path = rebuilt_root / "dataset_summary.json"
    if reuse_rebuilt_dataset and dataset_summary_path.exists():
        return load_json(dataset_summary_path)

    input_data_dir = Path(str(baseline_summary["input_data_dir"]))
    feature_branch = str(baseline_summary["feature_branch"])
    analytical_enabled = bool(baseline_summary.get("analytical_enabled", feature_branch != "no_analytical"))
    analytical_dir = (
        Path(str(baseline_summary["analytical_dir"]))
        if analytical_enabled and str(baseline_summary.get("analytical_dir", "")).strip()
        else None
    )
    return build_classed_dataset_artifacts(
        input_data_dir=input_data_dir,
        analytical_dir=analytical_dir,
        output_dir=rebuilt_root,
        feature_branch=feature_branch,
    )


def load_baseline_training_config(baseline_metrics: dict[str, Any], *, args: argparse.Namespace) -> dict[str, Any]:
    max_iter = int(args.max_iter_override) if int(args.max_iter_override) > 0 else int(baseline_metrics["max_iter"])
    return {
        "hidden_layers": tuple(int(value) for value in baseline_metrics["hidden_layers"]),
        "batch_size": int(baseline_metrics["batch_size"]),
        "max_iter": max_iter,
        "alpha": float(baseline_metrics["alpha"]),
        "learning_rate_init": float(baseline_metrics["learning_rate_init"]),
        "seed": int(baseline_metrics["seed"]),
        "log_target": bool(
            baseline_metrics.get("log_target_requested", baseline_metrics.get("log_target", True))
        ),
        "target_mode": str(baseline_metrics.get("target_mode", "direct_us")),
        "train_device": str(args.train_device),
        "npu_device_id": int(args.npu_device_id),
        "early_stopping_patience": int(args.early_stopping_patience),
        "export_onnx": not args.disable_onnx_export,
        "onnx_opset": int(args.onnx_opset),
    }


def compare_metric(baseline_value: float, new_value: float, metric_name: str) -> dict[str, float]:
    row = {
        "baseline": float(baseline_value),
        "new": float(new_value),
    }
    if metric_name in LOWER_IS_BETTER_METRICS:
        delta = float(new_value) - float(baseline_value)
        improvement_pct = (float(baseline_value) - float(new_value)) / float(baseline_value) * 100.0 if float(baseline_value) != 0.0 else 0.0
        row["delta"] = delta
        row["improvement_pct"] = improvement_pct
    else:
        delta = float(new_value) - float(baseline_value)
        row["delta"] = delta
        row["improvement_pct"] = delta * 100.0
    return row


def compare_predictions(baseline_prediction_csv: Path, new_prediction_csv: Path) -> dict[str, float]:
    baseline_df = pd.read_csv(
        baseline_prediction_csv,
        usecols=["row_uid", "abs_error_us", "ape"],
        low_memory=False,
    )
    new_df = pd.read_csv(
        new_prediction_csv,
        usecols=["row_uid", "abs_error_us", "ape"],
        low_memory=False,
    )
    merged = baseline_df.merge(new_df, on="row_uid", suffixes=("_baseline", "_new"))
    if merged.empty:
        return {
            "rows": 0,
            "better_abs_error_frac": 0.0,
            "better_ape_frac": 0.0,
        }
    return {
        "rows": int(len(merged)),
        "better_abs_error_frac": float(
            (merged["abs_error_us_new"] < merged["abs_error_us_baseline"]).mean()
        ),
        "better_ape_frac": float((merged["ape_new"] < merged["ape_baseline"]).mean()),
    }


def compare_group(
    *,
    baseline_root: Path,
    rebuilt_root: Path,
    output_dir: Path,
    model_group: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_model_dir = baseline_root / "models" / model_group
    baseline_metrics_path = baseline_model_dir / "metrics.json"
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(baseline_metrics_path)
    baseline_metrics = load_json(baseline_metrics_path)

    data_dir = rebuilt_root / "datasets" / model_group
    model_dir = output_dir / "models" / model_group
    model_dir.mkdir(parents=True, exist_ok=True)

    train_config = load_baseline_training_config(baseline_metrics, args=args)
    new_metrics = train_mlp.train_model(
        data_dir=data_dir,
        output_dir=model_dir,
        **train_config,
    )

    split_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split_name in DEFAULT_COMPARE_SPLITS:
        baseline_split = baseline_metrics["metrics"][split_name]
        new_split = new_metrics["metrics"][split_name]
        metric_rows = {
            metric_name: compare_metric(
                baseline_value=float(baseline_split[metric_name]),
                new_value=float(new_split[metric_name]),
                metric_name=metric_name,
            )
            for metric_name in ["mae_us", "rmse_us", "r2", "mape", "median_ape"]
        }
        prediction_summary = compare_predictions(
            baseline_model_dir / f"predictions_{split_name}.csv",
            model_dir / f"predictions_{split_name}.csv",
        )
        split_summary[split_name] = {
            "metrics": metric_rows,
            "prediction_summary": prediction_summary,
        }
        for metric_name, row in metric_rows.items():
            split_rows.append(
                {
                    "model_group": model_group,
                    "split": split_name,
                    "metric": metric_name,
                    **row,
                }
            )

    metrics_csv = output_dir / "comparison" / f"{model_group}_metric_comparison.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(metrics_csv, index=False)

    summary = {
        "model_group": model_group,
        "baseline_model_dir": str(baseline_model_dir),
        "new_model_dir": str(model_dir),
        "baseline_numeric_features": list(baseline_metrics["numeric_features"]),
        "new_numeric_features": list(new_metrics["numeric_features"]),
        "feature_count_baseline": int(baseline_metrics["feature_count"]),
        "feature_count_new": int(new_metrics["feature_count"]),
        "input_dim_after_encoding_baseline": int(baseline_metrics["input_dim_after_encoding"]),
        "input_dim_after_encoding_new": int(new_metrics["input_dim_after_encoding"]),
        "train_config": train_config,
        "splits": split_summary,
        "comparison_csv": str(metrics_csv),
    }
    with (output_dir / "comparison" / f"{model_group}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    baseline_summary_path = baseline_root / "dataset_summary.json"
    if not baseline_summary_path.exists():
        raise FileNotFoundError(baseline_summary_path)
    baseline_summary = load_json(baseline_summary_path)
    model_groups = resolve_model_groups(args, baseline_summary)
    output_dir = resolve_output_dir(args, baseline_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rebuilt_root = output_dir / "datasets_rebuilt"
    rebuilt_summary = rebuild_datasets(
        baseline_summary=baseline_summary,
        rebuilt_root=rebuilt_root,
        reuse_rebuilt_dataset=bool(args.reuse_rebuilt_dataset),
    )

    group_summaries = []
    for model_group in model_groups:
        group_summaries.append(
            compare_group(
                baseline_root=baseline_root,
                rebuilt_root=rebuilt_root,
                output_dir=output_dir,
                model_group=model_group,
                args=args,
            )
        )

    suite_summary = {
        "baseline_root": str(baseline_root),
        "output_dir": str(output_dir),
        "rebuilt_dataset_root": str(rebuilt_root),
        "rebuilt_dataset_summary": str(rebuilt_root / "dataset_summary.json"),
        "model_groups": model_groups,
        "group_summaries": group_summaries,
    }
    with (output_dir / "suite_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
