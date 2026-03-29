from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_mlp  # noqa: E402
from analyze_op_type_metrics import (  # noqa: E402
    combo_op_type_total_duration_weighted_mape,
    per_op_type_metrics,
    summarize_frame,
)

from analytical_calibrated.contracts import BASELINE_COMPARE_DIR, OP_CLASS_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one MLP per op class and combine their predictions back into full splits.",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "artifacts" / "latest" / "classed_op_mlp"),
        help="Output root produced by build_classed_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit model output root. Defaults to <data-root>/models.",
    )
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


def load_dataset_summary(data_root: Path) -> dict[str, Any]:
    summary_path = data_root / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combine_predictions(
    data_root: Path,
    model_root: Path,
    class_summaries: dict[str, Any],
) -> dict[str, Any]:
    merged_dataset_path = data_root / "classed_dataset_full.csv"
    merged_dataset = pd.read_csv(merged_dataset_path, low_memory=False)
    merged_dataset["row_uid"] = merged_dataset["row_uid"].astype(str)
    combined_dir = model_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        prediction_parts: list[pd.DataFrame] = []
        for op_class in OP_CLASS_ORDER:
            prediction_csv = model_root / op_class / f"predictions_{split_name}.csv"
            if not prediction_csv.exists():
                raise FileNotFoundError(prediction_csv)
            frame = pd.read_csv(prediction_csv, low_memory=False)
            frame["row_uid"] = frame["row_uid"].astype(str)
            prediction_parts.append(
                frame[
                    [column for column in ["row_uid", "target_mode", "target_us", "pred_us", "abs_error_us", "ape"] if column in frame.columns]
                ].assign(predicted_by_op_class=op_class)
            )
        merged_predictions = pd.concat(prediction_parts, ignore_index=True)
        if merged_predictions["row_uid"].duplicated().any():
            duplicated = int(merged_predictions["row_uid"].duplicated().sum())
            raise RuntimeError(f"Found {duplicated} duplicate row_uid values in combined {split_name} predictions")

        expected = merged_dataset[merged_dataset["split"] == split_name].copy()
        expected = expected[
            [
                "row_uid",
                "_source_order",
                "split",
                "case_id",
                "combo",
                "op_type",
                "op_class",
                "ana_calib_family",
            ]
        ].copy()
        combined = expected.merge(
            merged_predictions,
            on="row_uid",
            how="left",
            validate="one_to_one",
        )
        missing = int(combined["pred_us"].isna().sum())
        if missing > 0:
            raise RuntimeError(f"{split_name} combined predictions are missing {missing} rows")

        combined = combined.sort_values("_source_order", kind="stable").reset_index(drop=True)
        combined_csv = combined_dir / f"combined_predictions_{split_name}.csv"
        combined.to_csv(combined_csv, index=False)

        overall = summarize_frame(combined)
        overall["combo_op_type_total_duration_weighted_mape"] = combo_op_type_total_duration_weighted_mape(combined)
        op_type_df = per_op_type_metrics(combined)
        op_type_csv = combined_dir / f"combined_op_type_metrics_{split_name}.csv"
        op_type_df.to_csv(op_type_csv, index=False)

        class_rows: list[dict[str, Any]] = []
        for op_class, group in combined.groupby("op_class", sort=True):
            row = {"op_class": op_class, **summarize_frame(group)}
            row["combo_op_type_total_duration_weighted_mape"] = combo_op_type_total_duration_weighted_mape(group)
            class_rows.append(row)
        class_df = pd.DataFrame(class_rows).sort_values("op_class").reset_index(drop=True)
        class_csv = combined_dir / f"combined_op_class_metrics_{split_name}.csv"
        class_df.to_csv(class_csv, index=False)

        split_summaries[split_name] = {
            "combined_predictions_csv": str(combined_csv),
            "op_type_metrics_csv": str(op_type_csv),
            "op_class_metrics_csv": str(class_csv),
            "overall_metrics": overall,
        }

    payload = {
        "data_root": str(data_root),
        "model_root": str(model_root),
        "baseline_compare_dir": str(BASELINE_COMPARE_DIR),
        "classes": class_summaries,
        "splits": split_summaries,
    }
    with (combined_dir / "combined_metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def train_all_classes(
    data_root: Path,
    model_root: Path,
    *,
    hidden_layers: str,
    batch_size: int,
    max_iter: int,
    alpha: float,
    learning_rate_init: float,
    seed: int,
    train_device: str,
    npu_device_id: int,
    early_stopping_patience: int,
    log_target: bool,
    export_onnx: bool,
    onnx_opset: int,
) -> dict[str, Any]:
    dataset_summary = load_dataset_summary(data_root)
    models: dict[str, Any] = {}
    model_root.mkdir(parents=True, exist_ok=True)

    for op_class in OP_CLASS_ORDER:
        data_dir = data_root / "datasets" / op_class
        output_dir = model_root / op_class
        summary = train_mlp.train_model(
            data_dir=data_dir,
            output_dir=output_dir,
            hidden_layers=train_mlp.parse_hidden_layers(hidden_layers),
            batch_size=batch_size,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            seed=seed,
            log_target=log_target,
            target_mode="direct_us",
            train_device=train_device,
            npu_device_id=npu_device_id,
            early_stopping_patience=early_stopping_patience,
            export_onnx=export_onnx,
            onnx_opset=onnx_opset,
        )
        models[op_class] = summary

    combined_payload = combine_predictions(data_root, model_root, models)
    payload = {
        "dataset_summary": dataset_summary,
        "models": models,
        "combined": combined_payload,
    }
    with (model_root / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_root = Path(args.output_dir) if args.output_dir else data_root / "models"
    payload = train_all_classes(
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
    print(json.dumps(payload["combined"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
