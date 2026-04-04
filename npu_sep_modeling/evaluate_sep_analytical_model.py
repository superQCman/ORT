from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from fit_sep_analytical_model import (
    LABEL_COLUMN,
    LANE_COLUMN,
    OP_COLUMN,
    annotate_baselines,
    apply_calibration,
    load_optional_hardware_profile,
    load_split_frames,
)
from npu_sep_common import compute_regression_metrics, dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the calibrated analytical model against the baseline roofline.")
    parser.add_argument("--data-dir", required=True, help="Directory that contains train.csv/val.csv/test.csv.")
    parser.add_argument("--hardware-profile", default="", help="Optional hardware_profile_910b3.json path.")
    parser.add_argument("--calibration", required=True, help="Path to calibration.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for the comparison report.")
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--op-column", default=OP_COLUMN)
    parser.add_argument("--lane-column", default=LANE_COLUMN)
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in calibration file: {path}")
    return payload


def metrics_delta(base: dict[str, float], calibrated: dict[str, float]) -> dict[str, float]:
    return {
        "mae_delta": float(calibrated["mae"] - base["mae"]),
        "mape_delta": float(calibrated["mape"] - base["mape"]),
        "rmse_delta": float(calibrated["rmse"] - base["rmse"]),
    }


def compare_frame(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    baseline = compute_regression_metrics(df[label_col], df["baseline_pred_us"])
    calibrated = compute_regression_metrics(df[label_col], df["calibrated_pred_us"])
    lane_calibrated = compute_regression_metrics(df[label_col], df["lane_calibrated_pred_us"])
    global_calibrated = compute_regression_metrics(df[label_col], df["global_calibrated_pred_us"])
    return {
        "count": int(len(df)),
        "baseline": baseline,
        "calibrated": calibrated,
        "lane_calibrated": lane_calibrated,
        "global_calibrated": global_calibrated,
        "delta": metrics_delta(baseline, calibrated),
    }


def compare_groups(df: pd.DataFrame, label_col: str, group_col: str, pred_col: str) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for group, group_df in df.groupby(group_col, dropna=False):
        payload[str(group)] = {
            **compute_regression_metrics(group_df[label_col], group_df[pred_col]),
            "count": int(len(group_df)),
        }
    return payload


def compare_groups_with_baseline(df: pd.DataFrame, label_col: str, group_col: str, calibrated_pred_col: str) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for group, group_df in df.groupby(group_col, dropna=False):
        baseline = compute_regression_metrics(group_df[label_col], group_df["baseline_pred_us"])
        calibrated = compute_regression_metrics(group_df[label_col], group_df[calibrated_pred_col])
        payload[str(group)] = {
            "count": int(len(group_df)),
            "baseline": baseline,
            "calibrated": calibrated,
            "delta": metrics_delta(baseline, calibrated),
        }
    return payload


def format_float(value: float, digits: int = 3) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_report(calibration: dict[str, Any], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# NPU Analytical Model Evaluation")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Hardware profile: `{calibration.get('hardware_profile_path') or 'fallback defaults'}`")
    lines.append(f"- Label column: `{calibration.get('label_column')}`")
    lines.append(f"- Operator column: `{calibration.get('op_column')}`")
    lines.append(f"- Lane column: `{calibration.get('lane_column')}`")
    lines.append("")
    lines.append("## Baseline Defaults")
    defaults = calibration.get("baseline_defaults", {})
    lines.append(
        "- "
        + ", ".join(
            f"{key}={value}" for key, value in defaults.items()
        )
    )
    lines.append("")
    lines.append("## Overall Comparison")
    headers = [
        "split",
        "model",
        "MAE",
        "MAPE%",
        "RMSE",
    ]
    rows: list[list[str]] = []
    for split in ("train", "val", "test"):
        data = summary["overall"][split]
        for model_name in ("baseline", "calibrated", "lane_calibrated", "global_calibrated"):
            metrics = data[model_name]
            rows.append(
                [
                    split,
                    model_name,
                    format_float(metrics["mae"]),
                    format_float(metrics["mape"]),
                    format_float(metrics["rmse"]),
                ]
            )
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("## Baseline To Calibrated Delta")
    delta_headers = ["split", "MAE delta", "MAPE delta", "RMSE delta"]
    delta_rows: list[list[str]] = []
    for split in ("train", "val", "test"):
        delta = summary["overall"][split]["delta"]
        delta_rows.append(
            [
                split,
                format_float(delta["mae_delta"]),
                format_float(delta["mape_delta"]),
                format_float(delta["rmse_delta"]),
            ]
        )
    lines.append(markdown_table(delta_headers, delta_rows))
    lines.append("")
    lines.append("## By Op Name")
    for split in ("val", "test"):
        lines.append(f"### {split}")
        group_map = summary["by_op_name"][split]
        headers = [
            "op_name",
            "count",
            "baseline_MAE",
            "calibrated_MAE",
            "delta_MAE",
            "baseline_MAPE%",
            "calibrated_MAPE%",
            "delta_MAPE%",
            "baseline_RMSE",
            "calibrated_RMSE",
            "delta_RMSE",
        ]
        rows = []
        for op_name in sorted(group_map):
            metrics = group_map[op_name]
            rows.append(
                [
                    op_name,
                    str(metrics["count"]),
                    format_float(metrics["baseline"]["mae"]),
                    format_float(metrics["calibrated"]["mae"]),
                    format_float(metrics["delta"]["mae_delta"]),
                    format_float(metrics["baseline"]["mape"]),
                    format_float(metrics["calibrated"]["mape"]),
                    format_float(metrics["delta"]["mape_delta"]),
                    format_float(metrics["baseline"]["rmse"]),
                    format_float(metrics["calibrated"]["rmse"]),
                    format_float(metrics["delta"]["rmse_delta"]),
                ]
            )
        lines.append(markdown_table(headers, rows))
        lines.append("")
    lines.append("## By Lane")
    for split in ("val", "test"):
        lines.append(f"### {split}")
        group_map = summary["by_lane"][split]
        headers = [
            "lane",
            "count",
            "baseline_MAE",
            "calibrated_MAE",
            "delta_MAE",
            "baseline_MAPE%",
            "calibrated_MAPE%",
            "delta_MAPE%",
            "baseline_RMSE",
            "calibrated_RMSE",
            "delta_RMSE",
        ]
        rows = []
        for lane in sorted(group_map):
            metrics = group_map[lane]
            rows.append(
                [
                    lane,
                    str(metrics["count"]),
                    format_float(metrics["baseline"]["mae"]),
                    format_float(metrics["calibrated"]["mae"]),
                    format_float(metrics["delta"]["mae_delta"]),
                    format_float(metrics["baseline"]["mape"]),
                    format_float(metrics["calibrated"]["mape"]),
                    format_float(metrics["delta"]["mape_delta"]),
                    format_float(metrics["baseline"]["rmse"]),
                    format_float(metrics["calibrated"]["rmse"]),
                    format_float(metrics["delta"]["rmse_delta"]),
                ]
            )
        lines.append(markdown_table(headers, rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def evaluate_model(
    data_dir: Path,
    hardware_profile_path: str,
    calibration_path: Path,
    label_col: str,
    op_col: str,
    lane_col: str,
) -> dict[str, Any]:
    frames = load_split_frames(data_dir)
    hardware_profile = load_optional_hardware_profile(hardware_profile_path)
    calibration = load_calibration(calibration_path)
    op_params = calibration.get("by_op_name", {})
    lane_params = calibration.get("by_lane", {})
    global_params = calibration.get("global", {"scale": 1.0, "bias_us": 0.0})

    annotated_frames = {split: annotate_baselines(frame, hardware_profile) for split, frame in frames.items()}
    calibrated_frames = {
        split: apply_calibration(frame, op_params, lane_params, global_params, op_col, lane_col)
        for split, frame in annotated_frames.items()
    }

    summary: dict[str, Any] = {"overall": {}, "by_op_name": {}, "by_lane": {}}
    for split, frame in calibrated_frames.items():
        summary["overall"][split] = compare_frame(frame, label_col)
        summary["by_op_name"][split] = compare_groups_with_baseline(frame, label_col, op_col, "calibrated_pred_us")
        summary["by_lane"][split] = compare_groups_with_baseline(frame, label_col, lane_col, "lane_calibrated_pred_us")
    return {
        "calibration": calibration,
        "summary": summary,
        "calibrated_frames": calibrated_frames,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    calibration_path = Path(args.calibration)
    result = evaluate_model(Path(args.data_dir), args.hardware_profile, calibration_path, args.label_column, args.op_column, args.lane_column)
    report_text = render_report(result["calibration"], result["summary"])
    report_path = output_dir / "comparison_report.md"
    summary_path = output_dir / "comparison_summary.json"
    report_path.write_text(report_text, encoding="utf-8")
    dump_json(summary_path, result["summary"])
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
