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
    apply_physical_model,
    comparison_metrics_for_frame,
    component_means_for_frame,
    group_physical_metrics,
    load_optional_hardware_profile,
    load_split_frames,
)
from npu_sep_common import compute_regression_metrics, dump_json, ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the calibrated analytical model against the baseline roofline.")
    parser.add_argument("--data-dir", required=True, help="Directory that contains train.csv/val.csv/test.csv.")
    parser.add_argument("--hardware-profile", default="", help="Optional hardware_profile_910b3.json path.")
    parser.add_argument("--calibration", required=True, help="Path to the fitted calibration.json produced by fit_sep_analytical_model.py.")
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
    lines.append(f"- Model variant: `{calibration.get('model_variant') or 'physical'}`")
    lines.append(f"- Cube memory mode: `{calibration.get('cube_memory_mode') or 'merged_into_launch_runtime'}`")
    lines.append(f"- Physical prediction column: `{calibration.get('physical_prediction_column') or 'physical_pred_us'}`")
    if calibration.get("calibration_fit_fraction") is not None:
        lines.append(f"- Calibration fit fraction: `{calibration.get('calibration_fit_fraction')}`")
        lines.append(f"- Calibration fit seed: `{calibration.get('calibration_seed')}`")
    queue_policy = calibration.get("queue_proxy_policy") or {}
    if queue_policy:
        lines.append(f"- Queue proxy policy: `{queue_policy.get('definition')}` in `{queue_policy.get('units')}`")
    lines.append("")
    subset = calibration.get("calibration_subset") or {}
    if subset:
        lines.append("## Calibration Subset")
        lines.append(
            "- "
            + ", ".join(
                [
                    f"fit_rows={subset.get('fit_rows')}",
                    f"heldout_rows={subset.get('heldout_rows')}",
                    f"fit_group_col={calibration.get('calibration_fit_group_column')}",
                ]
            )
        )
        lines.append("")
    lines.append("## Hardware Inputs")
    hardware = calibration.get("hardware_profile_effective") or {}
    hw_headers = ["key", "value"]
    hw_rows = []
    for key in [
        "device_name",
        "ai_core_count",
        "cube_count",
        "vector_count",
        "cube_peak_eff_gflops",
        "vector_peak_eff_gflops",
        "memory_bw_gbps",
        "h2d_bw_gbps",
        "d2h_bw_gbps",
    ]:
        hw_rows.append([key, format_float(float(hardware[key])) if isinstance(hardware.get(key), (int, float)) else str(hardware.get(key))])
    lines.append(markdown_table(hw_headers, hw_rows))
    lines.append("")
    lines.append("## Fitted Parameters")
    params = calibration.get("parameters") or {}
    merged_terms = set(calibration.get("merged_terms") or [])
    cube_bw = params.get("cube_memory_bw_gbps")
    vector_bw = params.get("vector_memory_bw_gbps")
    h2d_bw = params.get("transfer_h2d_bw_gbps")
    d2h_bw = params.get("transfer_d2h_bw_gbps")
    lines.append(
        "- "
        + ", ".join(
            [
                f"queueing_scale={format_float(float(params.get('queueing_scale', float('nan'))))}",
                f"cube_memory_bw_gbps={cube_bw if cube_bw is not None else 'merged'}{' (merged)' if 'cube_memory_us' in merged_terms else ''}",
                f"vector_memory_bw_gbps={vector_bw}{' (merged)' if 'vector_memory_us' in merged_terms else ''}",
                f"h2d_bw_gbps={h2d_bw}{' (merged)' if 'transfer_h2d_us' in merged_terms else ''}",
                f"d2h_bw_gbps={d2h_bw}{' (merged)' if 'transfer_d2h_us' in merged_terms else ''}",
            ]
        )
    )
    if merged_terms:
        lines.append(f"- Merged terms: `{', '.join(sorted(merged_terms))}`")
    lines.append("")
    launch_map = params.get("launch_runtime_us_by_op_name") or {}
    if launch_map:
        lines.append("### Launch Runtime by Op")
        headers = ["op_name", "launch_runtime_us"]
        rows = [[op_name, format_float(float(value))] for op_name, value in sorted(launch_map.items())]
        lines.append(markdown_table(headers, rows))
        lines.append("")
    lines.append("## Overall Comparison")
    headers = ["split", "model", "MAE", "MAPE%", "RMSE"]
    rows = []
    for split in ("train_fit", "train_heldout", "train", "val", "test"):
        if split not in summary["overall"]:
            continue
        data = summary["overall"][split]
        for model_name in ("baseline", "physical"):
            metrics = data[model_name]
            rows.append([split, model_name, format_float(metrics["mae"]), format_float(metrics["mape"]), format_float(metrics["rmse"])])
    lines.append(markdown_table(headers, rows))
    lines.append("")
    if "train_heldout" in summary["overall"]:
        lines.append("## Internal Train Holdout")
        holdout = summary["overall"]["train_heldout"]
        headers = ["model", "MAE", "MAPE%", "RMSE"]
        rows = []
        for model_name in ("baseline", "physical"):
            metrics = holdout[model_name]
            rows.append([model_name, format_float(metrics["mae"]), format_float(metrics["mape"]), format_float(metrics["rmse"])])
        lines.append(markdown_table(headers, rows))
        lines.append("")
    lines.append("## Component Means")
    headers = ["split", "launch_runtime_us", "queueing_us", "compute_us", "memory_us", "dominant_us", "pred_us"]
    rows = []
    for split in ("train_fit", "train_heldout", "train", "val", "test"):
        if split not in summary["components"]:
            continue
        comp = summary["components"][split]
        rows.append(
            [
                split,
                format_float(comp.get("physical_launch_runtime_us", float("nan"))),
                format_float(comp.get("queueing_us", float("nan"))),
                format_float(comp.get("physical_compute_us", float("nan"))),
                format_float(comp.get("physical_memory_us", float("nan"))),
                format_float(comp.get("physical_dominant_us", float("nan"))),
                format_float(comp.get("physical_pred_us", float("nan"))),
            ]
        )
    lines.append(markdown_table(headers, rows))
    lines.append("")
    lines.append("## Baseline To Physical Delta")
    delta_headers = ["split", "MAE delta", "MAPE delta", "RMSE delta"]
    delta_rows: list[list[str]] = []
    for split in ("train", "val", "test"):
        if split not in summary["overall"]:
            continue
        delta = summary["overall"][split]["delta"]
        delta_rows.append([split, format_float(delta["mae_delta"]), format_float(delta["mape_delta"]), format_float(delta["rmse_delta"])])
    lines.append(markdown_table(delta_headers, delta_rows))
    lines.append("")
    lines.append("## By Op Name")
    for split in ("val", "test"):
        if split not in summary["by_op_name"]:
            continue
        lines.append(f"### {split}")
        group_map = summary["by_op_name"][split]
        headers = [
            "op_name",
            "count",
            "baseline_MAE",
            "physical_MAE",
            "delta_MAE",
            "baseline_MAPE%",
            "physical_MAPE%",
            "delta_MAPE%",
            "baseline_RMSE",
            "physical_RMSE",
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
                    format_float(metrics["physical"]["mae"]),
                    format_float(metrics["delta"]["mae_delta"]),
                    format_float(metrics["baseline"]["mape"]),
                    format_float(metrics["physical"]["mape"]),
                    format_float(metrics["delta"]["mape_delta"]),
                    format_float(metrics["baseline"]["rmse"]),
                    format_float(metrics["physical"]["rmse"]),
                    format_float(metrics["delta"]["rmse_delta"]),
                ]
            )
        lines.append(markdown_table(headers, rows))
        lines.append("")
    lines.append("## By Lane")
    for split in ("val", "test"):
        if split not in summary["by_lane"]:
            continue
        lines.append(f"### {split}")
        group_map = summary["by_lane"][split]
        headers = [
            "lane",
            "count",
            "baseline_MAE",
            "physical_MAE",
            "delta_MAE",
            "baseline_MAPE%",
            "physical_MAPE%",
            "delta_MAPE%",
            "baseline_RMSE",
            "physical_RMSE",
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
                    format_float(metrics["physical"]["mae"]),
                    format_float(metrics["delta"]["mae_delta"]),
                    format_float(metrics["baseline"]["mape"]),
                    format_float(metrics["physical"]["mape"]),
                    format_float(metrics["delta"]["mape_delta"]),
                    format_float(metrics["baseline"]["rmse"]),
                    format_float(metrics["physical"]["rmse"]),
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
    physical_params = calibration.get("parameters") or calibration

    annotated_frames = {split: annotate_baselines(frame, hardware_profile) for split, frame in frames.items()}
    calibrated_frames = {
        split: apply_physical_model(frame, physical_params, hardware_profile, op_col, lane_col)
        for split, frame in annotated_frames.items()
    }

    summary: dict[str, Any] = {"overall": {}, "by_op_name": {}, "by_lane": {}, "components": {}}
    subset = calibration.get("calibration_subset") or {}
    for split, frame in calibrated_frames.items():
        summary["overall"][split] = comparison_metrics_for_frame(frame, label_col)
        summary["by_op_name"][split] = group_physical_metrics(frame, label_col, "physical_pred_us", op_col)
        summary["by_lane"][split] = group_physical_metrics(frame, label_col, "physical_pred_us", lane_col)
        summary["components"][split] = component_means_for_frame(frame)

    fit_indices = subset.get("fit_indices") or []
    heldout_indices = subset.get("heldout_indices") or []
    train_frame = calibrated_frames["train"]
    if fit_indices:
        fit_frame = train_frame.loc[fit_indices]
        summary["overall"]["train_fit"] = comparison_metrics_for_frame(fit_frame, label_col)
        summary["by_op_name"]["train_fit"] = group_physical_metrics(fit_frame, label_col, "physical_pred_us", op_col)
        summary["by_lane"]["train_fit"] = group_physical_metrics(fit_frame, label_col, "physical_pred_us", lane_col)
        summary["components"]["train_fit"] = component_means_for_frame(fit_frame)
    if heldout_indices:
        heldout_frame = train_frame.loc[heldout_indices]
        summary["overall"]["train_heldout"] = comparison_metrics_for_frame(heldout_frame, label_col)
        summary["by_op_name"]["train_heldout"] = group_physical_metrics(heldout_frame, label_col, "physical_pred_us", op_col)
        summary["by_lane"]["train_heldout"] = group_physical_metrics(heldout_frame, label_col, "physical_pred_us", lane_col)
        summary["components"]["train_heldout"] = component_means_for_frame(heldout_frame)
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
