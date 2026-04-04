from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from npu_sep_common import (
    CASE_ID,
    DEFAULT_CUBE_PEAK_EFF_GFLOPS,
    DEFAULT_MEMORY_BW_GBPS,
    DEFAULT_TRANSFER_BW_GBPS,
    DEFAULT_VECTOR_PEAK_EFF_GFLOPS,
    compute_regression_metrics,
    dump_json,
    ensure_dir,
    fit_scale_bias,
    hardware_value,
    lane_baseline_components,
    load_hardware_profile,
)


LABEL_COLUMN = "label_npu_dur_us"
OP_COLUMN = "op_name"
LANE_COLUMN = "npu_lane"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the calibrated analytical model for the NPU separation dataset.")
    parser.add_argument("--data-dir", required=True, help="Directory that contains train.csv/val.csv/test.csv.")
    parser.add_argument("--hardware-profile", default="", help="Optional hardware_profile_910b3.json path.")
    parser.add_argument("--output-dir", required=True, help="Directory for calibration.json and metrics_summary.json.")
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--op-column", default=OP_COLUMN)
    parser.add_argument("--lane-column", default=LANE_COLUMN)
    parser.add_argument(
        "--calibration-fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of train rows used to fit calibration parameters. Values below 1.0 use a stratified subset.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=42,
        help="Random seed used when selecting the calibration subset.",
    )
    return parser.parse_args()


def load_split_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        df["split"] = split
        frames[split] = df
    return frames


def load_optional_hardware_profile(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    return load_hardware_profile(path)


def annotate_baselines(df: pd.DataFrame, hardware_profile: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        baseline = lane_baseline_components(row, hardware_profile)
        row["baseline_pred_us"] = float(baseline["baseline_us"])
        row["baseline_compute_us"] = float(baseline["compute_us"])
        row["baseline_memory_us"] = float(baseline["memory_us"])
        row["baseline_data_bytes"] = float(baseline["data_bytes"])
        row["baseline_peak_gflops"] = baseline["peak_gflops"]
        row["baseline_bw_gbps"] = baseline["bw_gbps"]
        row["baseline_formula"] = baseline["formula"]
        rows.append(row)
    return pd.DataFrame(rows)


def select_calibration_subset(
    df: pd.DataFrame,
    group_col: str,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    if fraction >= 1.0:
        fit_df = df.copy()
        heldout_df = df.iloc[0:0].copy()
        fit_indices = [int(idx) for idx in fit_df.index.tolist()]
        return fit_df, heldout_df, fit_indices, []
    if fraction <= 0.0:
        raise ValueError("calibration-fit-fraction must be greater than 0")

    rng = random.Random(seed)
    fit_indices: list[int] = []
    for _, group_df in df.groupby(group_col, dropna=False):
        group_indices = [int(idx) for idx in group_df.index.tolist()]
        rng.shuffle(group_indices)
        target = max(1, int(math.ceil(len(group_indices) * fraction)))
        target = min(len(group_indices), target)
        fit_indices.extend(group_indices[:target])

    fit_indices = sorted(set(fit_indices))
    heldout_indices = [int(idx) for idx in df.index.tolist() if int(idx) not in fit_indices]
    fit_df = df.loc[fit_indices].copy()
    heldout_df = df.loc[heldout_indices].copy()
    return fit_df, heldout_df, fit_indices, heldout_indices


def fit_group_params(df: pd.DataFrame, group_col: str, label_col: str) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for group, group_df in df.groupby(group_col, dropna=False):
        baseline = group_df["baseline_pred_us"].astype(float)
        labels = group_df[label_col].astype(float)
        scale, bias = fit_scale_bias(baseline, labels)
        params[str(group)] = {
            "scale": float(scale),
            "bias_us": float(bias),
            "n": int(len(group_df)),
            "baseline_mae": float(compute_regression_metrics(labels, baseline)["mae"]),
        }
    return params


def apply_calibration(df: pd.DataFrame, op_params: dict[str, dict[str, Any]], lane_params: dict[str, dict[str, Any]], global_params: dict[str, Any], op_col: str, lane_col: str) -> pd.DataFrame:
    calibrated_rows: list[dict[str, Any]] = []
    global_scale = float(global_params["scale"])
    global_bias = float(global_params["bias_us"])
    for row in df.to_dict(orient="records"):
        baseline = float(row["baseline_pred_us"])
        op_key = str(row.get(op_col) or "")
        lane_key = str(row.get(lane_col) or "")
        op_param = op_params.get(op_key, global_params)
        lane_param = lane_params.get(lane_key, global_params)
        row["calibrated_pred_us"] = baseline * float(op_param["scale"]) + float(op_param["bias_us"])
        row["lane_calibrated_pred_us"] = baseline * float(lane_param["scale"]) + float(lane_param["bias_us"])
        row["global_calibrated_pred_us"] = baseline * global_scale + global_bias
        calibrated_rows.append(row)
    return pd.DataFrame(calibrated_rows)


def metrics_for_frame(df: pd.DataFrame, label_col: str, pred_cols: list[str]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    labels = df[label_col].astype(float)
    for pred_col in pred_cols:
        payload[pred_col] = {
            **compute_regression_metrics(labels, df[pred_col].astype(float)),
            "count": int(len(df)),
        }
    return payload


def group_metrics(df: pd.DataFrame, label_col: str, pred_col: str, group_col: str) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for group, group_df in df.groupby(group_col, dropna=False):
        payload[str(group)] = {
            **compute_regression_metrics(group_df[label_col].astype(float), group_df[pred_col].astype(float)),
            "count": int(len(group_df)),
        }
    return payload


def build_metrics_summary(
    frames: dict[str, pd.DataFrame],
    label_col: str,
    op_col: str,
    lane_col: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_op_name": {}, "by_lane": {}}
    for split, frame in frames.items():
        summary["overall"][split] = metrics_for_frame(
            frame,
            label_col,
            ["baseline_pred_us", "calibrated_pred_us", "lane_calibrated_pred_us", "global_calibrated_pred_us"],
        )
        summary["by_op_name"][split] = group_metrics(frame, label_col, "calibrated_pred_us", op_col)
        summary["by_lane"][split] = group_metrics(frame, label_col, "lane_calibrated_pred_us", lane_col)
    return summary


def fit_model(
    data_dir: Path,
    hardware_profile_path: str,
    output_dir: Path,
    label_col: str,
    op_col: str,
    lane_col: str,
    calibration_fit_fraction: float,
    calibration_seed: int,
) -> dict[str, Any]:
    frames = load_split_frames(data_dir)
    hardware_profile = load_optional_hardware_profile(hardware_profile_path)

    annotated_frames = {split: annotate_baselines(frame, hardware_profile) for split, frame in frames.items()}
    train = annotated_frames["train"]
    fit_train, heldout_train, fit_indices, heldout_indices = select_calibration_subset(
        train, op_col, calibration_fit_fraction, calibration_seed
    )

    global_scale, global_bias = fit_scale_bias(fit_train["baseline_pred_us"], fit_train[label_col])
    global_params = {"scale": float(global_scale), "bias_us": float(global_bias), "n": int(len(fit_train))}
    op_params = fit_group_params(fit_train, op_col, label_col)
    lane_params = fit_group_params(fit_train, lane_col, label_col)

    calibrated_frames = {
        split: apply_calibration(frame, op_params, lane_params, global_params, op_col, lane_col)
        for split, frame in annotated_frames.items()
    }

    metrics_summary = build_metrics_summary(calibrated_frames, label_col, op_col, lane_col)
    metrics_summary["overall"]["train_fit"] = metrics_for_frame(
        apply_calibration(fit_train, op_params, lane_params, global_params, op_col, lane_col),
        label_col,
        ["baseline_pred_us", "calibrated_pred_us", "lane_calibrated_pred_us", "global_calibrated_pred_us"],
    )
    metrics_summary["overall"]["train_heldout"] = metrics_for_frame(
        apply_calibration(heldout_train, op_params, lane_params, global_params, op_col, lane_col),
        label_col,
        ["baseline_pred_us", "calibrated_pred_us", "lane_calibrated_pred_us", "global_calibrated_pred_us"],
    ) if len(heldout_train) else {}

    calibration_payload = {
        "case_id": CASE_ID,
        "label_column": label_col,
        "op_column": op_col,
        "lane_column": lane_col,
        "hardware_profile_path": hardware_profile_path or None,
        "calibration_fit_fraction": float(calibration_fit_fraction),
        "calibration_seed": int(calibration_seed),
        "calibration_fit_group_column": op_col,
        "calibration_subset": {
            "fit_rows": int(len(fit_train)),
            "heldout_rows": int(len(heldout_train)),
            "fit_indices": fit_indices,
            "heldout_indices": heldout_indices,
        },
        "baseline_defaults": {
            "cube_peak_eff_gflops": DEFAULT_CUBE_PEAK_EFF_GFLOPS,
            "vector_peak_eff_gflops": DEFAULT_VECTOR_PEAK_EFF_GFLOPS,
            "memory_bw_gbps": DEFAULT_MEMORY_BW_GBPS,
            "transfer_bw_gbps": DEFAULT_TRANSFER_BW_GBPS,
        },
        "hardware_profile_effective": {
            "device_name": hardware_value(hardware_profile, "device_name", "910B3"),
            "ai_core_count": hardware_value(hardware_profile, "ai_core_count"),
            "cube_count": hardware_value(hardware_profile, "cube_count", 20),
            "vector_count": hardware_value(hardware_profile, "vector_count", 40),
            "cube_peak_eff_gflops": hardware_value(
                hardware_profile, "cube_peak_eff_gflops", DEFAULT_CUBE_PEAK_EFF_GFLOPS
            ),
            "vector_peak_eff_gflops": hardware_value(
                hardware_profile, "vector_peak_eff_gflops", DEFAULT_VECTOR_PEAK_EFF_GFLOPS
            ),
            "memory_bw_gbps": hardware_value(hardware_profile, "memory_bw_gbps", DEFAULT_MEMORY_BW_GBPS),
            "h2d_bw_gbps": hardware_value(hardware_profile, "h2d_bw_gbps", DEFAULT_TRANSFER_BW_GBPS),
            "d2h_bw_gbps": hardware_value(hardware_profile, "d2h_bw_gbps", DEFAULT_TRANSFER_BW_GBPS),
        },
        "global": global_params,
        "by_op_name": op_params,
        "by_lane": lane_params,
        "training_rows": int(len(train)),
        "validation_rows": int(len(calibrated_frames["val"])),
        "test_rows": int(len(calibrated_frames["test"])),
    }

    ensure_dir(output_dir)
    calibration_path = output_dir / "calibration.json"
    metrics_path = output_dir / "metrics_summary.json"
    dump_json(calibration_path, calibration_payload)
    dump_json(metrics_path, metrics_summary)
    return {
        "calibration_path": str(calibration_path),
        "metrics_path": str(metrics_path),
        "calibration": calibration_payload,
        "metrics_summary": metrics_summary,
        "calibrated_frames": calibrated_frames,
    }


def main() -> None:
    args = parse_args()
    result = fit_model(
        Path(args.data_dir),
        args.hardware_profile,
        Path(args.output_dir),
        args.label_column,
        args.op_column,
        args.lane_column,
        args.calibration_fit_fraction,
        args.calibration_seed,
    )
    print(f"Wrote {result['calibration_path']}")
    print(f"Wrote {result['metrics_path']}")


if __name__ == "__main__":
    main()
