from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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
    infer_transfer_direction,
    queue_proxy_components,
    safe_float,
)


LABEL_COLUMN = "label_npu_dur_us"
OP_COLUMN = "op_name"
LANE_COLUMN = "npu_lane"
PHYSICAL_PREDICTION_COLUMN = "physical_pred_us"
QUEUE_PROXY_COLUMN = "queue_proxy_us"
QUEUE_WAIT_PROXY_COLUMN = "queue_wait_proxy_us"
QUEUE_ENQUEUE_PROXY_COLUMN = "queue_enqueue_proxy_us"
DEFAULT_QUEUE_SCALE = 1.0
DEFAULT_BW_GBPS = DEFAULT_MEMORY_BW_GBPS
QUEUE_PROXY_SOURCE_COLUMNS = [QUEUE_WAIT_PROXY_COLUMN, QUEUE_ENQUEUE_PROXY_COLUMN, QUEUE_PROXY_COLUMN]
IDENTIFIABILITY_BW_THRESHOLD_GBPS = 10000.0


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
        queue_proxy = queue_proxy_components(row)
        row["baseline_pred_us"] = float(baseline["baseline_us"])
        row["baseline_compute_us"] = float(baseline["compute_us"])
        row["baseline_memory_us"] = float(baseline["memory_us"])
        row["baseline_data_bytes"] = float(baseline["data_bytes"])
        row["baseline_peak_gflops"] = baseline["peak_gflops"]
        row["baseline_bw_gbps"] = baseline["bw_gbps"]
        row["baseline_formula"] = baseline["formula"]
        row["queue_wait_proxy_us"] = float(queue_proxy["queue_wait_proxy_us"])
        row["queue_enqueue_proxy_us"] = float(queue_proxy["queue_enqueue_proxy_us"])
        row["queue_proxy_us"] = float(queue_proxy["queue_proxy_us"])
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


def build_physical_parameter_names(op_names: list[str], include_cube_memory: bool) -> list[str]:
    names = [f"launch_runtime_us::{op_name}" for op_name in op_names]
    names.append("queueing_scale")
    if include_cube_memory:
        names.append("cube_memory_bw_gbps")
    names.extend(
        [
            "vector_memory_bw_gbps",
            "transfer_h2d_bw_gbps",
            "transfer_d2h_bw_gbps",
        ]
    )
    return names


def get_effective_peak_values(hardware_profile: dict[str, Any]) -> dict[str, float]:
    return {
        "cube_peak_eff_gflops": float(
            safe_float(
                hardware_value(hardware_profile, "cube_peak_eff_gflops", DEFAULT_CUBE_PEAK_EFF_GFLOPS),
                DEFAULT_CUBE_PEAK_EFF_GFLOPS,
            )
            or DEFAULT_CUBE_PEAK_EFF_GFLOPS
        ),
        "vector_peak_eff_gflops": float(
            safe_float(
                hardware_value(hardware_profile, "vector_peak_eff_gflops", DEFAULT_VECTOR_PEAK_EFF_GFLOPS),
                DEFAULT_VECTOR_PEAK_EFF_GFLOPS,
            )
            or DEFAULT_VECTOR_PEAK_EFF_GFLOPS
        ),
    }


def infer_queue_proxy_us(row: dict[str, Any]) -> float:
    if QUEUE_PROXY_COLUMN in row and row.get(QUEUE_PROXY_COLUMN) is not None:
        value = float(safe_float(row.get(QUEUE_PROXY_COLUMN), 0.0) or 0.0)
        return max(value, 0.0)
    proxy = queue_proxy_components(row)
    return float(proxy["queue_proxy_us"])


def physical_row_terms(
    row: dict[str, Any],
    hardware_profile: dict[str, Any],
    launch_runtime_us_by_op_name: dict[str, float],
    default_launch_runtime_us: float,
    queueing_scale: float,
    vector_memory_bw_gbps: float,
    transfer_h2d_bw_gbps: float,
    transfer_d2h_bw_gbps: float,
    cube_memory_bw_gbps: float | None,
) -> dict[str, Any]:
    row_dict = dict(row)
    lane = str(row_dict.get(LANE_COLUMN) or row_dict.get("npu_lane") or "")
    op_name = str(row_dict.get(OP_COLUMN) or row_dict.get("op_name") or "")
    queue_proxy_us = infer_queue_proxy_us(row_dict)
    launch_runtime_us = float(launch_runtime_us_by_op_name.get(op_name, default_launch_runtime_us))
    queueing_us = max(queue_proxy_us * float(queueing_scale), 0.0)

    peaks = get_effective_peak_values(hardware_profile)
    cube_peak = peaks["cube_peak_eff_gflops"]
    vector_peak = peaks["vector_peak_eff_gflops"]

    input_bytes = float(safe_float(row_dict.get("input_bytes"), 0.0) or 0.0)
    output_bytes = float(safe_float(row_dict.get("output_bytes"), 0.0) or 0.0)
    activation_bytes = float(safe_float(row_dict.get("activation_bytes"), 0.0) or 0.0)
    parameter_bytes = float(safe_float(row_dict.get("parameter_bytes"), 0.0) or 0.0)

    compute_us = 0.0
    memory_us = 0.0
    memory_bytes = 0.0
    dominant_us = 0.0
    bw_gbps: float | None = None
    formula = "physical_fallback"

    if lane == "cube":
        m = float(safe_float(row_dict.get("matmul_m"), 0.0) or 0.0)
        k = float(safe_float(row_dict.get("matmul_k"), 0.0) or 0.0)
        n = float(safe_float(row_dict.get("matmul_n"), 0.0) or 0.0)
        flops = 2.0 * m * k * n
        compute_us = flops / max(cube_peak, 1e-9) / 1000.0
        memory_bytes = input_bytes + output_bytes + activation_bytes + parameter_bytes
        if cube_memory_bw_gbps is not None and cube_memory_bw_gbps > 0.0:
            memory_us = memory_bytes / cube_memory_bw_gbps / 1000.0
        dominant_us = max(compute_us, memory_us)
        bw_gbps = cube_memory_bw_gbps
        formula = "cube_launch_queue_max_roofline"
    elif lane == "vector":
        elem_count = float(safe_float(row_dict.get("vector_elem_count"), 0.0) or 0.0)
        if elem_count <= 0.0:
            elem_count = max(output_bytes / 4.0, input_bytes / 4.0, 0.0)
        compute_us = elem_count / max(vector_peak, 1e-9) / 1000.0
        memory_bytes = input_bytes + output_bytes + activation_bytes
        memory_us = memory_bytes / max(vector_memory_bw_gbps, 1e-9) / 1000.0
        dominant_us = max(compute_us, memory_us)
        bw_gbps = vector_memory_bw_gbps
        formula = "vector_launch_queue_max_roofline"
    elif lane == "transfer":
        transfer_bytes = max(input_bytes, output_bytes, activation_bytes, parameter_bytes)
        bw_gbps = transfer_h2d_bw_gbps if infer_transfer_direction(op_name) == "h2d" else transfer_d2h_bw_gbps
        memory_bytes = transfer_bytes
        memory_us = transfer_bytes / max(bw_gbps, 1e-9) / 1000.0
        dominant_us = memory_us
        formula = "transfer_launch_queue_bandwidth"
    else:
        memory_bytes = input_bytes + output_bytes + activation_bytes + parameter_bytes
        memory_us = memory_bytes / max(vector_memory_bw_gbps, 1e-9) / 1000.0
        dominant_us = memory_us
        bw_gbps = vector_memory_bw_gbps

    physical_pred_us = max(0.0, launch_runtime_us + queueing_us + dominant_us)
    return {
        "lane": lane,
        "op_name": op_name,
        "queue_proxy_us": queue_proxy_us,
        "queueing_us": queueing_us,
        "launch_runtime_us": launch_runtime_us,
        "compute_us": compute_us,
        "memory_us": memory_us,
        "dominant_us": dominant_us,
        "memory_bytes": memory_bytes,
        "bw_gbps": bw_gbps,
        "formula": formula,
        "physical_pred_us": physical_pred_us,
    }


def build_physical_theta(
    op_names: list[str],
    fit_df: pd.DataFrame,
    hardware_profile: dict[str, Any],
    include_cube_memory: bool,
) -> tuple[np.ndarray, list[str], list[float], list[float]]:
    launch_guess: list[float] = []
    for op_name in op_names:
        op_rows = fit_df[fit_df[OP_COLUMN] == op_name]
        op_labels = op_rows[LABEL_COLUMN].astype(float)
        baseline_gap = (op_labels - op_rows["baseline_pred_us"].astype(float)).clip(lower=0.0)
        if len(baseline_gap):
            guess = float(np.median(baseline_gap))
        elif len(op_rows):
            guess = float(np.median(op_labels) * 0.1)
        else:
            guess = 1.0
        launch_guess.append(max(1.0, guess))

    initial: list[float] = [*launch_guess, DEFAULT_QUEUE_SCALE]
    lower: list[float] = [0.0] * len(launch_guess) + [0.0]
    upper: list[float] = [1e6] * len(launch_guess) + [10.0]

    if include_cube_memory:
        initial.append(float(hardware_value(hardware_profile, "memory_bw_gbps", DEFAULT_BW_GBPS) or DEFAULT_BW_GBPS))
        lower.append(1.0)
        upper.append(1e6)

    initial.extend(
        [
            float(hardware_value(hardware_profile, "memory_bw_gbps", DEFAULT_BW_GBPS) or DEFAULT_BW_GBPS),
            float(hardware_value(hardware_profile, "h2d_bw_gbps", DEFAULT_TRANSFER_BW_GBPS) or DEFAULT_TRANSFER_BW_GBPS),
            float(hardware_value(hardware_profile, "d2h_bw_gbps", DEFAULT_TRANSFER_BW_GBPS) or DEFAULT_TRANSFER_BW_GBPS),
        ]
    )
    lower.extend([1.0, 1.0, 1.0])
    upper.extend([1e6, 1e6, 1e6])
    return np.asarray(initial, dtype=float), lower, upper, op_names


def unpack_physical_theta(
    theta: np.ndarray,
    op_names: list[str],
    include_cube_memory: bool,
) -> dict[str, Any]:
    cursor = 0
    launch_runtime_us_by_op_name: dict[str, float] = {}
    for op_name in op_names:
        launch_runtime_us_by_op_name[op_name] = float(max(theta[cursor], 0.0))
        cursor += 1
    queueing_scale = float(max(theta[cursor], 0.0))
    cursor += 1
    cube_memory_bw_gbps = None
    if include_cube_memory:
        cube_memory_bw_gbps = float(max(theta[cursor], 1e-9))
        cursor += 1
    vector_memory_bw_gbps = float(max(theta[cursor], 1e-9))
    cursor += 1
    transfer_h2d_bw_gbps = float(max(theta[cursor], 1e-9))
    cursor += 1
    transfer_d2h_bw_gbps = float(max(theta[cursor], 1e-9))
    default_launch_runtime_us = float(np.median(list(launch_runtime_us_by_op_name.values()))) if launch_runtime_us_by_op_name else 0.0
    return {
        "launch_runtime_us_by_op_name": launch_runtime_us_by_op_name,
        "default_launch_runtime_us": default_launch_runtime_us,
        "queueing_scale": queueing_scale,
        "cube_memory_bw_gbps": cube_memory_bw_gbps,
        "vector_memory_bw_gbps": vector_memory_bw_gbps,
        "transfer_h2d_bw_gbps": transfer_h2d_bw_gbps,
        "transfer_d2h_bw_gbps": transfer_d2h_bw_gbps,
        "include_cube_memory": include_cube_memory,
    }


def apply_physical_model(
    df: pd.DataFrame,
    calibration: dict[str, Any],
    hardware_profile: dict[str, Any],
    op_col: str,
    lane_col: str,
) -> pd.DataFrame:
    params = calibration.get("parameters") if isinstance(calibration.get("parameters"), dict) else calibration
    predicted_rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        terms = physical_row_terms(
            row,
            hardware_profile,
            params["launch_runtime_us_by_op_name"],
            params["default_launch_runtime_us"],
            params["queueing_scale"],
            params["vector_memory_bw_gbps"],
            params["transfer_h2d_bw_gbps"],
            params["transfer_d2h_bw_gbps"],
            params.get("cube_memory_bw_gbps"),
        )
        row["queue_proxy_us"] = terms["queue_proxy_us"]
        row["queueing_us"] = terms["queueing_us"]
        row["physical_launch_runtime_us"] = terms["launch_runtime_us"]
        row["physical_compute_us"] = terms["compute_us"]
        row["physical_memory_us"] = terms["memory_us"]
        row["physical_dominant_us"] = terms["dominant_us"]
        row["physical_pred_us"] = terms["physical_pred_us"]
        row["calibrated_pred_us"] = terms["physical_pred_us"]
        row["lane_calibrated_pred_us"] = terms["physical_pred_us"]
        row["global_calibrated_pred_us"] = terms["physical_pred_us"]
        row["physical_formula"] = terms["formula"]
        row["physical_memory_bw_gbps"] = terms["bw_gbps"]
        row["physical_model_variant"] = calibration.get("model_variant", params.get("model_variant", "physical"))
        predicted_rows.append(row)
    return pd.DataFrame(predicted_rows)


def component_means_for_frame(df: pd.DataFrame) -> dict[str, float]:
    payload: dict[str, float] = {}
    for column in [
        "physical_launch_runtime_us",
        "queueing_us",
        "physical_compute_us",
        "physical_memory_us",
        "physical_dominant_us",
        "physical_pred_us",
    ]:
        payload[column] = float(df[column].astype(float).mean()) if column in df.columns and len(df) else float("nan")
    return payload


def comparison_metrics_for_frame(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    baseline = compute_regression_metrics(df[label_col], df["baseline_pred_us"])
    physical = compute_regression_metrics(df[label_col], df["physical_pred_us"])
    return {
        "count": int(len(df)),
        "baseline": baseline,
        "physical": physical,
        "delta": {
            "mae_delta": float(physical["mae"] - baseline["mae"]),
            "mape_delta": float(physical["mape"] - baseline["mape"]),
            "rmse_delta": float(physical["rmse"] - baseline["rmse"]),
        },
    }


def group_physical_metrics(df: pd.DataFrame, label_col: str, pred_col: str, group_col: str) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for group, group_df in df.groupby(group_col, dropna=False):
        baseline = compute_regression_metrics(group_df[label_col], group_df["baseline_pred_us"])
        physical = compute_regression_metrics(group_df[label_col], group_df[pred_col])
        payload[str(group)] = {
            "count": int(len(group_df)),
            "baseline": baseline,
            "physical": physical,
            "delta": {
                "mae_delta": float(physical["mae"] - baseline["mae"]),
                "mape_delta": float(physical["mape"] - baseline["mape"]),
                "rmse_delta": float(physical["rmse"] - baseline["rmse"]),
            },
        }
    return payload


def fit_physical_variant(
    fit_df: pd.DataFrame,
    hardware_profile: dict[str, Any],
    include_cube_memory: bool,
    label_col: str = LABEL_COLUMN,
    op_col: str = OP_COLUMN,
    lane_col: str = LANE_COLUMN,
) -> dict[str, Any]:
    op_names = sorted(str(value) for value in fit_df[op_col].dropna().unique())
    theta0, lower, upper, op_names = build_physical_theta(op_names, fit_df, hardware_profile, include_cube_memory)
    records = fit_df.to_dict(orient="records")
    labels = fit_df[label_col].astype(float).to_numpy()
    denom = np.clip(np.abs(labels), 1.0, None)

    def residuals(theta: np.ndarray) -> np.ndarray:
        calibration = unpack_physical_theta(theta, op_names, include_cube_memory)
        preds = np.asarray(
            [
                physical_row_terms(
                    row,
                    hardware_profile,
                    calibration["launch_runtime_us_by_op_name"],
                    calibration["default_launch_runtime_us"],
                    calibration["queueing_scale"],
                    calibration["vector_memory_bw_gbps"],
                    calibration["transfer_h2d_bw_gbps"],
                    calibration["transfer_d2h_bw_gbps"],
                    calibration.get("cube_memory_bw_gbps"),
                )["physical_pred_us"]
                for row in records
            ],
            dtype=float,
        )
        return (preds - labels) / denom

    result = least_squares(
        residuals,
        theta0,
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=8000,
        x_scale="jac",
    )
    calibration = unpack_physical_theta(result.x, op_names, include_cube_memory)
    calibration["op_names"] = op_names
    calibration["optimizer"] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }
    calibration["fit_metrics"] = compute_regression_metrics(labels, residuals(result.x) * denom + labels)
    return calibration


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
    summary: dict[str, Any] = {"overall": {}, "by_op_name": {}, "by_lane": {}, "components": {}}
    for split, frame in frames.items():
        summary["overall"][split] = comparison_metrics_for_frame(frame, label_col)
        summary["by_op_name"][split] = group_physical_metrics(frame, label_col, "physical_pred_us", op_col)
        summary["by_lane"][split] = group_physical_metrics(frame, label_col, "physical_pred_us", lane_col)
        summary["components"][split] = component_means_for_frame(frame)
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

    candidate_results: list[dict[str, Any]] = []
    for include_cube_memory in (True, False):
        calibration = fit_physical_variant(
            fit_train,
            hardware_profile,
            include_cube_memory=include_cube_memory,
            label_col=label_col,
            op_col=op_col,
            lane_col=lane_col,
        )
        calibration["model_variant"] = "full_physical" if include_cube_memory else "reduced_physical"
        calibration["cube_memory_mode"] = "fitted" if include_cube_memory else "merged_into_launch_runtime"
        calibration["merged_terms"] = [] if include_cube_memory else ["cube_memory_us"]

        calibrated_frames = {
            split: apply_physical_model(frame, calibration, hardware_profile, op_col, lane_col)
            for split, frame in annotated_frames.items()
        }
        fit_frame = apply_physical_model(fit_train, calibration, hardware_profile, op_col, lane_col)
        heldout_frame = (
            apply_physical_model(heldout_train, calibration, hardware_profile, op_col, lane_col)
            if len(heldout_train)
            else heldout_train
        )
        fit_metrics = comparison_metrics_for_frame(fit_frame, label_col)
        heldout_metrics = comparison_metrics_for_frame(heldout_frame, label_col) if len(heldout_train) else fit_metrics
        score = float(heldout_metrics["physical"]["mape"] if len(heldout_train) else fit_metrics["physical"]["mape"])
        candidate_results.append(
            {
                "calibration": calibration,
                "frames": calibrated_frames,
                "fit_frame": fit_frame,
                "heldout_frame": heldout_frame,
                "fit_metrics": fit_metrics,
                "heldout_metrics": heldout_metrics,
                "score": score,
            }
        )

    candidate_results.sort(key=lambda item: (item["score"], 0 if item["calibration"]["model_variant"] == "reduced_physical" else 1))
    best = candidate_results[0]
    calibration = best["calibration"]
    merged_terms = list(calibration.get("merged_terms", []))
    if calibration.get("cube_memory_bw_gbps") is not None and float(calibration["cube_memory_bw_gbps"]) >= IDENTIFIABILITY_BW_THRESHOLD_GBPS:
        merged_terms.append("cube_memory_us")
        calibration["cube_memory_mode"] = "merged_into_launch_runtime"
    if float(calibration["vector_memory_bw_gbps"]) >= IDENTIFIABILITY_BW_THRESHOLD_GBPS:
        merged_terms.append("vector_memory_us")
    if float(calibration["transfer_h2d_bw_gbps"]) >= IDENTIFIABILITY_BW_THRESHOLD_GBPS:
        merged_terms.append("transfer_h2d_us")
    if float(calibration["transfer_d2h_bw_gbps"]) >= IDENTIFIABILITY_BW_THRESHOLD_GBPS:
        merged_terms.append("transfer_d2h_us")
    calibration["merged_terms"] = sorted(set(merged_terms))
    calibrated_frames = best["frames"]
    metrics_summary = build_metrics_summary(calibrated_frames, label_col, op_col, lane_col)
    metrics_summary["overall"]["train_fit"] = best["fit_metrics"]
    if len(heldout_train):
        metrics_summary["overall"]["train_heldout"] = best["heldout_metrics"]
        metrics_summary["by_op_name"]["train_fit"] = group_physical_metrics(best["fit_frame"], label_col, "physical_pred_us", op_col)
        metrics_summary["by_lane"]["train_fit"] = group_physical_metrics(best["fit_frame"], label_col, "physical_pred_us", lane_col)
        metrics_summary["by_op_name"]["train_heldout"] = group_physical_metrics(best["heldout_frame"], label_col, "physical_pred_us", op_col)
        metrics_summary["by_lane"]["train_heldout"] = group_physical_metrics(best["heldout_frame"], label_col, "physical_pred_us", lane_col)
        metrics_summary["components"]["train_fit"] = component_means_for_frame(best["fit_frame"])
        metrics_summary["components"]["train_heldout"] = component_means_for_frame(best["heldout_frame"])

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
        "model_variant": calibration.get("model_variant"),
        "cube_memory_mode": calibration.get("cube_memory_mode"),
        "merged_terms": calibration.get("merged_terms", []),
        "physical_prediction_column": PHYSICAL_PREDICTION_COLUMN,
        "queue_proxy_columns": QUEUE_PROXY_SOURCE_COLUMNS,
        "queue_proxy_policy": {
            "definition": "cpu_main_Wait_avg + cpu_main_DistributionEnqueue_avg",
            "units": "microseconds",
            "notes": "Observed queueing proxy from trace-derived CPU scheduling stats; zero when unavailable.",
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
        "parameters": {
            "launch_runtime_us_by_op_name": calibration["launch_runtime_us_by_op_name"],
            "default_launch_runtime_us": calibration["default_launch_runtime_us"],
            "queueing_scale": calibration["queueing_scale"],
            "cube_memory_bw_gbps": calibration.get("cube_memory_bw_gbps"),
            "vector_memory_bw_gbps": calibration["vector_memory_bw_gbps"],
            "transfer_h2d_bw_gbps": calibration["transfer_h2d_bw_gbps"],
            "transfer_d2h_bw_gbps": calibration["transfer_d2h_bw_gbps"],
        },
        "optimizer": calibration.get("optimizer", {}),
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
