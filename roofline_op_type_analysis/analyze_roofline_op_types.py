from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_contract import TARGET_COLUMN
from feature_engineering import (  # noqa: E402
    HARDWARE_PROFILE_PATH,
    add_analytical_hardware_software_features,
    add_engineered_features,
    add_operator_hardware_context,
    load_hardware_features,
)


BOUND_ORDER = ["memory_bound", "near_ridge", "compute_bound"]
BOUND_COLORS = {
    "memory_bound": "#1b9e77",
    "near_ridge": "#d95f02",
    "compute_bound": "#7570b3",
}
BOUND_SHORT_LABELS = {
    "memory_bound": "M",
    "near_ridge": "R",
    "compute_bound": "C",
}

DEFAULT_INPUT_CSV = PROJECT_ROOT / "artifacts" / "latest" / "dataset_all_no_trace" / "dataset_full.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "latest" / "roofline_op_type_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roofline analysis for single-op rows and aggregate the result by op_type.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset CSV. Defaults to artifacts/latest/dataset_all_no_trace/dataset_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for CSV/PNG/JSON artifacts.",
    )
    parser.add_argument(
        "--hardware-profile",
        default=str(HARDWARE_PROFILE_PATH),
        help="Hardware profile YAML used for Roofline ceilings.",
    )
    parser.add_argument(
        "--min-optype-count",
        type=int,
        default=20,
        help="Minimum row count per op_type or op_type+thread group for main plots.",
    )
    parser.add_argument(
        "--ridge-band-low",
        type=float,
        default=0.8,
        help="Lower bound for near-ridge classification.",
    )
    parser.add_argument(
        "--ridge-band-high",
        type=float,
        default=1.25,
        help="Upper bound for near-ridge classification.",
    )
    parser.add_argument(
        "--thread-values",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit thread values to plot. Defaults to all thread values observed in the data.",
    )
    return parser.parse_args()


def import_matplotlib_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    return plt, TwoSlopeNorm


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def classify_ridge_gap(gap: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(gap, dtype=float)
    return np.where(
        values < low,
        "memory_bound",
        np.where(values > high, "compute_bound", "near_ridge"),
    )


def classify_scalar(gap: float, low: float, high: float) -> str:
    if gap < low:
        return "memory_bound"
    if gap > high:
        return "compute_bound"
    return "near_ridge"


def positive_clip(series: pd.Series | np.ndarray, floor: float) -> np.ndarray:
    return np.clip(np.asarray(series, dtype=float), a_min=floor, a_max=None)


def ensure_roofline_inputs(frame: pd.DataFrame, hardware_profile: Path) -> pd.DataFrame:
    # Rebuild Roofline-related derived columns from the raw/base columns so we do not
    # collide with an input table that already contains an older set of feat_/ana_/hw_ fields.
    work = frame.drop(
        columns=[column for column in frame.columns if column.startswith(("feat_", "ana_", "hw_"))],
        errors="ignore",
    ).copy()
    work = add_engineered_features(work)
    hardware_features = load_hardware_features(hardware_profile)
    for key, value in hardware_features.items():
        work[key] = float(value)

    work = add_operator_hardware_context(work, hardware_features=hardware_features)
    work = add_analytical_hardware_software_features(work)
    return work


def build_row_level_roofline(
    frame: pd.DataFrame,
    ridge_band_low: float,
    ridge_band_high: float,
) -> pd.DataFrame:
    work = frame.copy()
    work["actual_us"] = numeric_series(work, TARGET_COLUMN, default=0.0).clip(lower=0.0)
    work["ana_compute_ops"] = numeric_series(work, "ana_compute_ops", default=0.0).clip(lower=0.0)
    work["feat_io_bytes_sum"] = numeric_series(work, "feat_io_bytes_sum", default=0.0).clip(lower=0.0)

    active_cores = numeric_series(
        work,
        "hw_core_active_cores",
        default=numeric_series(work, "num_threads", default=1.0).clip(lower=1.0),
    ).clip(lower=1.0)
    cpu_clock_ghz = numeric_series(work, "hw_core_cpu_clock", default=1.0).clip(lower=1e-6)
    throughput_fma = numeric_series(work, "hw_instruction_fp_throughput_per_cycle_vector_sp_fma", default=np.nan)
    throughput_mul = numeric_series(work, "hw_instruction_fp_throughput_per_cycle_vector_sp_mul", default=0.0)
    simd_width_bits = numeric_series(work, "hw_instruction_simd_width_bits", default=128.0).clip(lower=32.0)

    vector_sp_throughput = throughput_fma.where(throughput_fma > 0.0, throughput_mul).fillna(0.0).clip(lower=0.0)
    peak_fp32_ops_per_us = vector_sp_throughput * (simd_width_bits / 32.0) * 2.0 * cpu_clock_ghz * 1e3 * active_cores
    mem_bandwidth_bytes_per_us = numeric_series(work, "hw_memory_bandwidth_gb_s_total", default=0.0).clip(lower=1e-6) * 1e3

    arithmetic_intensity = work["ana_compute_ops"] / work["feat_io_bytes_sum"].clip(lower=1.0)
    achieved_perf = work["ana_compute_ops"] / work["actual_us"].clip(lower=1e-9)
    ridge_point = peak_fp32_ops_per_us / mem_bandwidth_bytes_per_us
    ridge_gap = arithmetic_intensity / ridge_point.clip(lower=1e-9)
    bound_label = classify_ridge_gap(ridge_gap, ridge_band_low, ridge_band_high)

    row_level = pd.DataFrame(
        {
            "row_uid": work["row_uid"] if "row_uid" in work.columns else pd.Series("", index=work.index),
            "case_id": work["case_id"] if "case_id" in work.columns else pd.Series("", index=work.index),
            "source_name": work["source_name"] if "source_name" in work.columns else pd.Series("", index=work.index),
            "combo": work["combo"] if "combo" in work.columns else pd.Series("", index=work.index),
            "op_type": work["op_type"].fillna("__missing__").astype(str),
            "num_threads": numeric_series(work, "num_threads", default=1.0).round().astype(int),
            "actual_us": work["actual_us"].astype(float),
            "ana_compute_ops": work["ana_compute_ops"].astype(float),
            "feat_io_bytes_sum": work["feat_io_bytes_sum"].astype(float),
            "peak_fp32_ops_per_us": peak_fp32_ops_per_us.astype(float),
            "mem_bandwidth_bytes_per_us": mem_bandwidth_bytes_per_us.astype(float),
            "ridge_point_ops_per_byte": ridge_point.astype(float),
            "arithmetic_intensity_ops_per_byte": arithmetic_intensity.astype(float),
            "achieved_perf_ops_per_us": achieved_perf.astype(float),
            "ridge_gap": ridge_gap.astype(float),
            "bound_label": pd.Series(bound_label, index=work.index, dtype="object"),
        }
    )
    return row_level


def share_dict(series: pd.Series, total: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in BOUND_ORDER:
        value = float(series.get(label, 0.0))
        out[label] = value / total if total > 0.0 else 0.0
    return out


def dominant_label_from_shares(shares: dict[str, float]) -> str:
    return max(BOUND_ORDER, key=lambda label: (shares.get(label, 0.0), -BOUND_ORDER.index(label)))


def summarize_op_type_thread(
    row_level: pd.DataFrame,
    ridge_band_low: float,
    ridge_band_high: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (op_type, num_threads), group in row_level.groupby(["op_type", "num_threads"], sort=True, dropna=False):
        row_count = int(len(group))
        total_actual_us = float(group["actual_us"].clip(lower=0.0).sum())
        total_compute_ops = float(group["ana_compute_ops"].clip(lower=0.0).sum())
        total_io_bytes = float(group["feat_io_bytes_sum"].clip(lower=0.0).sum())
        arithmetic_intensity = total_compute_ops / max(total_io_bytes, 1.0)
        achieved_perf = total_compute_ops / max(total_actual_us, 1e-9)
        ridge_point = float(pd.to_numeric(group["ridge_point_ops_per_byte"], errors="coerce").median())
        ridge_gap = arithmetic_intensity / max(ridge_point, 1e-9)
        aggregated_label = classify_scalar(ridge_gap, ridge_band_low, ridge_band_high)

        row_count_shares = share_dict(group["bound_label"].value_counts(), float(row_count))
        duration_shares = share_dict(group.groupby("bound_label")["actual_us"].sum(), total_actual_us)
        dominant_duration_label = dominant_label_from_shares(duration_shares)

        rows.append(
            {
                "op_type": str(op_type),
                "num_threads": int(num_threads),
                "row_count": row_count,
                "total_actual_us": total_actual_us,
                "total_compute_ops": total_compute_ops,
                "total_io_bytes": total_io_bytes,
                "arithmetic_intensity_ops_per_byte": arithmetic_intensity,
                "achieved_perf_ops_per_us": achieved_perf,
                "ridge_point_ops_per_byte": ridge_point,
                "ridge_gap": ridge_gap,
                "aggregated_bound_label": aggregated_label,
                "dominant_duration_bound_label": dominant_duration_label,
                "memory_bound_row_share": row_count_shares["memory_bound"],
                "near_ridge_row_share": row_count_shares["near_ridge"],
                "compute_bound_row_share": row_count_shares["compute_bound"],
                "memory_bound_duration_share": duration_shares["memory_bound"],
                "near_ridge_duration_share": duration_shares["near_ridge"],
                "compute_bound_duration_share": duration_shares["compute_bound"],
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["total_actual_us", "row_count", "op_type", "num_threads"], ascending=[False, False, True, True]).reset_index(drop=True)


def summarize_op_type(row_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for op_type, group in row_level.groupby("op_type", sort=True, dropna=False):
        row_count = int(len(group))
        total_actual_us = float(group["actual_us"].clip(lower=0.0).sum())
        total_compute_ops = float(group["ana_compute_ops"].clip(lower=0.0).sum())
        total_io_bytes = float(group["feat_io_bytes_sum"].clip(lower=0.0).sum())
        unique_threads = sorted(int(value) for value in pd.to_numeric(group["num_threads"], errors="coerce").dropna().astype(int).unique().tolist())

        row_count_shares = share_dict(group["bound_label"].value_counts(), float(row_count))
        duration_shares = share_dict(group.groupby("bound_label")["actual_us"].sum(), total_actual_us)
        headline_label = dominant_label_from_shares(duration_shares)

        rows.append(
            {
                "op_type": str(op_type),
                "row_count": row_count,
                "thread_count": len(unique_threads),
                "thread_values": ",".join(str(value) for value in unique_threads),
                "total_actual_us": total_actual_us,
                "total_compute_ops": total_compute_ops,
                "total_io_bytes": total_io_bytes,
                "headline_bound_label": headline_label,
                "memory_bound_row_share": row_count_shares["memory_bound"],
                "near_ridge_row_share": row_count_shares["near_ridge"],
                "compute_bound_row_share": row_count_shares["compute_bound"],
                "memory_bound_duration_share": duration_shares["memory_bound"],
                "near_ridge_duration_share": duration_shares["near_ridge"],
                "compute_bound_duration_share": duration_shares["compute_bound"],
                "duration_weighted_ridge_gap_mean": float(np.average(
                    positive_clip(group["ridge_gap"], floor=1e-9),
                    weights=positive_clip(group["actual_us"], floor=1e-9),
                )),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["total_actual_us", "row_count", "op_type"], ascending=[False, False, True]).reset_index(drop=True)


def resolve_thread_values(observed: list[int], requested: list[int] | None) -> list[int]:
    if not requested:
        return observed
    selected = [value for value in requested if value in observed]
    if not selected:
        raise RuntimeError(
            f"No requested thread values are present in the dataset. requested={requested}, observed={observed}"
        )
    return sorted(set(selected))


def safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [safe_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_json_value(payload), handle, indent=2, ensure_ascii=False)


def plot_roofline_by_threads(
    summary_df: pd.DataFrame,
    row_level: pd.DataFrame,
    output_png: Path,
    thread_values: list[int],
    min_optype_count: int,
) -> str | None:
    plt, _ = import_matplotlib_pyplot()
    plot_df = summary_df[summary_df["row_count"] >= int(min_optype_count)].copy()
    plot_df = plot_df[plot_df["num_threads"].isin(thread_values)].copy()
    if plot_df.empty:
        return None

    x_positive = positive_clip(plot_df["arithmetic_intensity_ops_per_byte"], floor=1e-9)
    y_positive = positive_clip(plot_df["achieved_perf_ops_per_us"], floor=1e-9)
    roof_positive = positive_clip(
        row_level[row_level["num_threads"].isin(thread_values)]["peak_fp32_ops_per_us"],
        floor=1e-9,
    )
    x_min = max(float(np.min(x_positive)) / 1.5, 1e-6)
    x_max = max(float(np.max(x_positive)) * 1.8, x_min * 10.0)
    y_min = max(float(np.min(y_positive)) / 1.5, 1e-3)
    y_max = max(
        float(np.max(y_positive)) * 2.0,
        float(np.max(roof_positive)) * 1.15,
        y_min * 10.0,
    )

    num_threads = len(thread_values)
    ncols = 2 if num_threads > 1 else 1
    nrows = math.ceil(num_threads / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    size_base = np.sqrt(positive_clip(plot_df["total_actual_us"], floor=1.0))
    size_scale = 60.0 + 420.0 * (size_base - size_base.min()) / max(float(size_base.max() - size_base.min()), 1.0)
    plot_df["marker_size"] = size_scale

    for axis, thread_value in zip(axes_flat, thread_values):
        thread_df = plot_df[plot_df["num_threads"] == thread_value].copy()
        if thread_df.empty:
            axis.axis("off")
            continue

        roof_ref = row_level[row_level["num_threads"] == thread_value].copy()
        peak_perf = float(pd.to_numeric(roof_ref["peak_fp32_ops_per_us"], errors="coerce").median())
        bandwidth = float(pd.to_numeric(roof_ref["mem_bandwidth_bytes_per_us"], errors="coerce").median())
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        y_roof = np.minimum(peak_perf, bandwidth * x_grid)

        cluster_x_threshold = max(x_min * 20.0, 1e-5)
        cluster_y_threshold = max(y_min * 25.0, 1e-2)
        cluster_mask = (
            pd.to_numeric(thread_df["arithmetic_intensity_ops_per_byte"], errors="coerce").fillna(0.0).le(cluster_x_threshold)
            & pd.to_numeric(thread_df["achieved_perf_ops_per_us"], errors="coerce").fillna(0.0).le(cluster_y_threshold)
        )
        cluster_df = thread_df[cluster_mask].copy()
        labeled_df = thread_df[~cluster_mask].copy()

        for _, row in labeled_df.iterrows():
            x_value = max(float(row["arithmetic_intensity_ops_per_byte"]), 1e-6)
            y_value = max(float(row["achieved_perf_ops_per_us"]), 1e-3)
            axis.scatter(
                x_value,
                y_value,
                s=float(row["marker_size"]),
                c=BOUND_COLORS[str(row["aggregated_bound_label"])],
                alpha=0.9,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            axis.annotate(
                str(row["op_type"]),
                xy=(x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8.5,
                zorder=4,
            )

        if not cluster_df.empty:
            cluster_labels = (
                cluster_df.sort_values(["total_actual_us", "op_type"], ascending=[False, True])["op_type"]
                .astype(str)
                .tolist()
            )
            wrapped_labels: list[str] = []
            chunk_size = 3
            for idx in range(0, len(cluster_labels), chunk_size):
                wrapped_labels.append(", ".join(cluster_labels[idx : idx + chunk_size]))
            cluster_text = "Low-intensity cluster:\n" + "\n".join(wrapped_labels)
            axis.text(
                0.04,
                0.09,
                cluster_text,
                transform=axis.transAxes,
                fontsize=7.8,
                va="bottom",
                ha="left",
                zorder=2,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "alpha": 0.88,
                    "edgecolor": "#7f8c8d",
                },
            )

        ridge_point = float(pd.to_numeric(thread_df["ridge_point_ops_per_byte"], errors="coerce").median())
        axis.axvline(ridge_point, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2)
        axis.plot(
            x_grid,
            y_roof,
            color="#2c3e50",
            linewidth=2.4,
            label="Roofline ceiling",
            zorder=5,
            solid_capstyle="round",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, which="both", linestyle="--", alpha=0.28)
        axis.set_title(
            f"Threads = {thread_value} | ridge = {ridge_point:.3g} ops/byte",
            fontsize=11,
        )
        axis.set_xlabel("Arithmetic Intensity (ops/byte)")
        axis.set_ylabel("Achieved Performance (ops/us)")

    for axis in axes_flat[num_threads:]:
        axis.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=9)
        for label, color in [
            ("memory_bound", BOUND_COLORS["memory_bound"]),
            ("near_ridge", BOUND_COLORS["near_ridge"]),
            ("compute_bound", BOUND_COLORS["compute_bound"]),
        ]
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    figure.suptitle("Roofline View By Thread Count", fontsize=14, y=1.02)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_png)


def plot_op_type_bound_share(op_type_summary: pd.DataFrame, output_png: Path, min_optype_count: int) -> str | None:
    plt, _ = import_matplotlib_pyplot()
    plot_df = op_type_summary[op_type_summary["row_count"] >= int(min_optype_count)].copy()
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values("total_actual_us", ascending=False).reset_index(drop=True)

    figure_height = max(4.6, 0.5 * len(plot_df) + 1.5)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    left = np.zeros(len(plot_df), dtype=float)
    y_positions = np.arange(len(plot_df))

    for label in BOUND_ORDER:
        column = f"{label}_duration_share"
        values = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        axis.barh(
            y_positions,
            values,
            left=left,
            color=BOUND_COLORS[label],
            alpha=0.92,
            label=label,
        )
        left += values

    axis.set_yticks(y_positions)
    axis.set_yticklabels(plot_df["op_type"].tolist(), fontsize=9.5)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Duration Share")
    axis.set_title("Per-op_type Bound Mix (duration-weighted)")
    axis.grid(True, axis="x", linestyle="--", alpha=0.28)
    axis.legend(frameon=False, ncol=3, loc="lower right")

    for index, row in plot_df.iterrows():
        axis.text(
            1.005,
            index,
            f"{row['headline_bound_label']} | n={int(row['row_count'])}",
            va="center",
            fontsize=8.5,
        )

    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_png)


def plot_ridge_gap_heatmap(
    summary_df: pd.DataFrame,
    op_type_summary: pd.DataFrame,
    output_png: Path,
    thread_values: list[int],
    min_optype_count: int,
    ridge_band_low: float,
    ridge_band_high: float,
) -> str | None:
    plt, TwoSlopeNorm = import_matplotlib_pyplot()
    plot_df = summary_df[summary_df["row_count"] >= int(min_optype_count)].copy()
    plot_df = plot_df[plot_df["num_threads"].isin(thread_values)].copy()
    if plot_df.empty:
        return None

    ordered_op_types = op_type_summary[op_type_summary["row_count"] >= int(min_optype_count)]["op_type"].tolist()
    if not ordered_op_types:
        ordered_op_types = op_type_summary["op_type"].tolist()
    heat_df = plot_df.pivot(index="op_type", columns="num_threads", values="ridge_gap")
    heat_df = heat_df.reindex(index=ordered_op_types, columns=thread_values)
    if heat_df.empty:
        return None

    clipped = np.clip(heat_df.to_numpy(dtype=float), a_min=1e-4, a_max=1e4)
    log_values = np.log10(clipped)
    masked = np.ma.masked_invalid(log_values)

    finite_values = masked.compressed()
    max_abs = max(float(np.max(np.abs(finite_values))) if finite_values.size else 1.0, 1.0)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    figure_width = max(7.0, 1.7 * len(thread_values) + 3.5)
    figure_height = max(5.5, 0.45 * len(heat_df.index) + 1.8)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(masked, aspect="auto", cmap="coolwarm", norm=norm)

    axis.set_xticks(np.arange(len(thread_values)))
    axis.set_xticklabels([str(value) for value in thread_values])
    axis.set_yticks(np.arange(len(heat_df.index)))
    axis.set_yticklabels(heat_df.index.tolist(), fontsize=9)
    axis.set_xlabel("num_threads")
    axis.set_title("Aggregated ridge_gap by op_type and thread count")

    for row_index, op_type in enumerate(heat_df.index):
        for col_index, thread_value in enumerate(thread_values):
            gap = heat_df.loc[op_type, thread_value]
            if pd.isna(gap):
                continue
            bound_label = classify_scalar(float(gap), low=ridge_band_low, high=ridge_band_high)
            text = f"{BOUND_SHORT_LABELS[bound_label]}\n{float(gap):.2g}"
            axis.text(col_index, row_index, text, ha="center", va="center", fontsize=7.5, color="black")

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("log10(ridge_gap)")
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_png)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    hardware_profile = Path(args.hardware_profile)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    if not hardware_profile.exists():
        raise FileNotFoundError(hardware_profile)

    frame = pd.read_csv(input_csv)
    if "op_type" not in frame.columns:
        raise RuntimeError(f"Input CSV is missing required column: op_type ({input_csv})")
    if "num_threads" not in frame.columns:
        raise RuntimeError(f"Input CSV is missing required column: num_threads ({input_csv})")
    if TARGET_COLUMN not in frame.columns:
        raise RuntimeError(f"Input CSV is missing required column: {TARGET_COLUMN} ({input_csv})")

    work = ensure_roofline_inputs(frame, hardware_profile)
    row_level = build_row_level_roofline(
        work,
        ridge_band_low=float(args.ridge_band_low),
        ridge_band_high=float(args.ridge_band_high),
    )
    observed_thread_values = sorted(int(value) for value in row_level["num_threads"].dropna().astype(int).unique().tolist())
    thread_values = resolve_thread_values(observed_thread_values, args.thread_values)

    op_type_thread_summary = summarize_op_type_thread(
        row_level=row_level,
        ridge_band_low=float(args.ridge_band_low),
        ridge_band_high=float(args.ridge_band_high),
    )
    op_type_summary = summarize_op_type(row_level)

    output_dir.mkdir(parents=True, exist_ok=True)
    row_level_csv = output_dir / "row_level_roofline.csv"
    op_type_thread_csv = output_dir / "op_type_thread_summary.csv"
    op_type_csv = output_dir / "op_type_summary.csv"
    row_level.to_csv(row_level_csv, index=False)
    op_type_thread_summary.to_csv(op_type_thread_csv, index=False)
    op_type_summary.to_csv(op_type_csv, index=False)

    roofline_png = plot_roofline_by_threads(
        summary_df=op_type_thread_summary,
        row_level=row_level,
        output_png=output_dir / "roofline_by_threads.png",
        thread_values=thread_values,
        min_optype_count=int(args.min_optype_count),
    )
    bound_share_png = plot_op_type_bound_share(
        op_type_summary=op_type_summary,
        output_png=output_dir / "op_type_bound_share.png",
        min_optype_count=int(args.min_optype_count),
    )
    heatmap_png = plot_ridge_gap_heatmap(
        summary_df=op_type_thread_summary,
        op_type_summary=op_type_summary,
        output_png=output_dir / "op_type_ridge_gap_heatmap.png",
        thread_values=thread_values,
        min_optype_count=int(args.min_optype_count),
        ridge_band_low=float(args.ridge_band_low),
        ridge_band_high=float(args.ridge_band_high),
    )

    total_rows = int(len(row_level))
    total_duration_us = float(row_level["actual_us"].clip(lower=0.0).sum())
    row_count_summary = row_level["bound_label"].value_counts()
    duration_summary = row_level.groupby("bound_label")["actual_us"].sum()

    summary_payload = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "hardware_profile": str(hardware_profile),
        "ridge_band": {
            "low": float(args.ridge_band_low),
            "high": float(args.ridge_band_high),
        },
        "min_optype_count": int(args.min_optype_count),
        "thread_values": thread_values,
        "total_rows": total_rows,
        "total_duration_us": total_duration_us,
        "op_type_count": int(op_type_summary["op_type"].nunique()) if not op_type_summary.empty else 0,
        "bound_row_counts": {label: int(row_count_summary.get(label, 0)) for label in BOUND_ORDER},
        "bound_row_share": share_dict(row_count_summary, float(total_rows)),
        "bound_duration_us": {label: float(duration_summary.get(label, 0.0)) for label in BOUND_ORDER},
        "bound_duration_share": share_dict(duration_summary, total_duration_us),
        "top_runtime_op_types": safe_json_value(
            op_type_summary[
                [
                    "op_type",
                    "row_count",
                    "total_actual_us",
                    "headline_bound_label",
                    "memory_bound_duration_share",
                    "near_ridge_duration_share",
                    "compute_bound_duration_share",
                ]
            ]
            .head(10)
            .to_dict(orient="records")
        ),
        "output_files": {
            "row_level_csv": str(row_level_csv),
            "op_type_thread_summary_csv": str(op_type_thread_csv),
            "op_type_summary_csv": str(op_type_csv),
            "roofline_by_threads_png": roofline_png,
            "op_type_bound_share_png": bound_share_png,
            "op_type_ridge_gap_heatmap_png": heatmap_png,
        },
    }
    summary_json = output_dir / "roofline_summary.json"
    write_json(summary_json, summary_payload)

    print(f"row_level_csv={row_level_csv}")
    print(f"op_type_thread_summary_csv={op_type_thread_csv}")
    print(f"op_type_summary_csv={op_type_csv}")
    if roofline_png:
        print(f"roofline_by_threads_png={roofline_png}")
    if bound_share_png:
        print(f"op_type_bound_share_png={bound_share_png}")
    if heatmap_png:
        print(f"op_type_ridge_gap_heatmap_png={heatmap_png}")
    print(f"roofline_summary_json={summary_json}")
    if not op_type_summary.empty:
        preview = op_type_summary[["op_type", "row_count", "total_actual_us", "headline_bound_label"]].head(10)
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
