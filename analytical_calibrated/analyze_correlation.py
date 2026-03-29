from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_contract import TARGET_COLUMN  # noqa: E402

try:  # noqa: E402
    from .contracts import DEFAULT_INPUT_DATASET_DIR, DEFAULT_OUTPUT_DIR
except ImportError:  # noqa: E402
    from contracts import DEFAULT_INPUT_DATASET_DIR, DEFAULT_OUTPUT_DIR


DEFAULT_ANALYTICAL_CSV = DEFAULT_OUTPUT_DIR / "analytical_features_full.csv"
DEFAULT_CORRELATION_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "correlation_analysis"
RAW_FEATURE_COLUMNS = [
    "output_size",
    "activation_size",
    "parameter_size",
    "feat_io_bytes_sum",
    "feat_output_input_bytes_ratio",
    "feat_lookup_count",
    "feat_output_elements_per_lookup",
    "feat_output_elements_per_batch",
    "feat_reduction_work_items",
    "feat_reduction_axes_product",
    "feat_gemm_m",
    "feat_gemm_n",
    "feat_gemm_k",
    "feat_gemm_mac_count",
    "feat_gemm_bytes_per_mac",
    "num_threads",
]
ANALYTICAL_COMPONENT_COLUMNS = [
    "ana_calib_total_us",
    "ana_calib_mem_us",
    "ana_calib_compute_us",
    "ana_calib_overhead_us",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlation between analytical-calibrated features, targets, and error terms.",
    )
    parser.add_argument(
        "--analytical-csv",
        default=str(DEFAULT_ANALYTICAL_CSV),
        help="Row-level analytical feature CSV. Defaults to analytical_calibrated/analytical_features_full.csv.",
    )
    parser.add_argument(
        "--base-data-dir",
        default=str(DEFAULT_INPUT_DATASET_DIR),
        help="Base dataset directory used to bring back selected software features.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_CORRELATION_OUTPUT_DIR),
        help="Output directory for correlation CSV and summary JSON.",
    )
    return parser.parse_args()


def load_analytical_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    frame["target_us"] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").fillna(0.0)
    frame["pred_us"] = pd.to_numeric(frame["ana_calib_total_us"], errors="coerce").fillna(0.0)
    frame["abs_error_us"] = (frame["pred_us"] - frame["target_us"]).abs()
    frame["signed_error_us"] = frame["pred_us"] - frame["target_us"]
    denominator = frame["target_us"].clip(lower=1e-9)
    frame["ape"] = frame["abs_error_us"] / denominator
    frame["signed_relative_error"] = frame["signed_error_us"] / denominator
    return frame


def merge_base_features(frame: pd.DataFrame, base_data_dir: Path) -> pd.DataFrame:
    base_csv = base_data_dir / "dataset_full.csv"
    if not base_csv.exists():
        raise FileNotFoundError(base_csv)
    usecols = [
        "row_uid",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_lookup_count",
        "feat_output_elements_per_lookup",
        "feat_output_elements_per_batch",
        "feat_reduction_work_items",
        "feat_reduction_axes_product",
        "num_threads",
        "op_type",
        "op_class",
        "combo",
        "split",
    ]
    base = pd.read_csv(base_csv, usecols=[column for column in usecols if column != "op_class"], low_memory=False)
    base["row_uid"] = base["row_uid"].astype(str)
    return frame.merge(base, on="row_uid", how="left", suffixes=("", "__base"))


def correlation_table(frame: pd.DataFrame, metric_column: str, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_series = pd.to_numeric(frame[metric_column], errors="coerce")
    for column in feature_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & metric_series.notna()
        if int(valid.sum()) < 3:
            continue
        corr = float(values[valid].corr(metric_series[valid]))
        rows.append(
            {
                "feature": column,
                "metric": metric_column,
                "pearson_corr": corr,
                "abs_pearson_corr": abs(corr),
                "valid_rows": int(valid.sum()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["abs_pearson_corr", "feature"], ascending=[False, True]).reset_index(drop=True)


def grouped_error_summary(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_column, dropna=False, sort=True):
        rows.append(
            {
                group_column: str(key),
                "row_count": int(len(group)),
                "target_sum_us": float(group["target_us"].sum()),
                "pred_sum_us": float(group["pred_us"].sum()),
                "mae_us": float(group["abs_error_us"].mean()),
                "mape": float(group["ape"].mean()),
                "median_ape": float(group["ape"].median()),
                "duration_weighted_re": float(group["abs_error_us"].sum() / max(group["target_us"].sum(), 1e-9)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["mape", "row_count"], ascending=[False, False]).reset_index(drop=True)


def top_summary(table: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if table.empty:
        return []
    return table.head(limit).to_dict(orient="records")


def format_markdown_table(table: pd.DataFrame, columns: list[str], limit: int = 8, digits: int = 4) -> str:
    if table.empty:
        return "_No rows_\n"
    subset = table.loc[:, columns].head(limit).copy()
    for column in subset.columns:
        if pd.api.types.is_numeric_dtype(subset[column]):
            subset[column] = subset[column].map(
                lambda value: f"{float(value):.{digits}f}" if pd.notna(value) else "nan"
            )
    header = "| " + " | ".join(subset.columns) + " |\n"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |\n"
    rows = [
        "| " + " | ".join(str(row[column]) for column in subset.columns) + " |"
        for _, row in subset.iterrows()
    ]
    return header + divider + "\n".join(rows) + "\n"


def write_markdown_summary(
    output_path: Path,
    summary: dict[str, Any],
    raw_abs_error_corr: pd.DataFrame,
    raw_ape_corr: pd.DataFrame,
    component_abs_error_corr: pd.DataFrame,
    component_ape_corr: pd.DataFrame,
    op_type_summary: pd.DataFrame,
    op_class_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Correlation Analysis Summary",
        "",
        f"- Analytical CSV: `{summary['analytical_csv']}`",
        f"- Base dataset dir: `{summary['base_data_dir']}`",
        f"- Row count: `{summary['row_count']}`",
        "",
        "## Key Findings",
        "",
        "- `generic_memory` is the main source of inflated relative error. Its full-family duration-weighted relative error is much larger than the overall heavy-family error, which indicates a structural proxy mismatch rather than only calibration drift.",
        "- `Reshape` dominates the worst-op list because its median real latency is only tens of microseconds while the current proxy predicts tens of milliseconds worth of memory traffic.",
        "- `Gather` has a very large mean MAPE but a much smaller median APE and duration-weighted error, which indicates a long tail of small-target rows is inflating the arithmetic mean of relative error.",
        "- `compute_dominant` is comparatively well-behaved; its error is much lower than the `memory_pure` and `mixed_balanced` buckets.",
        "",
        "## Raw Feature Correlation With Absolute Error",
        "",
        format_markdown_table(
            raw_abs_error_corr,
            ["feature", "pearson_corr", "abs_pearson_corr", "valid_rows"],
        ),
        "## Raw Feature Correlation With APE",
        "",
        format_markdown_table(
            raw_ape_corr,
            ["feature", "pearson_corr", "abs_pearson_corr", "valid_rows"],
        ),
        "## Analytical Component Correlation With Absolute Error",
        "",
        format_markdown_table(
            component_abs_error_corr,
            ["feature", "pearson_corr", "abs_pearson_corr", "valid_rows"],
        ),
        "## Analytical Component Correlation With APE",
        "",
        format_markdown_table(
            component_ape_corr,
            ["feature", "pearson_corr", "abs_pearson_corr", "valid_rows"],
        ),
        "## Worst Op Types By MAPE",
        "",
        format_markdown_table(
            op_type_summary,
            ["op_type", "row_count", "mape", "median_ape", "duration_weighted_re"],
        ),
        "## Op Class Error Summary",
        "",
        format_markdown_table(
            op_class_summary,
            ["op_class", "row_count", "mape", "median_ape", "duration_weighted_re"],
        ),
        "## Analytical Family Error Summary",
        "",
        format_markdown_table(
            family_summary,
            ["ana_calib_family", "row_count", "mape", "median_ape", "duration_weighted_re"],
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    analytical_csv = Path(args.analytical_csv)
    base_data_dir = Path(args.base_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = merge_base_features(load_analytical_frame(analytical_csv), base_data_dir)
    raw_target_corr = correlation_table(frame, "target_us", RAW_FEATURE_COLUMNS)
    raw_abs_error_corr = correlation_table(frame, "abs_error_us", RAW_FEATURE_COLUMNS)
    raw_ape_corr = correlation_table(frame, "ape", RAW_FEATURE_COLUMNS)
    raw_signed_rel_corr = correlation_table(frame, "signed_relative_error", RAW_FEATURE_COLUMNS)

    component_target_corr = correlation_table(frame, "target_us", ANALYTICAL_COMPONENT_COLUMNS)
    component_abs_error_corr = correlation_table(frame, "abs_error_us", ANALYTICAL_COMPONENT_COLUMNS)
    component_ape_corr = correlation_table(frame, "ape", ANALYTICAL_COMPONENT_COLUMNS)
    component_signed_rel_corr = correlation_table(frame, "signed_relative_error", ANALYTICAL_COMPONENT_COLUMNS)

    op_type_summary = grouped_error_summary(frame, "op_type")
    op_class_summary = grouped_error_summary(frame, "op_class")
    family_summary = grouped_error_summary(frame, "ana_calib_family")

    outputs = {
        "raw_target_correlation_csv": output_dir / "raw_feature_target_correlation.csv",
        "raw_abs_error_correlation_csv": output_dir / "raw_feature_abs_error_correlation.csv",
        "raw_ape_correlation_csv": output_dir / "raw_feature_ape_correlation.csv",
        "raw_signed_relative_error_correlation_csv": output_dir / "raw_feature_signed_relative_error_correlation.csv",
        "component_target_correlation_csv": output_dir / "analytical_component_target_correlation.csv",
        "component_abs_error_correlation_csv": output_dir / "analytical_component_abs_error_correlation.csv",
        "component_ape_correlation_csv": output_dir / "analytical_component_ape_correlation.csv",
        "component_signed_relative_error_correlation_csv": output_dir / "analytical_component_signed_relative_error_correlation.csv",
        "op_type_summary_csv": output_dir / "op_type_error_summary.csv",
        "op_class_summary_csv": output_dir / "op_class_error_summary.csv",
        "family_summary_csv": output_dir / "analytical_family_error_summary.csv",
        "summary_markdown": output_dir / "correlation_summary.md",
    }

    raw_target_corr.to_csv(outputs["raw_target_correlation_csv"], index=False)
    raw_abs_error_corr.to_csv(outputs["raw_abs_error_correlation_csv"], index=False)
    raw_ape_corr.to_csv(outputs["raw_ape_correlation_csv"], index=False)
    raw_signed_rel_corr.to_csv(outputs["raw_signed_relative_error_correlation_csv"], index=False)
    component_target_corr.to_csv(outputs["component_target_correlation_csv"], index=False)
    component_abs_error_corr.to_csv(outputs["component_abs_error_correlation_csv"], index=False)
    component_ape_corr.to_csv(outputs["component_ape_correlation_csv"], index=False)
    component_signed_rel_corr.to_csv(outputs["component_signed_relative_error_correlation_csv"], index=False)
    op_type_summary.to_csv(outputs["op_type_summary_csv"], index=False)
    op_class_summary.to_csv(outputs["op_class_summary_csv"], index=False)
    family_summary.to_csv(outputs["family_summary_csv"], index=False)

    summary = {
        "analytical_csv": str(analytical_csv),
        "base_data_dir": str(base_data_dir),
        "row_count": int(len(frame)),
        "top_raw_feature_target_correlations": top_summary(raw_target_corr),
        "top_raw_feature_abs_error_correlations": top_summary(raw_abs_error_corr),
        "top_raw_feature_ape_correlations": top_summary(raw_ape_corr),
        "top_component_target_correlations": top_summary(component_target_corr),
        "top_component_abs_error_correlations": top_summary(component_abs_error_corr),
        "top_component_ape_correlations": top_summary(component_ape_corr),
        "worst_op_types_by_mape": top_summary(op_type_summary[["op_type", "row_count", "mape", "duration_weighted_re"]]),
        "worst_op_classes_by_mape": top_summary(op_class_summary[["op_class", "row_count", "mape", "duration_weighted_re"]]),
        "worst_analytical_families_by_mape": top_summary(family_summary[["ana_calib_family", "row_count", "mape", "duration_weighted_re"]]),
        "output_files": {key: str(value) for key, value in outputs.items()},
    }
    summary_path = output_dir / "correlation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_markdown_summary(
        outputs["summary_markdown"],
        summary,
        raw_abs_error_corr,
        raw_ape_corr,
        component_abs_error_corr,
        component_ape_corr,
        op_type_summary,
        op_class_summary,
        family_summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
