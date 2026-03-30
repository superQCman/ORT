from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "latest"
    / "classed_op_mlp_test_analytical_5_200_iter"
)
DEFAULT_MODEL_GROUP = "gather"
DEFAULT_TARGET_COLUMN = "label_operator_actual_dur_us"
DEFAULT_FEATURE_COLUMNS = (
    "ana_calib_mem_us",
    "ana_calib_compute_us",
    "ana_calib_total_us",
)
DEFAULT_ALL_BREAKDOWN_COLS = ("op_type",)
DEFAULT_TEST_BREAKDOWN_COLS = ("op_type", "case_id", "combo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the correlation between analytical proxy features and true latency "
            "for an existing classed_op_mlp dataset group."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory that contains classed_op_mlp dataset artifacts.",
    )
    parser.add_argument(
        "--model-group",
        default=DEFAULT_MODEL_GROUP,
        help="Model group / dataset subdirectory under <data-root>/datasets/.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
        help="Analytical feature columns to compare against the target latency.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COLUMN,
        help="Target latency column.",
    )
    parser.add_argument(
        "--all-breakdown-cols",
        nargs="+",
        default=list(DEFAULT_ALL_BREAKDOWN_COLS),
        help="Grouping columns for all-split breakdown CSVs.",
    )
    parser.add_argument(
        "--test-breakdown-cols",
        nargs="+",
        default=list(DEFAULT_TEST_BREAKDOWN_COLS),
        help="Grouping columns for test-split breakdown CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Optional output directory. Defaults to "
            "<data-root>/analysis/analytical_feature_correlation/<model-group>."
        ),
    )
    return parser.parse_args()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) <= 1:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_stats(frame: pd.DataFrame, feature_col: str, target_col: str) -> dict[str, Any]:
    clean = frame.copy()
    clean[feature_col] = _safe_numeric(clean[feature_col])
    clean[target_col] = _safe_numeric(clean[target_col])
    clean = clean.dropna(subset=[feature_col, target_col]).copy()
    if clean.empty:
        return {
            "rows": 0,
            "pearson_r": float("nan"),
            "spearman_rho": float("nan"),
            "linear_fit_slope_y_on_x": float("nan"),
            "linear_fit_intercept": float("nan"),
            "mean_actual_us": float("nan"),
            "mean_feature_us": float("nan"),
            "median_actual_us": float("nan"),
            "median_feature_us": float("nan"),
            "mape_vs_actual": float("nan"),
            "dwre_vs_actual": float("nan"),
        }

    x = clean[feature_col].to_numpy(dtype=float)
    y = clean[target_col].to_numpy(dtype=float)
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    slope, intercept = _linear_fit(x, y)
    denominator = np.clip(y, a_min=1e-9, a_max=None)
    abs_err = np.abs(x - y)
    return {
        "rows": int(len(clean)),
        "pearson_r": _safe_corr(x, y),
        "spearman_rho": _safe_corr(x_rank, y_rank),
        "linear_fit_slope_y_on_x": slope,
        "linear_fit_intercept": intercept,
        "mean_actual_us": float(np.mean(y)),
        "mean_feature_us": float(np.mean(x)),
        "median_actual_us": float(np.median(y)),
        "median_feature_us": float(np.median(x)),
        "mape_vs_actual": float(np.mean(abs_err / denominator)),
        "dwre_vs_actual": float(np.sum(abs_err) / np.sum(denominator)),
    }


def load_split_frames(group_dir: Path, split_names: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name in split_names:
        split_csv = group_dir / f"{split_name}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(split_csv)
        frame = pd.read_csv(split_csv, low_memory=False)
        frame["split"] = split_name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["row_uid"] = merged["row_uid"].astype(str)
    return merged


def split_summary(frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        split_df = frame[frame["split"] == split_name].copy()
        for feature_col in feature_cols:
            row = {"split": split_name, "feature": feature_col}
            row.update(compute_stats(split_df, feature_col, target_col))
            rows.append(row)
    for feature_col in feature_cols:
        row = {"split": "all", "feature": feature_col}
        row.update(compute_stats(frame, feature_col, target_col))
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_summary(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    if group_col not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_col, sort=True, dropna=False):
        for feature_col in feature_cols:
            row = {group_col: group_value, "feature": feature_col}
            row.update(compute_stats(group, feature_col, target_col))
            rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(
    model_group: str,
    target_col: str,
    split_df: pd.DataFrame,
    all_grouped: dict[str, pd.DataFrame],
    test_grouped: dict[str, pd.DataFrame],
) -> str:
    lines: list[str] = []
    lines.append(f"# Analytical Feature Correlation: {model_group}")
    lines.append("")
    lines.append(f"- Target column: `{target_col}`")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| split | feature | rows | Pearson r | Spearman rho | DWRE | MAPE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in split_df.iterrows():
        lines.append(
            f"| `{row['split']}` | `{row['feature']}` | {int(row['rows'])} | "
            f"{row['pearson_r']:.6f} | {row['spearman_rho']:.6f} | "
            f"{row['dwre_vs_actual'] * 100.0:.2f}% | {row['mape_vs_actual'] * 100.0:.2f}% |"
        )
    lines.append("")

    for group_col, grouped_df in all_grouped.items():
        if grouped_df.empty:
            continue
        lines.append(f"## All By {group_col}")
        lines.append("")
        lines.append(f"| {group_col} | feature | rows | Pearson r | Spearman rho | DWRE |")
        lines.append(f"| --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in grouped_df.iterrows():
            lines.append(
                f"| `{row[group_col]}` | `{row['feature']}` | {int(row['rows'])} | "
                f"{row['pearson_r']:.6f} | {row['spearman_rho']:.6f} | {row['dwre_vs_actual'] * 100.0:.2f}% |"
            )
        lines.append("")

    for group_col, grouped_df in test_grouped.items():
        if grouped_df.empty:
            continue
        lines.append(f"## Test By {group_col}")
        lines.append("")
        lines.append(f"| {group_col} | feature | rows | Pearson r | Spearman rho | DWRE |")
        lines.append(f"| --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in grouped_df.iterrows():
            lines.append(
                f"| `{row[group_col]}` | `{row['feature']}` | {int(row['rows'])} | "
                f"{row['pearson_r']:.6f} | {row['spearman_rho']:.6f} | {row['dwre_vs_actual'] * 100.0:.2f}% |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze(
    data_root: Path,
    model_group: str,
    feature_cols: list[str],
    target_col: str,
    output_dir: Path,
    all_breakdown_cols: list[str],
    test_breakdown_cols: list[str],
) -> dict[str, Any]:
    group_dir = data_root / "datasets" / model_group
    if not group_dir.exists():
        raise FileNotFoundError(group_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_split_frames(group_dir, ["train", "val", "test"])
    split_df = split_summary(frame, feature_cols, target_col)
    split_csv = output_dir / "split_summary.csv"
    split_df.to_csv(split_csv, index=False)

    all_grouped_outputs: dict[str, str] = {}
    all_grouped_frames: dict[str, pd.DataFrame] = {}
    for group_col in all_breakdown_cols:
        grouped_df = grouped_summary(frame, feature_cols, target_col, group_col)
        all_grouped_frames[group_col] = grouped_df
        if not grouped_df.empty:
            path = output_dir / f"all_by_{group_col}.csv"
            grouped_df.to_csv(path, index=False)
            all_grouped_outputs[group_col] = str(path)

    test_frame = frame[frame["split"] == "test"].copy()
    test_grouped_outputs: dict[str, str] = {}
    test_grouped_frames: dict[str, pd.DataFrame] = {}
    for group_col in test_breakdown_cols:
        grouped_df = grouped_summary(test_frame, feature_cols, target_col, group_col)
        test_grouped_frames[group_col] = grouped_df
        if not grouped_df.empty:
            path = output_dir / f"test_by_{group_col}.csv"
            grouped_df.to_csv(path, index=False)
            test_grouped_outputs[group_col] = str(path)

    summary_json_payload = {
        "data_root": str(data_root),
        "model_group": model_group,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "split_summary_csv": str(split_csv),
        "all_breakdown_csvs": all_grouped_outputs,
        "test_breakdown_csvs": test_grouped_outputs,
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        render_markdown(model_group, target_col, split_df, all_grouped_frames, test_grouped_frames),
        encoding="utf-8",
    )
    return {
        "split_summary_csv": str(split_csv),
        "all_breakdown_csvs": all_grouped_outputs,
        "test_breakdown_csvs": test_grouped_outputs,
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else data_root / "analysis" / "analytical_feature_correlation" / args.model_group
    )
    payload = analyze(
        data_root=data_root,
        model_group=args.model_group,
        feature_cols=list(args.feature_cols),
        target_col=args.target_col,
        output_dir=output_dir,
        all_breakdown_cols=list(args.all_breakdown_cols),
        test_breakdown_cols=list(args.test_breakdown_cols),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
