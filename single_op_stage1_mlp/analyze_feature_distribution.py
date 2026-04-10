from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-group feature distributions and flag concentrated features "
            "for the single-op MLP dataset."
        ),
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input dataset CSV, typically dataset_full.csv.",
    )
    parser.add_argument(
        "--feature-manifest",
        required=True,
        help="feature_columns.json emitted by dataset_builder.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--group-column",
        default="op_type",
        help="Column used to group rows before computing per-feature distributions.",
    )
    parser.add_argument(
        "--low-cardinality-threshold",
        type=int,
        default=3,
        help="Numeric features with <= this many unique values are flagged as concentrated.",
    )
    parser.add_argument(
        "--dominant-share-threshold",
        type=float,
        default=0.95,
        help="Features with a top-1 value share >= this threshold are flagged as concentrated.",
    )
    parser.add_argument(
        "--iqr-to-range-threshold",
        type=float,
        default=0.05,
        help="Numeric features with IQR/range <= this threshold are flagged as concentrated.",
    )
    parser.add_argument(
        "--summary-top-k",
        type=int,
        default=8,
        help="Maximum concentrated features shown per group in the Markdown summary.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def numeric_feature_stats(
    series: pd.Series,
    group_value: str,
    feature: str,
    low_cardinality_threshold: int,
    dominant_share_threshold: float,
    iqr_to_range_threshold: float,
) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    count = int(len(numeric))
    if count == 0:
        return {
            "group_value": group_value,
            "feature": feature,
            "kind": "numeric",
            "count": 0,
            "missing_count": int(series.isna().sum()),
            "unique_count": 0,
            "unique_ratio": 0.0,
            "top1_value": None,
            "top1_share": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "iqr_to_range": np.nan,
            "is_constant": False,
            "is_low_cardinality": False,
            "has_dominant_value": False,
            "has_narrow_iqr": False,
            "is_highly_concentrated": False,
        }

    values = numeric.to_numpy(dtype=float)
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    min_value = float(values.min())
    max_value = float(values.max())
    data_range = float(max_value - min_value)
    iqr = float(quantiles[4] - quantiles[2])
    iqr_to_range = float(iqr / data_range) if data_range > 0 else 0.0

    value_counts = numeric.value_counts(dropna=False)
    unique_count = int(numeric.nunique(dropna=True))
    top1_share = float(value_counts.iloc[0] / count)

    is_constant = unique_count == 1
    is_low_cardinality = unique_count <= low_cardinality_threshold
    has_dominant_value = top1_share >= dominant_share_threshold
    has_narrow_iqr = data_range > 0 and iqr_to_range <= iqr_to_range_threshold

    return {
        "group_value": group_value,
        "feature": feature,
        "kind": "numeric",
        "count": count,
        "missing_count": int(series.isna().sum()),
        "unique_count": unique_count,
        "unique_ratio": float(unique_count / count),
        "top1_value": clean_scalar(value_counts.index[0]),
        "top1_share": top1_share,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": min_value,
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": max_value,
        "range": data_range,
        "iqr": iqr,
        "iqr_to_range": iqr_to_range,
        "is_constant": is_constant,
        "is_low_cardinality": is_low_cardinality,
        "has_dominant_value": has_dominant_value,
        "has_narrow_iqr": has_narrow_iqr,
        "is_highly_concentrated": bool(
            is_constant or is_low_cardinality or has_dominant_value or has_narrow_iqr
        ),
    }


def categorical_feature_stats(series: pd.Series, group_value: str, feature: str) -> dict[str, Any]:
    categorical = series.fillna("__nan__").astype(str)
    count = int(len(categorical))
    value_counts = categorical.value_counts(dropna=False)
    unique_count = int(categorical.nunique(dropna=True))
    top1_share = float(value_counts.iloc[0] / count) if count else 0.0
    is_constant = unique_count == 1
    has_dominant_value = top1_share >= 0.95
    is_low_cardinality = unique_count <= 2

    return {
        "group_value": group_value,
        "feature": feature,
        "kind": "categorical",
        "count": count,
        "missing_count": int(series.isna().sum()),
        "unique_count": unique_count,
        "unique_ratio": float(unique_count / count) if count else 0.0,
        "top1_value": clean_scalar(value_counts.index[0]) if count else None,
        "top1_share": top1_share,
        "is_constant": is_constant,
        "is_low_cardinality": is_low_cardinality,
        "has_dominant_value": has_dominant_value,
        "is_highly_concentrated": bool(is_constant or is_low_cardinality or has_dominant_value),
    }


def summarize_group_counts(
    numeric_stats: pd.DataFrame,
    categorical_stats: pd.DataFrame,
    group_sizes: pd.Series,
) -> pd.DataFrame:
    numeric_summary = (
        numeric_stats.groupby("group_value")
        .agg(
            numeric_feature_count=("feature", "count"),
            concentrated_numeric_feature_count=("is_highly_concentrated", "sum"),
            constant_numeric_feature_count=("is_constant", "sum"),
        )
        .reset_index()
    )
    categorical_summary = (
        categorical_stats.groupby("group_value")
        .agg(
            categorical_feature_count=("feature", "count"),
            concentrated_categorical_feature_count=("is_highly_concentrated", "sum"),
            constant_categorical_feature_count=("is_constant", "sum"),
        )
        .reset_index()
    )
    merged = numeric_summary.merge(categorical_summary, on="group_value", how="outer")
    merged["group_row_count"] = merged["group_value"].map(group_sizes.to_dict()).astype(int)
    merged["concentrated_numeric_fraction"] = (
        merged["concentrated_numeric_feature_count"] / merged["numeric_feature_count"]
    )
    merged["concentrated_categorical_fraction"] = (
        merged["concentrated_categorical_feature_count"] / merged["categorical_feature_count"]
    )
    return merged.sort_values(
        ["concentrated_numeric_feature_count", "constant_numeric_feature_count", "group_value"],
        ascending=[False, False, True],
    )


def render_markdown_summary(
    dataset: pd.DataFrame,
    manifest: dict[str, Any],
    group_column: str,
    thresholds: dict[str, Any],
    group_summary: pd.DataFrame,
    concentrated_numeric: pd.DataFrame,
    concentrated_categorical: pd.DataFrame,
    top_k: int,
) -> str:
    lines: list[str] = []
    lines.append("# Feature Distribution Summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- input_rows: {len(dataset)}")
    lines.append(f"- group_column: {group_column}")
    lines.append(f"- group_count: {dataset[group_column].nunique(dropna=True)}")
    lines.append(f"- numeric_feature_count: {len(manifest.get('numeric_features', []))}")
    lines.append(f"- categorical_feature_count: {len(manifest.get('categorical_features', []))}")
    lines.append("")
    lines.append("## Concentration Rules")
    lines.append("")
    lines.append(f"- numeric_low_cardinality_threshold: <= {thresholds['low_cardinality_threshold']} unique values")
    lines.append(f"- numeric_dominant_share_threshold: top1_share >= {thresholds['dominant_share_threshold']:.2f}")
    lines.append(f"- numeric_iqr_to_range_threshold: IQR/range <= {thresholds['iqr_to_range_threshold']:.2f}")
    lines.append("- categorical_low_cardinality_threshold: <= 2 unique values")
    lines.append("- categorical_dominant_share_threshold: top1_share >= 0.95")
    lines.append("")
    lines.append("## Per-Group Overview")
    lines.append("")
    lines.append(
        "| group | rows | concentrated_numeric / total | constant_numeric | concentrated_categorical / total |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in group_summary.itertuples(index=False):
        lines.append(
            f"| {row.group_value} | {row.group_row_count} | "
            f"{row.concentrated_numeric_feature_count} / {row.numeric_feature_count} | "
            f"{row.constant_numeric_feature_count} | "
            f"{row.concentrated_categorical_feature_count} / {row.categorical_feature_count} |"
        )

    for group_value in group_summary["group_value"].tolist():
        lines.append("")
        lines.append(f"## {group_value}")
        lines.append("")
        numeric_rows = concentrated_numeric[concentrated_numeric["group_value"] == group_value].head(top_k)
        if numeric_rows.empty:
            lines.append("- No concentrated numeric features matched the current thresholds.")
        else:
            lines.append("- Top concentrated numeric features:")
            for row in numeric_rows.itertuples(index=False):
                lines.append(
                    "  - "
                    f"{row.feature}: unique={row.unique_count}, top1_share={row.top1_share:.3f}, "
                    f"p05={row.p05:.6g}, p50={row.p50:.6g}, p95={row.p95:.6g}, "
                    f"iqr_to_range={row.iqr_to_range:.6g}"
                )
        categorical_rows = concentrated_categorical[
            concentrated_categorical["group_value"] == group_value
        ].head(max(1, min(top_k, 4)))
        if categorical_rows.empty:
            lines.append("- No concentrated categorical features matched the current thresholds.")
        else:
            lines.append("- Top concentrated categorical features:")
            for row in categorical_rows.itertuples(index=False):
                lines.append(
                    "  - "
                    f"{row.feature}: unique={row.unique_count}, top1_value={row.top1_value}, "
                    f"top1_share={row.top1_share:.3f}"
                )

    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    feature_manifest = Path(args.feature_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(input_csv)
    manifest = load_manifest(feature_manifest)

    if args.group_column not in dataset.columns:
        raise RuntimeError(f"Group column {args.group_column!r} was not found in {input_csv}")

    numeric_features = [column for column in manifest.get("numeric_features", []) if column in dataset.columns]
    categorical_features = [
        column
        for column in manifest.get("categorical_features", [])
        if column in dataset.columns and column != args.group_column
    ]

    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    grouped = dataset.groupby(args.group_column, sort=True, dropna=False)
    for group_value_raw, group_df in grouped:
        group_value = str(group_value_raw)
        for feature in numeric_features:
            numeric_rows.append(
                numeric_feature_stats(
                    series=group_df[feature],
                    group_value=group_value,
                    feature=feature,
                    low_cardinality_threshold=max(1, int(args.low_cardinality_threshold)),
                    dominant_share_threshold=float(args.dominant_share_threshold),
                    iqr_to_range_threshold=float(args.iqr_to_range_threshold),
                )
            )
        for feature in categorical_features:
            categorical_rows.append(
                categorical_feature_stats(
                    series=group_df[feature],
                    group_value=group_value,
                    feature=feature,
                )
            )

    numeric_stats = pd.DataFrame(numeric_rows)
    categorical_stats = pd.DataFrame(categorical_rows)
    group_sizes = grouped.size()
    group_summary = summarize_group_counts(
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        group_sizes=group_sizes,
    )

    concentrated_numeric = numeric_stats[numeric_stats["is_highly_concentrated"]].copy()
    concentrated_numeric = concentrated_numeric.sort_values(
        [
            "group_value",
            "is_constant",
            "top1_share",
            "unique_count",
            "iqr_to_range",
            "feature",
        ],
        ascending=[True, False, False, True, True, True],
    )
    concentrated_categorical = categorical_stats[categorical_stats["is_highly_concentrated"]].copy()
    concentrated_categorical = concentrated_categorical.sort_values(
        ["group_value", "is_constant", "top1_share", "unique_count", "feature"],
        ascending=[True, False, False, True, True],
    )

    numeric_stats_path = output_dir / "numeric_feature_distribution_by_group.csv"
    categorical_stats_path = output_dir / "categorical_feature_distribution_by_group.csv"
    group_summary_path = output_dir / "group_concentration_summary.csv"
    concentrated_numeric_path = output_dir / "highly_concentrated_numeric_features.csv"
    concentrated_categorical_path = output_dir / "highly_concentrated_categorical_features.csv"
    summary_json_path = output_dir / "distribution_summary.json"
    summary_md_path = output_dir / "distribution_summary.md"

    numeric_stats.to_csv(numeric_stats_path, index=False)
    categorical_stats.to_csv(categorical_stats_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    concentrated_numeric.to_csv(concentrated_numeric_path, index=False)
    concentrated_categorical.to_csv(concentrated_categorical_path, index=False)

    thresholds = {
        "low_cardinality_threshold": int(args.low_cardinality_threshold),
        "dominant_share_threshold": float(args.dominant_share_threshold),
        "iqr_to_range_threshold": float(args.iqr_to_range_threshold),
    }
    summary_payload = {
        "input_csv": str(input_csv),
        "feature_manifest": str(feature_manifest),
        "group_column": args.group_column,
        "row_count": int(len(dataset)),
        "group_count": int(dataset[args.group_column].nunique(dropna=True)),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "thresholds": thresholds,
        "artifacts": {
            "numeric_feature_distribution_csv": str(numeric_stats_path),
            "categorical_feature_distribution_csv": str(categorical_stats_path),
            "group_concentration_summary_csv": str(group_summary_path),
            "highly_concentrated_numeric_features_csv": str(concentrated_numeric_path),
            "highly_concentrated_categorical_features_csv": str(concentrated_categorical_path),
            "summary_markdown": str(summary_md_path),
        },
    }
    write_json(summary_json_path, summary_payload)
    summary_md_path.write_text(
        render_markdown_summary(
            dataset=dataset,
            manifest=manifest,
            group_column=args.group_column,
            thresholds=thresholds,
            group_summary=group_summary,
            concentrated_numeric=concentrated_numeric,
            concentrated_categorical=concentrated_categorical,
            top_k=max(1, int(args.summary_top_k)),
        ),
        encoding="utf-8",
    )

    print(f"distribution_summary_json={summary_json_path}")
    print(f"distribution_summary_md={summary_md_path}")
    print(f"group_concentration_summary_csv={group_summary_path}")
    print(f"numeric_feature_distribution_csv={numeric_stats_path}")
    print(f"categorical_feature_distribution_csv={categorical_stats_path}")


if __name__ == "__main__":
    main()
