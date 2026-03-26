from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_contract import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from train_mlp import fit_preprocessor_state, transform_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze feature-feature and feature-target correlations for the single-op MLP dataset.",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input dataset CSV, e.g. dataset_full.csv, train.csv, val.csv, or test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for correlation CSVs, JSON summary, and heatmap PNGs.",
    )
    parser.add_argument(
        "--method",
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method used by pandas.DataFrame.corr.",
    )
    parser.add_argument(
        "--top-target-features",
        type=int,
        default=40,
        help="How many features to keep in the sorted feature-target heatmap.",
    )
    parser.add_argument(
        "--max-heatmap-features",
        type=int,
        default=80,
        help=(
            "Maximum number of features shown in the full encoded heatmap. "
            "If there are more, the script keeps the features most correlated with the target."
        ),
    )
    return parser.parse_args()


def import_matplotlib_pyplot() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to draw correlation heatmaps") from exc
    return plt


def load_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    df = pd.read_csv(input_csv)
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"Target column {TARGET_COLUMN!r} was not found in {input_csv}")
    return df


def available_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [column for column in BASELINE_NUMERIC_FEATURES if column in df.columns]
    categorical_features = [column for column in BASELINE_CATEGORICAL_FEATURES if column in df.columns]
    return numeric_features, categorical_features


def correlation_dataframe(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.corr(method=method, numeric_only=True)


def target_correlation_series(corr_df: pd.DataFrame) -> pd.Series:
    if corr_df.empty or TARGET_COLUMN not in corr_df.columns:
        return pd.Series(dtype=float)
    series = corr_df[TARGET_COLUMN].drop(labels=[TARGET_COLUMN], errors="ignore")
    return series.sort_values(key=lambda values: np.abs(values), ascending=False)


def _tick_step(size: int, max_ticks: int = 40) -> int:
    return max(1, int(np.ceil(size / max_ticks)))


def plot_matrix_heatmap(
    corr_df: pd.DataFrame,
    output_png: Path,
    title: str,
    center_zero: bool = True,
) -> None:
    if corr_df.empty:
        raise RuntimeError(f"Cannot draw empty heatmap: {output_png}")

    plt = import_matplotlib_pyplot()
    labels = list(corr_df.columns)
    size = len(labels)
    figure_size = min(max(8.0, size * 0.22), 28.0)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(
        corr_df.to_numpy(dtype=float),
        cmap="coolwarm",
        vmin=-1.0 if center_zero else None,
        vmax=1.0 if center_zero else None,
        aspect="auto",
    )
    step = _tick_step(size)
    tick_positions = list(range(0, size, step))
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels([labels[index] for index in tick_positions], rotation=90, fontsize=7)
    axis.set_yticklabels([labels[index] for index in tick_positions], fontsize=7)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def plot_target_heatmap(target_corr: pd.Series, output_png: Path, title: str) -> None:
    if target_corr.empty:
        raise RuntimeError(f"Cannot draw empty target-correlation heatmap: {output_png}")

    plt = import_matplotlib_pyplot()
    values = target_corr.to_numpy(dtype=float).reshape(-1, 1)
    labels = list(target_corr.index)
    figure_height = min(max(6.0, len(labels) * 0.28), 24.0)
    figure, axis = plt.subplots(figsize=(6.5, figure_height))
    image = axis.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks([0])
    axis.set_xticklabels([TARGET_COLUMN], rotation=0)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def plot_target_bar_chart(target_corr: pd.Series, output_png: Path, title: str) -> None:
    if target_corr.empty:
        raise RuntimeError(f"Cannot draw empty target-correlation bar chart: {output_png}")

    plt = import_matplotlib_pyplot()
    labels = list(target_corr.index)
    values = target_corr.to_numpy(dtype=float)
    colors = ["#d73027" if value >= 0 else "#4575b4" for value in values]

    figure_height = min(max(5.0, len(labels) * 0.45), 12.0)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    y_positions = np.arange(len(labels))
    axis.barh(y_positions, values, color=colors, alpha=0.9)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Correlation To Target")
    axis.set_title(title)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.grid(True, axis="x", linestyle="--", alpha=0.35)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def select_encoded_heatmap_columns(
    corr_df: pd.DataFrame,
    max_heatmap_features: int,
) -> list[str]:
    if corr_df.empty:
        return []

    feature_columns = [column for column in corr_df.columns if column != TARGET_COLUMN]
    if len(feature_columns) <= max_heatmap_features:
        return [*feature_columns, TARGET_COLUMN]

    target_corr = target_correlation_series(corr_df)
    selected = list(target_corr.index[:max_heatmap_features])
    return [*selected, TARGET_COLUMN]


def build_encoded_feature_frame(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessor_state = fit_preprocessor_state(
        train_df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    encoded_matrix = transform_features(df, preprocessor_state)
    encoded_df = pd.DataFrame(
        encoded_matrix,
        columns=preprocessor_state["transformed_feature_names"],
        index=df.index,
    )
    encoded_df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    return encoded_df, preprocessor_state


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(input_csv)
    numeric_features, categorical_features = available_feature_lists(dataset)

    numeric_df = dataset[numeric_features].copy()
    numeric_df[TARGET_COLUMN] = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    numeric_corr = correlation_dataframe(numeric_df, method=args.method)
    numeric_target_corr = target_correlation_series(numeric_corr)

    encoded_df, preprocessor_state = build_encoded_feature_frame(
        df=dataset,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    encoded_corr = correlation_dataframe(encoded_df, method=args.method)
    encoded_target_corr = target_correlation_series(encoded_corr)

    numeric_corr_path = output_dir / "numeric_feature_correlation.csv"
    numeric_corr.to_csv(numeric_corr_path)
    encoded_corr_path = output_dir / "encoded_feature_correlation.csv"
    encoded_corr.to_csv(encoded_corr_path)

    numeric_target_corr_path = output_dir / "numeric_feature_target_correlation.csv"
    numeric_target_corr.rename("correlation_to_target").rename_axis("feature").to_csv(
        numeric_target_corr_path,
        header=True,
    )
    encoded_target_corr_path = output_dir / "encoded_feature_target_correlation.csv"
    encoded_target_corr.rename("correlation_to_target").rename_axis("feature").to_csv(
        encoded_target_corr_path,
        header=True,
    )

    numeric_heatmap_path = output_dir / "numeric_feature_correlation_heatmap.png"
    plot_matrix_heatmap(
        corr_df=numeric_corr,
        output_png=numeric_heatmap_path,
        title=f"Numeric Feature Correlation ({args.method})",
    )

    encoded_heatmap_columns = select_encoded_heatmap_columns(
        corr_df=encoded_corr,
        max_heatmap_features=max(1, int(args.max_heatmap_features)),
    )
    encoded_heatmap_path = output_dir / "encoded_feature_correlation_heatmap.png"
    plot_matrix_heatmap(
        corr_df=encoded_corr.loc[encoded_heatmap_columns, encoded_heatmap_columns],
        output_png=encoded_heatmap_path,
        title=f"Encoded Feature Correlation ({args.method})",
    )

    top_target_count = max(1, int(args.top_target_features))
    numeric_target_heatmap_path = output_dir / "numeric_feature_target_heatmap.png"
    plot_target_heatmap(
        target_corr=numeric_target_corr.head(top_target_count),
        output_png=numeric_target_heatmap_path,
        title=f"Top Numeric Feature Correlation To Target ({args.method})",
    )
    numeric_target_bar_path = output_dir / "numeric_feature_target_top10_bar.png"
    plot_target_bar_chart(
        target_corr=numeric_target_corr.head(10),
        output_png=numeric_target_bar_path,
        title=f"Top 10 Numeric Feature Correlation To Target ({args.method})",
    )

    encoded_target_heatmap_path = output_dir / "encoded_feature_target_heatmap.png"
    plot_target_heatmap(
        target_corr=encoded_target_corr.head(top_target_count),
        output_png=encoded_target_heatmap_path,
        title=f"Top Encoded Feature Correlation To Target ({args.method})",
    )
    encoded_target_bar_path = output_dir / "encoded_feature_target_top10_bar.png"
    plot_target_bar_chart(
        target_corr=encoded_target_corr.head(10),
        output_png=encoded_target_bar_path,
        title=f"Top 10 Encoded Feature Correlation To Target ({args.method})",
    )

    summary = {
        "input_csv": str(input_csv),
        "row_count": int(len(dataset)),
        "method": args.method,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "encoded_input_dim": int(preprocessor_state["input_dim"]),
        "max_heatmap_features": int(args.max_heatmap_features),
        "top_target_features": int(args.top_target_features),
        "artifacts": {
            "numeric_feature_correlation_csv": str(numeric_corr_path),
            "encoded_feature_correlation_csv": str(encoded_corr_path),
            "numeric_feature_target_correlation_csv": str(numeric_target_corr_path),
            "encoded_feature_target_correlation_csv": str(encoded_target_corr_path),
            "numeric_feature_correlation_heatmap_png": str(numeric_heatmap_path),
            "encoded_feature_correlation_heatmap_png": str(encoded_heatmap_path),
            "numeric_feature_target_heatmap_png": str(numeric_target_heatmap_path),
            "encoded_feature_target_heatmap_png": str(encoded_target_heatmap_path),
            "numeric_feature_target_top10_bar_png": str(numeric_target_bar_path),
            "encoded_feature_target_top10_bar_png": str(encoded_target_bar_path),
        },
    }
    write_json(output_dir / "correlation_summary.json", summary)

    print(f"correlation_summary_json={output_dir / 'correlation_summary.json'}")
    print(f"numeric_heatmap_png={numeric_heatmap_path}")
    print(f"encoded_heatmap_png={encoded_heatmap_path}")
    print(f"numeric_target_heatmap_png={numeric_target_heatmap_path}")
    print(f"encoded_target_heatmap_png={encoded_target_heatmap_path}")
    print(f"numeric_target_top10_bar_png={numeric_target_bar_path}")
    print(f"encoded_target_top10_bar_png={encoded_target_bar_path}")


if __name__ == "__main__":
    main()
