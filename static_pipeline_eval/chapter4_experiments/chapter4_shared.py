from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from static_pipeline_eval.artifact_loader import (
    DEFAULT_ORT_ROOT,
    DEFAULT_ARTIFACT_ROOT as DEFAULT_SINGLE_OP_SOURCE,
    build_combo_specs,
    load_op_shapes_frame,
    load_prediction_frame,
    load_timeline_frame,
)
from static_pipeline_eval.schedule_engine import (
    analyze_embedding_execution,
    build_op_graph,
    compute_mean_batch_span,
    schedule_combo,
)

from .chapter4_config import (
    ABLATION_ARTIFACT_ROOT,
    BASELINE_MODEL_ROOT,
    CHAPTER4_DRAFT_PATH,
    CHAPTER4_OUTPUT_ROOT,
    E2E_ARTIFACT_ROOT,
    FIGURE_FILENAMES,
    MODEL_GROUP_ORDER,
    OOD_ARTIFACT_ROOT,
    OOD_BATCH_HOLDS,
    OOD_NUM_THREADS_HOLD,
    REPRESENTATIVE_OP_TYPES,
    SINGLE_OP_ARTIFACT_ROOT,
    TABLE_FILENAMES,
    TIMELINE_CASES,
)


@dataclass(frozen=True)
class SectionResult:
    name: str
    outputs: dict[str, str]


def ensure_output_layout(output_root: Path | None = None) -> dict[str, Path]:
    root = Path(output_root or CHAPTER4_OUTPUT_ROOT)
    layout = {
        "root": root,
        "single_op": root / "single_op",
        "e2e": root / "e2e",
        "ablation": root / "ablation",
        "figures": root / "figures",
        "tables": root / "tables",
        "manifests": root / "manifests",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _fmt_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 1000.0:
            return f"{value:,.3f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame, title: str | None = None) -> str:
    rows = frame.fillna("").copy()
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    if rows.empty:
        lines.append("_No rows_")
        lines.append("")
        return "\n".join(lines)
    headers = list(rows.columns)
    rendered_rows = [[_fmt_cell(cell) for cell in row] for row in rows.itertuples(index=False, name=None)]
    widths = [len(str(header)) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: Iterable[str]) -> str:
        cells = [str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines.append(render_row(headers))
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rendered_rows:
        lines.append(render_row(row))
    lines.append("")
    return "\n".join(lines)


def write_frame_csv_md(frame: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(frame, title), encoding="utf-8")
    return csv_path, md_path


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def plot_bar(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, *, xlabel: str = "", ylabel: str = "", color: str = "#4477aa") -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame[x].astype(str), frame[y].astype(float), color=color)
    _style_axes(ax, title, xlabel or x, ylabel or y)
    ax.tick_params(axis="x", rotation=30)
    return save_figure(fig, path)


def plot_grouped_bar(
    frame: pd.DataFrame,
    index_col: str,
    value_cols: list[str],
    path: Path,
    title: str,
    *,
    ylabel: str,
    legend_labels: list[str] | None = None,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(frame))
    width = 0.8 / max(1, len(value_cols))
    labels = legend_labels or value_cols
    for idx, col in enumerate(value_cols):
        offsets = [pos + (idx - (len(value_cols) - 1) / 2.0) * width for pos in x]
        ax.bar(offsets, frame[col].astype(float), width=width, label=labels[idx])
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame[index_col].astype(str).tolist(), rotation=30, ha="right")
    _style_axes(ax, title, index_col, ylabel)
    ax.legend(frameon=False, ncols=min(len(value_cols), 3))
    return save_figure(fig, path)


def plot_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    hue: str | None = None,
    reference_line: bool = True,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if hue and hue in frame.columns:
        for label, sub in frame.groupby(hue):
            ax.scatter(sub[x], sub[y], s=18, alpha=0.75, label=str(label))
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(frame[x], frame[y], s=18, alpha=0.75, color="#4477aa")
    if reference_line and not frame.empty:
        lo = float(min(frame[x].min(), frame[y].min()))
        hi = float(max(frame[x].max(), frame[y].max()))
        ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=1.0, linestyle="--")
    _style_axes(ax, title, x, y)
    return save_figure(fig, path)


def plot_heatmap(frame: pd.DataFrame, path: Path, title: str, *, xlabel: str = "", ylabel: str = "", annot: bool = True) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = frame.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels([str(col) for col in frame.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels([str(idx) for idx in frame.index])
    _style_axes(ax, title, xlabel, ylabel)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if annot:
        for row_idx, row_label in enumerate(frame.index):
            for col_idx, col_label in enumerate(frame.columns):
                value = matrix[row_idx, col_idx]
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    return save_figure(fig, path)


def plot_boxplot(values_by_label: dict[str, list[float]], path: Path, title: str, *, ylabel: str) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(values_by_label.keys())
    values = [values_by_label[label] for label in labels]
    ax.boxplot(values, labels=labels, showmeans=True)
    _style_axes(ax, title, "", ylabel)
    ax.tick_params(axis="x", rotation=25)
    return save_figure(fig, path)


def plot_line(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, *, group_col: str | None = None, ylabel: str | None = None) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    if group_col and group_col in frame.columns:
        for label, sub in frame.groupby(group_col):
            ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, label=str(label))
        ax.legend(frameon=False)
    else:
        ax.plot(frame[x], frame[y], marker="o", linewidth=1.5, color="#4477aa")
    _style_axes(ax, title, x, ylabel or y)
    return save_figure(fig, path)


def plot_gantt(frame: pd.DataFrame, path: Path, title: str, *, label_col: str, start_col: str, end_col: str, hue_col: str | None = None) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.3 * len(frame))))
    labels = frame[label_col].astype(str).tolist()
    colors = {
        "bottom": "#4C78A8",
        "embedding": "#F58518",
        "tail": "#54A24B",
        "barrier": "#B279A2",
    }
    for idx, row in enumerate(frame.itertuples(index=False)):
        color = colors.get(str(getattr(row, hue_col)) if hue_col else "bottom", "#4C78A8")
        ax.broken_barh([(float(getattr(row, start_col)), float(getattr(row, end_col)) - float(getattr(row, start_col)))], (idx - 0.35, 0.7), facecolors=color)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    _style_axes(ax, title, "time (us)", "")
    return save_figure(fig, path)


def plot_simple_graph(nodes: pd.DataFrame, edges: pd.DataFrame, path: Path, title: str) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 7))
    if nodes.empty:
        ax.text(0.5, 0.5, "No graph data", ha="center", va="center")
        ax.axis("off")
        return save_figure(fig, path)

    x_values = nodes["layer"].astype(float)
    y_positions = defaultdict(list)
    for layer in sorted(set(x_values)):
        layer_nodes = nodes[nodes["layer"] == layer].copy()
        for row_idx, row in enumerate(layer_nodes.itertuples(index=False)):
            y_positions[int(layer)].append(row_idx)
    positions: dict[int, tuple[float, float]] = {}
    for layer in sorted(set(x_values)):
        layer_nodes = nodes[nodes["layer"] == layer].copy()
        count = len(layer_nodes)
        for idx, row in enumerate(layer_nodes.itertuples(index=False)):
            y = (count - 1) / 2.0 - idx
            positions[int(row.node_idx)] = (float(layer), float(y))

    color_map = {
        "bottom": "#4C78A8",
        "embedding": "#F58518",
        "tail": "#54A24B",
        "constant": "#C7C7C7",
    }
    for edge in edges.itertuples(index=False):
        src = int(edge.src)
        dst = int(edge.dst)
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8, alpha=0.8))

    for node in nodes.itertuples(index=False):
        x, y = positions[int(node.node_idx)]
        color = color_map.get(str(node.partition), "#4C78A8")
        ax.scatter([x], [y], s=500, color=color, edgecolor="#222222", linewidth=0.7, zorder=3)
        ax.text(x, y, str(node.label), ha="center", va="center", fontsize=8, color="white", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("topological depth")
    ax.set_ylabel("node layer")
    ax.set_yticks([])
    ax.grid(False)
    return save_figure(fig, path)


def _table_path(layout: dict[str, Path], figure_no: str) -> Path:
    return layout["tables"] / TABLE_FILENAMES[figure_no]


def _figure_path(layout: dict[str, Path], figure_no: str) -> Path:
    return layout["figures"] / FIGURE_FILENAMES[figure_no]


def _write_summary_bundle(section_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    section_dir.mkdir(parents=True, exist_ok=True)
    return write_json(section_dir / f"{name}_summary.json", payload)


def _sanitize_list(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def _resolve_root(value: Path | None, default: Path) -> Path:
    return Path(value) if value is not None else Path(default)


def _metric_summary(metric_rows: list[dict[str, Any]], predicted_key: str, actual_key: str) -> dict[str, Any]:
    if not metric_rows:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "p50_ape": None,
            "p95_ape": None,
            "worst_combo": None,
        }
    frame = pd.DataFrame(metric_rows)
    worst_row = frame.sort_values("abs_error_us", ascending=False).iloc[0]
    return {
        "count": int(len(frame)),
        "mae_us": float(frame["abs_error_us"].mean()),
        "mape": float(frame["ape"].mean()),
        "p50_ape": float(frame["ape"].quantile(0.50)),
        "p95_ape": float(frame["ape"].quantile(0.95)),
        "worst_combo": {
            "case_id": worst_row["case_id"],
            "combo": worst_row["combo"],
            predicted_key: float(worst_row[predicted_key]),
            actual_key: float(worst_row[actual_key]),
            "abs_error_us": float(worst_row["abs_error_us"]),
            "ape": float(worst_row["ape"]),
        },
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _topological_layers(graph: dict[int, Any], node_subset: set[int] | None = None) -> dict[int, int]:
    subset = node_subset if node_subset is not None else set(graph)
    indegree = {node_idx: 0 for node_idx in subset}
    successors: dict[int, list[int]] = defaultdict(list)
    for node_idx in subset:
        node = graph[node_idx]
        for pred in node.predecessors:
            if pred in subset:
                indegree[node_idx] += 1
                successors[pred].append(node_idx)
    ready = sorted([node_idx for node_idx, degree in indegree.items() if degree == 0])
    layer = {node_idx: 0 for node_idx in ready}
    while ready:
        node_idx = ready.pop(0)
        for succ in sorted(successors.get(node_idx, [])):
            layer[succ] = max(layer.get(succ, 0), layer[node_idx] + 1)
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    return layer


def _graph_subframe(graph: dict[int, Any], keep_predicate) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_nodes = [node for node in graph.values() if keep_predicate(node)]
    kept_indices = {node.node_idx for node in kept_nodes}
    nodes = []
    edges = []
    layers = _topological_layers(graph, kept_indices)
    for node in kept_nodes:
        nodes.append(
            {
                "node_idx": node.node_idx,
                "label": node.op_type,
                "partition": node.partition,
                "layer": layers.get(node.node_idx, 0),
            }
        )
        for pred in node.predecessors:
            if pred in kept_indices:
                edges.append({"src": pred, "dst": node.node_idx})
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _load_group_metrics(training_summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_group in training_summary["combined"]["model_group_order"]:
        group_summary = training_summary["models"][model_group]
        test_metrics = group_summary["metrics"]["test"]
        rows.append(
            {
                "model_group": model_group,
                "feature_count": group_summary["feature_count"],
                "input_dim_after_encoding": group_summary["input_dim_after_encoding"],
                "best_epoch": group_summary["best_epoch"],
                "best_validation_loss": group_summary["best_validation_loss"],
                "test_mae_us": test_metrics["mae_us"],
                "test_rmse_us": test_metrics["rmse_us"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
                "test_median_ape": test_metrics["median_ape"],
            }
        )
    return pd.DataFrame(rows)


def _load_ablation_frames(ablation_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(Path(ablation_root).rglob("ablation_summary.json")):
        summary = read_json(summary_path)
        model_group = summary["model_group"]
        for row in summary.get("summary_rows", []):
            rows.append(
                {
                    "model_group": model_group,
                    "variant": row["variant"],
                    "dropped_features": row["dropped_features"],
                    "dropped_feature_count": row["dropped_feature_count"],
                    "numeric_feature_count": row["numeric_feature_count"],
                    "test_mae_us": row["test_mae_us"],
                    "test_mape": row["test_mape"],
                    "test_median_ape": row["test_median_ape"],
                    "test_r2": row["test_r2"],
                    "test_mae_us_delta_vs_baseline": row["test_mae_us_delta_vs_baseline"],
                    "test_mape_delta_vs_baseline": row["test_mape_delta_vs_baseline"],
                    "test_median_ape_delta_vs_baseline": row["test_median_ape_delta_vs_baseline"],
                    "test_r2_delta_vs_baseline": row["test_r2_delta_vs_baseline"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["dropped_features"] = frame["dropped_features"].replace("", "(baseline)")
    return frame


def _load_generalization_summary(path: Path) -> pd.DataFrame:
    summary = read_json(path)
    rows: list[dict[str, Any]] = []
    for scheme_name in ("leave_one_case_out", "leave_one_combo_out"):
        scheme = summary.get(scheme_name, {})
        for split_name in ("train", "test"):
            split = scheme.get(split_name, {})
            for family_row in split.get("family_summary", []):
                rows.append(
                    {
                        "scheme": scheme_name,
                        "split": split_name,
                        "family": family_row["family"],
                        "mean_mape": family_row["mean_mape"],
                        "median_mape": family_row["median_mape"],
                        "max_mape": family_row["max_mape"],
                        "mean_dwre": family_row["mean_dwre"],
                        "median_dwre": family_row["median_dwre"],
                        "max_dwre": family_row["max_dwre"],
                        "folds": family_row["folds"],
                        "total_rows": family_row["total_rows"],
                        "total_actual_us": family_row["total_actual_us"],
                        "total_abs_error_us": family_row["total_abs_error_us"],
                    }
                )
            for fold_row in split.get("fold_macro", []):
                rows.append(
                    {
                        "scheme": scheme_name,
                        "split": split_name,
                        "family": "__fold_macro__",
                        "fold": fold_row["fold"],
                        "macro_mape": fold_row["macro_mape"],
                        "total_rows": fold_row["total_rows"],
                        "actual_sum_us": fold_row["actual_sum_us"],
                        "abs_error_sum_us": fold_row["abs_error_sum_us"],
                        "duration_weighted_relative_error": fold_row["duration_weighted_relative_error"],
                    }
                )
    return pd.DataFrame(rows)


def _compute_longest_path(graph: dict[int, Any], duration_by_node: dict[int, float]) -> tuple[list[int], float]:
    indegree = {node_idx: len([pred for pred in node.predecessors if pred in graph]) for node_idx, node in graph.items()}
    successors: dict[int, list[int]] = defaultdict(list)
    for node_idx, node in graph.items():
        for pred in node.predecessors:
            if pred in graph:
                successors[pred].append(node_idx)
    ready = sorted([node_idx for node_idx, degree in indegree.items() if degree == 0])
    best_total: dict[int, float] = {}
    parent: dict[int, int | None] = {}
    while ready:
        node_idx = ready.pop(0)
        node_duration = float(duration_by_node.get(node_idx, 0.0))
        best_pred = None
        best_pred_total = 0.0
        for pred in graph[node_idx].predecessors:
            if pred in best_total and best_total[pred] > best_pred_total:
                best_pred_total = best_total[pred]
                best_pred = pred
        best_total[node_idx] = best_pred_total + node_duration
        parent[node_idx] = best_pred
        for succ in sorted(successors.get(node_idx, [])):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
                ready.sort()
    if not best_total:
        return [], 0.0
    sink = max(best_total, key=best_total.get)
    path: list[int] = []
    current = sink
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path, float(best_total[sink])


def _slice_predictions(prediction_df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    frame = prediction_df.copy()
    for column, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            frame = frame[frame[column].isin(list(value))]
        else:
            frame = frame[frame[column] == value]
    return frame


def _evaluate_predictions(prediction_df: pd.DataFrame) -> dict[str, Any]:
    actual = prediction_df["target_us"].astype(float)
    pred = prediction_df["pred_us"].astype(float)
    abs_error = (pred - actual).abs()
    ape = abs_error / actual.replace(0.0, pd.NA)
    return {
        "rows": int(len(prediction_df)),
        "mae_us": float(abs_error.mean()) if not abs_error.empty else None,
        "rmse_us": float(((pred - actual) ** 2).mean() ** 0.5) if not pred.empty else None,
        "mape": float(ape.fillna(0.0).mean()) if not ape.empty else None,
        "median_ape": float(ape.fillna(0.0).median()) if not ape.empty else None,
    }


def run_platform_summary(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
    baseline_model_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    baseline_root = _resolve_root(baseline_model_root, BASELINE_MODEL_ROOT)

    single_op_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    e2e_summary = read_json(e2e_root / "summary.json")
    baseline_metrics = read_json(baseline_root / "metrics.json")

    rows = [
        {
            "source": "single_op_classed_dataset",
            "artifact_root": str(single_op_root),
            "rows": int(len(single_op_df)),
            "cases": int(single_op_df["case_id"].nunique()),
            "combos": int(single_op_df[["case_id", "combo"]].drop_duplicates().shape[0]),
            "batch_sizes": _sanitize_list(sorted(single_op_df["batch_size"].dropna().astype(int).unique().tolist())),
            "inter_threads": _sanitize_list(sorted(single_op_df["inter_threads"].dropna().astype(int).unique().tolist())),
            "notes": single_op_root.name,
        },
        {
            "source": "static_pipeline_eval",
            "artifact_root": str(e2e_root),
            "rows": int(e2e_summary["combo_counts"]["total_test_combos"]),
            "cases": int(single_op_df["case_id"].nunique()),
            "combos": int(e2e_summary["combo_counts"]["total_test_combos"]),
            "batch_sizes": _sanitize_list(sorted(single_op_df["batch_size"].dropna().astype(int).unique().tolist())),
            "inter_threads": _sanitize_list(sorted(single_op_df["inter_threads"].dropna().astype(int).unique().tolist())),
            "notes": e2e_root.name,
        },
        {
            "source": "model_all_no_trace_baseline",
            "artifact_root": str(baseline_root),
            "rows": 1,
            "cases": 1,
            "combos": 1,
            "batch_sizes": "n/a",
            "inter_threads": "n/a",
            "notes": "baseline comparison model",
        },
    ]
    frame = pd.DataFrame(rows)
    csv_path, md_path = write_frame_csv_md(frame, tables_dir / TABLE_FILENAMES["4-1"], tables_dir / "table_4_1_platform_dataset.md", "Table 4-1 Platform and Dataset Overview")

    plot_frame = pd.DataFrame(
        [
            {"metric": "single_op rows", "value": len(single_op_df)},
            {"metric": "single_op combos", "value": single_op_df[["case_id", "combo"]].drop_duplicates().shape[0]},
            {"metric": "e2e combos", "value": e2e_summary["combo_counts"]["total_test_combos"]},
            {"metric": "e2e full combos", "value": e2e_summary["combo_counts"]["full_combo_count"]},
            {"metric": "baseline test rows", "value": 1},
        ]
    )
    fig_path = plot_bar(
        plot_frame,
        "metric",
        "value",
        figures_dir / FIGURE_FILENAMES["4-1"],
        "Figure 4-1 Platform and Dataset Overview",
        ylabel="count",
        color="#4C78A8",
    )

    summary = {
        "single_op_rows": int(len(single_op_df)),
        "single_op_combos": int(single_op_df[["case_id", "combo"]].drop_duplicates().shape[0]),
        "e2e_combos": int(e2e_summary["combo_counts"]["total_test_combos"]),
        "full_e2e_combos": int(e2e_summary["combo_counts"]["full_combo_count"]),
        "baseline_test_mape": baseline_metrics["metrics"]["test"]["mape"],
    }
    manifest = {
        "table_csv": str(csv_path),
        "table_md": str(md_path),
        "figure": str(fig_path),
        "summary": summary,
    }
    write_json(layout["manifests"] / "platform_summary.json", manifest)
    return SectionResult(name="platform", outputs=manifest)


def run_single_op_core(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    baseline_model_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["single_op"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    baseline_root = _resolve_root(baseline_model_root, BASELINE_MODEL_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)

    training_summary = read_json(single_op_root / "models" / "training_summary.json")
    group_metrics = _load_group_metrics(training_summary)
    baseline_metrics = read_json(baseline_root / "metrics.json")
    combined_op_type_metrics = pd.read_csv(
        single_op_root / "models" / "combined" / "combined_op_type_metrics_test.csv",
        low_memory=False,
    )
    comparison_summary = read_json(
        single_op_root / "models" / "comparison" / "comparison_summary.json"
    )
    combined_predictions = pd.read_csv(
        single_op_root / "models" / "combined" / "combined_predictions_test.csv",
        low_memory=False,
    )

    group_metrics = group_metrics.sort_values("model_group").reset_index(drop=True)
    group_metrics["delta_mape_vs_baseline"] = group_metrics["test_mape"] - float(baseline_metrics["metrics"]["test"]["mape"])
    group_metrics["delta_mae_vs_baseline"] = group_metrics["test_mae_us"] - float(baseline_metrics["metrics"]["test"]["mae_us"])
    group_metrics["baseline_test_mape"] = float(baseline_metrics["metrics"]["test"]["mape"])

    table_4_2 = group_metrics[
        [
            "model_group",
            "feature_count",
            "input_dim_after_encoding",
            "best_epoch",
            "best_validation_loss",
            "test_mae_us",
            "test_rmse_us" if "test_rmse_us" in group_metrics.columns else "test_mae_us",
            "test_r2" if "test_r2" in group_metrics.columns else "best_validation_loss",
            "test_mape",
            "test_median_ape",
            "delta_mape_vs_baseline",
        ]
    ].copy()

    table_4_3 = combined_op_type_metrics[
        [
            "op_type",
            "row_count",
            "mae_us",
            "rmse_us",
            "r2",
            "mape",
            "median_ape",
            "p90_ape",
            "bias_mean_us",
        ]
    ].sort_values("mape", ascending=False).reset_index(drop=True)

    representative_rows: list[dict[str, Any]] = []
    for op_type in REPRESENTATIVE_OP_TYPES:
        subset = combined_predictions[combined_predictions["op_type"] == op_type]
        if subset.empty:
            continue
        actual = subset["target_us"].astype(float)
        pred = subset["pred_us"].astype(float)
        abs_error = (pred - actual).abs()
        ape = abs_error / actual.replace(0.0, pd.NA)
        representative_rows.append(
            {
                "op_type": op_type,
                "rows": int(len(subset)),
                "target_mean_us": float(actual.mean()),
                "pred_mean_us": float(pred.mean()),
                "mae_us": float(abs_error.mean()),
                "mape": float(ape.fillna(0.0).mean()),
                "median_ape": float(ape.fillna(0.0).median()),
                "p90_ape": float(ape.fillna(0.0).quantile(0.9)),
            }
        )
    table_4_4 = pd.DataFrame(representative_rows).sort_values("mape", ascending=False).reset_index(drop=True)

    csv_42, md_42 = write_frame_csv_md(table_4_2, tables_dir / TABLE_FILENAMES["4-2"], tables_dir / "table_4_2_single_op_group_metrics.md", "Table 4-2 Single-op group metrics")
    csv_43, md_43 = write_frame_csv_md(table_4_3, tables_dir / TABLE_FILENAMES["4-3"], tables_dir / "table_4_3_single_op_optype_metrics.md", "Table 4-3 Single-op op-type metrics")
    csv_44, md_44 = write_frame_csv_md(table_4_4, tables_dir / TABLE_FILENAMES["4-4"], tables_dir / "table_4_4_single_op_representative_ops.md", "Table 4-4 Representative operator metrics")

    plot_grouped_bar(
        table_4_2[["model_group", "test_mape", "delta_mape_vs_baseline"]].rename(columns={"test_mape": "mape", "delta_mape_vs_baseline": "delta"}),
        "model_group",
        ["mape", "delta"],
        figures_dir / FIGURE_FILENAMES["4-2"],
        "Figure 4-2 Single-op Group MAPE and Baseline Delta",
        ylabel="value",
        legend_labels=["test MAPE", "delta vs baseline"],
    )

    plot_bar(
        table_4_2,
        "model_group",
        "test_mae_us",
        figures_dir / FIGURE_FILENAMES["4-3"],
        "Figure 4-3 Single-op Group MAE",
        ylabel="MAE (us)",
        color="#F58518",
    )

    plot_bar(
        table_4_3.head(8),
        "op_type",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-4"],
        "Figure 4-4 Single-op Op-type MAPE",
        ylabel="MAPE",
        color="#54A24B",
    )

    graph_case, graph_combo = TIMELINE_CASES[0]
    prediction_df = load_prediction_frame(single_op_root, split="test")
    combo_spec_map = {
        (spec.case_id, spec.combo): spec
        for spec in build_combo_specs(prediction_df, ort_root=Path(DEFAULT_ORT_ROOT))
    }
    graph = build_op_graph(load_op_shapes_frame(combo_spec_map[(graph_case, graph_combo)].artifact_paths.shape_csv))
    representative_nodes = {node.node_idx for node in graph.values() if node.op_type in set(REPRESENTATIVE_OP_TYPES)}
    for node_idx in list(representative_nodes):
        representative_nodes.update(graph[node_idx].predecessors)
        representative_nodes.update(graph[node_idx].successors)
    if not representative_nodes:
        representative_nodes = {node_idx for node_idx, node in graph.items() if node.op_type != "Constant"}
    nodes_frame, edges_frame = _graph_subframe(
        graph,
        lambda node: node.node_idx in representative_nodes
        or node.op_type in set(REPRESENTATIVE_OP_TYPES)
        or node.partition in {"bottom", "tail"},
    )
    if nodes_frame.empty:
        nodes_frame, edges_frame = _graph_subframe(graph, lambda node: node.op_type != "Constant")
    plot_simple_graph(
        nodes_frame,
        edges_frame,
        figures_dir / FIGURE_FILENAMES["4-5"],
        f"Figure 4-5 Representative Graph ({graph_case} / {graph_combo})",
    )

    scatter_rows: list[dict[str, Any]] = []
    for op_type in REPRESENTATIVE_OP_TYPES:
        subset = combined_predictions[combined_predictions["op_type"] == op_type].copy()
        if subset.empty:
            continue
        subset["actual_us"] = subset["target_us"].astype(float)
        subset["predicted_us"] = subset["pred_us"].astype(float)
        subset["residual_us"] = subset["predicted_us"] - subset["actual_us"]
        scatter_rows.append(subset.sample(n=min(250, len(subset)), random_state=42))
    scatter_frame = pd.concat(scatter_rows, ignore_index=True) if scatter_rows else pd.DataFrame(columns=["actual_us", "predicted_us", "op_type"])
    if not scatter_frame.empty:
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for op_type, sub in scatter_frame.groupby("op_type"):
            ax.scatter(sub["actual_us"], sub["predicted_us"], s=12, alpha=0.6, label=op_type)
        lo = float(min(scatter_frame["actual_us"].min(), scatter_frame["predicted_us"].min()))
        hi = float(max(scatter_frame["actual_us"].max(), scatter_frame["predicted_us"].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#222222")
        ax.set_title("Figure 4-6 Representative Op Scatter")
        ax.set_xlabel("actual (us)")
        ax.set_ylabel("predicted (us)")
        ax.legend(frameon=False, fontsize=8)
        save_figure(fig, figures_dir / FIGURE_FILENAMES["4-6"])

    history_rows: list[dict[str, Any]] = []
    for model_group in MODEL_GROUP_ORDER:
        history_path = single_op_root / "models" / model_group / "training_history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path, low_memory=False)
        history = history.assign(model_group=model_group)
        history_rows.append(history[["epoch", "val_loss", "model_group"]])
    if history_rows:
        history_frame = pd.concat(history_rows, ignore_index=True)
        plot_line(
            history_frame,
            "epoch",
            "val_loss",
            figures_dir / FIGURE_FILENAMES["4-7"],
            "Figure 4-7 Single-op Validation Loss Curves",
            group_col="model_group",
            ylabel="validation loss",
        )

    residual_rows = []
    for op_type, sub in combined_predictions.groupby("op_type"):
        abs_error = (sub["pred_us"].astype(float) - sub["target_us"].astype(float)).abs()
        residual_rows.append({"op_type": op_type, "abs_error_us": float(abs_error.mean()), "mape": float((abs_error / sub["target_us"].replace(0.0, pd.NA)).fillna(0.0).mean())})
    residual_frame = pd.DataFrame(residual_rows).sort_values("mape", ascending=False).reset_index(drop=True)
    if not residual_frame.empty:
        plot_bar(
            residual_frame.head(12),
            "op_type",
            "mape",
            figures_dir / FIGURE_FILENAMES["4-8"],
            "Figure 4-8 Single-op Residual Distribution",
            ylabel="MAPE",
            color="#B279A2",
        )

    section_payload = {
        "tables": {
            "4-2": str(csv_42),
            "4-3": str(csv_43),
            "4-4": str(csv_44),
        },
        "figures": {
            "4-2": str(figures_dir / FIGURE_FILENAMES["4-2"]),
            "4-3": str(figures_dir / FIGURE_FILENAMES["4-3"]),
            "4-4": str(figures_dir / FIGURE_FILENAMES["4-4"]),
            "4-5": str(figures_dir / FIGURE_FILENAMES["4-5"]),
            "4-6": str(figures_dir / FIGURE_FILENAMES["4-6"]),
            "4-7": str(figures_dir / FIGURE_FILENAMES["4-7"]),
            "4-8": str(figures_dir / FIGURE_FILENAMES["4-8"]),
        },
        "summary": {
            "baseline_test_mape": float(baseline_metrics["metrics"]["test"]["mape"]),
            "group_test_mape_mean": float(group_metrics["test_mape"].mean()),
            "group_test_mape_worst": float(group_metrics["test_mape"].max()),
            "representative_op_types": list(REPRESENTATIVE_OP_TYPES),
            "e2e_reference_root": str(e2e_root),
        },
    }
    _write_summary_bundle(section_dir, "single_op_core", section_payload)
    return SectionResult(name="single_op_core", outputs=section_payload)


def run_single_op_ood(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
    ood_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["single_op"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    ood_root = _resolve_root(ood_artifact_root, OOD_ARTIFACT_ROOT)

    prediction_df = load_prediction_frame(single_op_root, split="test")
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    joined = prediction_df.merge(
        dataset_df[["row_uid", "num_threads"]],
        on="row_uid",
        how="left",
        validate="one_to_one",
    )
    slices = []
    batch_slice = _slice_predictions(joined, batch_size=list(OOD_BATCH_HOLDS))
    thread_slice = _slice_predictions(joined, num_threads=OOD_NUM_THREADS_HOLD)
    for name, frame in [("unseen_batch_size", batch_slice), ("unseen_num_threads", thread_slice)]:
        metrics = _evaluate_predictions(frame)
        metrics.update(
            {
                "slice_name": name,
                "rows": int(len(frame)),
                "batch_sizes": _sanitize_list(sorted(frame["batch_size"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
                "num_threads": _sanitize_list(sorted(frame["num_threads"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
                "inter_threads": _sanitize_list(sorted(frame["inter_threads"].dropna().astype(int).unique().tolist())) if not frame.empty else "",
            }
        )
        slices.append(metrics)
    ood_frame = pd.DataFrame(slices)

    generalization_frame = _load_generalization_summary(ood_root / "summary.json")
    generalization_frame = generalization_frame[generalization_frame["family"] != "__fold_macro__"].copy()
    if not generalization_frame.empty:
        generalization_frame = generalization_frame[generalization_frame["split"] == "test"].copy()

    csv_44 = tables_dir / "ood_slice_summary.csv"
    md_44 = tables_dir / "ood_slice_summary.md"
    write_frame_csv_md(ood_frame, csv_44, md_44, "Single-op OOD Slice Summary")
    generalization_csv = tables_dir / "ood_generalization_reference.csv"
    generalization_md = tables_dir / "ood_generalization_reference.md"
    write_frame_csv_md(
        generalization_frame.sort_values(["scheme", "family"]),
        generalization_csv,
        generalization_md,
        "Analytical Generalization Reference",
    )

    plot_frame = ood_frame.copy()
    if not plot_frame.empty:
        plot_frame["label"] = plot_frame["slice_name"].map(
            {
                "unseen_batch_size": "batch holdout",
                "unseen_num_threads": "thread holdout",
            }
        )
        plot_bar(
            plot_frame,
            "label",
            "mape",
            figures_dir / FIGURE_FILENAMES["4-9"],
            "Figure 4-9 Single-op OOD Slice Error",
            ylabel="MAPE",
            color="#E45756",
        )

    if not generalization_frame.empty:
        scheme_summary = (
            generalization_frame.groupby(["scheme", "split"], as_index=False)["mean_mape"]
            .mean()
            .sort_values(["scheme", "split"])
        )
        plt = _import_pyplot()
        fig, ax = plt.subplots(figsize=(10, 5))
        for scheme, sub in scheme_summary.groupby("scheme"):
            ax.plot(sub["split"], sub["mean_mape"], marker="o", linewidth=1.5, label=scheme)
        ax.set_title("Figure 4-10 Analytical Generalization Reference")
        ax.set_xlabel("split")
        ax.set_ylabel("mean MAPE")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)
        save_figure(fig, figures_dir / FIGURE_FILENAMES["4-10"])

    payload = {
        "tables": {
            "ood_slice_summary": str(csv_44),
            "ood_generalization_reference": str(generalization_csv),
        },
        "figures": {
            "4-9": str(figures_dir / FIGURE_FILENAMES["4-9"]),
            "4-10": str(figures_dir / FIGURE_FILENAMES["4-10"]),
        },
        "summary": {
            "ood_slices": slices,
            "generalization_reference_rows": int(len(generalization_frame)),
        },
    }
    _write_summary_bundle(section_dir, "single_op_ood", payload)
    return SectionResult(name="single_op_ood", outputs=payload)


def run_single_op_ablation(
    output_root: Path | None = None,
    *,
    ablation_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["ablation"]

    ablation_root = _resolve_root(ablation_artifact_root, ABLATION_ARTIFACT_ROOT)
    ablation_frame = _load_ablation_frames(ablation_root)
    if ablation_frame.empty:
        payload = {"tables": {}, "figures": {}, "summary": {"rows": 0}}
        _write_summary_bundle(section_dir, "single_op_ablation", payload)
        return SectionResult(name="single_op_ablation", outputs=payload)

    baseline_rows = ablation_frame[ablation_frame["variant"] == "(baseline)"].copy()
    variant_rows = ablation_frame[ablation_frame["variant"] != "(baseline)"].copy()
    table_4_7 = ablation_frame.sort_values(["model_group", "test_mape_delta_vs_baseline"]).reset_index(drop=True)
    csv_47, md_47 = write_frame_csv_md(table_4_7, tables_dir / TABLE_FILENAMES["4-7"], tables_dir / "table_4_7_single_op_ablation_summary.md", "Table 4-7 Single-op Ablation Summary")

    pivot = variant_rows.pivot_table(
        index="variant",
        columns="model_group",
        values="test_mape_delta_vs_baseline",
        aggfunc="mean",
    ).fillna(0.0)
    if not pivot.empty:
        plot_heatmap(
            pivot,
            figures_dir / FIGURE_FILENAMES["4-16"],
            "Figure 4-16 Ablation Delta MAPE",
            xlabel="model group",
            ylabel="variant",
        )

    group_best = (
        variant_rows.sort_values(["model_group", "test_mape_delta_vs_baseline"])
        .groupby("model_group", as_index=False)
        .first()
    )
    if not group_best.empty:
        plot_bar(
            group_best,
            "model_group",
            "test_mape_delta_vs_baseline",
            figures_dir / FIGURE_FILENAMES["4-19"],
            "Figure 4-19 Best Ablation Improvement",
            ylabel="delta MAPE vs baseline",
            color="#54A24B",
        )

    if not variant_rows.empty:
        variant_rows = variant_rows.copy()
        variant_rows["feature_count_drop"] = variant_rows["dropped_feature_count"].astype(int)
        plot_grouped_bar(
            variant_rows.head(min(12, len(variant_rows))).assign(label=lambda frame: frame["model_group"] + " / " + frame["variant"]),
            "label",
            ["test_mape", "test_mape_delta_vs_baseline"],
            figures_dir / FIGURE_FILENAMES["4-17"],
            "Figure 4-17 Ablation Absolute and Delta MAPE",
            ylabel="MAPE",
            legend_labels=["test MAPE", "delta vs baseline"],
        )

    payload = {
        "tables": {"4-7": str(csv_47)},
        "figures": {
            "4-16": str(figures_dir / FIGURE_FILENAMES["4-16"]),
            "4-17": str(figures_dir / FIGURE_FILENAMES["4-17"]),
            "4-19": str(figures_dir / FIGURE_FILENAMES["4-19"]),
        },
        "summary": {
            "rows": int(len(ablation_frame)),
            "model_groups": list(sorted(ablation_frame["model_group"].unique().tolist())),
        },
    }
    _write_summary_bundle(section_dir, "single_op_ablation", payload)
    return SectionResult(name="single_op_ablation", outputs=payload)


def run_e2e_core(
    output_root: Path | None = None,
    *,
    e2e_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]

    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    summary = read_json(e2e_root / "summary.json")
    full_metrics = pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)

    grouped = full_metrics.groupby(["batch_size", "inter_threads"], as_index=False).agg(
        combo_count=("combo", "count"),
        mean_mape=("ape", "mean"),
        mean_abs_error_us=("abs_error_us", "mean"),
        p95_ape=("ape", lambda values: float(pd.Series(values).quantile(0.95))),
    )
    grouped = grouped.sort_values(["batch_size", "inter_threads"]).reset_index(drop=True)

    table_5 = pd.DataFrame(
        [
            {
                "metric": "total_test_combos",
                "value": summary["combo_counts"]["total_test_combos"],
            },
            {
                "metric": "full_combo_count",
                "value": summary["combo_counts"]["full_combo_count"],
            },
            {
                "metric": "full_graph_count",
                "value": summary["full_graph_metrics"]["count"],
            },
            {
                "metric": "full_graph_mape",
                "value": summary["full_graph_metrics"]["mape"],
            },
            {
                "metric": "full_graph_p95_ape",
                "value": summary["full_graph_metrics"]["p95_ape"],
            },
            {
                "metric": "worst_combo_case",
                "value": summary["full_graph_metrics"]["worst_combo"]["case_id"],
            },
            {
                "metric": "worst_combo_combo",
                "value": summary["full_graph_metrics"]["worst_combo"]["combo"],
            },
        ]
    )
    csv_5, md_5 = write_frame_csv_md(table_5, tables_dir / TABLE_FILENAMES["4-5"], tables_dir / "table_4_5_e2e_static_summary.md", "Table 4-5 Static E2E Summary")
    grouped_csv = tables_dir / "table_4_5_e2e_batch_inter_threads_grouped.csv"
    grouped_md = tables_dir / "table_4_5_e2e_batch_inter_threads_grouped.md"
    write_frame_csv_md(grouped, grouped_csv, grouped_md, "Table 4-5b E2E grouped summary")

    plot_scatter(
        full_metrics.assign(actual=full_metrics["actual_e2e_us"], predicted=full_metrics["predicted_e2e_us"]),
        "actual",
        "predicted",
        figures_dir / FIGURE_FILENAMES["4-11"],
        "Figure 4-11 E2E Predicted vs Actual",
        hue=None,
    )

    heatmap = grouped.pivot(index="batch_size", columns="inter_threads", values="mean_mape").sort_index()
    if not heatmap.empty:
        plot_heatmap(
            heatmap,
            figures_dir / FIGURE_FILENAMES["4-12"],
            "Figure 4-12 E2E Mean MAPE by Batch Size and inter_threads",
            xlabel="inter_threads",
            ylabel="batch_size",
        )

    distribution = grouped.groupby("inter_threads", as_index=False).agg(mean_mape=("mean_mape", "mean"), p95_ape=("p95_ape", "mean"))
    if not distribution.empty:
        plot_grouped_bar(
            distribution,
            "inter_threads",
            ["mean_mape", "p95_ape"],
            figures_dir / FIGURE_FILENAMES["4-13"],
            "Figure 4-13 E2E Error Distribution by inter_threads",
            ylabel="value",
            legend_labels=["mean MAPE", "p95 APE"],
        )

    payload = {
        "tables": {
            "4-5": str(csv_5),
            "4-5b": str(grouped_csv),
        },
        "figures": {
            "4-11": str(figures_dir / FIGURE_FILENAMES["4-11"]),
            "4-12": str(figures_dir / FIGURE_FILENAMES["4-12"]),
            "4-13": str(figures_dir / FIGURE_FILENAMES["4-13"]),
        },
        "summary": summary,
    }
    _write_summary_bundle(section_dir, "e2e_core", payload)
    return SectionResult(name="e2e_core", outputs=payload)


def run_e2e_sum_baseline(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    prediction_df = load_prediction_frame(single_op_root, split="test")
    combo_specs = build_combo_specs(prediction_df, ort_root=Path(DEFAULT_ORT_ROOT))
    rows: list[dict[str, Any]] = []
    for combo_spec in combo_specs:
        combo_rows = prediction_df[(prediction_df["case_id"] == combo_spec.case_id) & (prediction_df["combo"] == combo_spec.combo)].copy()
        op_shapes_df = load_op_shapes_frame(combo_spec.artifact_paths.shape_csv)
        graph = build_op_graph(op_shapes_df)
        schedule_result = schedule_combo(combo_spec, graph, combo_rows)
        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        actual = compute_mean_batch_span(timeline_df, [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices])
        sum_pred = float(combo_rows["pred_us"].astype(float).sum())
        rows.append(
            {
                "case_id": combo_spec.case_id,
                "combo": combo_spec.combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "sum_predicted_us": sum_pred,
                "static_predicted_us": float(schedule_result.predicted_full_graph_us),
                "actual_e2e_us": float(actual.mean_span_us),
                "sum_abs_error_us": abs(sum_pred - float(actual.mean_span_us)),
                "static_abs_error_us": abs(float(schedule_result.predicted_full_graph_us) - float(actual.mean_span_us)),
                "sum_ape": abs(sum_pred - float(actual.mean_span_us)) / float(actual.mean_span_us) if actual.mean_span_us else 0.0,
                "static_ape": abs(float(schedule_result.predicted_full_graph_us) - float(actual.mean_span_us)) / float(actual.mean_span_us) if actual.mean_span_us else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(
        [
            {
                "metric": "rows",
                "value": int(len(frame)),
            },
            {
                "metric": "sum_mape",
                "value": float(frame["sum_ape"].mean()) if not frame.empty else None,
            },
            {
                "metric": "static_mape",
                "value": float(frame["static_ape"].mean()) if not frame.empty else None,
            },
            {
                "metric": "mape_delta",
                "value": float(frame["sum_ape"].mean() - frame["static_ape"].mean()) if not frame.empty else None,
            },
        ]
    )
    csv_6, md_6 = write_frame_csv_md(summary_frame, tables_dir / TABLE_FILENAMES["4-6"], tables_dir / "table_4_6_e2e_sum_baseline.md", "Table 4-6 Simple Sum Baseline Summary")

    if not frame.empty:
        comparison = frame[["combo", "sum_ape", "static_ape"]].copy()
        comparison["combo"] = comparison["combo"].astype(str)
        plot_grouped_bar(
            comparison.head(min(18, len(comparison))),
            "combo",
            ["sum_ape", "static_ape"],
            figures_dir / FIGURE_FILENAMES["4-18"],
            "Figure 4-18 Simple Sum vs Static Scheduler",
            ylabel="APE",
            legend_labels=["simple sum", "static scheduler"],
        )

    payload = {
        "tables": {"4-6": str(csv_6)},
        "figures": {"4-18": str(figures_dir / FIGURE_FILENAMES["4-18"])},
        "summary": {
            "rows": int(len(frame)),
            "sum_mape": float(frame["sum_ape"].mean()) if not frame.empty else None,
            "static_mape": float(frame["static_ape"].mean()) if not frame.empty else None,
        },
    }
    _write_summary_bundle(section_dir, "e2e_sum_baseline", payload)
    return SectionResult(name="e2e_sum_baseline", outputs=payload)


def run_timeline_cases(
    output_root: Path | None = None,
    *,
    single_op_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["e2e"]
    timeline_dir = section_dir / "timeline_cases"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    prediction_df = load_prediction_frame(single_op_root, split="test")
    combo_specs = build_combo_specs(prediction_df, ort_root=Path(DEFAULT_ORT_ROOT))
    combo_map = {(spec.case_id, spec.combo): spec for spec in combo_specs}

    summary_rows: list[dict[str, Any]] = []
    gantt_frames: list[pd.DataFrame] = []
    critical_frames: list[pd.DataFrame] = []
    for case_id, combo in TIMELINE_CASES:
        combo_spec = combo_map[(case_id, combo)]
        combo_rows = prediction_df[(prediction_df["case_id"] == case_id) & (prediction_df["combo"] == combo)].copy()
        graph = build_op_graph(load_op_shapes_frame(combo_spec.artifact_paths.shape_csv))
        schedule_result = schedule_combo(combo_spec, graph, combo_rows)
        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        actual = compute_mean_batch_span(timeline_df, [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices])
        actual_obs = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)
        duration_by_node = {int(row.op_idx): float(row.pred_us) for row in combo_rows.itertuples(index=False)}
        critical_path, critical_us = _compute_longest_path(graph, duration_by_node)

        node_span_map = schedule_result.node_span_map()
        path_rows = []
        for step, node_idx in enumerate(critical_path):
            node = graph[node_idx]
            span = node_span_map.get(node_idx)
            if span is None:
                continue
            path_rows.append(
                {
                    "case_id": case_id,
                    "combo": combo,
                    "step": step,
                    "node_idx": node_idx,
                    "node_name": node.node_name,
                    "op_type": node.op_type,
                    "partition": node.partition,
                    "predicted_start_us": span.start_us,
                    "predicted_end_us": span.end_us,
                    "predicted_duration_us": span.duration_us,
                }
            )
        critical_frame = pd.DataFrame(path_rows)
        critical_frames.append(critical_frame)

        gantt_rows = []
        for span in schedule_result.task_spans:
            gantt_rows.append(
                {
                    "label": span.task_id,
                    "task_kind": span.task_kind,
                    "partition": span.partition,
                    "start_us": span.start_us,
                    "end_us": span.end_us,
                    "duration_us": span.duration_us,
                }
            )
        gantt_frame = pd.DataFrame(gantt_rows)
        gantt_frames.append(gantt_frame)

        case_dir = timeline_dir / f"{case_id}__{combo}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            case_dir / "summary.json",
            {
                "case_id": case_id,
                "combo": combo,
                "predicted_full_graph_us": float(schedule_result.predicted_full_graph_us),
                "actual_full_graph_us": float(actual.mean_span_us),
                "predicted_critical_path_us": critical_us,
                "embedding_launch_order": actual_obs.representative_launch_order,
                "max_gather_concurrency": actual_obs.representative_max_gather_concurrency,
            },
        )
        gantt_frame.to_csv(case_dir / "task_spans.csv", index=False)
        critical_frame.to_csv(case_dir / "critical_path.csv", index=False)

        summary_rows.append(
            {
                "case_id": case_id,
                "combo": combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "predicted_full_graph_us": float(schedule_result.predicted_full_graph_us),
                "actual_full_graph_us": float(actual.mean_span_us),
                "predicted_critical_path_us": critical_us,
                "critical_path_nodes": len(critical_path),
                "ape": abs(float(schedule_result.predicted_full_graph_us) - float(actual.mean_span_us)) / float(actual.mean_span_us) if actual.mean_span_us else 0.0,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    csv_path, md_path = write_frame_csv_md(summary_frame, tables_dir / "table_4_6_timeline_case_summary.csv", tables_dir / "table_4_6_timeline_case_summary.md", "Table 4-6 Timeline Case Summary")

    if gantt_frames:
        plot_frame = pd.concat(gantt_frames, ignore_index=True)
        if not plot_frame.empty:
            plt = _import_pyplot()
            fig, axes = plt.subplots(len(TIMELINE_CASES), 1, figsize=(13, 3.5 * len(TIMELINE_CASES)), sharex=False)
            if len(TIMELINE_CASES) == 1:
                axes = [axes]
            for ax, (case_id, combo), gantt_frame in zip(axes, TIMELINE_CASES, gantt_frames):
                for row in gantt_frame.itertuples(index=False):
                    color = "#4C78A8" if row.partition == "bottom" else "#F58518" if row.partition == "embedding" else "#54A24B"
                    ax.broken_barh([(float(row.start_us), float(row.duration_us))], (0, 1), facecolors=color, alpha=0.7)
                ax.set_yticks([])
                ax.set_title(f"{case_id} / {combo}")
                ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
            axes[-1].set_xlabel("time (us)")
            save_figure(fig, figures_dir / FIGURE_FILENAMES["4-14"])

    if critical_frames:
        critical_plot = pd.concat(
            [
                frame.assign(case_label=f"{row['case_id']} / {row['combo']}")
                for frame, row in zip(critical_frames, summary_rows)
            ],
            ignore_index=True,
        )
        if not critical_plot.empty:
            plt = _import_pyplot()
            fig, ax = plt.subplots(figsize=(13, 5))
            colors = {"bottom": "#4C78A8", "embedding": "#F58518", "tail": "#54A24B"}
            for case_label, sub in critical_plot.groupby("case_label"):
                left = 0.0
                for row in sub.itertuples(index=False):
                    ax.bar(case_label, row.predicted_duration_us, bottom=left, color=colors.get(row.partition, "#7F7F7F"))
                    left += row.predicted_duration_us
            ax.set_title("Figure 4-15 Critical Path Breakdown")
            ax.set_ylabel("predicted duration (us)")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            save_figure(fig, figures_dir / FIGURE_FILENAMES["4-15"])

    payload = {
        "tables": {"timeline_case_summary": str(csv_path)},
        "figures": {
            "4-14": str(figures_dir / FIGURE_FILENAMES["4-14"]),
            "4-15": str(figures_dir / FIGURE_FILENAMES["4-15"]),
        },
        "summary": summary_rows,
    }
    _write_summary_bundle(section_dir, "timeline_cases", payload)
    return SectionResult(name="timelines", outputs=payload)


def build_figures_catalog(output_root: Path | None = None) -> SectionResult:
    layout = ensure_output_layout(output_root)
    rows = []
    for figure_no, filename in FIGURE_FILENAMES.items():
        rows.append(
            {
                "figure_no": figure_no,
                "filename": filename,
                "path": str(layout["figures"] / filename),
            }
        )
    frame = pd.DataFrame(rows).sort_values("figure_no").reset_index(drop=True)
    csv_path, md_path = write_frame_csv_md(frame, layout["manifests"] / "figures_catalog.csv", layout["manifests"] / "figures_catalog.md", "Chapter 4 Figure Catalog")
    payload = {"csv": str(csv_path), "md": str(md_path), "rows": rows}
    write_json(layout["manifests"] / "figures_catalog.json", payload)
    return SectionResult(name="figures", outputs=payload)


def build_chapter4_draft(output_root: Path | None = None) -> SectionResult:
    layout = ensure_output_layout(output_root)
    single_op_summary = read_json(layout["single_op"] / "single_op_core_summary.json") if (layout["single_op"] / "single_op_core_summary.json").exists() else {}
    ood_summary = read_json(layout["single_op"] / "single_op_ood_summary.json") if (layout["single_op"] / "single_op_ood_summary.json").exists() else {}
    e2e_summary = read_json(layout["e2e"] / "e2e_core_summary.json") if (layout["e2e"] / "e2e_core_summary.json").exists() else {}
    ablation_summary = read_json(layout["ablation"] / "single_op_ablation_summary.json") if (layout["ablation"] / "single_op_ablation_summary.json").exists() else {}
    timeline_summary = read_json(layout["e2e"] / "timeline_cases_summary.json") if (layout["e2e"] / "timeline_cases_summary.json").exists() else {}
    figures_catalog = read_json(layout["manifests"] / "figures_catalog.json") if (layout["manifests"] / "figures_catalog.json").exists() else {"rows": []}

    single_op_metrics = single_op_summary.get("summary", {})
    e2e_metrics = e2e_summary.get("summary", {})
    ood_rows = ood_summary.get("summary", [])
    ablation_rows = ablation_summary.get("summary", [])
    timeline_rows = timeline_summary.get("summary", [])

    lines = [
        "# 第四章实验结果",
        "",
        "## 4.1 实验平台与数据采集方法",
        "",
        f"- 单算子统一实验目录：`{SINGLE_OP_ARTIFACT_ROOT}`",
        f"- 整图静态聚合目录：`{E2E_ARTIFACT_ROOT}`",
        f"- 基线目录：`{BASELINE_MODEL_ROOT}`",
        f"- 统一输出目录：`{layout['root']}`",
        "",
        f"表 4-1 记录了平台与数据规模概览，图 4-1 对应统一统计视图。",
        "",
        "## 4.2 算子级性能建模实验",
        "",
        f"- 表 4-2 汇总五个模型组的测试指标与训练口径。",
        f"- 表 4-3 展示代表算子 `{' / '.join(REPRESENTATIVE_OP_TYPES)}` 的测试误差统计。",
        f"- 图 4-2 到图 4-8 展示模型组误差、baseline 对比、代表算子图、散点与训练曲线。",
        "",
        f"- 章节内单算子核心结果摘要：group mean MAPE = {single_op_metrics.get('group_test_mape_mean', 0.0):.6f}, worst group MAPE = {single_op_metrics.get('group_test_mape_worst', 0.0):.6f}.",
        "",
        f"### 4.2.1 OOD 泛化",
        "",
        f"- 章节 OOD 规则采用 batch holdout `{_sanitize_list(OOD_BATCH_HOLDS)}` 与 `num_threads={OOD_NUM_THREADS_HOLD}`。",
        f"- OOD 切片摘要写入单独的 CSV/JSON 文件，图 4-9 到图 4-10 展示 slice 与 generalization 参考结果。",
        f"- 当前已写入 {len(ood_rows)} 条 OOD slice 记录。",
        "",
        "## 4.3 整图性能聚合实验",
        "",
        f"- 表 4-5 汇总静态调度器在 `v1_300_iter_quick_nodrop` 上的主指标。",
        f"- 图 4-11 到图 4-13 给出预测-真实散点、batch/inter_threads 分组热力图与误差分布。",
        f"- 图 4-14 与图 4-15 来自典型时间线与关键路径导出。",
        "",
        f"- E2E 结果摘要：full_graph MAPE = {e2e_metrics.get('full_graph_metrics', {}).get('mape', 0.0):.6f}, worst combo = {e2e_metrics.get('full_graph_metrics', {}).get('worst_combo', {}).get('case_id', 'n/a')} / {e2e_metrics.get('full_graph_metrics', {}).get('worst_combo', {}).get('combo', 'n/a')}.",
        "",
        "## 4.4 消融实验与误差分析",
        "",
        f"- 表 4-6 汇总简单求和基线与静态调度器的差异。",
        f"- 表 4-7 汇总单算子特征消融实验结果。",
        f"- 图 4-16、图 4-17、图 4-18、图 4-19 分别对应特征消融热图、增量误差、求和基线与最佳消融结果。",
        "",
        f"- 总体消融样本数：{ablation_summary.get('summary', {}).get('rows', len(ablation_rows))}。",
        "",
        "## 4.5 本章小结",
        "",
        f"- 关键时间线案例数：{len(timeline_rows) if isinstance(timeline_rows, list) else 0}。",
        f"- 统一 figure catalog 共收录 {len(figures_catalog.get('rows', []))} 张图。",
        f"- 本章草稿由 `{CHAPTER4_DRAFT_PATH.name}` 自动生成，并可由总入口重复刷新。",
        "",
    ]
    draft = "\n".join(lines).rstrip() + "\n"
    CHAPTER4_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAPTER4_DRAFT_PATH.write_text(draft, encoding="utf-8")
    return SectionResult(name="draft", outputs={"draft_path": str(CHAPTER4_DRAFT_PATH)})


def build_run_manifest(
    *,
    output_root: Path | None = None,
    sections: list[SectionResult],
    single_op_artifact_root: Path = SINGLE_OP_ARTIFACT_ROOT,
    e2e_artifact_root: Path = E2E_ARTIFACT_ROOT,
) -> Path:
    layout = ensure_output_layout(output_root)
    manifest = {
        "output_root": str(layout["root"]),
        "single_op_artifact_root": str(single_op_artifact_root),
        "e2e_artifact_root": str(e2e_artifact_root),
        "python_version": sys.version.split()[0],
        "execution_order": [section.name for section in sections],
        "generated_files": [
            path
            for section in sections
            for path in section.outputs.values()
            if isinstance(path, str)
        ],
        "sections": [
            {
                "name": section.name,
                "outputs": section.outputs,
            }
            for section in sections
        ],
    }
    return write_json(layout["manifests"] / "run_manifest.json", manifest)
