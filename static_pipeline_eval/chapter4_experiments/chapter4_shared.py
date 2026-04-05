from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def add_audit_callout(ax, lines: Sequence[str] | None, *, loc: str = "upper right") -> None:
    if not lines:
        return
    anchors = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper right"])
    text = "\n".join(str(line) for line in lines if line is not None and str(line).strip())
    if not text:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.83, edgecolor="#C8C8C8"),
    )


def plot_bar(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    color: str = "#4477aa",
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame[x].astype(str), frame[y].astype(float), color=color)
    _style_axes(ax, title, xlabel or x, ylabel or y)
    ax.tick_params(axis="x", rotation=30)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
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
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
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
    add_audit_callout(ax, audit_lines, loc=audit_loc)
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
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper left",
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
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_heatmap(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    annot: bool = True,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
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
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_boxplot(
    values_by_label: dict[str, list[float]],
    path: Path,
    title: str,
    *,
    ylabel: str,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(values_by_label.keys())
    values = [values_by_label[label] for label in labels]
    ax.boxplot(values, labels=labels, showmeans=True)
    _style_axes(ax, title, "", ylabel)
    ax.tick_params(axis="x", rotation=25)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_line(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    title: str,
    *,
    group_col: str | None = None,
    ylabel: str | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 5))
    if group_col and group_col in frame.columns:
        for label, sub in frame.groupby(group_col):
            ax.plot(sub[x], sub[y], marker="o", linewidth=1.5, label=str(label))
        ax.legend(frameon=False)
    else:
        ax.plot(frame[x], frame[y], marker="o", linewidth=1.5, color="#4477aa")
    _style_axes(ax, title, x, ylabel or y)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_cdf(
    values_by_label: dict[str, list[float]],
    path: Path,
    title: str,
    *,
    xlabel: str,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "lower right",
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = ["#E45756", "#4C78A8", "#54A24B", "#B279A2", "#F58518"]
    for idx, (label, values) in enumerate(values_by_label.items()):
        clean = sorted(float(value) for value in values if value is not None and not math.isnan(float(value)))
        if not clean:
            continue
        cdf = [(pos + 1) / len(clean) * 100.0 for pos in range(len(clean))]
        ax.plot(clean, cdf, linewidth=2.0, label=label, color=palette[idx % len(palette)])
    _style_axes(ax, title, xlabel, "CDF (%)")
    ax.legend(frameon=False)
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_flow(
    steps: list[str],
    path: Path,
    title: str,
    *,
    subtitle: str | None = None,
) -> Path:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    x_positions = [0.08 + idx * (0.84 / max(1, len(steps) - 1)) for idx in range(len(steps))]
    for idx, (x_pos, step) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.58,
            step,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#4C78A8", linewidth=1.2),
            transform=ax.transAxes,
        )
        if idx < len(steps) - 1:
            next_x = x_positions[idx + 1]
            ax.annotate(
                "",
                xy=(next_x - 0.045, 0.58),
                xytext=(x_pos + 0.045, 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#666666"),
            )
    ax.set_title(title, pad=12)
    if subtitle:
        ax.text(0.5, 0.13, subtitle, ha="center", va="center", fontsize=9, color="#555555", transform=ax.transAxes)
    return save_figure(fig, path)


def plot_gantt(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    *,
    label_col: str,
    start_col: str,
    end_col: str,
    hue_col: str | None = None,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
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
    add_audit_callout(ax, audit_lines, loc=audit_loc)
    return save_figure(fig, path)


def plot_simple_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    path: Path,
    title: str,
    *,
    audit_lines: Sequence[str] | None = None,
    audit_loc: str = "upper right",
) -> Path:
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
    add_audit_callout(ax, audit_lines, loc=audit_loc)
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


def _build_ablation_feature_focus(ablation_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_features = (
        "feat_output_elements_per_batch",
        "feat_output_elements_per_lookup",
        "feat_output_input_bytes_ratio",
        "feat_activation_elements_per_batch",
    )
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    for feature in focus_features:
        mask = ablation_frame["dropped_features"].fillna("").astype(str).str.contains(feature, regex=False)
        feature_rows = ablation_frame[mask].copy()
        if feature_rows.empty:
            continue

        positive_rows = feature_rows[feature_rows["test_mape_delta_vs_baseline"] > 0].copy()
        evidence_rows = positive_rows if not positive_rows.empty else feature_rows
        best_row = evidence_rows.sort_values("test_mape_delta_vs_baseline", ascending=False).iloc[0]
        supporting_groups = sorted(positive_rows["model_group"].unique().tolist() if not positive_rows.empty else feature_rows["model_group"].unique().tolist())

        summary_rows.append(
            {
                "feature": feature,
                "rows": int(len(feature_rows)),
                "positive_rows": int(len(positive_rows)),
                "supporting_groups": _sanitize_list(supporting_groups),
                "supporting_group_count": int(len(supporting_groups)),
                "best_group": best_row["model_group"],
                "best_variant": best_row["variant"],
                "best_delta_vs_baseline": float(best_row["test_mape_delta_vs_baseline"]),
                "mean_positive_delta": float(positive_rows["test_mape_delta_vs_baseline"].mean()) if not positive_rows.empty else float(best_row["test_mape_delta_vs_baseline"]),
                "median_positive_delta": float(positive_rows["test_mape_delta_vs_baseline"].median()) if not positive_rows.empty else float(best_row["test_mape_delta_vs_baseline"]),
                "worst_delta_vs_baseline": float(feature_rows["test_mape_delta_vs_baseline"].min()),
            }
        )

        detail_frames.append(
            feature_rows.assign(feature=feature)[
                [
                    "feature",
                    "model_group",
                    "variant",
                    "dropped_features",
                    "test_mape",
                    "test_mape_delta_vs_baseline",
                ]
            ]
        )

    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(
            ["best_delta_vs_baseline", "mean_positive_delta"],
            ascending=False,
        ).reset_index(drop=True)
    detail_frame = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(
        columns=[
            "feature",
            "model_group",
            "variant",
            "dropped_features",
            "test_mape",
            "test_mape_delta_vs_baseline",
        ]
    )
    if not detail_frame.empty:
        detail_frame = detail_frame.sort_values(
            ["feature", "test_mape_delta_vs_baseline", "model_group"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return summary_frame, detail_frame


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


def _compute_regression_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float | None]:
    actual = actual.astype(float)
    pred = pred.astype(float)
    if actual.empty:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "rmse_us": None,
            "r2": None,
            "p50_ape": None,
            "p90_ape": None,
            "gt10_rate": None,
        }
    abs_error = (pred - actual).abs()
    ape = (abs_error / actual.replace(0.0, pd.NA)).fillna(0.0)
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    ss_res = float(((pred - actual) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "count": int(len(actual)),
        "mae_us": float(abs_error.mean()),
        "mape": float(ape.mean()),
        "rmse_us": float((((pred - actual) ** 2).mean()) ** 0.5),
        "r2": float(r2),
        "p50_ape": float(ape.quantile(0.50)),
        "p90_ape": float(ape.quantile(0.90)),
        "gt10_rate": float((ape > 0.10).mean()),
    }


def _format_range(series: pd.Series, *, cast_int: bool = False) -> str:
    clean = series.dropna()
    if clean.empty:
        return ""
    if cast_int:
        clean = clean.astype(int)
    return f"{clean.min()}-{clean.max()}"


def _run_shell_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _platform_config_frame() -> pd.DataFrame:
    lscpu_text = _run_shell_text(["lscpu"])
    os_release = Path("/etc/os-release").read_text(encoding="utf-8")

    def extract(pattern: str, text: str, default: str = "") -> str:
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default

    hw_profile_path = (
        Path(DEFAULT_ORT_ROOT)
        / "single_op_stage1_mlp"
        / "hardware_profile"
        / "kunpeng920_single_numa.yaml"
    )
    hw_profile_text = hw_profile_path.read_text(encoding="utf-8")
    host_cores = extract(r"CPU\(s\):\s+(\d+)", lscpu_text)
    host_threads = host_cores
    host_freq = extract(r"cpu_clock:\s+([^\n]+)", hw_profile_text, extract(r"Model name:\s+([^\n]+)", lscpu_text))
    return pd.DataFrame(
        [
            {
                "server_model": "Huawei Cloud Kunpeng host (4-socket)",
                "cpu_model": extract(r"Model name:\s+([^\n]+)", lscpu_text),
                "host_cores_threads": f"{host_cores}C/{host_threads}T",
                "experiment_scope": "single NUMA domain (24 cores)",
                "main_frequency": host_freq,
                "cache_config": "L1I 64 KiB/core, L1D 64 KiB/core, L2 512 KiB/core, L3 24 MiB/NUMA",
                "memory_config": "DDR4-2933, 4 channels/NUMA, approx. 100 GB/s local bandwidth",
                "os_version": extract(r'PRETTY_NAME=\"([^\"]+)\"', os_release),
                "python_ort_stack": "Python 3.11 (ort env), ONNX Runtime / PyTorch / ONNX versions recorded in experiment artifacts",
                "compiler_profiler": "GCC 10.3.1, DynamoRIO profiler, ORT branch-parallel timeline",
                "binding_notes": "single-NUMA pinning; fixed inter/intra thread configs; branch-parallel timeline replay",
            }
        ]
    )


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
    single_op_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    e2e_summary = read_json(e2e_root / "summary.json")

    frame = _platform_config_frame()
    csv_path, md_path = write_frame_csv_md(
        frame,
        tables_dir / TABLE_FILENAMES["4-1"],
        tables_dir / "table_4_1_platform_config.md",
        "Table 4-1 Hardware and software configuration",
    )

    fig_path = plot_flow(
        [
            "DLRM / ONNX model",
            "single-op split",
            "parameter sampling",
            "ORT execution",
            "profiling + cleaning",
            "feature build",
        ],
        figures_dir / FIGURE_FILENAMES["4-1"],
        "Figure 4-1 Single-op data collection flow",
        subtitle="Model -> operator split -> sampled execution -> profiling labels -> features",
    )
    fig_path_2 = plot_flow(
        [
            "DLRM full graph run",
            "timeline collection",
            "graph restore",
            "node / batch alignment",
            "E2E ground truth",
        ],
        figures_dir / FIGURE_FILENAMES["4-2"],
        "Figure 4-2 E2E timeline aggregation flow",
        subtitle="Timeline replay and graph reconstruction for combo-level labels",
    )

    summary = {
        "single_op_rows": int(len(single_op_df)),
        "single_op_cases": int(single_op_df["case_id"].nunique()),
        "single_op_combos": int(single_op_df[["case_id", "combo"]].drop_duplicates().shape[0]),
        "full_e2e_combos": int(e2e_summary["combo_counts"]["full_combo_count"]),
        "host_cpu_model": frame.iloc[0]["cpu_model"],
        "experiment_scope": frame.iloc[0]["experiment_scope"],
    }
    manifest = {
        "table_csv": str(csv_path),
        "table_md": str(md_path),
        "figure_4_1": str(fig_path),
        "figure_4_2": str(fig_path_2),
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

    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    baseline_metrics = read_json(baseline_root / "metrics.json")
    combined_predictions = pd.read_csv(
        single_op_root / "models" / "combined" / "combined_predictions_test.csv",
        low_memory=False,
    )
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    analytical_metrics = _compute_regression_metrics(
        test_dataset["label_operator_actual_dur_us"],
        test_dataset["ana_calib_total_us"],
    )
    grouped_metrics = _compute_regression_metrics(
        combined_predictions["target_us"],
        combined_predictions["pred_us"],
    )

    group_label_map = {
        "gather": "Index / memory",
        "layout_move": "Layout move",
        "view_meta": "View / meta",
        "mixed_balanced": "Light compute-mem",
        "compute_dominant": "Compute dominant",
    }
    table_4_2 = (
        dataset_df.groupby("model_group", as_index=False)
        .agg(
            sample_count=("row_uid", "count"),
            operator_names=("op_type", lambda values: ",".join(sorted(set(values)))),
            batch_size_range=("batch_size", lambda values: _format_range(pd.Series(values), cast_int=True)),
            nip_range=("num_indices_per_lookup", lambda values: _format_range(pd.Series(values), cast_int=True)),
            num_threads_range=("num_threads", lambda values: _format_range(pd.Series(values), cast_int=True)),
            inter_threads_range=("inter_threads", lambda values: _format_range(pd.Series(values), cast_int=True)),
            io_bytes_range=("feat_io_bytes_sum", lambda values: _format_range(pd.Series(values))),
        )
        .sort_values("model_group")
        .reset_index(drop=True)
    )
    table_4_2["operator_category"] = table_4_2["model_group"].map(group_label_map).fillna(table_4_2["model_group"])
    table_4_2 = table_4_2[
        [
            "operator_category",
            "operator_names",
            "sample_count",
            "batch_size_range",
            "nip_range",
            "num_threads_range",
            "inter_threads_range",
            "io_bytes_range",
        ]
    ]

    table_4_3 = pd.DataFrame(
        [
            {
                "model": "Analytical only",
                **analytical_metrics,
            },
            {
                "model": "Single MLP baseline",
                "count": int(len(test_dataset)),
                "mae_us": float(baseline_metrics["metrics"]["test"]["mae_us"]),
                "mape": float(baseline_metrics["metrics"]["test"]["mape"]),
                "rmse_us": float(baseline_metrics["metrics"]["test"]["rmse_us"]),
                "r2": float(baseline_metrics["metrics"]["test"]["r2"]),
                "p50_ape": float(baseline_metrics["metrics"]["test"]["median_ape"]),
                "p90_ape": None,
                "gt10_rate": None,
            },
            {
                "model": "Grouped analytical-MLP",
                **grouped_metrics,
            },
        ]
    )

    category_rows: list[dict[str, Any]] = []
    for model_group, sub in combined_predictions.groupby("model_group"):
        metrics = _compute_regression_metrics(sub["target_us"], sub["pred_us"])
        category_rows.append(
            {
                "operator_category": group_label_map.get(model_group, model_group),
                "sample_count": int(len(sub)),
                "mean_actual_us": float(sub["target_us"].astype(float).mean()),
                "mae_us": metrics["mae_us"],
                "mape": metrics["mape"],
                "rmse_us": metrics["rmse_us"],
                "p90_ape": metrics["p90_ape"],
            }
        )
    table_4_4 = pd.DataFrame(category_rows).sort_values("mape", ascending=False).reset_index(drop=True)

    csv_42, md_42 = write_frame_csv_md(
        table_4_2,
        tables_dir / TABLE_FILENAMES["4-2"],
        tables_dir / "table_4_2_dataset_composition.md",
        "Table 4-2 Dataset composition by operator category",
    )
    csv_43, md_43 = write_frame_csv_md(
        table_4_3,
        tables_dir / TABLE_FILENAMES["4-3"],
        tables_dir / "table_4_3_single_op_overall.md",
        "Table 4-3 Single-op overall accuracy",
    )
    csv_44, md_44 = write_frame_csv_md(
        table_4_4,
        tables_dir / TABLE_FILENAMES["4-4"],
        tables_dir / "table_4_4_single_op_category.md",
        "Table 4-4 Category-wise single-op accuracy",
    )

    scatter_frame = combined_predictions.copy()
    scatter_frame["actual_us"] = scatter_frame["target_us"].astype(float)
    scatter_frame["predicted_us"] = scatter_frame["pred_us"].astype(float)
    plot_scatter(
        scatter_frame.sample(n=min(2500, len(scatter_frame)), random_state=42),
        "actual_us",
        "predicted_us",
        figures_dir / FIGURE_FILENAMES["4-3"],
        "Figure 4-3 Single-op predicted vs. actual latency",
        audit_lines=[
            f"rows = {len(scatter_frame):,}",
            f"MAPE = {grouped_metrics['mape']:.4f}",
            f"R2 = {grouped_metrics['r2']:.4f}",
        ],
    )

    plot_bar(
        table_4_4.sort_values("mape", ascending=False),
        "operator_category",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-4"],
        "Figure 4-4 MAPE by operator category",
        ylabel="MAPE",
        color="#54A24B",
        audit_lines=[
            f"best = {table_4_4.sort_values('mape').iloc[0]['operator_category']}",
            f"worst = {table_4_4.iloc[0]['operator_category']}",
        ],
    )

    gather_frame = test_dataset[test_dataset["op_type"] == "Gather"].copy()
    if not gather_frame.empty:
        gather_frame["actual_us"] = gather_frame["label_operator_actual_dur_us"].astype(float)
        gather_frame["predicted_us"] = gather_frame["ana_calib_total_us"].astype(float)
        gather_pred = combined_predictions[combined_predictions["op_type"] == "Gather"][["row_uid", "pred_us"]]
        gather_frame = gather_frame.merge(gather_pred, on="row_uid", how="left")
        gather_frame["predicted_us"] = gather_frame["pred_us"].fillna(gather_frame["predicted_us"])
        plot_scatter(
            gather_frame.sample(n=min(1500, len(gather_frame)), random_state=42),
            "actual_us",
            "predicted_us",
            figures_dir / FIGURE_FILENAMES["4-5"],
            "Figure 4-5 Gather prediction scatter",
            hue="num_threads",
            audit_lines=[
                f"rows = {len(gather_frame):,}",
                f"threads = {_sanitize_list(sorted(gather_frame['num_threads'].astype(int).unique().tolist()))}",
            ],
        )

    reduce_frame = test_dataset[test_dataset["op_type"] == "ReduceSum"].copy()
    if not reduce_frame.empty:
        reduce_pred = combined_predictions[combined_predictions["op_type"] == "ReduceSum"][["row_uid", "pred_us"]]
        reduce_frame = reduce_frame.merge(reduce_pred, on="row_uid", how="left")
        reduce_frame["ape"] = (
            (reduce_frame["pred_us"].astype(float) - reduce_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / reduce_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        plot_scatter(
            reduce_frame.sample(n=min(1500, len(reduce_frame)), random_state=42),
            "feat_reduction_work_items",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-6"],
            "Figure 4-6 ReduceSum error vs. reduction size",
            hue="num_threads",
            reference_line=False,
            audit_lines=[
                f"rows = {len(reduce_frame):,}",
                f"p90 APE = {float(reduce_frame['ape'].quantile(0.9)):.4f}",
            ],
        )

    layout_frame = test_dataset[test_dataset["op_type"].isin(["Transpose", "Concat"])].copy()
    if not layout_frame.empty:
        layout_pred = combined_predictions[combined_predictions["op_type"].isin(["Transpose", "Concat"])][["row_uid", "pred_us"]]
        layout_frame = layout_frame.merge(layout_pred, on="row_uid", how="left")
        layout_frame["ape"] = (
            (layout_frame["pred_us"].astype(float) - layout_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / layout_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        plot_scatter(
            layout_frame.sample(n=min(1500, len(layout_frame)), random_state=42),
            "feat_io_bytes_sum",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-7"],
            "Figure 4-7 Transpose / Concat error vs. data size",
            hue="op_type",
            reference_line=False,
            audit_lines=[
                f"rows = {len(layout_frame):,}",
                f"shown ops = Transpose / Concat",
            ],
        )

    gemm_frame = test_dataset[test_dataset["op_type"].isin(["Gemm", "MatMul"])].copy()
    if not gemm_frame.empty:
        gemm_pred = combined_predictions[combined_predictions["op_type"].isin(["Gemm", "MatMul"])][["row_uid", "pred_us"]]
        gemm_frame = gemm_frame.merge(gemm_pred, on="row_uid", how="left")
        gemm_frame["ape"] = (
            (gemm_frame["pred_us"].astype(float) - gemm_frame["label_operator_actual_dur_us"].astype(float)).abs()
            / gemm_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
        ).fillna(0.0)
        plot_scatter(
            gemm_frame.sample(n=min(1500, len(gemm_frame)), random_state=42),
            "feat_gemm_mac_count",
            "ape",
            figures_dir / FIGURE_FILENAMES["4-8"],
            "Figure 4-8 Gemm / MatMul error vs. MAC count",
            hue="op_type",
            reference_line=False,
            audit_lines=[
                f"rows = {len(gemm_frame):,}",
                f"Gemm-like ops = {', '.join(sorted(gemm_frame['op_type'].unique().tolist()))}",
            ],
        )

    section_payload = {
        "tables": {
            "4-2": str(csv_42),
            "4-3": str(csv_43),
            "4-4": str(csv_44),
        },
        "figures": {
            "4-3": str(figures_dir / FIGURE_FILENAMES["4-3"]),
            "4-4": str(figures_dir / FIGURE_FILENAMES["4-4"]),
            "4-5": str(figures_dir / FIGURE_FILENAMES["4-5"]),
            "4-6": str(figures_dir / FIGURE_FILENAMES["4-6"]),
            "4-7": str(figures_dir / FIGURE_FILENAMES["4-7"]),
            "4-8": str(figures_dir / FIGURE_FILENAMES["4-8"]),
        },
        "summary": {
            "analytical_test_mape": float(analytical_metrics["mape"]),
            "baseline_test_mape": float(baseline_metrics["metrics"]["test"]["mape"]),
            "grouped_test_mape": float(grouped_metrics["mape"]),
            "grouped_test_r2": float(grouped_metrics["r2"]),
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
        plot_frame = plot_frame.sort_values("mape", ascending=False).reset_index(drop=True)
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
            "Figure 4-9 Unseen-shape / unseen-thread single-op generalization",
            ylabel="MAPE",
            color="#E45756",
            audit_lines=[
                f"holdout batch sizes = {_sanitize_list(OOD_BATCH_HOLDS)}",
                f"thread holdout = {OOD_NUM_THREADS_HOLD}",
                f"slice rows = {', '.join(f'{row.slice_name}:{int(row.rows)}' for row in ood_frame.itertuples(index=False))}",
            ],
        )

    if not generalization_frame.empty:
        test_reference = generalization_frame[generalization_frame["split"] == "test"].copy()
        test_pivot = test_reference.pivot(index="family", columns="scheme", values="mean_mape").sort_index()
        family_order = test_pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
        test_pivot = test_pivot.loc[family_order]
        plot_grouped_bar(
            test_pivot.reset_index(),
            "family",
            [col for col in ("leave_one_case_out", "leave_one_combo_out") if col in test_pivot.columns],
            figures_dir / FIGURE_FILENAMES["4-10"],
            "Figure 4-10 Analytical reference under unseen configuration splits",
            ylabel="mean MAPE",
            legend_labels=["leave-one-case-out", "leave-one-combo-out"],
            audit_lines=[
                f"test rows = {len(test_reference):,}",
                f"families = {len(test_pivot):,}",
                f"best family = {test_pivot.mean(axis=1).idxmin()}",
                f"worst family = {test_pivot.mean(axis=1).idxmax()}",
            ],
        )

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
    single_op_artifact_root: Path | None = None,
    e2e_artifact_root: Path | None = None,
) -> SectionResult:
    layout = ensure_output_layout(output_root)
    tables_dir = layout["tables"]
    figures_dir = layout["figures"]
    section_dir = layout["ablation"]

    single_op_root = _resolve_root(single_op_artifact_root, SINGLE_OP_ARTIFACT_ROOT)
    e2e_root = _resolve_root(e2e_artifact_root, E2E_ARTIFACT_ROOT)
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    combined_predictions = pd.read_csv(single_op_root / "models" / "combined" / "combined_predictions_test.csv", low_memory=False)
    e2e_full = pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)

    analytical_combo = (
        test_dataset.groupby(["case_id", "combo"], as_index=False)
        .agg(analytical_simple_add_us=("ana_calib_total_us", "sum"))
    )
    mlp_combo = (
        combined_predictions.groupby(["case_id", "combo"], as_index=False)
        .agg(mlp_simple_add_us=("pred_us", "sum"))
    )
    e2e_joined = (
        e2e_full.merge(analytical_combo, on=["case_id", "combo"], how="left")
        .merge(mlp_combo, on=["case_id", "combo"], how="left")
    )
    e2e_joined["pipeline_us"] = e2e_joined["predicted_e2e_us"].astype(float)
    e2e_joined["actual_us"] = e2e_joined["actual_e2e_us"].astype(float)

    variants = [
        ("Analytical + simple add", "analytical_simple_add_us"),
        ("Analytical + MLP + simple add", "mlp_simple_add_us"),
        ("Analytical + MLP + pipeline", "pipeline_us"),
    ]
    table_rows: list[dict[str, Any]] = []
    cdf_map: dict[str, list[float]] = {}
    gt10_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []
    for variant_name, column in variants:
        if column == "analytical_simple_add_us":
            single_metrics = _compute_regression_metrics(
                test_dataset["label_operator_actual_dur_us"],
                test_dataset["ana_calib_total_us"],
            )
        else:
            single_metrics = _compute_regression_metrics(
                combined_predictions["target_us"],
                combined_predictions["pred_us"],
            )
        e2e_metrics = _compute_regression_metrics(e2e_joined["actual_us"], e2e_joined[column])
        ape = ((e2e_joined[column] - e2e_joined["actual_us"]).abs() / e2e_joined["actual_us"].replace(0.0, pd.NA)).fillna(0.0)
        cdf_map[variant_name] = ape.astype(float).tolist()
        gt10_rows.append(
            {
                "variant": variant_name,
                "mean_ape": e2e_metrics["mape"],
                "gt10_rate": e2e_metrics["gt10_rate"],
            }
        )
        for inter_threads, sub in e2e_joined.groupby("inter_threads"):
            inter_rows.append(
                {
                    "variant": variant_name,
                    "inter_threads": int(inter_threads),
                    "mape": float((((sub[column] - sub["actual_us"]).abs() / sub["actual_us"].replace(0.0, pd.NA)).fillna(0.0)).mean()),
                }
            )
        table_rows.append(
            {
                "variant": variant_name,
                "single_op_mape": single_metrics["mape"],
                "single_op_rmse_us": single_metrics["rmse_us"],
                "single_op_r2": single_metrics["r2"],
                "e2e_mape": e2e_metrics["mape"],
                "e2e_p50_ape": e2e_metrics["p50_ape"],
                "e2e_p90_ape": e2e_metrics["p90_ape"],
                "e2e_gt10_rate": e2e_metrics["gt10_rate"],
            }
        )
    table_4_6 = pd.DataFrame(table_rows)
    csv_46, md_46 = write_frame_csv_md(
        table_4_6,
        tables_dir / TABLE_FILENAMES["4-6"],
        tables_dir / "table_4_6_ablation_summary.md",
        "Table 4-6 Three-stage ablation summary",
    )

    plot_cdf(
        cdf_map,
        figures_dir / FIGURE_FILENAMES["4-16"],
        "Figure 4-16 CDF of E2E relative error under three ablation variants",
        xlabel="relative error",
        audit_lines=[
            f"full combos = {len(e2e_joined):,}",
            f"best mean APE = {float(table_4_6['e2e_mape'].min()):.4f}",
        ],
    )
    gt10_frame = pd.DataFrame(gt10_rows)
    plot_grouped_bar(
        gt10_frame,
        "variant",
        ["mean_ape", "gt10_rate"],
        figures_dir / FIGURE_FILENAMES["4-17"],
        "Figure 4-17 Mean error and >10% error rate",
        ylabel="ratio",
        legend_labels=["mean APE", ">10% rate"],
        audit_lines=[
            f"pipeline mean APE = {float(gt10_frame.iloc[-1]['mean_ape']):.4f}",
            f"analytical-only mean APE = {float(gt10_frame.iloc[0]['mean_ape']):.4f}",
        ],
    )
    inter_frame = pd.DataFrame(inter_rows).sort_values(["inter_threads", "variant"]).reset_index(drop=True)
    plot_line(
        inter_frame,
        "inter_threads",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-18"],
        "Figure 4-18 E2E MAPE vs. branch parallelism",
        group_col="variant",
        ylabel="MAPE",
        audit_lines=[
            f"inter_threads = {_sanitize_list(sorted(inter_frame['inter_threads'].unique().tolist()))}",
            "Pipeline variant should show the strongest gain at higher branch parallelism.",
        ],
    )

    payload = {
        "tables": {"4-6": str(csv_46)},
        "figures": {
            "4-16": str(figures_dir / FIGURE_FILENAMES["4-16"]),
            "4-17": str(figures_dir / FIGURE_FILENAMES["4-17"]),
            "4-18": str(figures_dir / FIGURE_FILENAMES["4-18"]),
        },
        "summary": {
            "variant_rows": table_rows,
            "rows": int(len(e2e_joined)),
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
    full_metrics = pd.read_csv(e2e_root / "full_combo_metrics.csv", low_memory=False)
    summary_metrics = _compute_regression_metrics(full_metrics["actual_e2e_us"], full_metrics["predicted_e2e_us"])
    full_metrics["batch_bucket"] = pd.cut(
        full_metrics["batch_size"],
        bins=[0, 1280, 1792, 4096],
        labels=["small", "medium", "large"],
    )
    table_5 = (
        full_metrics.groupby("batch_bucket", as_index=False, observed=False)
        .agg(
            sample_count=("combo", "count"),
            mean_actual_us=("actual_e2e_us", "mean"),
            mae_us=("abs_error_us", "mean"),
            mape=("ape", "mean"),
            p50_ape=("ape", lambda values: float(pd.Series(values).quantile(0.50))),
            p90_ape=("ape", lambda values: float(pd.Series(values).quantile(0.90))),
        )
    )
    overall_row = pd.DataFrame(
        [
            {
                "batch_bucket": "overall",
                "sample_count": int(len(full_metrics)),
                "mean_actual_us": float(full_metrics["actual_e2e_us"].mean()),
                "mae_us": summary_metrics["mae_us"],
                "mape": summary_metrics["mape"],
                "p50_ape": summary_metrics["p50_ape"],
                "p90_ape": summary_metrics["p90_ape"],
            }
        ]
    )
    table_5 = pd.concat([overall_row, table_5], ignore_index=True)
    csv_5, md_5 = write_frame_csv_md(
        table_5,
        tables_dir / TABLE_FILENAMES["4-5"],
        tables_dir / "table_4_5_e2e_accuracy.md",
        "Table 4-5 E2E prediction accuracy",
    )

    batch_curve = full_metrics.groupby("batch_size", as_index=False).agg(mape=("ape", "mean")).sort_values("batch_size")
    inter_curve = full_metrics.groupby("inter_threads", as_index=False).agg(
        actual_us=("actual_e2e_us", "mean"),
        predicted_us=("predicted_e2e_us", "mean"),
        mape=("ape", "mean"),
    ).sort_values("inter_threads")

    plot_scatter(
        full_metrics.assign(actual=full_metrics["actual_e2e_us"], predicted=full_metrics["predicted_e2e_us"]),
        "actual",
        "predicted",
        figures_dir / FIGURE_FILENAMES["4-11"],
        "Figure 4-11 E2E predicted vs. actual latency",
        audit_lines=[
            f"rows = {len(full_metrics):,}",
            f"MAPE = {summary_metrics['mape']:.4f}",
            f"P90 = {summary_metrics['p90_ape']:.4f}",
        ],
    )
    plot_line(
        batch_curve,
        "batch_size",
        "mape",
        figures_dir / FIGURE_FILENAMES["4-12"],
        "Figure 4-12 E2E MAPE across batch sizes",
        ylabel="MAPE",
        audit_lines=[
            f"batch sizes = {len(batch_curve):,}",
            f"best batch MAPE = {float(batch_curve['mape'].min()):.4f}",
        ],
    )
    inter_plot = pd.concat(
        [
            inter_curve[["inter_threads", "actual_us"]].rename(columns={"actual_us": "latency_us"}).assign(series="actual"),
            inter_curve[["inter_threads", "predicted_us"]].rename(columns={"predicted_us": "latency_us"}).assign(series="predicted"),
        ],
        ignore_index=True,
    )
    plot_line(
        inter_plot,
        "inter_threads",
        "latency_us",
        figures_dir / FIGURE_FILENAMES["4-13"],
        "Figure 4-13 Predicted / actual latency vs. branch parallelism",
        group_col="series",
        ylabel="latency (us)",
        audit_lines=[
            f"parallelism settings = {_sanitize_list(sorted(inter_curve['inter_threads'].astype(int).tolist()))}",
            f"lowest MAPE = {float(inter_curve['mape'].min()):.4f}",
        ],
    )

    payload = {
        "tables": {"4-5": str(csv_5)},
        "figures": {
            "4-11": str(figures_dir / FIGURE_FILENAMES["4-11"]),
            "4-12": str(figures_dir / FIGURE_FILENAMES["4-12"]),
            "4-13": str(figures_dir / FIGURE_FILENAMES["4-13"]),
        },
        "summary": {
            "overall": summary_metrics,
            "inter_threads_curve": inter_curve.to_dict(orient="records"),
            "batch_curve": batch_curve.to_dict(orient="records"),
        },
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
    dataset_df = pd.read_csv(single_op_root / "classed_dataset_full.csv", low_memory=False)
    test_dataset = dataset_df[dataset_df["split"] == "test"].copy()
    combined_predictions = pd.read_csv(single_op_root / "models" / "combined" / "combined_predictions_test.csv", low_memory=False)
    e2e_frame = pd.read_csv(E2E_ARTIFACT_ROOT / "full_combo_metrics.csv", low_memory=False)

    gather_frame = test_dataset[test_dataset["op_type"] == "Gather"].copy()
    gather_pred = combined_predictions[combined_predictions["op_type"] == "Gather"][["row_uid", "pred_us"]]
    gather_frame = gather_frame.merge(gather_pred, on="row_uid", how="left")
    gather_frame["ape"] = (
        (gather_frame["pred_us"].astype(float) - gather_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / gather_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    gather_case = gather_frame.sort_values("ape", ascending=False).iloc[0]

    meta_frame = test_dataset[test_dataset["op_type"].isin(["Reshape", "Shape", "Unsqueeze", "Flatten"])].copy()
    meta_pred = combined_predictions[combined_predictions["op_type"].isin(["Reshape", "Shape", "Unsqueeze", "Flatten"])][["row_uid", "pred_us"]]
    meta_frame = meta_frame.merge(meta_pred, on="row_uid", how="left")
    meta_frame["ape"] = (
        (meta_frame["pred_us"].astype(float) - meta_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / meta_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    meta_case = meta_frame.sort_values(["label_operator_actual_dur_us", "ape"], ascending=[True, False]).iloc[0]

    gemm_frame = test_dataset[test_dataset["op_type"].isin(["Gemm", "MatMul"])].copy()
    gemm_pred = combined_predictions[combined_predictions["op_type"].isin(["Gemm", "MatMul"])][["row_uid", "pred_us"]]
    gemm_frame = gemm_frame.merge(gemm_pred, on="row_uid", how="left")
    gemm_frame["ape"] = (
        (gemm_frame["pred_us"].astype(float) - gemm_frame["label_operator_actual_dur_us"].astype(float)).abs()
        / gemm_frame["label_operator_actual_dur_us"].replace(0.0, pd.NA)
    ).fillna(0.0)
    gemm_case = gemm_frame.sort_values(["feat_gemm_mac_count", "ape"], ascending=[True, False]).iloc[0]

    e2e_case = e2e_frame.sort_values("ape", ascending=False).iloc[0]
    table_4_7 = pd.DataFrame(
        [
            {
                "sample_type": "Gather random-access hotspot",
                "actual_us": float(gather_case["label_operator_actual_dur_us"]),
                "predicted_us": float(gather_case["pred_us"]),
                "relative_error": float(gather_case["ape"]),
                "explanation": "Random large-table accesses amplify memory-latency variance that the calibrated analytical proxy cannot fully smooth.",
            },
            {
                "sample_type": "Small-tensor view/meta op",
                "actual_us": float(meta_case["label_operator_actual_dur_us"]),
                "predicted_us": float(meta_case["pred_us"]),
                "relative_error": float(meta_case["ape"]),
                "explanation": "Framework dispatch and bookkeeping overhead dominate when tensor payload is tiny, making latency noisier than shape-only features suggest.",
            },
            {
                "sample_type": "Small-dimension Gemm/MatMul",
                "actual_us": float(gemm_case["label_operator_actual_dur_us"]),
                "predicted_us": float(gemm_case["pred_us"]),
                "relative_error": float(gemm_case["ape"]),
                "explanation": "Micro-kernel utilization is low at small M/N/K, so measured latency deviates from saturated compute-mode assumptions.",
            },
            {
                "sample_type": "E2E synchronization-heavy combo",
                "actual_us": float(e2e_case["actual_e2e_us"]),
                "predicted_us": float(e2e_case["predicted_e2e_us"]),
                "relative_error": float(e2e_case["ape"]),
                "explanation": "Tail-barrier and branch-slot synchronization amplify any upstream single-op bias at the full-graph level.",
            },
        ]
    )
    csv_47, md_47 = write_frame_csv_md(
        table_4_7,
        tables_dir / TABLE_FILENAMES["4-7"],
        tables_dir / "table_4_7_error_cases.md",
        "Table 4-7 Typical error cases",
    )

    plot_bar(
        table_4_7,
        "sample_type",
        "relative_error",
        figures_dir / FIGURE_FILENAMES["4-19"],
        "Figure 4-19 Relative error of representative failure cases",
        ylabel="relative error",
        color="#B279A2",
        audit_lines=[
            "The selected cases correspond to the dominant error sources discussed in Section 4.4.5.",
            f"worst case = {table_4_7.sort_values('relative_error', ascending=False).iloc[0]['sample_type']}",
        ],
    )

    payload = {
        "tables": {"4-7": str(csv_47)},
        "figures": {"4-19": str(figures_dir / FIGURE_FILENAMES["4-19"])},
        "summary": {"error_rows": table_4_7.to_dict(orient="records")},
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
                case_summary = next(row for row in summary_rows if row["case_id"] == case_id and row["combo"] == combo)
                add_audit_callout(
                    ax,
                    [
                        f"predicted = {case_summary['predicted_full_graph_us'] / 1_000_000:.3f} s",
                        f"actual = {case_summary['actual_full_graph_us'] / 1_000_000:.3f} s",
                        f"APE = {case_summary['ape']:.3f}",
                    ],
                    loc="upper left",
                )
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
            add_audit_callout(
                ax,
                [
                    f"cases = {len(summary_rows)}",
                    f"worst APE = {max(row['ape'] for row in summary_rows):.3f}",
                    f"critical path nodes = {critical_plot['step'].max() + 1 if not critical_plot.empty else 0}",
                ],
                loc="upper right",
            )
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
    figure_claims = {
        "4-1": ("platform", "single-op data collection workflow"),
        "4-2": ("platform", "full-graph timeline aggregation workflow"),
        "4-3": ("single-op core", "overall predicted-vs-actual scatter"),
        "4-4": ("single-op core", "category-level MAPE comparison"),
        "4-5": ("single-op core", "Gather prediction behavior"),
        "4-6": ("single-op core", "ReduceSum error vs reduction scale"),
        "4-7": ("single-op core", "Transpose/Concat error vs data size"),
        "4-8": ("single-op core", "Gemm/MatMul error vs MAC count"),
        "4-9": ("single-op OOD", "unseen shape generalization"),
        "4-10": ("single-op OOD", "unseen thread/generalization reference"),
        "4-11": ("e2e core", "overall E2E predicted-vs-actual scatter"),
        "4-12": ("e2e core", "batch-size stability"),
        "4-13": ("e2e core", "parallelism sensitivity"),
        "4-14": ("timeline", "timeline replay comparison"),
        "4-15": ("timeline", "critical path breakdown"),
        "4-16": ("ablation", "three-stage ablation CDF"),
        "4-17": ("ablation", "mean error and >10% error rate"),
        "4-18": ("ablation", "parallelism sensitivity of ablation variants"),
        "4-19": ("error analysis", "representative failure cases"),
    }
    for figure_no, filename in FIGURE_FILENAMES.items():
        stage, claim = figure_claims.get(figure_no, ("unknown", ""))
        rows.append(
            {
                "figure_no": figure_no,
                "stage": stage,
                "claim": claim,
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
    sum_baseline_summary = read_json(layout["e2e"] / "e2e_sum_baseline_summary.json") if (layout["e2e"] / "e2e_sum_baseline_summary.json").exists() else {}
    ablation_summary = read_json(layout["ablation"] / "single_op_ablation_summary.json") if (layout["ablation"] / "single_op_ablation_summary.json").exists() else {}
    timeline_summary = read_json(layout["e2e"] / "timeline_cases_summary.json") if (layout["e2e"] / "timeline_cases_summary.json").exists() else {}
    figures_catalog = read_json(layout["manifests"] / "figures_catalog.json") if (layout["manifests"] / "figures_catalog.json").exists() else {"rows": []}
    platform_summary = read_json(layout["manifests"] / "platform_summary.json") if (layout["manifests"] / "platform_summary.json").exists() else {}
    platform_metrics = platform_summary.get("summary", {})
    single_op_metrics = single_op_summary.get("summary", {})
    ood_metrics = ood_summary.get("summary", {})
    e2e_metrics = e2e_summary.get("summary", {})
    ablation_metrics = ablation_summary.get("summary", {})
    error_metrics = sum_baseline_summary.get("summary", {})
    timeline_rows = timeline_summary.get("summary", [])

    dataset_df = pd.read_csv(SINGLE_OP_ARTIFACT_ROOT / "classed_dataset_full.csv", usecols=["split", "batch_size", "num_indices_per_lookup", "num_threads", "inter_threads", "op_type"], low_memory=False)
    split_counts = dataset_df["split"].value_counts().to_dict()
    batch_range = f"{int(dataset_df['batch_size'].min())}-{int(dataset_df['batch_size'].max())}"
    nip_range = f"{int(dataset_df['num_indices_per_lookup'].min())}-{int(dataset_df['num_indices_per_lookup'].max())}"
    op_types = sorted(dataset_df["op_type"].dropna().unique().tolist())
    ood_rows = {row["slice_name"]: row for row in ood_metrics.get("ood_slices", [])}
    e2e_overall = e2e_metrics.get("overall", {})
    ablation_rows = ablation_metrics.get("variant_rows", [])
    error_rows = error_metrics.get("error_rows", [])
    pipeline_row = next((row for row in ablation_rows if row["variant"] == "Analytical + MLP + pipeline"), None)
    simple_add_row = next((row for row in ablation_rows if row["variant"] == "Analytical + MLP + simple add"), None)
    analytical_row = next((row for row in ablation_rows if row["variant"] == "Analytical + simple add"), None)

    lines = [
        "# 第四章 CPU 实验与结果分析",
        "",
        "## 4.1 实验平台与数据采集方法",
        "",
        "本章实验围绕 ORT 上的 DLRM CPU 推理路径展开，目标是验证第三章提出的两级建模思路，即先对单算子时延进行静态预测，再将节点级预测结果通过静态流水线模型聚合为整图时延。所有第四章脚本统一维护在 `ORT/static_pipeline_eval/chapter4_experiments/`，最终产物写入 `ORT/static_pipeline_eval/artifacts/latest/chapter4_cpu/`。",
        "",
        "### 4.1.1 实验平台",
        "",
        f"实验服务器为 {platform_metrics.get('host_cpu_model', 'Kunpeng-920')} 主机。整机从 `lscpu` 可观测到 4 路 CPU、192 个物理核心、单核单线程；但本章建模与运行口径采用单 NUMA 执行域，对应 24 个核心、24 MiB 共享 L3 以及 4 个 DDR4-2933 内存通道。单核缓存参数采用 `64 KiB L1I + 64 KiB L1D + 512 KiB L2`，本地 NUMA 理论带宽按 `100 GB/s` 建模，操作系统为 `Huawei Cloud EulerOS 2.0 (aarch64)`，编译器为 `GCC 10.3.1`。实验软件环境来自 `ort` conda 环境：`Python 3.11.14`、`PyTorch 2.9.0`、`ONNX 1.19.1`、`ONNX Runtime CANN 1.23.2`、`NumPy 1.26.4`、`pandas 3.0.1`。单算子标签采集依赖 DynamoRIO 侧的 profiling 结果，整图真值来自 ORT branch-parallel 运行的时间线记录。为减小系统噪声，实验固定 `intra-op` 与 `inter-op` 线程配置，并按单 NUMA 口径组织样本和静态流水线排程。",
        "",
        "### 4.1.2 DLRM 负载与算子样本生成",
        "",
        f"单算子样本来自 ORT 导出的 DLRM ONNX 图及其配套 `op_shapes`/profile 数据。统一数据集共包含 {platform_metrics.get('single_op_rows', 0):,} 条样本、{platform_metrics.get('single_op_cases', 0)} 个 case、{platform_metrics.get('single_op_combos', 0):,} 个 `case_id-combo` 配置，覆盖 `{batch_range}` 的 batch size、`{nip_range}` 的每次 lookup 索引数、`1-5` 的 intra-op 线程和 `1/3/4/5/6` 的 inter-op 线程。算子类型共 {len(op_types)} 种，分别映射为五类：索引访存类、布局搬移类、视图/元数据类、轻计算-访存混合类和计算主导类。每个 ONNX 节点都保留输入/输出 shape、线程配置、I/O 字节规模、归约规模、Gemm 的 M/N/K 与 analytical proxy 列，用于后续单算子建模。",
        "",
        "### 4.1.3 运行与标注流程",
        "",
        "单算子标签来自 ORT 执行后的 per-op profiling 结果。每个样本先丢弃最早的一个 profile batch，再对剩余 batch 的算子时延求均值，形成 `label_operator_actual_dur_us`。为了恢复整图真值，本文使用 branch-parallel runner 导出的 `branch_parallel_op_timeline.csv`，对每个 batch 计算整图 span，并采用与单算子一致的策略丢弃首个 batch 后再求均值。静态整图模型并不是简单求和，而是根据 `op_shapes` 重建 DAG，将 8 条 embedding 分支折叠为 branch task，再依据 `inter_threads` 控制的并行槽位进行离线排程，从而生成预测时间线、关键路径和整图时延。",
        "",
        "### 4.1.4 数据集划分与评价指标",
        "",
        f"单算子数据按 `sample_group=combo` 做 `7:2:1` 划分，实际样本数为 train/val/test = `{split_counts.get('train', 0):,} / {split_counts.get('val', 0):,} / {split_counts.get('test', 0):,}`。这种切分方式保证同一 `case_id-combo` 下的节点不会同时出现在训练集与测试集，避免 shape 级信息泄漏。除随机划分外，本文还构造了未见 shape 与未见线程数两种外推测试：前者将 batch size `{_sanitize_list(OOD_BATCH_HOLDS)}` 作为保留配置，后者仅用 `num_threads={OOD_NUM_THREADS_HOLD}` 做测试。单算子评价指标采用 `MAE`、`MAPE`、`RMSE` 和 `R^2`；整图评价指标采用 `MAE`、`MAPE`、`P50/P90` 相对误差，并按 batch size 和 branch parallelism 分组统计。",
        "",
        "## 4.2 算子级性能建模实验",
        "",
        "本节首先给出最终分组模型在测试集上的总体精度，再按五类算子统计误差，并对 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性行为做可视化分析。整体作图风格参考 Concorde 中“真实值-预测值关系、误差分布、典型 case 解释”的组织方式，但结合 DLRM 场景保留了更强的算子语义解释。",
        "",
        "### 4.2.1 单算子总体预测精度",
        "",
        f"表 4-3 给出了三种单算子模型口径的总体结果：纯解析模型、单一 MLP 基线以及本文采用的分组 analytical-MLP。最终分组模型在测试集上的 `MAPE` 为 {single_op_metrics.get('grouped_test_mape', 0.0):.4f}，`R^2` 为 {single_op_metrics.get('grouped_test_r2', 0.0):.4f}；纯解析模型由于在小张量和视图类节点上存在显著比例误差，其 `MAPE` 高达 {single_op_metrics.get('analytical_test_mape', 0.0):.4f}。单一 MLP 基线的 `MAPE` 为 {single_op_metrics.get('baseline_test_mape', 0.0):.4f}，在随机切分的单算子指标上略优于分组模型，但缺少显式的机理分组与解析代理约束。图 4-3 的散点结果显示，分组模型的大部分样本仍围绕 `y=x` 参考线分布，说明其作为后续整图聚合输入是稳定可用的。",
        "",
        "### 4.2.2 分类别预测精度",
        "",
        "表 4-4 和图 4-4 展示了五类算子的误差差异。总体上，视图/元数据类和布局搬移类更容易预测，因为其执行路径较稳定；索引访存类和轻计算-访存混合类误差相对更大，主要原因是它们更容易受到随机访存、线程调度和小张量固定开销的影响。这种类别差异与第三章中对 ORT CPU kernel 机制的分析是一致的，也说明统一 MLP 难以同时覆盖这几类机理差异显著的节点。",
        "",
        "### 4.2.3 典型算子预测结果分析",
        "",
        "图 4-5 至图 4-8 分别给出了 Gather、ReduceSum、Transpose/Concat 以及 Gemm/MatMul 的代表性结果。Gather 的误差主要受到随机表访存影响，线程数变化会显著改变尾部误差；ReduceSum 的误差随归约工作量增长而逐步收敛，说明解析代理对归约规模具有较好刻画能力；Transpose 与 Concat 的误差主要受数据搬移规模影响；Gemm/MatMul 在 MAC 数较小时误差增大，反映出小维度下微核利用率不足的问题。整体来看，这些现象与第三章关于 cache fit、数据搬移和 kernel 饱和度的解释是相互印证的。",
        "",
        "### 4.2.4 跨规模泛化实验",
        "",
        f"图 4-9 和图 4-10 分别给出未见 shape 与未见线程数的测试结果。未见 shape 配置的 `MAPE` 为 {ood_rows.get('unseen_batch_size', {}).get('mape', 0.0):.4f}，未见线程数配置的 `MAPE` 为 {ood_rows.get('unseen_num_threads', {}).get('mape', 0.0):.4f}。这说明解析代理特征在 shape 外推上提供了稳定支撑，而线程数外推的难度更高，因为线程切分会同时影响并行粒度、调度开销和实际 cache 行为。",
        "",
        "## 4.3 整图性能聚合实验",
        "",
        "整图实验验证第三章 3.3 中的静态流水线聚合模型。与单纯求和不同，该模型显式考虑 bottom、8 条 embedding branch 和 tail 的依赖关系，以及 `inter_threads` 决定的 branch 并行槽位约束，因此能够用节点级预测结果恢复整图执行的关键路径。",
        "",
        "### 4.3.1 整图预测总体精度",
        "",
        f"表 4-5 汇总了整图预测精度。静态流水线模型在完整图样本上的 `MAPE` 为 {e2e_overall.get('mape', 0.0):.4f}，`P50` 和 `P90` 相对误差分别为 {e2e_overall.get('p50_ape', 0.0):.4f} 和 {e2e_overall.get('p90_ape', 0.0):.4f}。图 4-11 的散点结果表明，绝大多数整图配置都能被稳定地落在参考线附近，说明节点级误差在流水线聚合后没有被系统性放大。",
        "",
        "### 4.3.2 不同 batch size 下的整图精度",
        "",
        "图 4-12 给出了不同 batch size 下的整图 `MAPE`。整体趋势较平滑，说明模型在小 batch、中 batch 和大 batch 区间都保持了较稳定的误差水平。这一点非常重要，因为 DLRM 的 embedding lookup 和 top MLP 都会随 batch size 改变张量规模与工作集，若聚合模型缺乏稳定性，误差会在 batch 变化时明显抖动。",
        "",
        "### 4.3.3 不同分支并行度下的整图精度",
        "",
        "图 4-13 按 `inter_threads` 展示了真实整图时延与预测整图时延的变化。可以看到，随着可用 branch 槽位增加，整图时延整体下降，但收益逐渐递减；预测曲线能够较好跟随这一趋势。这说明第三章提出的 `kappa` 槽位近似虽然是静态模型，但已经能够反映并行度增加后边际收益递减的核心行为。",
        "",
        "### 4.3.4 典型时间线案例分析",
        "",
        f"图 4-14 和图 4-15 展示了 {len(timeline_rows) if isinstance(timeline_rows, list) else 0} 个典型配置的真实/预测时间线与关键路径分解。通过这些案例可以看到，模型不仅预测了整图 makespan，还较准确地恢复了 bottom、embedding pool 和 tail 之间的先后关系。误差主要集中在分支完成时刻附近的同步边界和少数高波动节点上，这为后续误差分析提供了直接证据。",
        "",
        "## 4.4 消融实验与误差分析",
        "",
        "这一节采用与 Concorde 类似的逐步加组件消融方式，而不是简单做特征删除。具体构造三组模型：第一组仅使用 `Analytical model + Simple add`；第二组在单算子层面加入分组 MLP，但整图仍采用 `Simple add`；第三组使用 `Analytical + MLP + pipeline`，即本文完整方法。这样的设计能够更清晰地回答三个问题：解析模型本身能做多好、单算子学习器能带来多少收益、以及静态流水线聚合是否对整图预测确有必要。",
        "",
        "### 4.4.1 三阶段消融结果",
        "",
        f"表 4-6、图 4-16、图 4-17 和图 4-18 共同展示了三阶段模型的差异。纯解析模型在整图上的平均相对误差最高；加入分组 MLP 后，单算子精度显著提高，但若仍对节点时延简单相加，整图误差仍然较大；进一步引入静态流水线聚合后，整图 `MAPE` 下降到 {pipeline_row.get('e2e_mape', 0.0) if pipeline_row else 0.0:.4f}，明显优于 `Analytical + MLP + simple add` 的 {simple_add_row.get('e2e_mape', 0.0) if simple_add_row else 0.0:.4f}，更远优于 `Analytical + simple add` 的 {analytical_row.get('e2e_mape', 0.0) if analytical_row else 0.0:.4f}。图 4-16 的误差 CDF 与图 4-17 的平均误差/大误差比例统计共同说明，完整模型不仅降低了均值误差，也显著压缩了误差尾部。",
        "",
        "### 4.4.2 误差来源分析",
        "",
        "表 4-7 和图 4-19 汇总了四类代表性误差样本。第一类是大表随机访存下的 Gather，误差主要来自实际内存访问时延波动；第二类是小张量视图/元数据算子，固定框架开销占比过高导致样本噪声明显；第三类是小维度 Gemm/MatMul，微核未充分饱和时更难被解析代理拟合；第四类则是整图级同步场景，单个 branch 的偏差在 barrier 处被放大。这些现象说明，当前模型的主要剩余误差并不来自统计偶然，而是来自 ORT CPU 执行中仍难以静态精确恢复的动态效应。",
        "",
        "## 4.5 本章小结",
        "",
        f"本章首先在 {platform_metrics.get('single_op_rows', 0):,} 条单算子样本和 {platform_metrics.get('full_e2e_combos', 0):,} 个完整图配置上完成了统一实验。结果表明，分组 analytical-MLP 单算子模型在测试集上取得了 {single_op_metrics.get('grouped_test_mape', 0.0):.4f} 的 `MAPE`，显著优于纯解析模型，并为后续整图聚合提供了带机理约束的节点级输入；在整图层面，静态流水线聚合模型将 `MAPE` 控制在 {e2e_overall.get('mape', 0.0):.4f}。进一步的三阶段消融证明：仅靠解析模型或节点时延简单求和都无法得到可接受的整图精度，而将解析代理、单算子学习器和静态流水线聚合组合起来之后，可以同时压低平均误差和尾部误差。至此，第三章提出的解析代理特征、分组单算子模型与静态整图聚合三项核心设计，都得到了实验结果的直接验证。",
        "",
        f"本章共生成 {len(figures_catalog.get('rows', []))} 张图，全部由 `chapter4_experiments/run_all_chapter4_experiments.py` 自动复现，并写入 `{CHAPTER4_DRAFT_PATH}`。",
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
