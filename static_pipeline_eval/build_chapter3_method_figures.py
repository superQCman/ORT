from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "latest"
    / "chapter4_cpu_single_only"
    / "e2e"
    / "timeline_cases"
    / "case_10_3_3__bs2048_nip2000"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "latest" / "chapter3_method_figures"


PALETTE = {
    "bottom": "#4C78A8",
    "embedding": "#F58518",
    "tail": "#54A24B",
    "barrier": "#B279A2",
    "ink": "#243447",
    "muted": "#6B7785",
    "panel": "#F6F8FB",
    "line": "#C8D0D9",
    "accent": "#D6DEE8",
}


@dataclass(frozen=True)
class BranchSpan:
    label: str
    start_us: float
    end_us: float
    duration_us: float


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    return plt, font_manager, FancyArrowPatch, FancyBboxPatch, Polygon


def _pick_font(font_manager, candidates: list[str]) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def _configure_style(plt, font_manager) -> None:
    serif_font = _pick_font(font_manager, ["STIXGeneral", "DejaVu Serif", "Liberation Serif"])
    sans_font = _pick_font(font_manager, ["DejaVu Sans", "Liberation Sans"])
    plt.rcParams.update(
        {
            "font.family": serif_font,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    plt.rcParams["font.sans-serif"] = [sans_font]


def _save_all_formats(fig, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    return outputs


def _panel(ax, label: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=PALETTE["line"],
            facecolor=PALETTE["panel"],
            zorder=0,
        )
    )
    ax.text(0.04, 0.95, label, fontsize=12, fontweight="bold", color=PALETTE["ink"], va="top")


def _box(ax, x: float, y: float, w: float, h: float, text: str, *, face: str, edge: str | None = None, text_color: str = "white", fontsize: float = 10.5, rounded: float = 0.018, lw: float = 1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={rounded}",
            linewidth=lw,
            edgecolor=edge or face,
            facecolor=face,
        )
    )
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def _diamond(ax, x: float, y: float, w: float, h: float, text: str, *, face: str, edge: str | None = None, text_color: str = "white", fontsize: float = 10.5) -> None:
    points = [
        (x + w / 2.0, y + h),
        (x + w, y + h / 2.0),
        (x + w / 2.0, y),
        (x, y + h / 2.0),
    ]
    ax.add_patch(Polygon(points, closed=True, facecolor=face, edgecolor=edge or face, linewidth=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", color=text_color, fontsize=fontsize)


def _arrow(ax, p0: tuple[float, float], p1: tuple[float, float], *, color: str = PALETTE["muted"], lw: float = 1.2, style: str = "->", connectionstyle: str = "arc3") -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            linewidth=lw,
            color=color,
            mutation_scale=10,
            connectionstyle=connectionstyle,
        )
    )


def _draw_structure_figure(output_dir: Path) -> list[Path]:
    plt, font_manager, fancy_arrow_patch, fancy_box_patch, polygon = _import_matplotlib()
    global FancyArrowPatch, FancyBboxPatch, Polygon
    FancyArrowPatch = fancy_arrow_patch
    FancyBboxPatch = fancy_box_patch
    Polygon = polygon
    _configure_style(plt, font_manager)

    fig = plt.figure(figsize=(15.2, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    _panel(ax_left, "(a)")
    _panel(ax_right, "(b)")

    col_x = [0.08, 0.25, 0.42, 0.59, 0.79]
    top_y = 0.73
    mid_y = 0.48
    mid_h = 0.082
    branch_gap = 0.035
    op_w = 0.115
    task_w = 0.095
    tail_w = 0.14
    tail_h = 0.09

    bottom_nodes = [(col_x[0], top_y, "Bottom\nop 1"), (col_x[1], top_y, "Bottom\nop 2"), (col_x[2], top_y, "Bottom\nop 3"), (col_x[3], top_y, "Bottom\nop 4")]
    for idx, (x, y, label) in enumerate(bottom_nodes):
        _box(ax_left, x, y, op_w, 0.10, label, face=PALETTE["bottom"], fontsize=10.6)
        if idx:
            prev_x, prev_y, _ = bottom_nodes[idx - 1]
            _arrow(ax_left, (prev_x + op_w, prev_y + 0.05), (x, y + 0.05))

    branch_cols = [col_x[0] + 0.015, col_x[1] + 0.015, col_x[2] + 0.015, 0.70]
    branch_labels = ["emb_l0", "emb_l1", "emb_l2", "emb_l7"]
    for x, branch_label in zip(branch_cols, branch_labels):
        ax_left.text(x + 0.05, mid_y + 0.12, branch_label, ha="center", va="bottom", fontsize=9.6, color=PALETTE["muted"])
        _box(ax_left, x, mid_y, 0.10, mid_h, "Gather", face="#FDBA73", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=9.6, rounded=0.016)
        _box(ax_left, x, mid_y - (mid_h + branch_gap), 0.10, mid_h, "Reshape", face="#FFE1BF", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=9.2, rounded=0.016)
        _box(ax_left, x, mid_y - 2 * (mid_h + branch_gap), 0.10, mid_h, "Reduce\nSum", face="#FBCFA7", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=8.9, rounded=0.016)
        _arrow(ax_left, (x + 0.05, mid_y), (x + 0.05, mid_y - branch_gap))
        _arrow(ax_left, (x + 0.05, mid_y - (mid_h + branch_gap)), (x + 0.05, mid_y - (mid_h + 2 * branch_gap)))

    ax_left.text(0.545, mid_y - 0.02, "⋯", fontsize=30, color=PALETTE["muted"], ha="center", va="center")
    ax_left.text(0.545, mid_y - (mid_h + branch_gap) - 0.01, "⋯", fontsize=30, color=PALETTE["muted"], ha="center", va="center")
    ax_left.text(0.545, mid_y - 2 * (mid_h + branch_gap), "⋯", fontsize=30, color=PALETTE["muted"], ha="center", va="center")

    tail_x = 0.83
    _box(ax_left, tail_x, 0.50, tail_w, tail_h, "Feature\ninteraction", face=PALETTE["tail"], fontsize=10.3)
    _box(ax_left, tail_x, 0.33, tail_w, tail_h, "Top MLP\n/ output", face="#6FBF85", edge=PALETTE["tail"], fontsize=10.3)
    _arrow(ax_left, (col_x[3] + op_w, top_y + 0.05), (tail_x, 0.545), connectionstyle="arc3,rad=-0.16")
    for x in branch_cols:
        _arrow(ax_left, (x + 0.05, mid_y - 2 * (mid_h + branch_gap)), (tail_x, 0.545), connectionstyle="arc3,rad=0.04")
    _arrow(ax_left, (tail_x + 0.085, 0.50), (tail_x + 0.085, 0.42))

    task_bottom = [(col_x[0], top_y, r"$u_1$"), (col_x[1], top_y, r"$u_2$"), (col_x[2], top_y, r"$u_3$"), (col_x[3], top_y, r"$u_4$")]
    for idx, (x, y, label) in enumerate(task_bottom):
        _box(ax_right, x + 0.02, y, task_w, 0.09, label, face=PALETTE["bottom"], fontsize=11.8)
        if idx:
            prev_x, prev_y, _ = task_bottom[idx - 1]
            _arrow(ax_right, (prev_x + 0.02 + task_w, prev_y + 0.045), (x + 0.02, y + 0.045))
    ax_right.text(col_x[0] + 0.02, 0.86, r"$\mathcal{U}_{\mathrm{bot}}$", fontsize=11.0, color=PALETTE["bottom"], fontweight="bold")

    task_branches = [(col_x[0] + 0.03, 0.42, r"$B_0$"), (col_x[1] + 0.03, 0.42, r"$B_1$"), (col_x[2] + 0.03, 0.42, r"$B_2$"), (col_x[4] - 0.045, 0.42, r"$B_7$")]
    for x, y, label in task_branches:
        _box(ax_right, x, y, task_w, 0.09, label, face=PALETTE["embedding"], fontsize=11.8)
    ax_right.text(0.57, 0.465, "⋯", fontsize=28, color=PALETTE["muted"], ha="center", va="center")
    ax_right.text(col_x[0] + 0.03, 0.58, r"$\mathcal{U}_{\mathrm{emb}}=\{B_0,\ldots,B_7\}$", fontsize=11.0, color=PALETTE["embedding"], fontweight="bold")

    _diamond(ax_right, 0.79, 0.40, 0.11, 0.11, r"$u_{\mathrm{bar}}$", face=PALETTE["barrier"], fontsize=11.5)
    for x, y, _ in task_bottom:
        _arrow(ax_right, (x + 0.02 + task_w / 2.0, y), (0.845, 0.51), connectionstyle="arc3,rad=-0.10")
    for x, y, _ in task_branches:
        _arrow(ax_right, (x + task_w / 2.0, y + 0.09), (0.845, 0.40), connectionstyle="arc3,rad=0.05")

    ax_right.text(0.79, 0.26, r"$\mathcal{U}_{\mathrm{tail}}$", fontsize=11.0, color=PALETTE["tail"], fontweight="bold")
    tail_nodes = [(0.74, 0.15, r"$v_1$"), (0.86, 0.15, r"$v_2$"), (0.74, 0.04, r"$v_3$"), (0.86, 0.04, r"$v_4$")]
    for x, y, label in tail_nodes:
        _box(ax_right, x, y, 0.08, 0.08, label, face=PALETTE["tail"], fontsize=11.0)
    _arrow(ax_right, (0.845, 0.40), (0.78, 0.23))
    _arrow(ax_right, (0.82, 0.19), (0.86, 0.19))
    _arrow(ax_right, (0.78, 0.15), (0.78, 0.12))
    _arrow(ax_right, (0.82, 0.08), (0.86, 0.08))

    outputs = _save_all_formats(fig, output_dir / "fig_3_3_4_task_graph_construction")
    plt.close(fig)
    return outputs


def _assign_branch_slots(branch_frame: pd.DataFrame, slot_count: int) -> list[tuple[str, BranchSpan]]:
    branches = branch_frame.sort_values(["start_us", "label"]).to_dict("records")
    slot_release_us = [0.0 for _ in range(slot_count)]
    slot_contents: list[list[BranchSpan]] = [[] for _ in range(slot_count)]
    for row in branches:
        best_slot = min(range(slot_count), key=lambda idx: (max(slot_release_us[idx], float(row["start_us"])), slot_release_us[idx], idx))
        start_us = float(row["start_us"])
        end_us = float(row["end_us"])
        slot_contents[best_slot].append(
            BranchSpan(
                label=str(row["label"]).replace("branch:", "B"),
                start_us=start_us,
                end_us=end_us,
                duration_us=end_us - start_us,
            )
        )
        slot_release_us[best_slot] = end_us
    flattened: list[tuple[str, BranchSpan]] = []
    for idx, spans in enumerate(slot_contents, start=1):
        lane = f"Slot {idx}"
        for span in spans:
            flattened.append((lane, span))
    return flattened


def _compress_tail_segments(critical_path: pd.DataFrame) -> list[dict[str, float | str]]:
    tail_rows = critical_path[critical_path["partition"] == "tail"].copy()
    if tail_rows.empty:
        return []
    segments = [
        ("Pre-tail", 0, 11),
        ("Top MLP", 11, 16),
        ("Output", 16, len(tail_rows)),
    ]
    compressed = []
    for label, start_idx, end_idx in segments:
        sub = tail_rows.iloc[start_idx:end_idx]
        if sub.empty:
            continue
        compressed.append(
            {
                "label": label,
                "start_us": float(sub["predicted_start_us"].min()),
                "end_us": float(sub["predicted_end_us"].max()),
            }
        )
    return compressed


def _draw_timeline_figure(case_dir: Path, output_dir: Path) -> list[Path]:
    plt, font_manager, fancy_arrow_patch, fancy_box_patch, polygon = _import_matplotlib()
    global FancyArrowPatch, FancyBboxPatch, Polygon
    FancyArrowPatch = fancy_arrow_patch
    FancyBboxPatch = fancy_box_patch
    Polygon = polygon
    _configure_style(plt, font_manager)

    task_spans = pd.read_csv(case_dir / "task_spans.csv")
    critical_path = pd.read_csv(case_dir / "critical_path.csv")
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))

    branch_frame = task_spans[task_spans["task_kind"] == "branch"].copy()
    bottom_frame = task_spans[task_spans["partition"] == "bottom"].copy()
    slot_count = int(summary.get("max_gather_concurrency") or 1)
    branch_slots = _assign_branch_slots(branch_frame, slot_count)
    tail_segments = _compress_tail_segments(critical_path)

    barrier_us = float(task_spans.loc[task_spans["label"] == "barrier:tail", "start_us"].iloc[0])
    finish_us = float(task_spans["end_us"].max())
    total_ms = finish_us / 1000.0

    fig = plt.figure(figsize=(14.0, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.45], hspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[1, 0])
    y_positions = {
        "Bottom ops": 5.6,
        **{f"Slot {idx}": 4.7 - 0.9 * (idx - 1) for idx in range(1, slot_count + 1)},
        "Barrier": 1.6,
        "Tail summary": 0.7,
    }

    for lane, y in y_positions.items():
        ax.hlines(y, 0.0, total_ms, color=PALETTE["accent"], linewidth=1.0, zorder=0)
        ax.text(-0.16 * total_ms / 7.0, y, lane, ha="right", va="center", fontsize=11, color=PALETTE["ink"])

    if not bottom_frame.empty:
        bottom_start_ms = float(bottom_frame["start_us"].min()) / 1000.0
        bottom_end_ms = float(bottom_frame["end_us"].max()) / 1000.0
        bottom_duration_ms = bottom_end_ms - bottom_start_ms
        ax.broken_barh(
            [(bottom_start_ms, bottom_duration_ms)],
            (y_positions["Bottom ops"] - 0.24, 0.48),
            facecolors=PALETTE["bottom"],
            edgecolors="white",
            linewidth=1.0,
        )

    for lane, span in branch_slots:
        y = y_positions[lane]
        start_ms = span.start_us / 1000.0
        duration_ms = span.duration_us / 1000.0
        ax.broken_barh([(start_ms, duration_ms)], (y - 0.26, 0.52), facecolors=PALETTE["embedding"], edgecolors="white", linewidth=1.0)
        ax.text(start_ms + duration_ms / 2.0, y, span.label, ha="center", va="center", color="white", fontsize=10.2)

    barrier_ms = barrier_us / 1000.0
    ax.axvspan(barrier_ms, finish_us / 1000.0, ymin=0.0, ymax=0.22, facecolor="#F6EEF5", alpha=0.8, zorder=0)
    ax.vlines(barrier_ms, y_positions["Tail summary"] - 0.35, y_positions["Bottom ops"] + 0.40, colors=PALETTE["barrier"], linewidth=2.2, linestyles=(0, (4, 2)))
    ax.text(barrier_ms + total_ms * 0.006, y_positions["Barrier"], r"$u_{\mathrm{bar}}$", color=PALETTE["barrier"], fontsize=12, fontweight="bold", va="center")

    finish_ms = finish_us / 1000.0

    ax.broken_barh(
        [(barrier_ms, finish_ms - barrier_ms)],
        (y_positions["Tail summary"] - 0.24, 0.48),
        facecolors=PALETTE["tail"],
        edgecolors="white",
        linewidth=1.0,
    )
    ax.text(
        barrier_ms + (finish_ms - barrier_ms) / 2.0,
        y_positions["Tail summary"],
        r"$\mathcal{U}_{\mathrm{tail}}$",
        color="white",
        fontsize=10.0,
        ha="center",
        va="center",
    )

    ax.annotate(
        "",
        xy=(finish_ms, y_positions["Tail summary"] - 0.55),
        xytext=(0.0, y_positions["Tail summary"] - 0.55),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"], lw=1.2),
    )
    ax.text(finish_ms / 2.0, y_positions["Tail summary"] - 0.80, r"$\hat{M}$", ha="center", va="center", fontsize=12, color=PALETTE["ink"])

    ax.set_xlim(-0.06 * total_ms, total_ms * 1.03)
    ax.set_ylim(-0.15, 6.2)
    ax.set_xlabel("Scheduled time (ms)")
    ax.set_yticks([])
    ax.grid(True, axis="x", color=PALETTE["accent"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["line"])

    local_tail_ms = max((finish_us - barrier_us) / 1000.0, 1e-6)
    ax_zoom.axvline(0.0, color=PALETTE["barrier"], linewidth=2.0, linestyle=(0, (4, 2)))
    tiny_label_y = [0.93, 0.84, 0.75]
    tiny_label_idx = 0
    for segment in tail_segments:
        start_ms = (float(segment["start_us"]) - barrier_us) / 1000.0
        duration_ms = (float(segment["end_us"]) - float(segment["start_us"])) / 1000.0
        ax_zoom.broken_barh([(start_ms, duration_ms)], (0.25, 0.55), facecolors=PALETTE["tail"], edgecolors="white", linewidth=1.0)
        center_ms = start_ms + duration_ms / 2.0
        if duration_ms >= 18.0:
            ax_zoom.text(
                center_ms,
                0.525,
                str(segment["label"]),
                ha="center",
                va="center",
                fontsize=9.6,
                color="white",
            )
        else:
            label_y = tiny_label_y[min(tiny_label_idx, len(tiny_label_y) - 1)]
            tiny_label_idx += 1
            ax_zoom.plot([center_ms, center_ms], [0.82, 0.68], color=PALETTE["muted"], linewidth=0.8)
            ax_zoom.text(center_ms, label_y, str(segment["label"]), ha="center", va="bottom", fontsize=8.8, color=PALETTE["ink"])
    ax_zoom.text(0.0, 0.10, r"$u_{\mathrm{bar}}$", color=PALETTE["barrier"], fontsize=11.0, fontweight="bold", ha="left")
    ax_zoom.set_xlim(0.0, local_tail_ms)
    ax_zoom.set_ylim(0.0, 1.0)
    ax_zoom.set_yticks([])
    ax_zoom.set_xlabel("Time after barrier (ms)")
    ax_zoom.grid(True, axis="x", color=PALETTE["accent"], linewidth=0.8)
    for spine in ax_zoom.spines.values():
        spine.set_color(PALETTE["line"])
        spine.set_linewidth(0.8)
    ax_zoom.spines["left"].set_visible(False)
    ax_zoom.text(-0.015 * local_tail_ms, 0.52, "Tail critical chain", ha="right", va="center", fontsize=10.8, color=PALETTE["ink"])

    outputs = _save_all_formats(fig, output_dir / "fig_3_3_5_static_schedule_timeline")
    plt.close(fig)
    return outputs


def build_figures(case_dir: Path, output_dir: Path) -> dict[str, list[str]]:
    outputs = {
        "3.3.4": [str(path) for path in _draw_structure_figure(output_dir)],
        "3.3.5": [str(path) for path in _draw_timeline_figure(case_dir, output_dir)],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["manifest"] = [str(manifest_path)]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-style Chapter 3 method figures.")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR, help="Timeline-case directory containing task_spans.csv, critical_path.csv, and summary.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where the figure files will be written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figures(args.case_dir, args.output_dir)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
