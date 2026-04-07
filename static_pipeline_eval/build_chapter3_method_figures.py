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


def _panel(ax, title: str) -> None:
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
    ax.text(0.04, 0.94, title, fontsize=13, fontweight="bold", color=PALETTE["ink"], va="top")


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

    fig = plt.figure(figsize=(13.6, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.12)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    _panel(ax_left, "(a) Original operator DAG")
    _panel(ax_right, "(b) Hybrid task DAG used by the scheduler")

    bottom_nodes = [(0.08, 0.76, "Bottom\nop 1"), (0.23, 0.76, "Bottom\nop 2"), (0.38, 0.76, "Bottom\nop 3"), (0.53, 0.76, "Bottom\nop 4")]
    for idx, (x, y, label) in enumerate(bottom_nodes):
        _box(ax_left, x, y, 0.11, 0.10, label, face=PALETTE["bottom"])
        if idx:
            prev_x, prev_y, _ = bottom_nodes[idx - 1]
            _arrow(ax_left, (prev_x + 0.11, prev_y + 0.05), (x, y + 0.05))

    branch_columns = [0.09, 0.22, 0.35, 0.61]
    branch_labels = ["emb_l0", "emb_l1", "emb_l2", "emb_l7"]
    for x, branch_label in zip(branch_columns, branch_labels):
        ax_left.text(x + 0.045, 0.655, branch_label, ha="center", va="bottom", fontsize=9.5, color=PALETTE["muted"])
        _box(ax_left, x, 0.54, 0.09, 0.08, "Gather", face="#FDBA73", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=9.5)
        _box(ax_left, x, 0.42, 0.09, 0.08, "Reshape", face="#FFE1BF", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=9.2)
        _box(ax_left, x, 0.30, 0.09, 0.08, "Reduce\nSum", face="#FBCFA7", edge=PALETTE["embedding"], text_color=PALETTE["ink"], fontsize=9.0)
        _arrow(ax_left, (x + 0.045, 0.54), (x + 0.045, 0.50))
        _arrow(ax_left, (x + 0.045, 0.42), (x + 0.045, 0.38))

    ax_left.text(0.495, 0.46, "⋯", fontsize=28, color=PALETTE["muted"], ha="center", va="center")
    ax_left.text(0.495, 0.34, "⋯", fontsize=28, color=PALETTE["muted"], ha="center", va="center")

    _box(ax_left, 0.74, 0.54, 0.16, 0.09, "Feature\ninteraction", face=PALETTE["tail"], fontsize=10.2)
    _box(ax_left, 0.74, 0.37, 0.16, 0.09, "Top MLP\n/ output", face="#6FBF85", edge=PALETTE["tail"], fontsize=10.2)
    _arrow(ax_left, (0.64, 0.81), (0.74, 0.585), connectionstyle="arc3,rad=-0.18")
    for x in branch_columns:
        _arrow(ax_left, (x + 0.045, 0.30), (0.74, 0.585), connectionstyle="arc3,rad=0.08")
    _arrow(ax_left, (0.82, 0.54), (0.82, 0.46))
    ax_left.text(0.08, 0.10, "Constants are omitted; each embedding branch contains\nthree scheduled operators: Gather, Reshape, and ReduceSum.", color=PALETTE["muted"], fontsize=9.4)

    task_bottom = [(0.07, 0.76, r"$u_1$"), (0.20, 0.76, r"$u_2$"), (0.33, 0.76, r"$u_3$"), (0.46, 0.76, r"$u_4$")]
    for idx, (x, y, label) in enumerate(task_bottom):
        _box(ax_right, x, y, 0.09, 0.09, label, face=PALETTE["bottom"], fontsize=11.5)
        if idx:
            prev_x, prev_y, _ = task_bottom[idx - 1]
            _arrow(ax_right, (prev_x + 0.09, prev_y + 0.045), (x, y + 0.045))
    ax_right.text(0.07, 0.69, r"$\mathcal{U}_{\mathrm{bot}}$", color=PALETTE["bottom"], fontsize=11.5, fontweight="bold")

    task_branches = [(0.08, 0.47, r"$B_0$"), (0.21, 0.47, r"$B_1$"), (0.34, 0.47, r"$B_2$"), (0.60, 0.47, r"$B_7$")]
    for x, y, label in task_branches:
        _box(ax_right, x, y, 0.09, 0.09, label, face=PALETTE["embedding"], fontsize=11.5)
    ax_right.text(0.445, 0.515, "⋯", fontsize=30, color=PALETTE["muted"], ha="center", va="center")
    ax_right.text(0.08, 0.40, r"$B_j = \mathrm{Gather}_j + \mathrm{Reshape}_j + \mathrm{ReduceSum}_j$", color=PALETTE["muted"], fontsize=10.5)
    ax_right.text(0.08, 0.62, r"$\mathcal{U}_{\mathrm{emb}}=\{B_0,\ldots,B_7\}$", color=PALETTE["embedding"], fontsize=11.5, fontweight="bold")

    _diamond(ax_right, 0.70, 0.45, 0.14, 0.12, r"$u_{\mathrm{bar}}$", face=PALETTE["barrier"], fontsize=12)
    for x, y, _ in task_bottom:
        _arrow(ax_right, (x + 0.045, y), (0.77, 0.57), connectionstyle="arc3,rad=-0.15")
    for x, y, _ in task_branches:
        _arrow(ax_right, (x + 0.045, y + 0.09), (0.77, 0.45), connectionstyle="arc3,rad=0.08")

    tail_nodes = [(0.72, 0.26, r"$v_1$"), (0.84, 0.26, r"$v_2$"), (0.72, 0.12, r"$v_3$"), (0.84, 0.12, r"$v_4$")]
    for x, y, label in tail_nodes:
        _box(ax_right, x, y, 0.09, 0.09, label, face=PALETTE["tail"], fontsize=11.5)
    _arrow(ax_right, (0.77, 0.45), (0.765, 0.35))
    _arrow(ax_right, (0.81, 0.305), (0.84, 0.305))
    _arrow(ax_right, (0.765, 0.26), (0.765, 0.21))
    _arrow(ax_right, (0.81, 0.165), (0.84, 0.165))
    ax_right.text(0.72, 0.06, r"$\mathcal{U}_{\mathrm{tail}}$", color=PALETTE["tail"], fontsize=11.5, fontweight="bold")

    fig.suptitle("Static task-graph construction for branch-parallel DLRM inference", fontsize=15, fontweight="bold", y=0.99)
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
        ("Concat + reshape", 0, 5),
        ("Interaction", 5, 11),
        ("Top MLP block 1", 11, 16),
        ("Top MLP block 2 / output", 16, len(tail_rows)),
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

    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    y_positions = {
        "Bottom ops": 5.5,
        **{f"Slot {idx}": 4.6 - 0.9 * (idx - 1) for idx in range(1, slot_count + 1)},
        "Barrier": 1.55,
        "Tail critical chain": 0.55,
    }

    for lane, y in y_positions.items():
        ax.hlines(y, 0.0, total_ms, color=PALETTE["accent"], linewidth=1.0, zorder=0)
        ax.text(-0.20 * total_ms / 7.0, y, lane, ha="right", va="center", fontsize=11, color=PALETTE["ink"])

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
    ax.vlines(barrier_ms, y_positions["Tail critical chain"] - 0.35, y_positions["Bottom ops"] + 0.40, colors=PALETTE["barrier"], linewidth=2.2, linestyles=(0, (4, 2)))
    ax.text(barrier_ms + total_ms * 0.008, y_positions["Barrier"], r"$u_{\mathrm{bar}}$", color=PALETTE["barrier"], fontsize=12, fontweight="bold", va="center")
    ax.text(
        barrier_ms + total_ms * 0.008,
        y_positions["Barrier"] - 0.26,
        r"$s(u_{\mathrm{bar}})=\max(\max f(u), \max f(B_j))$",
        color=PALETTE["muted"],
        fontsize=9.5,
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
    )

    finish_ms = finish_us / 1000.0

    ax.broken_barh(
        [(barrier_ms, finish_ms - barrier_ms)],
        (y_positions["Tail critical chain"] - 0.24, 0.48),
        facecolors=PALETTE["tail"],
        edgecolors="white",
        linewidth=1.0,
    )
    branch3 = branch_frame[branch_frame["label"] == "branch:3"].iloc[0]
    wait_ms = float(branch3["start_us"]) / 1000.0
    ax.annotate(
        r"$s(B_3)=\max\{r(B_3), a_{\min}\}$",
        xy=(wait_ms, y_positions["Slot 2"] + 0.28),
        xytext=(wait_ms + total_ms * 0.06, y_positions["Slot 2"] + 0.95),
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1.2),
        fontsize=10.2,
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.92),
    )
    ax.text(
        wait_ms + total_ms * 0.06,
        y_positions["Slot 2"] + 0.60,
        "B3 can start only after the earliest slot releases.",
        fontsize=9.5,
        color=PALETTE["muted"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
    )

    ax.annotate(
        "",
        xy=(finish_ms, y_positions["Tail critical chain"] - 0.55),
        xytext=(0.0, y_positions["Tail critical chain"] - 0.55),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"], lw=1.2),
    )
    ax.text(finish_ms / 2.0, y_positions["Tail critical chain"] - 0.78, r"predicted full-graph latency $\hat{M}$", ha="center", va="center", fontsize=11, color=PALETTE["ink"])

    ax.set_xlim(-0.06 * total_ms, total_ms * 1.03)
    ax.set_ylim(-0.15, 6.2)
    ax.set_xlabel("Scheduled time (ms)")
    ax.set_yticks([])
    ax.grid(True, axis="x", color=PALETTE["accent"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["line"])

    ax.text(
        0.0,
        6.05,
        "Static timeline generation under slot-limited FIFO branch scheduling",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.0,
        5.78,
        f"Example aligned with the actual scheduler artifact: case_10_3_3 / bs2048_nip2000, kappa = {slot_count}.",
        fontsize=10.2,
        color=PALETTE["muted"],
    )

    inset = ax.inset_axes([0.70, 0.06, 0.25, 0.16])
    inset.set_facecolor("white")
    local_tail_ms = max((finish_us - barrier_us) / 1000.0, 1e-6)
    inset_labels = {
        "Concat + reshape": "Concat\n+ reshape",
        "Interaction": "Interaction",
        "Top MLP block 1": "Top MLP\nblock 1",
        "Top MLP block 2 / output": "Top MLP\nblock 2",
    }
    for segment in tail_segments:
        start_ms = (float(segment["start_us"]) - barrier_us) / 1000.0
        duration_ms = (float(segment["end_us"]) - float(segment["start_us"])) / 1000.0
        inset.broken_barh([(start_ms, duration_ms)], (0.1, 0.55), facecolors=PALETTE["tail"], edgecolors="white", linewidth=1.0)
        inset.text(
            start_ms + duration_ms / 2.0,
            0.375,
            inset_labels.get(str(segment["label"]), str(segment["label"])),
            ha="center",
            va="center",
            fontsize=7.4,
            color="white",
            linespacing=0.9,
        )
    inset.set_xlim(0.0, local_tail_ms)
    inset.set_ylim(0.0, 0.8)
    inset.set_yticks([])
    inset.grid(True, axis="x", color=PALETTE["accent"], linewidth=0.6)
    inset.set_title("Tail zoom", fontsize=9.5, color=PALETTE["ink"], pad=2.0)
    inset.set_xlabel("ms after barrier", fontsize=8.6)
    inset.tick_params(axis="x", labelsize=8.2)
    for spine in inset.spines.values():
        spine.set_color(PALETTE["line"])
        spine.set_linewidth(0.8)

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
