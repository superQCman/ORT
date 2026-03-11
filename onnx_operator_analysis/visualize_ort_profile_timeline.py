#!/usr/bin/env python3
"""
Visualize ORT profiling Node events as a lane-based timeline and summarize
operator concurrency.

Outputs:
  - <stem>_operator_timeline.html
  - <stem>_operator_timeline.csv
  - <stem>_concurrency_segments.csv

The HTML is self-contained and uses only inline SVG + CSS so it can be opened
directly in a browser without extra dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from extract_cpu_thread_usage import find_parallel_ops, load_profile

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def parse_node_events(events: Sequence[dict], provider_filter: str | None = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, ev in enumerate(events):
        if ev.get("cat") != "Node":
            continue
        dur = int(ev.get("dur", 0) or 0)
        if dur <= 0:
            continue

        args = ev.get("args", {})
        provider = str(args.get("provider", "") or "")
        if provider_filter and provider != provider_filter:
            continue

        ts = int(ev.get("ts", 0) or 0)
        row = {
            "event_id": idx,
            "node_name": str(ev.get("name", "") or ""),
            "op_name": str(args.get("op_name", "") or "unknown"),
            "provider": provider,
            "node_index": str(args.get("node_index", "") or ""),
            "ts_us": ts,
            "dur_us": dur,
            "end_us": ts + dur,
            "output_size": str(args.get("output_size", "") or ""),
            "activation_size": str(args.get("activation_size", "") or ""),
            "parameter_size": str(args.get("parameter_size", "") or ""),
        }
        rows.append(row)

    rows.sort(key=lambda row: (int(row["ts_us"]), int(row["end_us"]), str(row["node_name"])))
    return rows


def assign_lanes(nodes: List[Dict[str, object]]) -> int:
    """Assign each node interval to the earliest available lane."""
    active_heap: List[tuple[int, int]] = []
    free_lanes: List[int] = []
    next_lane = 0

    for node in nodes:
        ts_us = int(node["ts_us"])
        while active_heap and active_heap[0][0] <= ts_us:
            _, lane = heappop(active_heap)
            heappush(free_lanes, lane)

        if free_lanes:
            lane = heappop(free_lanes)
        else:
            lane = next_lane
            next_lane += 1

        node["lane"] = lane
        heappush(active_heap, (int(node["end_us"]), lane))

    return next_lane


def build_occupancy_segments(nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build sweep-line occupancy intervals for the timeline."""
    boundaries: List[tuple[int, int, int]] = []
    for idx, node in enumerate(nodes):
        boundaries.append((int(node["ts_us"]), 0, idx))
        boundaries.append((int(node["end_us"]), 1, idx))
    boundaries.sort()

    active: set[int] = set()
    prev_time: int | None = None
    segments: List[Dict[str, object]] = []

    for time_pt, event_type, idx in boundaries:
        if prev_time is not None and time_pt > prev_time and active:
            active_rows = [nodes[i] for i in sorted(active)]
            providers = sorted({str(row["provider"]) for row in active_rows if row["provider"]})
            op_names = [str(row["op_name"]) for row in active_rows]
            node_indices = [str(row["node_index"]) for row in active_rows if str(row["node_index"])]
            segments.append(
                {
                    "start_us": prev_time,
                    "end_us": time_pt,
                    "dur_us": time_pt - prev_time,
                    "concurrency": len(active_rows),
                    "providers": "|".join(providers),
                    "op_names": "|".join(op_names),
                    "node_indices": "|".join(node_indices),
                }
            )

        prev_time = time_pt
        if event_type == 0:
            active.add(idx)
        else:
            active.discard(idx)

    return segments


def summarize_timeline(nodes: Sequence[Dict[str, object]], occupancy_segments: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not nodes:
        return {
            "event_count": 0,
            "provider_counts": {},
            "op_counts": {},
            "min_ts_us": 0,
            "max_end_us": 0,
            "wall_time_us": 0,
            "total_active_us": 0,
            "total_parallel_us": 0,
            "parallel_pct": 0.0,
            "max_concurrency": 0,
            "avg_concurrency_when_active": 0.0,
        }

    min_ts_us = min(int(node["ts_us"]) for node in nodes)
    max_end_us = max(int(node["end_us"]) for node in nodes)
    total_active_us = sum(int(seg["dur_us"]) for seg in occupancy_segments)
    total_parallel_us = sum(int(seg["dur_us"]) for seg in occupancy_segments if int(seg["concurrency"]) >= 2)
    weighted_concurrency = sum(int(seg["dur_us"]) * int(seg["concurrency"]) for seg in occupancy_segments)

    provider_counts = Counter(str(node["provider"]) for node in nodes)
    op_counts = Counter(str(node["op_name"]) for node in nodes)

    return {
        "event_count": len(nodes),
        "provider_counts": dict(provider_counts),
        "op_counts": dict(op_counts),
        "min_ts_us": min_ts_us,
        "max_end_us": max_end_us,
        "wall_time_us": max_end_us - min_ts_us,
        "total_active_us": total_active_us,
        "total_parallel_us": total_parallel_us,
        "parallel_pct": (100.0 * total_parallel_us / total_active_us) if total_active_us else 0.0,
        "max_concurrency": max((int(seg["concurrency"]) for seg in occupancy_segments), default=0),
        "avg_concurrency_when_active": (weighted_concurrency / total_active_us) if total_active_us else 0.0,
    }


def hash_color(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue}, 58%, 68%)"


def format_us(value: int) -> str:
    return f"{value:,} us"


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def to_rows(nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for node in nodes:
        out.append(
            {
                "event_id": node["event_id"],
                "lane": node["lane"],
                "node_index": node["node_index"],
                "node_name": node["node_name"],
                "op_name": node["op_name"],
                "provider": node["provider"],
                "ts_us": node["ts_us"],
                "dur_us": node["dur_us"],
                "end_us": node["end_us"],
                "output_size": node["output_size"],
                "activation_size": node["activation_size"],
                "parameter_size": node["parameter_size"],
            }
        )
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_counts_table(title: str, counter: Dict[str, int], limit: int = 12) -> str:
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    if not rows:
        return ""
    body = "\n".join(
        f"<tr><td>{html.escape(name or '(empty)')}</td><td>{count}</td></tr>"
        for name, count in rows
    )
    return (
        f"<section><h3>{html.escape(title)}</h3>"
        "<table><thead><tr><th>Name</th><th>Count</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def render_pair_table(pair_rows: Sequence[Dict[str, object]], limit: int = 15) -> str:
    if not pair_rows:
        return "<p>No overlapping operator pairs detected.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['op_a']))}</td>"
        f"<td>{html.escape(str(row['provider_a']))}</td>"
        f"<td>{html.escape(str(row['op_b']))}</td>"
        f"<td>{html.escape(str(row['provider_b']))}</td>"
        f"<td>{row['overlap_count']}</td>"
        f"<td>{row['total_overlap_us']}</td>"
        "</tr>"
        for row in pair_rows[:limit]
    )
    return (
        "<table><thead><tr>"
        "<th>op_a</th><th>provider_a</th><th>op_b</th><th>provider_b</th>"
        "<th>overlap_count</th><th>total_overlap_us</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def render_segment_table(segments: Sequence[Dict[str, object]], limit: int = 20) -> str:
    parallel_only = [seg for seg in segments if int(seg["concurrency"]) >= 2]
    parallel_only.sort(key=lambda seg: (-int(seg["concurrency"]), -int(seg["dur_us"]), int(seg["start_us"])))
    if not parallel_only:
        return "<p>No concurrency segments detected.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{seg['concurrency']}</td>"
        f"<td>{seg['start_us']}</td>"
        f"<td>{seg['end_us']}</td>"
        f"<td>{seg['dur_us']}</td>"
        f"<td>{html.escape(truncate_text(str(seg['op_names']), 120))}</td>"
        "</tr>"
        for seg in parallel_only[:limit]
    )
    return (
        "<table><thead><tr>"
        "<th>concurrency</th><th>start_us</th><th>end_us</th><th>dur_us</th><th>op_names</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def render_html(
    json_path: Path,
    html_path: Path,
    nodes: Sequence[Dict[str, object]],
    occupancy_segments: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    pair_rows: Sequence[Dict[str, object]],
    lane_count: int,
) -> None:
    min_ts_us = int(summary["min_ts_us"])
    max_end_us = int(summary["max_end_us"])
    span_us = max(max_end_us - min_ts_us, 1)

    chart_width = 1600
    left_pad = 120
    right_pad = 20
    top_pad = 24
    lane_height = 22
    plot_width = chart_width - left_pad - right_pad
    lane_chart_height = max(lane_count * lane_height + 40, 160)
    max_concurrency = max(int(summary["max_concurrency"]), 1)
    occupancy_height = 140
    svg_height = occupancy_height + lane_chart_height + 100

    def x_for(value: int) -> float:
        return left_pad + ((value - min_ts_us) / span_us) * plot_width

    occupancy_rects = []
    for seg in occupancy_segments:
        start = int(seg["start_us"])
        end = int(seg["end_us"])
        concurrency = int(seg["concurrency"])
        x = x_for(start)
        width = max(x_for(end) - x, 1.0)
        bar_height = (concurrency / max_concurrency) * (occupancy_height - 40)
        y = top_pad + (occupancy_height - 20) - bar_height
        occupancy_rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{bar_height:.2f}" '
            f'class="occupancy" data-count="{concurrency}"><title>'
            f'concurrency={concurrency} start={start}us end={end}us dur={end - start}us'
            f"</title></rect>"
        )

    lane_rects = []
    lane_labels = []
    for lane in range(lane_count):
        y = occupancy_height + 30 + lane * lane_height
        lane_labels.append(
            f'<text x="{left_pad - 10}" y="{y + 13}" class="lane-label">Lane {lane}</text>'
        )

    for node in nodes:
        start = int(node["ts_us"])
        end = int(node["end_us"])
        dur = int(node["dur_us"])
        lane = int(node["lane"])
        x = x_for(start)
        width = max(x_for(end) - x, 1.0)
        y = occupancy_height + 30 + lane * lane_height
        fill = hash_color(f"{node['provider']}::{node['op_name']}")
        label = str(node["op_name"])
        title = (
            f"node_name={node['node_name']}\n"
            f"op_name={node['op_name']}\n"
            f"provider={node['provider']}\n"
            f"node_index={node['node_index']}\n"
            f"lane={lane}\n"
            f"start_us={start}\n"
            f"dur_us={dur}\n"
            f"end_us={end}\n"
            f"output_size={node['output_size']}\n"
            f"activation_size={node['activation_size']}\n"
            f"parameter_size={node['parameter_size']}"
        )
        lane_rects.append(
            f'<g><rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="16" rx="3" '
            f'fill="{fill}" class="event"><title>{html.escape(title)}</title></rect>'
            + (
                f'<text x="{x + 3:.2f}" y="{y + 12:.2f}" class="event-label">{html.escape(truncate_text(label, 20))}</text>'
                if width >= 38
                else ""
            )
            + "</g>"
        )

    tick_count = 10
    ticks = []
    for idx in range(tick_count + 1):
        value = min_ts_us + int(span_us * idx / tick_count)
        x = x_for(value)
        ticks.append(
            f'<line x1="{x:.2f}" y1="{top_pad}" x2="{x:.2f}" y2="{svg_height - 30}" class="tick" />'
            f'<text x="{x:.2f}" y="{svg_height - 8}" class="tick-label">{value - min_ts_us} us</text>'
        )

    provider_table = render_counts_table("Provider Event Counts", summary["provider_counts"], limit=8)
    op_table = render_counts_table("Top Op Counts", summary["op_counts"], limit=12)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ORT Profile Timeline - {html.escape(json_path.name)}</title>
  <style>
    body {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      margin: 24px;
      color: #10212b;
      background: #f5f7f9;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 12px;
      margin: 18px 0 24px 0;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d6dde3;
      border-radius: 8px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .card .label {{
      font-size: 12px;
      color: #5b6b76;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d6dde3;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
      overflow-x: auto;
    }}
    svg {{
      background: #fbfcfd;
      border: 1px solid #e0e6eb;
      border-radius: 8px;
    }}
    .tick {{
      stroke: #dbe4ea;
      stroke-width: 1;
    }}
    .tick-label {{
      font-size: 10px;
      text-anchor: middle;
      fill: #6d7c87;
    }}
    .lane-label {{
      font-size: 11px;
      text-anchor: end;
      fill: #6d7c87;
    }}
    .event {{
      stroke: rgba(16, 33, 43, 0.35);
      stroke-width: 0.7;
    }}
    .event-label {{
      font-size: 10px;
      fill: #0e2430;
      pointer-events: none;
    }}
    .occupancy {{
      fill: #3b82f6;
      opacity: 0.55;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(360px, 1fr));
      gap: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e7edf1;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f4f8fb;
    }}
    .note {{
      color: #5b6b76;
      font-size: 12px;
      margin-top: 10px;
    }}
  </style>
</head>
<body>
  <h1>ORT Profile Timeline</h1>
  <div class="note">Source JSON: {html.escape(str(json_path))}</div>

  <div class="summary">
    <div class="card"><div class="label">Node events</div><div class="value">{summary['event_count']}</div></div>
    <div class="card"><div class="label">Lane count</div><div class="value">{lane_count}</div></div>
    <div class="card"><div class="label">Max concurrency</div><div class="value">{summary['max_concurrency']}</div></div>
    <div class="card"><div class="label">Parallel active time</div><div class="value">{format_pct(float(summary['parallel_pct']))}</div></div>
    <div class="card"><div class="label">Wall time span</div><div class="value">{format_us(int(summary['wall_time_us']))}</div></div>
    <div class="card"><div class="label">Active time</div><div class="value">{format_us(int(summary['total_active_us']))}</div></div>
    <div class="card"><div class="label">Parallel time</div><div class="value">{format_us(int(summary['total_parallel_us']))}</div></div>
    <div class="card"><div class="label">Avg concurrency when active</div><div class="value">{float(summary['avg_concurrency_when_active']):.2f}</div></div>
  </div>

  <section class="panel">
    <h2>Timeline</h2>
    <svg width="{chart_width}" height="{svg_height}" viewBox="0 0 {chart_width} {svg_height}">
      {''.join(ticks)}
      <text x="{left_pad}" y="16" class="lane-label" text-anchor="start">Concurrency occupancy</text>
      {''.join(occupancy_rects)}
      {''.join(lane_labels)}
      {''.join(lane_rects)}
    </svg>
    <div class="note">Top chart shows instantaneous concurrency. Lower chart assigns each Node event to a lane; overlapping bars mean real overlap in the ORT profile timeline. Hover a bar to see full node details.</div>
  </section>

  <div class="panel-grid">
    <section class="panel">
      <h2>Top Overlap Pairs</h2>
      {render_pair_table(pair_rows)}
    </section>
    <section class="panel">
      <h2>Top Parallel Segments</h2>
      {render_segment_table(occupancy_segments)}
    </section>
  </div>

  <div class="panel-grid">
    <section class="panel">
      {provider_table}
    </section>
    <section class="panel">
      {op_table}
    </section>
  </div>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def load_font(size: int):
    if ImageFont is None:
        return None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_raster_image(
    image_path: Path,
    nodes: Sequence[Dict[str, object]],
    occupancy_segments: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    lane_count: int,
) -> None:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG/JPG export.")

    min_ts_us = int(summary["min_ts_us"])
    max_end_us = int(summary["max_end_us"])
    span_us = max(max_end_us - min_ts_us, 1)

    width = 2200
    left_pad = 150
    right_pad = 40
    top_pad = 28
    plot_width = width - left_pad - right_pad
    occupancy_height = 180
    lane_height = 26
    lane_chart_top = 320
    lane_chart_height = max(lane_count * lane_height + 40, 180)
    summary_height = 170
    bottom_pad = 80
    height = summary_height + occupancy_height + lane_chart_height + bottom_pad

    image = Image.new("RGB", (width, height), (245, 247, 249))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    section_font = load_font(18)
    body_font = load_font(14)
    small_font = load_font(12)

    draw.text((24, 20), "ORT Profile Timeline", fill=(16, 33, 43), font=title_font)
    draw.text((24, 58), f"Node events: {summary['event_count']}", fill=(72, 92, 104), font=body_font)

    cards = [
        ("Lane count", str(lane_count)),
        ("Max concurrency", str(summary["max_concurrency"])),
        ("Parallel active %", format_pct(float(summary["parallel_pct"]))),
        ("Wall time", format_us(int(summary["wall_time_us"]))),
        ("Parallel time", format_us(int(summary["total_parallel_us"]))),
        ("Avg concurrency", f"{float(summary['avg_concurrency_when_active']):.2f}"),
    ]
    card_x = 24
    card_y = 95
    card_w = 320
    card_h = 56
    for idx, (label, value) in enumerate(cards):
        x = card_x + (idx % 3) * (card_w + 16)
        y = card_y + (idx // 3) * (card_h + 12)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=10, fill=(255, 255, 255), outline=(214, 221, 227))
        draw.text((x + 12, y + 9), label, fill=(91, 107, 118), font=small_font)
        draw.text((x + 12, y + 28), value, fill=(16, 33, 43), font=section_font)

    occ_top = summary_height
    draw.text((24, occ_top - 28), "Concurrency occupancy", fill=(16, 33, 43), font=section_font)
    draw.rounded_rectangle((left_pad, occ_top, left_pad + plot_width, occ_top + occupancy_height), radius=8, fill=(251, 252, 253), outline=(224, 230, 235))

    max_concurrency = max(int(summary["max_concurrency"]), 1)

    def x_for(value: int) -> float:
        return left_pad + ((value - min_ts_us) / span_us) * plot_width

    tick_count = 10
    for idx in range(tick_count + 1):
        value = min_ts_us + int(span_us * idx / tick_count)
        x = x_for(value)
        draw.line((x, occ_top, x, height - bottom_pad + 8), fill=(219, 228, 234), width=1)
        label = f"{value - min_ts_us} us"
        draw.text((x - 24, height - bottom_pad + 16), label, fill=(109, 124, 135), font=small_font)

    for seg in occupancy_segments:
        start = int(seg["start_us"])
        end = int(seg["end_us"])
        concurrency = int(seg["concurrency"])
        x0 = x_for(start)
        x1 = max(x_for(end), x0 + 1)
        bar_height = (concurrency / max_concurrency) * (occupancy_height - 36)
        y0 = occ_top + occupancy_height - 18 - bar_height
        y1 = occ_top + occupancy_height - 18
        fill = (59, 130, 246) if concurrency >= 2 else (147, 197, 253)
        draw.rectangle((x0, y0, x1, y1), fill=fill)

    lane_top = lane_chart_top
    draw.text((24, lane_top - 28), "Operator lanes", fill=(16, 33, 43), font=section_font)
    draw.rounded_rectangle((left_pad, lane_top, left_pad + plot_width, lane_top + lane_chart_height), radius=8, fill=(251, 252, 253), outline=(224, 230, 235))

    for lane in range(lane_count):
        y = lane_top + 18 + lane * lane_height
        draw.text((32, y), f"Lane {lane}", fill=(109, 124, 135), font=small_font)

    for node in nodes:
        start = int(node["ts_us"])
        end = int(node["end_us"])
        lane = int(node["lane"])
        x0 = x_for(start)
        x1 = max(x_for(end), x0 + 1)
        y0 = lane_top + 18 + lane * lane_height
        y1 = y0 + 16
        fill = hash_color(f"{node['provider']}::{node['op_name']}")
        draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=fill, outline=(35, 55, 66))
        if x1 - x0 >= 60:
            label = truncate_text(str(node["op_name"]), 16)
            draw.text((x0 + 4, y0 + 2), label, fill=(14, 36, 48), font=small_font)

    note = "Overlapping bars indicate actual overlap in ORT Node events. Blue occupancy bars above 1 mean concurrent operator execution."
    draw.text((24, height - 40), note, fill=(91, 107, 118), font=small_font)

    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(image_path, quality=95)
    else:
        image.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ORT profile operator timeline and concurrency.")
    parser.add_argument("json_path", help="Path to ort_cann_profile_*.json")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to the JSON directory.")
    parser.add_argument("--provider", default=None, help="Optional provider filter, e.g. CPUExecutionProvider")
    parser.add_argument("--png", action="store_true", help="Also export a PNG timeline image.")
    parser.add_argument("--jpg", action="store_true", help="Also export a JPG timeline image.")
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_profile(str(json_path))
    nodes = parse_node_events(events, provider_filter=args.provider)
    if not nodes:
        raise SystemExit("No Node events found for the requested filter.")

    lane_count = assign_lanes(nodes)
    occupancy_segments = build_occupancy_segments(nodes)
    parallel_pairs, _ = find_parallel_ops(events)
    if args.provider:
        parallel_pairs = [
            row
            for row in parallel_pairs
            if row.get("provider_a") == args.provider and row.get("provider_b") == args.provider
        ]
    summary = summarize_timeline(nodes, occupancy_segments)

    stem = json_path.stem
    timeline_csv = out_dir / f"{stem}_operator_timeline.csv"
    concurrency_csv = out_dir / f"{stem}_concurrency_segments.csv"
    html_path = out_dir / f"{stem}_operator_timeline.html"
    png_path = out_dir / f"{stem}_operator_timeline.png"
    jpg_path = out_dir / f"{stem}_operator_timeline.jpg"

    write_csv(timeline_csv, to_rows(nodes))
    write_csv(concurrency_csv, [seg for seg in occupancy_segments if int(seg["concurrency"]) >= 2])
    render_html(json_path, html_path, nodes, occupancy_segments, summary, parallel_pairs, lane_count)
    if args.png:
        render_raster_image(png_path, nodes, occupancy_segments, summary, lane_count)
    if args.jpg:
        render_raster_image(jpg_path, nodes, occupancy_segments, summary, lane_count)

    print(f"Node events             : {summary['event_count']}")
    print(f"Lane count              : {lane_count}")
    print(f"Wall time span (us)     : {summary['wall_time_us']}")
    print(f"Active time (us)        : {summary['total_active_us']}")
    print(f"Parallel time (us)      : {summary['total_parallel_us']}")
    print(f"Parallel active time %  : {summary['parallel_pct']:.2f}")
    print(f"Max concurrency         : {summary['max_concurrency']}")
    print(f"Avg concurrency active  : {summary['avg_concurrency_when_active']:.2f}")
    print(f"Saved timeline CSV      : {timeline_csv}")
    print(f"Saved concurrency CSV   : {concurrency_csv}")
    print(f"Saved timeline HTML     : {html_path}")
    if args.png:
        print(f"Saved timeline PNG      : {png_path}")
    if args.jpg:
        print(f"Saved timeline JPG      : {jpg_path}")


if __name__ == "__main__":
    main()
