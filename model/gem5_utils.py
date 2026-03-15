from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from model_utils import infer_combo_from_path, parse_trace_op_idx, safe_float, split_combo


STAT_RE = re.compile(r"^(?P<name>\S+)\s+(?P<value>\S+)")
CPU_METRIC_RE = re.compile(r"^system\.cpu(?P<cpu>\d+)\.(?P<metric>numCycles|ipc|cpi)$")


def parse_gem5_stats_file(path: Path) -> dict[str, Any]:
    scalar_metrics: dict[str, float] = {}
    cpu_metrics: dict[int, dict[str, float]] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("-"):
                continue
            match = STAT_RE.match(line)
            if not match:
                continue

            name = match.group("name")
            value = safe_float(match.group("value"))
            if value is None:
                continue

            cpu_match = CPU_METRIC_RE.match(name)
            if cpu_match:
                cpu_id = int(cpu_match.group("cpu"))
                metric = cpu_match.group("metric")
                cpu_metrics.setdefault(cpu_id, {})[metric] = value
                continue

            if name in {"simSeconds", "simInsts", "simOps", "hostSeconds", "hostTickRate", "simFreq", "system.clk_domain.clock"}:
                scalar_metrics[name] = value

    active = []
    for cpu_id, metrics in sorted(cpu_metrics.items()):
        cycles = metrics.get("numCycles", 0.0)
        ipc = metrics.get("ipc")
        cpi = metrics.get("cpi")
        if cycles and cycles > 0:
            active.append(
                {
                    "cpu_id": cpu_id,
                    "cycles": cycles,
                    "ipc": ipc if ipc is not None and math.isfinite(ipc) else None,
                    "cpi": cpi if cpi is not None and math.isfinite(cpi) else None,
                }
            )

    cycle_sum = sum(item["cycles"] for item in active)
    cycle_max = max((item["cycles"] for item in active), default=0.0)
    sim_insts = scalar_metrics.get("simInsts")
    sim_seconds = scalar_metrics.get("simSeconds")
    sim_freq = scalar_metrics.get("simFreq")
    tick_period = scalar_metrics.get("system.clk_domain.clock")

    weighted_ipc = None
    if sim_insts is not None and cycle_sum > 0:
        weighted_ipc = sim_insts / cycle_sum

    elapsed_ns_from_cycles = None
    if sim_freq and tick_period and cycle_max > 0:
        elapsed_ns_from_cycles = cycle_max * tick_period / sim_freq * 1e9

    return {
        "label_gem5_sim_seconds": sim_seconds,
        "label_gem5_sim_us": sim_seconds * 1e6 if sim_seconds is not None else None,
        "label_gem5_sim_insts": sim_insts,
        "label_gem5_sim_ops": scalar_metrics.get("simOps"),
        "label_gem5_host_seconds": scalar_metrics.get("hostSeconds"),
        "label_gem5_sum_core_cycles": cycle_sum,
        "label_gem5_elapsed_cycles": cycle_max,
        "label_gem5_weighted_ipc": weighted_ipc,
        "label_gem5_active_cpu_count": float(len(active)),
        "label_gem5_mean_active_ipc": (
            sum(item["ipc"] for item in active if item["ipc"] is not None)
            / max(1, sum(1 for item in active if item["ipc"] is not None))
            if active
            else None
        ),
        "label_gem5_elapsed_ns_from_max_cycles": elapsed_ns_from_cycles,
    }


def collect_gem5_label_rows(stats_roots: list[Path], default_combo: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in stats_roots:
        for stats_path in sorted(root.rglob("stats.txt")):
            op_name = stats_path.parent.name
            op_idx = parse_trace_op_idx(op_name)
            combo = infer_combo_from_path(stats_path) or default_combo
            batch_size, num_indices = split_combo(combo)
            row = {
                "combo": combo,
                "batch_size": batch_size,
                "num_indices_per_lookup": num_indices,
                "op_idx": op_idx,
                "gem5_op_name": op_name,
                "gem5_stats_path": str(stats_path),
                "gem5_root": str(root),
            }
            row.update(parse_gem5_stats_file(stats_path))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "label_gem5_host_seconds" in df.columns:
        df = df.sort_values(
            by=["combo", "op_idx", "label_gem5_host_seconds"],
            ascending=[True, True, False],
            na_position="last",
        )
    df = df.drop_duplicates(subset=["combo", "op_idx"], keep="first")
    return df.reset_index(drop=True)
