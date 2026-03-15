#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from hardware_utils import write_hardware_profile


CLASS_BLOCK_RE = {
    "l1i": re.compile(r"class L1ICache\(Cache\):(.*?)class L1DCache", re.S),
    "l1d": re.compile(r"class L1DCache\(Cache\):(.*?)class L2Cache", re.S),
    "l2": re.compile(r"class L2Cache\(Cache\):(.*?)class L3Cache", re.S),
    "l3": re.compile(r"class L3Cache\(Cache\):(.*?)(?:# -----------------------------|\ndef make_ddr4)", re.S),
}


def _capture(block: str, pattern: str, cast: type = str, default: Any = None) -> Any:
    match = re.search(pattern, block)
    if not match:
        return default
    value = match.group(1)
    return cast(value) if cast is not str else value


def _cache_section(text: str, key: str) -> dict[str, Any]:
    match = CLASS_BLOCK_RE[key].search(text)
    if not match:
        raise ValueError(f"Could not locate cache class block for {key}")
    block = match.group(1)
    return {
        "size": _capture(block, r'size="([^"]+)"'),
        "assoc": _capture(block, r"assoc = (\d+)", int),
        "tag_latency_cycles": _capture(block, r"tag_latency = (\d+)", int),
        "data_latency_cycles": _capture(block, r"data_latency = (\d+)", int),
        "response_latency_cycles": _capture(block, r"response_latency = (\d+)", int),
        "mshrs": _capture(block, r"mshrs = (\d+)", int),
    }


def build_profile(config_path: Path, paper_path: Path | None) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")

    def arg_default(name: str, cast: type = str, default: Any = None) -> Any:
        pattern = rf'parser\.add_argument\("{re.escape(name)}",.*?default=([^,\n)]+)'
        match = re.search(pattern, text, re.S)
        if not match:
            return default
        raw = match.group(1).strip().strip('"').strip("'")
        return cast(raw) if cast is not str else raw

    widths = {
        "fetch_width": _capture(text, r"c\.fetchWidth = (\d+)", int),
        "decode_width": _capture(text, r"c\.decodeWidth = (\d+)", int),
        "rename_width": _capture(text, r"c\.renameWidth = (\d+)", int),
        "dispatch_width": _capture(text, r"c\.dispatchWidth = (\d+)", int),
        "issue_width": _capture(text, r"c\.issueWidth = (\d+)", int),
        "wb_width": _capture(text, r"c\.wbWidth = (\d+)", int),
        "commit_width": _capture(text, r"c\.commitWidth = (\d+)", int),
    }

    profile = {
        "profile_name": "kunpeng920_gem5",
        "source_config": str(config_path.resolve()),
        "source_paper": str(paper_path.resolve()) if paper_path else "",
        "notes": [
            "Values are extracted from the current gem5 configuration, not from marketing SKU sheets.",
            "The Huawei Research paper is kept as a cross-check reference because published silicon and the gem5 approximation are not identical.",
        ],
        "core": {
            "total_cores": _capture(text, r"total_cores = (\d+)", int),
            "cores_per_die": _capture(text, r"cores_per_die = (\d+)", int),
            "cpu_clock": arg_default("--cpu-clock"),
            "rob_entries": arg_default("--rob-entries", int),
            "lq_entries": arg_default("--lq-entries", int),
            "sq_entries": arg_default("--sq-entries", int),
            **widths,
        },
        "cache": {
            "l1i": _cache_section(text, "l1i"),
            "l1d": _cache_section(text, "l1d"),
            "l2": _cache_section(text, "l2"),
            "l3_per_die": {
                **_cache_section(text, "l3"),
                "size": arg_default("--l3-size-per-die"),
            },
            "cacheline_bytes": 64,
        },
        "memory": {
            "numa_nodes": 2,
            "mem_size0": arg_default("--mem-size0"),
            "mem_size1": arg_default("--mem-size1"),
            "local_mem_delay": arg_default("--local-mem-delay"),
            "remote_mem_delay": arg_default("--remote-mem-delay"),
            "dram_model": _capture(text, r"ctrl\.dram = ([A-Za-z0-9_]+)\(\)", str, "DDR4_2400_16x4"),
        },
        "paper_cross_check": {
            "paper_core_l1i_kib": 64,
            "paper_core_l1d_kib": 64,
            "paper_core_l2_kib": 512,
            "paper_total_l3_mib": 64,
            "paper_ddr_channels": 8,
            "paper_max_ddr_mtps": 2933,
            "paper_pipeline_width": 4,
        },
    }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a reusable hardware profile from kunpeng920.py")
    parser.add_argument(
        "--config",
        default="/data/qc/dlrm/ops_profile/concorde/simulatoin/kunpeng920.py",
        help="Path to the gem5 config script",
    )
    parser.add_argument(
        "--paper",
        default="/data/qc/dlrm/ORT/鲲鹏920.pdf",
        help="Optional PDF reference path recorded in the profile metadata",
    )
    parser.add_argument(
        "--output",
        default="/data/qc/dlrm/ORT/model/hardware_profiles/kunpeng920_gem5.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    profile = build_profile(Path(args.config), Path(args.paper) if args.paper else None)
    write_hardware_profile(Path(args.output), profile)
    print(Path(args.output))


if __name__ == "__main__":
    main()
