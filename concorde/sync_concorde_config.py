#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml


SIZE_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KiB|MiB|GiB|KB|MB|GB)\s*$")
FREQ_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)\s*$")
TIME_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ns|us|ms|s)\s*$")


def parse_size_bytes(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = SIZE_RE.match(str(value).strip())
    if not match:
        return None
    scale = {
        "KB": 1000.0,
        "MB": 1000.0**2,
        "GB": 1000.0**3,
        "KiB": 1024.0,
        "MiB": 1024.0**2,
        "GiB": 1024.0**3,
    }[match.group("unit")]
    return int(float(match.group("value")) * scale)


def parse_freq_ghz(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = FREQ_RE.match(str(value).strip())
    if not match:
        return None
    scale = {
        "Hz": 1e-9,
        "kHz": 1e-6,
        "MHz": 1e-3,
        "GHz": 1.0,
    }[match.group("unit")]
    return float(match.group("value")) * scale


def parse_time_ns(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = TIME_RE.match(str(value).strip())
    if not match:
        return None
    scale = {
        "ns": 1.0,
        "us": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }[match.group("unit")]
    return float(match.group("value")) * scale


def ns_to_cycles(time_ns: float | None, ghz: float | None, fallback: int) -> int:
    if time_ns is None or ghz is None or ghz <= 0:
        return fallback
    return int(round(time_ns * ghz))


def build_concorde_config(profile: dict, args: argparse.Namespace) -> dict:
    core = profile.get("core", {})
    cache = profile.get("cache", {})
    memory = profile.get("memory", {})

    cpu_clock_ghz = parse_freq_ghz(core.get("cpu_clock"))
    memory_latency_cycles = ns_to_cycles(parse_time_ns(memory.get("local_mem_delay")), cpu_clock_ghz, fallback=60)

    return {
        "rob": {
            "entries": int(core.get("rob_entries", 128)),
            "window_size": int(args.window_size),
        },
        "pipeline": {
            "fetch_width": int(core.get("fetch_width", 4)),
            "decode_width": int(core.get("decode_width", 4)),
            "rename_width": int(core.get("rename_width", 4)),
            "commit_width": int(core.get("commit_width", 4)),
            "issue_widths": {
                "alu": int(args.issue_width_alu),
                "fp": int(args.issue_width_fp),
                "ls": int(args.issue_width_ls),
            },
        },
        "load_store_pipes": {
            "load_store_pipes": int(args.load_store_pipes),
            "load_only_pipes": int(args.load_only_pipes),
        },
        "icache": {
            "size_bytes": int(parse_size_bytes(cache.get("l1i", {}).get("size")) or 64 * 1024),
            "line_size": int(cache.get("cacheline_bytes", 64)),
            "max_fills": int(cache.get("l1i", {}).get("mshrs", 8)),
            "fill_latency": int(args.icache_fill_latency),
            "fetch_width": int(core.get("fetch_width", 4)),
        },
        "fetch_buffer": {
            "entries": int(args.fetch_buffer_entries),
        },
        "cache_hierarchy": {
            "line_size": int(cache.get("cacheline_bytes", 64)),
            "l1": {
                "size_bytes": int(parse_size_bytes(cache.get("l1d", {}).get("size")) or 64 * 1024),
                "associativity": int(cache.get("l1d", {}).get("assoc", 4)),
                "hit_latency": int(cache.get("l1d", {}).get("data_latency_cycles", 1)),
            },
            "l2": {
                "size_bytes": int(parse_size_bytes(cache.get("l2", {}).get("size")) or 512 * 1024),
                "associativity": int(cache.get("l2", {}).get("assoc", 8)),
                "hit_latency": int(cache.get("l2", {}).get("data_latency_cycles", 10)),
            },
            "l3": {
                "size_bytes": int(parse_size_bytes(cache.get("l3_per_die", {}).get("size")) or 32_000_000),
                "associativity": int(cache.get("l3_per_die", {}).get("assoc", 16)),
                "hit_latency": int(cache.get("l3_per_die", {}).get("data_latency_cycles", 20)),
            },
            "memory": {
                "latency": int(memory_latency_cycles),
            },
        },
        "shared_llc": {
            "enabled": False,
            "size_bytes": int(parse_size_bytes(cache.get("l3_per_die", {}).get("size")) or 32_000_000),
            "associativity": int(cache.get("l3_per_die", {}).get("assoc", 16)),
            "line_size": int(cache.get("cacheline_bytes", 64)),
            "hit_latency": int(cache.get("l3_per_die", {}).get("data_latency_cycles", 20)),
            "num_banks": int(args.shared_llc_banks),
            "mshr_entries": int(cache.get("l3_per_die", {}).get("mshrs", 128)),
            "memory": {
                "latency": int(memory_latency_cycles),
                "bandwidth_gbps": float(args.memory_bandwidth_gbps),
                "num_channels": int(args.memory_channels),
            },
        },
        "branch_prediction": {
            "simple": {
                "misprediction_rate": float(args.simple_misprediction_rate),
                "seed": 1,
            },
            "tage": {
                "num_tables": 8,
                "table_size": 2048,
                "tag_bits": 10,
                "ghr_bits": 200,
                "base_size": 4096,
                "counter_bits": 3,
                "usefulness_bits": 2,
                "seed": 1,
            },
        },
        "analysis": {
            "top_n": 50,
            "cdf": {
                "quantile_step": 0.02,
                "tail_quantile": 0.9,
                "separate_figs": False,
                "output": {
                    "png_dpi": 200,
                    "dir": "./result",
                },
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Concorde config from the ORT hardware profile.")
    parser.add_argument(
        "--hardware-profile",
        default="/data/qc/dlrm/ORT/model/hardware_profiles/kunpeng920_gem5.yaml",
        help="Path to the ORT hardware profile YAML",
    )
    parser.add_argument(
        "--output",
        default="/data/qc/dlrm/ORT/concorde/config/kunpeng920_gem5.yaml",
        help="Output path for the Concorde config YAML",
    )
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--issue-width-alu", type=int, default=3)
    parser.add_argument("--issue-width-fp", type=int, default=2)
    parser.add_argument("--issue-width-ls", type=int, default=2)
    parser.add_argument("--load-store-pipes", type=int, default=2)
    parser.add_argument("--load-only-pipes", type=int, default=10)
    parser.add_argument("--fetch-buffer-entries", type=int, default=64)
    parser.add_argument("--icache-fill-latency", type=int, default=40)
    parser.add_argument("--shared-llc-banks", type=int, default=16)
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=37.0)
    parser.add_argument("--memory-channels", type=int, default=8)
    parser.add_argument("--simple-misprediction-rate", type=float, default=0.05)
    args = parser.parse_args()

    profile_path = Path(args.hardware_profile)
    with profile_path.open("r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}

    config = build_concorde_config(profile, args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)

    print(out_path)


if __name__ == "__main__":
    main()
