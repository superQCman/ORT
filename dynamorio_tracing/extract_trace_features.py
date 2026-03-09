#!/usr/bin/env python3
"""
extract_trace_features.py
从 drrio_traces_per_op_multithread 下所有算子的 drmemtrace 提取:
  - basic_counts  : total/per-thread 指令数、loads、stores 等
  - drcachesim    : L1I/L1D/L2/LLC 命中率、miss 数
    - reuse_time    : 平均 reuse time 和前 10 个距离桶统计
    - reuse_distance: 平均/中位数/标准差、唯一访问和 cache line 统计
  - opcode_mix    : 指令类别分布（math/branch/load/store/fp/simd 等）

结果在 summary.csv 基础上追加列，输出到 trace_features.csv，
可直接用于神经网络训练。

用法:
    python3 extract_trace_features.py [--ops_dir DIR] [--out CSV] [--jobs N]
                                      [--cache_conf FILE] [--use_physical]
"""

import os
import re
import subprocess
import csv
import json
import argparse
import concurrent.futures
from pathlib import Path

# ──────────────────────────────────────────────
DRRUN = "/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun"
CACHE_CONF = "/data/qc/dlrm/ops_profile/concorde/test/cache_nl.conf"
OPS_DIR = "/data/qc/dlrm/ORT/dynamorio_tracing/drrio_traces_per_op_multithread"
OUT_CSV = "/data/qc/dlrm/ORT/dynamorio_tracing/trace_features.csv"
# ──────────────────────────────────────────────

# ── 正则 ──────────────────────────────────────
RE_BASIC_TOTAL = re.compile(
    r"Total counts:\s+"
    r"(\d+)\s+total \(fetched\) instructions\s+"
    r"(\d+)\s+total unique \(fetched\) instructions\s+"
    r"\d+\s+total non-fetched instructions\s+"
    r"(\d+)\s+total prefetches\s+"
    r"(\d+)\s+total data loads\s+"
    r"(\d+)\s+total data stores\s+"
    r"\d+\s+total icache flushes\s+"
    r"\d+\s+total dcache flushes\s+"
    r"(\d+)\s+total threads",
    re.DOTALL,
)

# cache sim: 按 cache 名称解析
# 注意：LLC/L2 的命中率行标签是 "Local miss rate" 而非 "Miss rate"
RE_CACHE_BLOCK = re.compile(
    r"([\w]+) \(size=(\d+)[^)]*\) stats:\s*"
    r"Hits:\s+(\d+)\s+"
    r"Misses:\s+(\d+)\s+"
    r"Compulsory misses:\s+(\d+).*?"
    r"(?:Local miss rate|Miss rate):\s+([\d.]+)%",
    re.DOTALL,
)

# opcode_mix 类别行（形如  134020544 :      math）
RE_OPCODE_CAT = re.compile(r"^\s+(\d+)\s+:\s+([\w ]+)$", re.MULTILINE)

# opcode_mix 单条指令行（形如  272765505 :       ldr）
RE_OPCODE_INSTR = re.compile(r"^\s+(\d+)\s+:\s+(\S+)$", re.MULTILINE)

RE_REUSE_TIME_SUMMARY = re.compile(
    r"Total accesses:\s+(\d+)\s+"
    r"Total instructions:\s+(\d+)\s+"
    r"Mean reuse time:\s+([\d.]+)",
    re.DOTALL,
)

RE_REUSE_TIME_BIN = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%$",
    re.MULTILINE,
)

RE_REUSE_DISTANCE_SUMMARY = re.compile(
    r"Total accesses:\s+(\d+)\s+"
    r"Instruction accesses:\s+(\d+)\s+"
    r"Data accesses:\s+(\d+)\s+"
    r"Unique accesses:\s+(\d+)\s+"
    r"Unique cache lines accessed:\s+(\d+)\s+"
    r"Distance limit:\s+(\d+)\s+"
    r"Pruned addresses:\s+(\d+)\s+"
    r"Pruned address hits:\s+(\d+)\s+"
    r"\s*Reuse distance mean:\s+([\d.]+)\s+"
    r"Reuse distance median:\s+([\d.]+)\s+"
    r"Reuse distance standard deviation:\s+([\d.]+)",
    re.DOTALL,
)

# ── 辅助 ──────────────────────────────────────

def find_drmem_dir(op_path: Path):
    """找到算子目录下的 drmemtrace.*.dir"""
    dirs = sorted(op_path.glob("drmemtrace.*.dir"))
    if dirs:
        return dirs[0]
    # 也搜索一层子目录（有些 trace 被放在子目录下）
    for child in op_path.iterdir():
        if child.is_dir():
            dirs = sorted(child.glob("drmemtrace.*.dir"))
            if dirs:
                return dirs[0]
    return None


def run_tool(args, timeout=300):
    """运行 drrun 工具，返回 stdout+stderr 字符串"""
    try:
        r = subprocess.run(
            args,
            capture_output=True, text=True, timeout=timeout
        )
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"
    except Exception as e:
        return f"__ERROR__: {e}"


def build_drrun_args(tool_name: str, drmem_dir: Path, use_physical: bool,
                     extra_args=None):
    args = [DRRUN, "-t", "drmemtrace", "-indir", str(drmem_dir)]
    if use_physical:
        args.append("-use_physical")
    args.extend(["-tool", tool_name])
    if extra_args:
        args.extend(extra_args)
    return args


def build_drcachesim_args(drmem_dir: Path, cache_conf: str, use_physical: bool):
    args = [DRRUN, "-t", "drcachesim", "-indir", str(drmem_dir)]
    if use_physical:
        args.append("-use_physical")
    args.extend(["-config_file", cache_conf])
    return args


# ── basic_counts 解析 ─────────────────────────

def parse_basic_counts(text: str) -> dict:
    m = RE_BASIC_TOTAL.search(text)
    if not m:
        return {}
    total_instr, unique_instr, prefetch, loads, stores, threads = m.groups()
    # load/store 比
    total_mem = int(loads) + int(stores)
    return {
        "total_instructions": int(total_instr),
        "unique_instructions": int(unique_instr),
        "total_loads": int(loads),
        "total_stores": int(stores),
        "total_mem_ops": total_mem,
        "load_store_ratio": round(int(loads) / max(int(stores), 1), 4),
        "mem_intensity": round(total_mem / max(int(total_instr), 1), 6),
        "total_prefetches": int(prefetch),
        "num_threads": int(threads),
    }


# ── drcachesim 解析 ───────────────────────────

def parse_cache_sim(text: str) -> dict:
    result = {}
    for m in RE_CACHE_BLOCK.finditer(text):
        name, size, hits, misses, comp_misses, miss_rate = m.groups()
        name_lower = name.lower()
        # 只关心 L1D、L1I、L2/P0L2(所谓私有L2)、LLC
        prefix = ""
        if re.match(r"[Pp](\d)[Ll]1[Dd]$", name):
            core = re.match(r"[Pp](\d)", name).group(1)
            prefix = f"core{core}_L1D"
        elif re.match(r"[Pp](\d)[Ll]1[Ii]$", name):
            core = re.match(r"[Pp](\d)", name).group(1)
            prefix = f"core{core}_L1I"
        elif re.match(r"[Pp](\d)[Ll]2$", name):
            core = re.match(r"[Pp](\d)", name).group(1)
            prefix = f"core{core}_L2"
        elif name.upper() == "LLC":
            prefix = "LLC"
        else:
            continue
        result[f"{prefix}_hits"] = int(hits)
        result[f"{prefix}_misses"] = int(misses)
        result[f"{prefix}_compulsory_misses"] = int(comp_misses)
        result[f"{prefix}_miss_rate_pct"] = float(miss_rate)

    # 汇总所有 core 的 L1D / L1I / L2
    for level in ("L1D", "L1I", "L2"):
        hits_list = [v for k, v in result.items() if k.endswith(f"{level}_hits")]
        misses_list = [v for k, v in result.items() if k.endswith(f"{level}_misses")]
        if hits_list:
            total_h = sum(hits_list)
            total_m = sum(misses_list)
            result[f"total_{level}_hits"] = total_h
            result[f"total_{level}_misses"] = total_m
            result[f"total_{level}_miss_rate_pct"] = round(
                total_m / max(total_h + total_m, 1) * 100, 4
            )
    return result


# ── reuse_time / reuse_distance 解析 ─────────────────

def parse_reuse_time(text: str) -> dict:
    result = {}
    agg_text = text.split("==================================================", 1)[0]
    m = RE_REUSE_TIME_SUMMARY.search(agg_text)
    if not m:
        return result

    total_accesses, total_instructions, mean_reuse_time = m.groups()
    total_accesses = int(total_accesses)
    total_instructions = int(total_instructions)
    result.update({
        "reuse_time_total_accesses": total_accesses,
        "reuse_time_total_instructions": total_instructions,
        "reuse_time_mean": float(mean_reuse_time),
        "reuse_time_access_per_instruction": round(
            total_accesses / max(total_instructions, 1), 6
        ),
    })

    bin_cumulative_10 = None
    for m in RE_REUSE_TIME_BIN.finditer(agg_text):
        distance, count, percent, cumulative = m.groups()
        distance = int(distance)
        if 1 <= distance <= 10:
            result[f"reuse_time_bin_{distance}_count"] = int(count)
            result[f"reuse_time_bin_{distance}_pct"] = float(percent)
            if distance == 10:
                bin_cumulative_10 = float(cumulative)
    if bin_cumulative_10 is not None:
        result["reuse_time_top10_cumulative_pct"] = bin_cumulative_10
    return result


def parse_reuse_distance(text: str) -> dict:
    result = {}
    agg_text = text.split("==================================================", 1)[0]
    m = RE_REUSE_DISTANCE_SUMMARY.search(agg_text)
    if not m:
        return result

    (
        total_accesses,
        instruction_accesses,
        data_accesses,
        unique_accesses,
        unique_cache_lines,
        distance_limit,
        pruned_addresses,
        pruned_address_hits,
        mean_distance,
        median_distance,
        std_distance,
    ) = m.groups()

    total_accesses = int(total_accesses)
    instruction_accesses = int(instruction_accesses)
    data_accesses = int(data_accesses)
    unique_accesses = int(unique_accesses)
    unique_cache_lines = int(unique_cache_lines)
    pruned_addresses = int(pruned_addresses)
    pruned_address_hits = int(pruned_address_hits)

    result.update({
        "reuse_distance_total_accesses": total_accesses,
        "reuse_distance_instruction_accesses": instruction_accesses,
        "reuse_distance_data_accesses": data_accesses,
        "reuse_distance_unique_accesses": unique_accesses,
        "reuse_distance_unique_cache_lines": unique_cache_lines,
        "reuse_distance_distance_limit": int(distance_limit),
        "reuse_distance_pruned_addresses": pruned_addresses,
        "reuse_distance_pruned_address_hits": pruned_address_hits,
        "reuse_distance_mean": float(mean_distance),
        "reuse_distance_median": float(median_distance),
        "reuse_distance_std": float(std_distance),
        "reuse_distance_unique_access_ratio": round(
            unique_accesses / max(total_accesses, 1), 6
        ),
        "reuse_distance_unique_cache_lines_per_k_accesses": round(
            unique_cache_lines / max(total_accesses, 1) * 1000, 6
        ),
        "reuse_distance_data_access_ratio": round(
            data_accesses / max(total_accesses, 1), 6
        ),
    })
    return result


# ── opcode_mix 解析 ───────────────────────────

CATEGORY_COLS = [
    "math", "branch", "load", "store",
    "fp store simd", "fp load simd", "fp convert",
    "simd", "fp move", "fp math"
]

# 归一化列名（去空格/改斜线）
def cat_col(name: str) -> str:
    return "opc_" + name.strip().replace(" ", "_")


def parse_opcode_mix(text: str) -> dict:
    result = {}
    total_instr = 0
    # 找"sets of categories"行之后的类别统计
    cat_section = re.search(r"\d+ : sets of categories\n(.*)", text, re.DOTALL)
    if cat_section:
        for m in RE_OPCODE_CAT.finditer(cat_section.group(1)):
            cnt, cat = int(m.group(1)), m.group(2).strip()
            col = cat_col(cat)
            result[col] = cnt
            total_instr += cnt
    # 单条指令计数（top opcodes）
    for m in RE_OPCODE_INSTR.finditer(text):
        cnt, instr = int(m.group(1)), m.group(2).strip()
        result[f"instr_{instr}"] = cnt
    # 比例特征
    if total_instr > 0:
        for cat in CATEGORY_COLS:
            col = cat_col(cat)
            if col in result:
                result[f"{col}_ratio"] = round(result[col] / total_instr, 6)
    return result


# ── 单算子完整提取 ────────────────────────────

def extract_one_op(op_dir: Path, cache_conf: str, use_physical: bool = False) -> dict:
    record = {
        "op_name": op_dir.name,
        "op_idx": int(op_dir.name.split("_")[0]),
        "op_type": op_dir.name.split("_")[1] if len(op_dir.name.split("_")) > 1 else "",
    }

    drmem_dir = find_drmem_dir(op_dir)
    if drmem_dir is None:
        record["error"] = "no_drmem_dir"
        return record

    record["drmem_dir"] = str(drmem_dir)

    # 1. basic_counts
    out = run_tool(build_drrun_args("basic_counts", drmem_dir, use_physical))
    if out.startswith("__"):
        record["error"] = f"basic_counts:{out}"
    else:
        record.update(parse_basic_counts(out))

    # 2. drcachesim
    out = run_tool(build_drcachesim_args(drmem_dir, cache_conf, use_physical))
    if out.startswith("__"):
        record["error"] = record.get("error", "") + f"|drcachesim:{out}"
    else:
        record.update(parse_cache_sim(out))

    # 3. opcode_mix
    out = run_tool(build_drrun_args("opcode_mix", drmem_dir, use_physical))
    if out.startswith("__"):
        record["error"] = record.get("error", "") + f"|opcode_mix:{out}"
    else:
        record.update(parse_opcode_mix(out))

    # 4. reuse_time
    out = run_tool(build_drrun_args("reuse_time", drmem_dir, use_physical))
    if out.startswith("__"):
        record["error"] = record.get("error", "") + f"|reuse_time:{out}"
    else:
        record.update(parse_reuse_time(out))

    # 5. reuse_distance
    out = run_tool(build_drrun_args("reuse_distance", drmem_dir, use_physical))
    if out.startswith("__"):
        record["error"] = record.get("error", "") + f"|reuse_distance:{out}"
    else:
        record.update(parse_reuse_distance(out))

    return record


# ── 主流程 ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops_dir", default=OPS_DIR)
    parser.add_argument("--out", default=OUT_CSV)
    parser.add_argument("--jobs", type=int, default=4,
                        help="并行 worker 数（每个 worker 串行跑 3 个 drrun 命令）")
    parser.add_argument("--cache_conf", default=CACHE_CONF)
    parser.add_argument("--use_physical", action="store_true",
                        help="向 drrun/drcachesim 透传 -use_physical，默认关闭")
    args = parser.parse_args()

    ops_root = Path(args.ops_dir)
    op_dirs = sorted(
        [d for d in ops_root.iterdir() if d.is_dir() and re.match(r"^\d+_", d.name)]
    )
    print(f"找到 {len(op_dirs)} 个算子目录")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(extract_one_op, op_dir, args.cache_conf, args.use_physical): op_dir
            for op_dir in op_dirs
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            op_dir = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"op_name": op_dir.name, "error": str(e)}
            records.append(rec)
            done += 1
            print(f"  [{done}/{len(op_dirs)}] {op_dir.name}  "
                  f"instr={rec.get('total_instructions','?')}  "
                  f"LLC_miss={rec.get('LLC_miss_rate_pct','?')}%")

    # 按 op_idx 排序
    records.sort(key=lambda r: r.get("op_idx", 9999))

    # 收集所有列（保证列顺序稳定）
    key_cols = ["op_idx", "op_name", "op_type",
                "total_instructions", "unique_instructions",
                "total_loads", "total_stores", "total_mem_ops",
                "load_store_ratio", "mem_intensity",
                "total_prefetches", "num_threads",
                "reuse_time_total_accesses", "reuse_time_total_instructions",
                "reuse_time_mean", "reuse_time_access_per_instruction",
                "reuse_distance_total_accesses",
                "reuse_distance_instruction_accesses",
                "reuse_distance_data_accesses",
                "reuse_distance_unique_accesses",
                "reuse_distance_unique_cache_lines",
                "reuse_distance_mean", "reuse_distance_median",
                "reuse_distance_std", "reuse_distance_unique_access_ratio",
                "reuse_distance_unique_cache_lines_per_k_accesses",
                "reuse_distance_data_access_ratio"]
    extra_cols = []
    for r in records:
        for k in r:
            if k not in key_cols and k not in extra_cols and k not in ("drmem_dir", "error"):
                extra_cols.append(k)

    # opcode_ratio 类列放最后
    def col_sort_key(c):
        if c.startswith("opc_") and c.endswith("_ratio"):
            return (3, c)
        if c.startswith("opc_"):
            return (2, c)
        if c.startswith("instr_"):
            return (4, c)
        if c.startswith("reuse_time_bin_"):
            return (1, c)
        if c.startswith("reuse_distance_"):
            return (1, c)
        if "miss_rate" in c:
            return (1, c)
        return (0, c)

    extra_cols.sort(key=col_sort_key)
    all_cols = key_cols + extra_cols + ["drmem_dir", "error"]

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"\n写入 {out_path}  ({len(records)} 行, {len(all_cols)} 列)")

    # 打印样例
    if records:
        r0 = records[0]
        print("\n── 第一行样例(部分字段) ──")
        for k in key_cols + ["total_L1D_miss_rate_pct", "LLC_miss_rate_pct",
                              "reuse_time_bin_1_pct",
                              "reuse_time_top10_cumulative_pct",
                              "opc_math_ratio", "opc_branch_ratio",
                              "opc_load_ratio", "opc_store_ratio"]:
            if k in r0:
                print(f"  {k}: {r0[k]}")


if __name__ == "__main__":
    main()
