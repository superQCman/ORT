#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONCORDE_ROOT = ROOT / "ORT" / "concorde"
DEFAULT_DRRUN = Path("/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun")
DEFAULT_BINARY_STREAMER = ROOT / "ORT" / "concorde" / "run_binary_trace_backend.sh"
DEFAULT_NATIVE_ANALYZER = ROOT / "ORT" / "concorde" / "run_native_concorde_analyzer.sh"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ort-concorde")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

if str(CONCORDE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONCORDE_ROOT))

from src.analysis import branch_type_distribution, plot_cdf_bundle  # type: ignore  # noqa: E402
from src.bandwidth import (  # type: ignore  # noqa: E402
    fetch_buffers_throughput,
    icache_fills_resp_times,
    icache_fills_throughput,
    pipes_throughput_bounds,
    static_bandwidth_throughputs,
)
from src.branch_prediction import SimplePredictor, TAGEPredictor, classify_branch_type  # type: ignore  # noqa: E402
from src.config import ArchConfig, init_config  # type: ignore  # noqa: E402
from src.feature_extraction import (  # type: ignore  # noqa: E402
    build_all_series,
    build_ml_input,
    extract_arch_params,
    extract_performance_features,
)
from src.rob_model import rob_throughput_model  # type: ignore  # noqa: E402
from src.trace_parser import parse_trace, parse_trace_with_shared_llc  # type: ignore  # noqa: E402
from trace_stream_parser import (  # noqa: E402
    parse_compact_trace_cache,
    parse_compact_trace_cache_with_shared_llc,
    parse_compact_trace_with_shared_llc_stream,
    parse_compact_trace_stream,
    parse_trace_stream,
    parse_trace_with_shared_llc_stream,
)


COMBO_RE = re.compile(r"(bs\d+_nip\d+)")
OP_RE = re.compile(r"^(?P<op_idx>\d+)_")


def resolve_drmemtrace_dir(path: Path) -> Path:
    if (path / "trace").is_dir() and any((path / "trace").glob("*.trace.zip")):
        return path

    candidates = sorted(path.glob("drmemtrace*.dir"))
    for candidate in candidates:
        if (candidate / "trace").is_dir():
            return candidate

    raise FileNotFoundError(f"Could not locate drmemtrace directory under {path}")


def infer_combo(path: Path) -> str | None:
    for part in path.parts:
        match = COMBO_RE.search(part)
        if match:
            return match.group(1)
    return None


def infer_op_name(path: Path) -> str:
    if path.name.startswith("drmemtrace"):
        return path.parent.name
    return path.name


def infer_op_idx(op_name: str) -> int | None:
    match = OP_RE.match(op_name)
    if not match:
        return None
    return int(match.group("op_idx"))


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def run_view(trace_dir: Path, out_path: Path, drrun_path: Path) -> Path:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(drrun_path),
        "-t",
        "drmemtrace",
        "-tool",
        "view",
        "-indir",
        str(trace_dir),
    ]
    with out_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    return out_path


def tee_lines(lines, tee_file):
    for line in lines:
        tee_file.write(line)
        yield line


def parse_trace_from_view_stream(
    trace_dir: Path,
    drrun_path: Path,
    use_shared_llc: bool,
    view_log_path: Path | None = None,
):
    cmd = [
        str(drrun_path),
        "-t",
        "drmemtrace",
        "-tool",
        "view",
        "-indir",
        str(trace_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    line_source = proc.stdout
    if view_log_path is not None:
        view_log_path.parent.mkdir(parents=True, exist_ok=True)
        with view_log_path.open("w", encoding="utf-8") as tee_f:
            line_source = tee_lines(proc.stdout, tee_f)
            parsed = (
                parse_trace_with_shared_llc_stream(line_source)
                if use_shared_llc
                else parse_trace_stream(line_source)
            )
    else:
        parsed = (
            parse_trace_with_shared_llc_stream(line_source)
            if use_shared_llc
            else parse_trace_stream(line_source)
        )

    return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return parsed


def parse_trace_from_binary_stream(
    trace_dir: Path,
    streamer_path: Path,
    use_shared_llc: bool,
    stream_log_path: Path | None = None,
):
    if not streamer_path.exists():
        raise FileNotFoundError(
            f"Binary trace streamer not found: {streamer_path}. "
            "Build it first with ORT/concorde/build_binary_trace_backend.sh"
        )

    cmd = [
        str(streamer_path),
        "--trace-dir",
        str(trace_dir),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    line_source = proc.stdout
    if stream_log_path is not None:
        stream_log_path.parent.mkdir(parents=True, exist_ok=True)
        with stream_log_path.open("w", encoding="utf-8") as tee_f:
            line_source = tee_lines(proc.stdout, tee_f)
            parsed = (
                parse_compact_trace_with_shared_llc_stream(line_source)
                if use_shared_llc
                else parse_compact_trace_stream(line_source)
            )
    else:
        parsed = (
            parse_compact_trace_with_shared_llc_stream(line_source)
            if use_shared_llc
            else parse_compact_trace_stream(line_source)
        )

    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
    return_code = proc.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd, output=None, stderr=stderr_text)
    return parsed


def materialize_compact_trace_cache(
    trace_dir: Path,
    streamer_path: Path,
    cache_path: Path,
    max_instructions: int = 0,
) -> Path:
    if not streamer_path.exists():
        raise FileNotFoundError(
            f"Binary trace streamer not found: {streamer_path}. "
            "Build it first with ORT/concorde/build_binary_trace_backend.sh"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(streamer_path),
        "--trace-dir",
        str(trace_dir),
        "--output",
        str(cache_path),
        "--output-format",
        "compact-bin",
    ]
    if max_instructions > 0:
        cmd.extend(["--max-instructions", str(max_instructions)])
    subprocess.run(cmd, check=True)
    return cache_path


NATIVE_CONFIG_KEYS = [
    "rob.entries",
    "rob.window_size",
    "pipeline.fetch_width",
    "pipeline.decode_width",
    "pipeline.rename_width",
    "pipeline.commit_width",
    "pipeline.issue_widths.alu",
    "pipeline.issue_widths.fp",
    "pipeline.issue_widths.ls",
    "load_store_pipes.load_store_pipes",
    "load_store_pipes.load_only_pipes",
    "cache_hierarchy.line_size",
    "cache_hierarchy.l1.size_bytes",
    "cache_hierarchy.l1.associativity",
    "cache_hierarchy.l1.hit_latency",
    "cache_hierarchy.l2.size_bytes",
    "cache_hierarchy.l2.associativity",
    "cache_hierarchy.l2.hit_latency",
    "cache_hierarchy.l3.size_bytes",
    "cache_hierarchy.l3.associativity",
    "cache_hierarchy.l3.hit_latency",
    "cache_hierarchy.memory.latency",
    "icache.max_fills",
    "icache.fill_latency",
    "icache.size_bytes",
    "icache.line_size",
    "icache.fetch_width",
    "fetch_buffer.entries",
    "branch_prediction.simple.misprediction_rate",
    "branch_prediction.simple.seed",
    "branch_prediction.tage.num_tables",
    "branch_prediction.tage.table_size",
    "branch_prediction.tage.tag_bits",
    "branch_prediction.tage.ghr_bits",
    "branch_prediction.tage.base_size",
    "branch_prediction.tage.counter_bits",
    "branch_prediction.tage.usefulness_bits",
    "branch_prediction.tage.seed",
]


def write_native_config(path: Path, config: ArchConfig) -> Path:
    lines = []
    for key in NATIVE_CONFIG_KEYS:
        value = config.get(key, None)
        if value is None:
            continue
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_single_thread_results_from_native(native_results: dict[str, Any], config: ArchConfig) -> dict[str, Any]:
    throughput_series = native_results["throughput_series"]
    branch_prediction = native_results["branch_prediction"]
    perf_features = extract_performance_features(throughput_series, branch_prediction["tage"])
    arch_params = extract_arch_params(config)
    ml_input = build_ml_input(perf_features, arch_params)

    return {
        "tid": int(native_results.get("tid", native_results.get("target_tid", -1))),
        "shard_index": int(native_results.get("shard_index", native_results.get("target_shard_index", -1))),
        "input_id": int(native_results.get("input_id", native_results.get("target_input_id", -1))),
        "stream_name": str(native_results.get("stream_name", native_results.get("target_stream_name", ""))),
        "instruction_count": int(native_results["instruction_count"]),
        "window_size": int(native_results["window_size"]),
        "rob": {
            "avg_ipc": float(native_results["rob_avg_ipc"]),
            "thr_chunks": throughput_series.get("ROB.thr_chunks", []),
        },
        "branch_prediction": branch_prediction,
        "throughput_series": throughput_series,
        "performance_features": perf_features,
        "arch_params": arch_params,
        "ml_input": ml_input,
        "timings": native_results.get("timings", {}),
    }


def build_results_from_native(native_results: dict[str, Any], config: ArchConfig) -> dict[str, Any]:
    thread_native_results = native_results.get("thread_results", [])
    if thread_native_results:
        thread_results = [build_single_thread_results_from_native(item, config) for item in thread_native_results]
    else:
        thread_results = [build_single_thread_results_from_native(native_results, config)]

    target_tid = int(native_results.get("target_tid", thread_results[0]["tid"]))
    primary = next((item for item in thread_results if item["tid"] == target_tid), thread_results[0])
    primary_results = dict(primary)
    primary_results["target_tid"] = target_tid
    primary_results["target_shard_index"] = int(
        native_results.get("target_shard_index", primary.get("shard_index", -1))
    )
    primary_results["target_input_id"] = int(
        native_results.get("target_input_id", primary.get("input_id", -1))
    )
    primary_results["target_stream_name"] = str(
        native_results.get("target_stream_name", primary.get("stream_name", ""))
    )
    primary_results["thread_count"] = len(thread_results)
    primary_results["thread_results"] = thread_results
    primary_results["timings"] = native_results.get("timings", {})
    return primary_results


def run_native_analysis(
    analyzer_path: Path,
    trace_dir: Path,
    config: ArchConfig,
    output_dir: Path,
    max_instructions: int,
    analysis_workers: int,
) -> dict[str, Any]:
    if not analyzer_path.exists():
        raise FileNotFoundError(
            f"Native analyzer not found: {analyzer_path}. "
            "Build it first with ORT/concorde/build_binary_trace_backend.sh"
        )

    native_config_path = write_native_config(output_dir / "native_config.env", config)
    native_output_path = output_dir / "native_analysis.json"
    cmd = [
        str(analyzer_path),
        "--trace-dir",
        str(trace_dir),
        "--config",
        str(native_config_path),
        "--output-json",
        str(native_output_path),
    ]
    if max_instructions > 0:
        cmd.extend(["--max-instructions", str(max_instructions)])
    if analysis_workers > 0:
        cmd.extend(["--worker-count", str(analysis_workers)])
    subprocess.run(cmd, check=True)
    return json.loads(native_output_path.read_text(encoding="utf-8"))


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def log_stage(message: str) -> None:
    print(f"[concorde] {message}", file=sys.stderr, flush=True)


def build_named_row(prefix: str, values: list[float], names: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    row = dict(metadata)
    for name, value in zip(names, values):
        row[f"{prefix}{sanitize_name(name)}"] = float(value)
    return row


def compute_branch_mispred_rate_local(
    instrs_sorted,
    predictor,
    only_conditional: bool = True,
    require_label: bool = True,
) -> dict[str, Any]:
    total = 0
    misp = 0
    by_type: dict[str, dict[str, int]] = {}

    for ins in instrs_sorted:
        if not hasattr(ins, "branch_instr_pc"):
            continue

        actual_taken = getattr(ins, "branch_taken", None)
        if require_label and actual_taken is None:
            continue

        br_type = classify_branch_type(ins.mnemonic)
        if only_conditional and br_type != "Conditional Branch":
            continue

        is_misp, pred = predictor.update_and_count(ins.branch_instr_pc, actual_taken)
        total += 1
        by_type.setdefault(br_type, {"total": 0, "misp": 0})
        by_type[br_type]["total"] += 1
        if is_misp:
            misp += 1
            by_type[br_type]["misp"] += 1

    return {
        "total": total,
        "misp": misp,
        "misp_rate": (misp / total) if total > 0 else 0.0,
        "by_type": {
            name: {
                "total": stats["total"],
                "misp": stats["misp"],
                "misp_rate": (stats["misp"] / stats["total"]) if stats["total"] > 0 else 0.0,
            }
            for name, stats in by_type.items()
        },
    }


def analyze_trace(parsed, config: ArchConfig, source_label: str) -> dict[str, Any]:
    timings: dict[str, float] = {}
    analyze_start = time.perf_counter()

    (
        record_types,
        markers,
        mnemonics,
        instruction_classify_arr,
        load_instr_list,
        cache_line_access,
        hit_level_counter,
        instrs_by_tid,
    ) = parsed

    if not instrs_by_tid:
        raise ValueError(f"No instructions parsed from {source_label}")

    step_start = time.perf_counter()
    target_tid = max(instrs_by_tid.items(), key=lambda item: len(item[1]))[0]
    instrs_sorted = instrs_by_tid[target_tid]
    timings["select_target_tid"] = time.perf_counter() - step_start

    rob_entries = int(config.get("rob.entries", 192))
    window_size = int(config.get("rob.window_size", 400))

    step_start = time.perf_counter()
    rob_res = rob_throughput_model(instrs_sorted, ROB=rob_entries, k=window_size)
    timings["rob_model"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    static_thr = static_bandwidth_throughputs(instrs_sorted, k=window_size)
    timings["static_bandwidth"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    pipes_thr = pipes_throughput_bounds(instrs_sorted, k=window_size)
    timings["pipes_bounds"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    ic_thr = icache_fills_throughput(instrs_sorted, k=window_size)
    timings["icache_throughput"] = time.perf_counter() - step_start

    fb_entries = int(config.get("fetch_buffer.entries", 64))
    decode_width = int(config.get("pipeline.decode_width", 4))
    step_start = time.perf_counter()
    ready_time = icache_fills_resp_times(instrs_sorted)
    timings["icache_ready_time"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    fb_res = fetch_buffers_throughput(
        instrs_sorted,
        k=window_size,
        fb_entries=fb_entries,
        decode_width=decode_width,
        ready_time=ready_time,
    )
    timings["fetch_buffer_model"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    br_type_dist = branch_type_distribution(instrs_sorted, k=window_size)
    timings["branch_type_distribution"] = time.perf_counter() - step_start

    simple_cfg = config.get("branch_prediction.simple", {}) or {}
    simple_bp = SimplePredictor(
        p=float(simple_cfg.get("misprediction_rate", 0.05)),
        seed=int(simple_cfg.get("seed", 1)),
    )
    step_start = time.perf_counter()
    br_simple = compute_branch_mispred_rate_local(instrs_sorted, simple_bp, only_conditional=True, require_label=True)
    timings["branch_predictor_simple"] = time.perf_counter() - step_start

    tage_cfg = config.get("branch_prediction.tage", {}) or {}
    tage_bp = TAGEPredictor(
        num_tables=int(tage_cfg.get("num_tables", 8)),
        table_size=int(tage_cfg.get("table_size", 2048)),
        tag_bits=int(tage_cfg.get("tag_bits", 10)),
        ghr_bits=int(tage_cfg.get("ghr_bits", 200)),
        base_size=int(tage_cfg.get("base_size", 4096)),
        ctr_bits=int(tage_cfg.get("counter_bits", 3)),
        u_bits=int(tage_cfg.get("usefulness_bits", 2)),
        seed=int(tage_cfg.get("seed", 1)),
    )
    step_start = time.perf_counter()
    br_tage = compute_branch_mispred_rate_local(instrs_sorted, tage_bp, only_conditional=True, require_label=True)
    timings["branch_predictor_tage"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    all_series = build_all_series(rob_res, static_thr, pipes_thr, ic_thr, fb_res, br_type_dist)
    timings["build_all_series"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    perf_features = extract_performance_features(all_series, br_tage)
    timings["extract_performance_features"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    arch_params = extract_arch_params(config)
    timings["extract_arch_params"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    ml_input = build_ml_input(perf_features, arch_params)
    timings["build_ml_input"] = time.perf_counter() - step_start
    timings["analyze_total"] = time.perf_counter() - analyze_start

    return {
        "config": config.config_data,
        "target_tid": int(target_tid),
        "instruction_count": len(instrs_sorted),
        "window_size": window_size,
        "record_types": dict(record_types),
        "markers": dict(markers),
        "mnemonics": dict(mnemonics),
        "cache_hit_levels": dict(hit_level_counter),
        "load_instruction_count": len(load_instr_list),
        "cache_line_access_count": len(cache_line_access),
        "instruction_classify_arr": json_ready(instruction_classify_arr),
        "rob": {
            "avg_ipc": rob_res["avg_ipc"],
            "thr_chunks": rob_res["thr_chunks"],
        },
        "branch_prediction": {
            "simple": br_simple,
            "tage": br_tage,
            "branch_type_distribution": json_ready(br_type_dist),
        },
        "timings": timings,
        "throughput_series": json_ready(all_series),
        "performance_features": {
            "cdf_vectors": json_ready(perf_features["cdf_vectors"]),
            "branch_misp_rates": perf_features["branch_misp_rates"],
            "feature_vector": json_ready(perf_features["feature_vector"]),
            "feature_names": perf_features["feature_names"],
            "z_dim": perf_features["z_dim"],
        },
        "arch_params": {
            "param_vector": json_ready(arch_params["param_vector"]),
            "param_names": arch_params["param_names"],
            "p_dim": arch_params["p_dim"],
        },
        "ml_input": {
            "input_vector": json_ready(ml_input["input_vector"]),
            "input_names": ml_input["input_names"],
            "z_dim": ml_input["z_dim"],
            "p_dim": ml_input["p_dim"],
            "total_dim": ml_input["total_dim"],
        },
    }


def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run Concorde-style analytical modeling on an ORT per-op offline trace.")
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Path to an ORT per-op directory or a drmemtrace.*.dir directory",
    )
    parser.add_argument(
        "--config",
        default="/data/qc/dlrm/ORT/concorde/config/kunpeng920_gem5.yaml",
        help="Concorde config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated Concorde artifacts; defaults under ORT/concorde/artifacts/<combo>/<op_name>",
    )
    parser.add_argument(
        "--drrun",
        default=str(DEFAULT_DRRUN),
        help="Path to DynamoRIO drrun",
    )
    parser.add_argument(
        "--trace-backend",
        choices=("view", "binary"),
        default="view",
        help="Trace reader backend: use DynamoRIO view or the direct offline binary streamer",
    )
    parser.add_argument(
        "--binary-streamer",
        default=str(DEFAULT_BINARY_STREAMER),
        help="Path to the direct offline binary streamer executable",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=("auto", "python", "native"),
        default="auto",
        help="Analysis implementation backend. 'native' offloads full-trace modeling to C++.",
    )
    parser.add_argument(
        "--native-analyzer",
        default=str(DEFAULT_NATIVE_ANALYZER),
        help="Path to the native Concorde analyzer executable wrapper",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=0,
        help="Optional native analyzer worker count for thread-sharded parallel analysis; 0 lets the backend choose.",
    )
    parser.add_argument(
        "--compact-trace-cache",
        default="",
        help="Optional reusable compact binary trace cache path for the binary backend",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=0,
        help="Optional hard cap on parsed instructions for fast-budget runs",
    )
    parser.add_argument(
        "--shared-llc",
        action="store_true",
        help="Use the shared-LLC parser/model path",
    )
    parser.add_argument(
        "--emit-plots",
        action="store_true",
        help="Emit CDF plots in addition to JSON/CSV outputs",
    )
    parser.add_argument("--hardware-name", default="kunpeng920_gem5")
    parser.add_argument("--combo", default="")
    parser.add_argument("--op-name", default="")
    parser.add_argument("--op-idx", type=int, default=-1)
    parser.add_argument(
        "--view-log",
        default="",
        help="Optional existing drmemtrace -tool view log to parse directly",
    )
    parser.add_argument(
        "--materialize-view-log",
        action="store_true",
        help="Persist trace_view.log under the output directory while parsing; default is streaming parse without the huge text artifact",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Concorde config not found: {config_path}")
    config = init_config(str(config_path))

    input_path = Path(args.trace_dir).resolve()
    drmemtrace_dir = resolve_drmemtrace_dir(input_path)

    combo = args.combo or infer_combo(drmemtrace_dir) or "unknown_combo"
    op_name = args.op_name or infer_op_name(input_path)
    op_idx = args.op_idx if args.op_idx >= 0 else infer_op_idx(op_name)

    default_output_dir = ROOT / "ORT" / "concorde" / "artifacts" / combo / op_name
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_trace_cache = Path(args.compact_trace_cache).resolve() if args.compact_trace_cache else None
    native_analyzer = Path(args.native_analyzer).resolve()
    use_native_analysis = False
    if args.analysis_backend == "native":
        use_native_analysis = True
    elif args.analysis_backend == "auto":
        use_native_analysis = (
            args.trace_backend == "binary"
            and not args.shared_llc
            and not args.view_log
            and not args.materialize_view_log
            and compact_trace_cache is None
        )

    stage_timings: dict[str, float] = {}
    view_log: Path | None = None
    log_stage(
        f"starting trace analysis backend={args.trace_backend} trace_dir={drmemtrace_dir} output_dir={output_dir}"
    )
    if use_native_analysis:
        if args.trace_backend != "binary":
            raise ValueError("Native analysis backend currently requires --trace-backend binary")
        if args.shared_llc:
            raise ValueError("Native analysis backend does not support --shared-llc yet")
        if args.view_log:
            raise ValueError("Native analysis backend does not support --view-log")
        log_stage(f"running native concorde analyzer {native_analyzer}")
        native_start = time.perf_counter()
        native_results = run_native_analysis(
            analyzer_path=native_analyzer,
            trace_dir=drmemtrace_dir,
            config=config,
            output_dir=output_dir,
            max_instructions=args.max_instructions,
            analysis_workers=args.analysis_workers,
        )
        stage_timings["trace_parse_total"] = float(native_results.get("timings", {}).get("trace_parse_total", 0.0))
        stage_timings["native_wrapper_total"] = time.perf_counter() - native_start
        results = build_results_from_native(native_results, config)
    elif args.view_log:
        log_stage(f"parsing existing view log {args.view_log}")
        parse_start = time.perf_counter()
        view_log = Path(args.view_log).resolve()
        parsed = parse_trace_with_shared_llc(str(view_log)) if args.shared_llc else parse_trace(str(view_log))
        stage_timings["trace_parse_total"] = time.perf_counter() - parse_start
    else:
        if args.materialize_view_log:
            view_log = output_dir / ("trace_view.log" if args.trace_backend == "view" else "trace_pseudoview.log")
        if args.trace_backend == "binary":
            parse_start = time.perf_counter()
            if compact_trace_cache is not None:
                if compact_trace_cache.exists() and compact_trace_cache.stat().st_size > len(b"CTRCBIN1"):
                    log_stage(f"reusing compact trace cache {compact_trace_cache}")
                else:
                    log_stage(f"materializing compact trace cache {compact_trace_cache}")
                    cache_start = time.perf_counter()
                    materialize_compact_trace_cache(
                        trace_dir=drmemtrace_dir,
                        streamer_path=Path(args.binary_streamer).resolve(),
                        cache_path=compact_trace_cache,
                        max_instructions=args.max_instructions,
                    )
                    stage_timings["trace_cache_materialize_total"] = time.perf_counter() - cache_start
                log_stage(f"parsing compact trace cache {compact_trace_cache}")
                parsed = (
                    parse_compact_trace_cache_with_shared_llc(compact_trace_cache, max_instructions=args.max_instructions)
                    if args.shared_llc
                    else parse_compact_trace_cache(compact_trace_cache, max_instructions=args.max_instructions)
                )
            else:
                log_stage(f"streaming binary trace from {drmemtrace_dir}")
                parsed = parse_trace_from_binary_stream(
                    trace_dir=drmemtrace_dir,
                    streamer_path=Path(args.binary_streamer).resolve(),
                    use_shared_llc=args.shared_llc,
                    stream_log_path=view_log,
                )
            stage_timings["trace_parse_total"] = time.perf_counter() - parse_start
        else:
            log_stage(f"streaming drmemtrace view from {drmemtrace_dir}")
            parse_start = time.perf_counter()
            parsed = parse_trace_from_view_stream(
                trace_dir=drmemtrace_dir,
                drrun_path=Path(args.drrun),
                use_shared_llc=args.shared_llc,
                view_log_path=view_log,
            )
            stage_timings["trace_parse_total"] = time.perf_counter() - parse_start
        log_stage("running analytical models")
        analyze_start = time.perf_counter()
        results = analyze_trace(
            parsed=parsed,
            config=config,
            source_label=str(view_log) if view_log is not None else str(drmemtrace_dir),
        )
        stage_timings["analyze_trace_wrapper"] = time.perf_counter() - analyze_start
    log_stage(f"trace parse finished in {results['timings'].get('trace_parse_total', stage_timings.get('trace_parse_total', 0.0)):.3f}s")

    if use_native_analysis:
        stage_timings["analyze_trace_wrapper"] = results["timings"].get("native_total", 0.0)
    stage_timings["wall_total"] = time.perf_counter() - wall_start
    results["timings"].update(stage_timings)
    log_stage(
        "analysis finished "
        f"parse={results['timings']['trace_parse_total']:.3f}s "
        f"analyze={results['timings']['analyze_trace_wrapper']:.3f}s "
        f"wall={results['timings']['wall_total']:.3f}s"
    )

    metadata = {
        "hardware_name": args.hardware_name,
        "combo": combo,
        "op_name": op_name,
        "op_idx": op_idx,
        "trace_dir": str(drmemtrace_dir),
        "view_log": str(view_log) if view_log is not None else "",
        "compact_trace_cache": str(compact_trace_cache) if compact_trace_cache is not None else "",
        "trace_backend": args.trace_backend,
        "analysis_backend": args.analysis_backend if not use_native_analysis else "native",
        "config_path": str(config_path),
        "shared_llc": bool(args.shared_llc),
        "max_instructions": int(args.max_instructions),
        "target_tid": results["target_tid"],
        "target_shard_index": results.get("target_shard_index", -1),
        "target_input_id": results.get("target_input_id", -1),
        "target_stream_name": results.get("target_stream_name", ""),
        "instruction_count": results["instruction_count"],
        "window_size": results["window_size"],
        "thread_count": int(results.get("thread_count", 1)),
        "timing_keys": sorted(results["timings"].keys()),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "throughput_series.json").write_text(
        json.dumps(results["throughput_series"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "cdf_vectors.json").write_text(
        json.dumps(json_ready(results["performance_features"]["cdf_vectors"]), indent=2),
        encoding="utf-8",
    )
    (output_dir / "branch_prediction.json").write_text(
        json.dumps(results["branch_prediction"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "thread_results.json").write_text(
        json.dumps(json_ready(results.get("thread_results", [])), indent=2),
        encoding="utf-8",
    )

    perf_row = build_named_row(
        prefix="concorde_",
        values=results["performance_features"]["feature_vector"],
        names=results["performance_features"]["feature_names"],
        metadata=metadata,
    )
    ml_row = build_named_row(
        prefix="concorde_",
        values=results["ml_input"]["input_vector"],
        names=results["ml_input"]["input_names"],
        metadata=metadata,
    )
    write_single_row_csv(output_dir / "performance_distribution_row.csv", perf_row)
    write_single_row_csv(output_dir / "ml_input_row.csv", ml_row)
    thread_summary_rows = []
    for thread_result in results.get("thread_results", []):
        thread_summary_rows.append(
            {
                **metadata,
                "thread_tid": thread_result["tid"],
                "thread_shard_index": thread_result.get("shard_index", -1),
                "thread_input_id": thread_result.get("input_id", -1),
                "thread_stream_name": thread_result.get("stream_name", ""),
                "thread_instruction_count": thread_result["instruction_count"],
                "thread_rob_avg_ipc": thread_result["rob"]["avg_ipc"],
                "thread_tage_misp_rate": thread_result["branch_prediction"]["tage"]["misp_rate"],
                "thread_z_dim": thread_result["performance_features"]["z_dim"],
                "thread_p_dim": thread_result["arch_params"]["p_dim"],
                "thread_ml_input_dim": thread_result["ml_input"]["total_dim"],
            }
        )
    if thread_summary_rows:
        with (output_dir / "thread_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(thread_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(thread_summary_rows)

    if args.emit_plots:
        plot_cdf_bundle(
            series_dict=results["throughput_series"],
            title=f"Concorde Throughput CDF: {op_name}",
            out_path_png=str(output_dir / "throughput_cdf.png"),
            out_path_pdf=str(output_dir / "throughput_cdf.pdf"),
            drop_inf=True,
            xlim=None,
            show_tail_zoom=True,
            tail_quantile=0.9,
            separate_figs=False,
        )

    summary = {
        **metadata,
        "z_dim": results["performance_features"]["z_dim"],
        "p_dim": results["arch_params"]["p_dim"],
        "ml_input_dim": results["ml_input"]["total_dim"],
        "rob_avg_ipc": results["rob"]["avg_ipc"],
        "tage_misp_rate": results["branch_prediction"]["tage"]["misp_rate"],
        "thread_results": [
            {
                "tid": thread_result["tid"],
                "shard_index": thread_result.get("shard_index", -1),
                "input_id": thread_result.get("input_id", -1),
                "stream_name": thread_result.get("stream_name", ""),
                "instruction_count": thread_result["instruction_count"],
                "rob_avg_ipc": thread_result["rob"]["avg_ipc"],
                "tage_misp_rate": thread_result["branch_prediction"]["tage"]["misp_rate"],
            }
            for thread_result in results.get("thread_results", [])
        ],
        "timings": results["timings"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_stage("artifacts written")
    print(json.dumps({"output_dir": str(output_dir), "timings": results["timings"]}, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
