#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYZER = ROOT / "ORT" / "concorde" / "run_concorde_trace_analysis.py"


def discover_ops(trace_root: Path, combo_filter: str, max_combos: int, max_ops: int) -> list[tuple[str, Path]]:
    combo_dirs = sorted(path for path in trace_root.iterdir() if path.is_dir())
    if combo_filter:
        combo_dirs = [path for path in combo_dirs if combo_filter in path.name]
    if max_combos > 0:
        combo_dirs = combo_dirs[:max_combos]

    discovered: list[tuple[str, Path]] = []
    for combo_dir in combo_dirs:
        op_dirs = sorted(path for path in combo_dir.iterdir() if path.is_dir() and path.name[:1].isdigit())
        if max_ops > 0:
            op_dirs = op_dirs[:max_ops]
        for op_dir in op_dirs:
            discovered.append((combo_dir.name, op_dir))
    return discovered


def run_one(
    analyzer: Path,
    config_path: Path,
    hardware_name: str,
    output_root: Path,
    trace_cache_root: Path | None,
    analysis_backend: str,
    native_analyzer: Path,
    analysis_workers: int,
    combo: str,
    op_dir: Path,
    emit_plots: bool,
    shared_llc: bool,
    materialize_view_log: bool,
    trace_backend: str,
    binary_streamer: Path,
    max_instructions: int,
    resume: bool,
) -> dict[str, str]:
    op_name = op_dir.name
    out_dir = output_root / combo / op_name
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "run.log"
    compact_trace_cache = None
    if trace_cache_root is not None:
        suffix = ".full.bin" if max_instructions <= 0 else f".max{max_instructions}.bin"
        compact_trace_cache = trace_cache_root / combo / f"{op_name}{suffix}"

    if resume and summary_path.exists():
        return {
            "combo": combo,
            "op_name": op_name,
            "status": "skipped",
            "output_dir": str(out_dir),
            "seconds": "0.0",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(analyzer),
        "--trace-dir",
        str(op_dir),
        "--config",
        str(config_path),
        "--output-dir",
        str(out_dir),
        "--hardware-name",
        hardware_name,
        "--analysis-backend",
        analysis_backend,
        "--native-analyzer",
        str(native_analyzer),
    ]
    if analysis_workers > 0:
        cmd.extend(["--analysis-workers", str(analysis_workers)])
    if emit_plots:
        cmd.append("--emit-plots")
    if shared_llc:
        cmd.append("--shared-llc")
    if materialize_view_log:
        cmd.append("--materialize-view-log")
    if trace_backend != "view":
        cmd.extend(["--trace-backend", trace_backend, "--binary-streamer", str(binary_streamer)])
    if compact_trace_cache is not None:
        cmd.extend(["--compact-trace-cache", str(compact_trace_cache)])
    if max_instructions > 0:
        cmd.extend(["--max-instructions", str(max_instructions)])

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start

    return {
        "combo": combo,
        "op_name": op_name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": str(proc.returncode),
        "output_dir": str(out_dir),
        "log_path": str(log_path),
        "seconds": f"{elapsed:.3f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Concorde analysis over a sweep directory of per-op traces.")
    parser.add_argument(
        "--trace-root",
        required=True,
        help="Root directory like ORT/dynamorio_tracing/drrio_traces_sweep",
    )
    parser.add_argument(
        "--config",
        default="/data/qc/dlrm/ORT/concorde/config/kunpeng920_gem5.yaml",
        help="Concorde config YAML",
    )
    parser.add_argument(
        "--output-root",
        default="/data/qc/dlrm/ORT/concorde/artifacts",
        help="Root directory for per-op Concorde artifacts",
    )
    parser.add_argument("--hardware-name", default="kunpeng920_gem5")
    parser.add_argument("--combo-filter", default="")
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--max-ops", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip ops whose summary.json already exists")
    parser.add_argument("--emit-plots", action="store_true")
    parser.add_argument("--shared-llc", action="store_true")
    parser.add_argument("--materialize-view-log", action="store_true")
    parser.add_argument(
        "--trace-backend",
        choices=("view", "binary"),
        default="view",
        help="Trace reader backend for each per-op analysis job",
    )
    parser.add_argument(
        "--binary-streamer",
        default=str(ROOT / "ORT" / "concorde" / "run_binary_trace_backend.sh"),
        help="Path to the direct offline binary streamer executable",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=("auto", "python", "native"),
        default="auto",
        help="Analysis implementation backend for each per-op job",
    )
    parser.add_argument(
        "--native-analyzer",
        default=str(ROOT / "ORT" / "concorde" / "run_native_concorde_analyzer.sh"),
        help="Path to the native Concorde analyzer executable wrapper",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=0,
        help="Optional native analyzer worker count for each per-op job; 0 lets the backend choose.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional sweep summary CSV; defaults to <output-root>/sweep_summary.csv",
    )
    parser.add_argument(
        "--trace-cache-root",
        default="",
        help="Optional shared root for reusable compact binary trace caches",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        default=0,
        help="Optional hard cap on parsed instructions per operator for fast-budget runs",
    )
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    trace_cache_root = Path(args.trace_cache_root).resolve() if args.trace_cache_root else None
    analyzer = DEFAULT_ANALYZER.resolve()
    binary_streamer = Path(args.binary_streamer).resolve()
    native_analyzer = Path(args.native_analyzer).resolve()
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else output_root / "sweep_summary.csv"

    ops = discover_ops(trace_root, args.combo_filter, args.max_combos, args.max_ops)
    if not ops:
        raise FileNotFoundError(f"No operator trace directories found under {trace_root}")

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(
                run_one,
                analyzer,
                config_path,
                args.hardware_name,
                output_root,
                trace_cache_root,
                args.analysis_backend,
                native_analyzer,
                args.analysis_workers,
                combo,
                op_dir,
                args.emit_plots,
                args.shared_llc,
                args.materialize_view_log,
                args.trace_backend,
                binary_streamer,
                args.max_instructions,
                args.resume,
            )
            for combo, op_dir in ops
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row["status"] == "failed":
                print(
                    f"failed: {row['combo']}/{row['op_name']} "
                    f"(returncode={row.get('returncode', 'unknown')}, "
                    f"log_path={row.get('log_path', '')})"
                )
            else:
                print(f"{row['status']}: {row['combo']}/{row['op_name']}")

    rows.sort(key=lambda row: (row["combo"], row["op_name"]))
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row.keys():
            if name not in fieldnames:
                fieldnames.append(name)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(summary_csv)


if __name__ == "__main__":
    main()
