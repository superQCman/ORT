from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

from npu_sep_common import dump_json, ensure_dir, load_json


PROJECT_DIR = Path(__file__).resolve().parent
ASCEND_ENV_SH = Path("/data/qc/Ascend/ascend-toolkit/set_env.sh")
ORT_ENV_PYTHON = Path("/data/qc/anaconda3/envs/ort/bin/python")
ORT_ENV_LIB = Path("/data/qc/anaconda3/envs/ort/lib")


DEVICE_ROW_RE = re.compile(r"^\|\s*(?P<npu_id>\d+)\s+(?P<name>[A-Za-z0-9_.-]+)\s+\|")
CHIP_COUNT_RE = re.compile(r"^\s*Chip Count\s*:\s*(?P<value>\d+)\s*$")
AICORE_COUNT_RE = re.compile(r"^\s*Aicore Count\s*:\s*(?P<value>\d+)\s*$")
AICORE_FREQ_RE = re.compile(r"^\s*Aicore Freq\(MHZ\)\s*:\s*(?P<value>\d+)\s*$")
AICORE_CUR_FREQ_RE = re.compile(r"^\s*Aicore curFreq\(MHZ\)\s*:\s*(?P<value>\d+)\s*$")
DEFAULT_CUBE_COUNT = 20
DEFAULT_VECTOR_COUNT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Ascend 910B3 hardware inputs for NPU separation modeling.")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive hardware_profile_910b3.json.")
    parser.add_argument("--device-id", type=int, default=0, help="NPU device id to probe.")
    return parser.parse_args()


def run_command(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_npu_smi_info() -> dict[str, Any]:
    code, stdout, stderr = run_command(["npu-smi", "info"])
    diagnostics: list[str] = []
    if code != 0:
        diagnostics.append(f"npu-smi info failed: {stderr.strip() or stdout.strip()}")
        return {
            "device_ids": [],
            "device_name": None,
            "device_count": None,
            "diagnostics": diagnostics,
        }

    device_rows: list[tuple[int, str]] = []
    for line in stdout.splitlines():
        match = DEVICE_ROW_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if ":" in name:
            continue
        device_rows.append((int(match.group("npu_id")), name))

    device_name = device_rows[0][1] if device_rows else None
    diagnostics.append(f"parsed {len(device_rows)} device rows from npu-smi info")
    return {
        "device_ids": [item[0] for item in device_rows],
        "device_name": device_name,
        "device_count": len(device_rows) or None,
        "diagnostics": diagnostics,
        "raw_output": stdout,
    }


def parse_npu_smi_board(device_id: int) -> dict[str, Any]:
    code, stdout, stderr = run_command(["npu-smi", "info", "-t", "board", "-i", str(device_id)])
    if code != 0:
        return {
            "chip_count": None,
            "diagnostics": [f"npu-smi board probe failed: {stderr.strip() or stdout.strip()}"],
        }

    chip_count = None
    for line in stdout.splitlines():
        chip_match = CHIP_COUNT_RE.match(line)
        if chip_match is not None:
            chip_count = int(chip_match.group("value"))
    return {
        "chip_count": chip_count,
        "diagnostics": [],
        "raw_output": stdout,
    }


def parse_npu_smi_common(device_id: int) -> dict[str, Any]:
    code, stdout, stderr = run_command(["npu-smi", "info", "-t", "common", "-i", str(device_id)])
    if code != 0:
        return {
            "ai_core_count": None,
            "frequency_mhz": None,
            "current_frequency_mhz": None,
            "diagnostics": [f"npu-smi common probe failed: {stderr.strip() or stdout.strip()}"],
        }

    ai_core_count = None
    frequency_mhz = None
    current_frequency_mhz = None
    for line in stdout.splitlines():
        count_match = AICORE_COUNT_RE.match(line)
        if count_match is not None:
            ai_core_count = int(count_match.group("value"))
            continue
        freq_match = AICORE_FREQ_RE.match(line)
        if freq_match is not None:
            frequency_mhz = int(freq_match.group("value"))
            continue
        cur_freq_match = AICORE_CUR_FREQ_RE.match(line)
        if cur_freq_match is not None:
            current_frequency_mhz = int(cur_freq_match.group("value"))
            continue

    return {
        "ai_core_count": ai_core_count,
        "frequency_mhz": frequency_mhz,
        "current_frequency_mhz": current_frequency_mhz,
        "diagnostics": [],
        "raw_output": stdout,
    }


def probe_ascend_dmi() -> dict[str, Any]:
    path = shutil.which("ascend-dmi")
    if path is None:
        return {
            "available": False,
            "path": None,
            "diagnostics": ["ascend-dmi not found"],
        }
    code, stdout, stderr = run_command([path, "-h"])
    return {
        "available": code == 0,
        "path": path,
        "diagnostics": [stderr.strip() or stdout.strip()] if code != 0 else [],
    }


def locate_microbench_python() -> Path | None:
    candidates: list[Path] = []
    env_python = os.environ.get("NPU_SEP_MODELING_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(ORT_ENV_PYTHON)
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def locate_latest_msprof_profile(msprof_root: Path) -> Path | None:
    if not msprof_root.exists():
        return None
    candidates = []
    for path in msprof_root.glob("PROF_*"):
        if not path.is_dir():
            continue
        output_dir = path / "mindstudio_profiler_output"
        if not any(output_dir.glob("task_time_*.csv")) or not any(output_dir.glob("op_summary_*.csv")):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def first_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def case_lookup(microbench_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = microbench_summary.get("cases") or []
    lookup: dict[str, dict[str, Any]] = {}
    for case in cases:
        if isinstance(case, dict) and case.get("name"):
            lookup[str(case["name"])] = case
    return lookup


def parse_shape_signature(signature: str | None) -> tuple[tuple[int, ...], ...]:
    if signature is None:
        return ()
    text = signature.strip().strip('"')
    if not text:
        return ()
    shapes: list[tuple[int, ...]] = []
    for part in text.split(";"):
        part = part.strip().strip('"')
        if not part:
            shapes.append(())
            continue
        dims = tuple(int(dim.strip()) for dim in part.split(",") if dim.strip())
        shapes.append(dims)
    return tuple(shapes)


def case_shape_signature(case: dict[str, Any]) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    input_shapes = tuple(tuple(int(dim) for dim in shape) for shape in case.get("input_shapes") or [])
    output_shape = tuple(int(dim) for dim in (case.get("output_shape") or []))
    return input_shapes, output_shape


def summarize_msprof_profile(msprof_root: Path, microbench_summary: dict[str, Any]) -> dict[str, Any]:
    profile_dir = locate_latest_msprof_profile(msprof_root)
    if profile_dir is None:
        return {
            "available": False,
            "diagnostics": [f"msprof output not found under {msprof_root}"],
        }

    output_dir = profile_dir / "mindstudio_profiler_output"
    task_time_candidates = sorted(output_dir.glob("task_time_*.csv"))
    op_summary_candidates = sorted(output_dir.glob("op_summary_*.csv"))
    if not task_time_candidates or not op_summary_candidates:
        return {
            "available": False,
            "profile_dir": str(profile_dir),
            "diagnostics": [f"msprof CSVs missing under {output_dir}"],
        }

    task_rows = read_csv_dicts(task_time_candidates[-1])
    op_rows = read_csv_dicts(op_summary_candidates[-1])
    cases = case_lookup(microbench_summary)

    matmul_case = cases.get("cube_matmul")
    add_case = cases.get("vector_add")
    relu_case = cases.get("vector_relu")
    transfer_h2d_case = cases.get("transfer_h2d")
    transfer_d2h_case = cases.get("transfer_d2h")

    def matched_op_rows(case: dict[str, Any], op_types: set[str]) -> list[dict[str, str]]:
        input_sig, output_sig = case_shape_signature(case)
        rows: list[dict[str, str]] = []
        for row in op_rows:
            if str(row.get("OP Type") or "").strip() not in op_types:
                continue
            if parse_shape_signature(row.get("Input Shapes")) != input_sig:
                continue
            if parse_shape_signature(row.get("Output Shapes")) != output_sig:
                continue
            rows.append(row)
        return rows

    def op_throughput(case: dict[str, Any], op_types: set[str], time_field: str) -> tuple[float | None, int]:
        rows = matched_op_rows(case, op_types)
        total_time = sum(first_float(row.get(time_field)) or 0.0 for row in rows)
        total_count = len(rows)
        if total_time <= 0.0 or total_count <= 0:
            return None, total_count
        total_units = float(case.get("units_per_run") or 0.0) * float(total_count)
        return float(total_units / total_time / 1_000.0), total_count

    cube_peak = None
    cube_count = 0
    if matmul_case is not None:
        cube_peak, cube_count = op_throughput(matmul_case, {"MatMulV3"}, "aicore_time(us)")

    vector_add_peak = None
    vector_add_count = 0
    if add_case is not None:
        vector_add_peak, vector_add_count = op_throughput(add_case, {"Add"}, "aiv_time(us)")

    vector_relu_peak = None
    vector_relu_count = 0
    if relu_case is not None:
        vector_relu_peak, vector_relu_count = op_throughput(relu_case, {"Relu"}, "aiv_time(us)")

    vector_candidates = [value for value in (vector_add_peak, vector_relu_peak) if value is not None]
    vector_peak = median(vector_candidates) if vector_candidates else None

    def transfer_bandwidth(case: dict[str, Any], direction: str) -> tuple[float | None, int]:
        if case is None:
            return None, 0
        bytes_per_run = float(case.get("input_bytes_per_run") or 0.0) if direction == "h2d" else float(case.get("output_bytes_per_run") or 0.0)
        if bytes_per_run <= 0.0:
            return None, 0

        # Use task-time adjacency on actual NPU work items.  The transfer benchmark
        # cases are MatMul graphs; we classify the surrounding PCIE DMA blocks by
        # whether the dominant DMA time lands before or after the short AI_CORE
        # block in a stream.
        stream_groups: dict[str, list[dict[str, str]]] = {}
        for row in task_rows:
            stream_id = str(row.get("stream_id") or "").strip()
            if not stream_id:
                continue
            stream_groups.setdefault(stream_id, []).append(row)

        total_time = 0.0
        total_count = 0
        for rows in stream_groups.values():
            rows = sorted(rows, key=lambda row: first_int(row.get("task_id")) or 0)
            for idx, row in enumerate(rows):
                if str(row.get("kernel_type") or "").strip() != "AI_CORE":
                    continue
                ai_us = first_float(row.get("task_time(us)")) or 0.0
                if ai_us <= 0.0 or ai_us > 500.0:
                    continue

                prev_dma_us = 0.0
                cursor = idx - 1
                while cursor >= 0 and str(rows[cursor].get("kernel_type") or "").strip() == "PCIE_DMA_SQE":
                    prev_dma_us += first_float(rows[cursor].get("task_time(us)")) or 0.0
                    cursor -= 1

                next_dma_us = 0.0
                cursor = idx + 1
                while cursor < len(rows) and str(rows[cursor].get("kernel_type") or "").strip() == "PCIE_DMA_SQE":
                    next_dma_us += first_float(rows[cursor].get("task_time(us)")) or 0.0
                    cursor += 1

                if direction == "h2d":
                    dominant_us = prev_dma_us
                    if dominant_us <= 0.0 or dominant_us <= next_dma_us:
                        continue
                else:
                    dominant_us = next_dma_us
                    if dominant_us <= 0.0 or dominant_us <= prev_dma_us:
                        continue

                total_time += dominant_us
                total_count += 1

        if total_time <= 0.0 or total_count <= 0:
            measured_time_us = first_float(case.get("measured_time_us")) or 0.0
            target_op_time_us = first_float(case.get("target_op_time_us")) or 0.0
            measure_runs = first_int(case.get("measure_runs")) or 0
            effective_time_us = measured_time_us - target_op_time_us
            if effective_time_us <= 0.0:
                effective_time_us = measured_time_us
            if effective_time_us <= 0.0 or measure_runs <= 0:
                return None, total_count
            total_bytes = bytes_per_run * float(measure_runs)
            return float(total_bytes / effective_time_us / 1_000.0), measure_runs
        total_bytes = bytes_per_run * float(total_count)
        return float(total_bytes / total_time / 1_000.0), total_count

    h2d_bw, h2d_count = transfer_bandwidth(transfer_h2d_case, "h2d")
    d2h_bw, d2h_count = transfer_bandwidth(transfer_d2h_case, "d2h")

    return {
        "available": bool(cube_peak is not None and vector_peak is not None and h2d_bw is not None and d2h_bw is not None),
        "profile_dir": str(profile_dir),
        "op_summary_path": str(op_summary_candidates[-1]),
        "task_time_path": str(task_time_candidates[-1]),
        "cube_peak_eff_gflops": cube_peak,
        "vector_peak_eff_gflops": vector_peak,
        "h2d_bw_gbps": h2d_bw,
        "d2h_bw_gbps": d2h_bw,
        "transfer_h2d_count": h2d_count,
        "transfer_d2h_count": d2h_count,
        "cube_case_count": cube_count,
        "vector_add_case_count": vector_add_count,
        "vector_relu_case_count": vector_relu_count,
        "diagnostics": [],
    }


def probe_microbenchmarks(output_dir: Path, device_id: int) -> dict[str, Any]:
    msprof_path = shutil.which("msprof")
    status = {
        "available": False,
        "onnx_available": None,
        "onnxruntime_available": None,
        "runner_python": None,
        "msprof_path": msprof_path,
        "msprof_output_root": None,
        "msprof_profile_dir": None,
        "msprof_op_summary_path": None,
        "msprof_task_time_path": None,
        "benchmark_root": None,
        "summary_path": None,
        "diagnostics": [],
        "cube_peak_eff_gflops": None,
        "vector_peak_eff_gflops": None,
        "h2d_bw_gbps": None,
        "d2h_bw_gbps": None,
    }

    runner_python = locate_microbench_python()
    if runner_python is None:
        status["diagnostics"].append("microbench skipped: no usable Python runner found")
        return status

    benchmark_root = ensure_dir(output_dir / "microbench")
    summary_path = benchmark_root / "microbench_summary.json"
    msprof_root = output_dir / "msprof"
    benchmark_script = PROJECT_DIR / "npu_microbench.py"
    if not benchmark_script.exists():
        status["diagnostics"].append(f"microbench skipped: missing benchmark script {benchmark_script}")
        return status

    env = os.environ.copy()
    if runner_python == ORT_ENV_PYTHON and ORT_ENV_LIB.exists():
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{ORT_ENV_LIB}:{existing_ld}" if existing_ld else str(ORT_ENV_LIB)

    cmd = [
        str(runner_python),
        str(benchmark_script),
        "--output-dir",
        str(output_dir),
        "--device-id",
        str(device_id),
        "--summary-json",
        str(summary_path),
    ]
    if msprof_path is not None:
        msprof_root = ensure_dir(msprof_root)
        application_cmd = " ".join(shlex.quote(part) for part in cmd)
        msprof_cmd = [
            msprof_path,
            f"--output={msprof_root}",
            "--task-time=on",
            "--ai-core=on",
            "--aic-mode=task-based",
            f'--application={application_cmd}',
        ]
        if ASCEND_ENV_SH.exists():
            shell_cmd = (
                f"source {shlex.quote(str(ASCEND_ENV_SH))} >/dev/null 2>&1 && "
                f"export LD_LIBRARY_PATH={shlex.quote(env.get('LD_LIBRARY_PATH', ''))}:$LD_LIBRARY_PATH && "
                f"{' '.join(shlex.quote(part) for part in msprof_cmd)}"
            )
            proc = subprocess.run(["bash", "-lc", shell_cmd], check=False, capture_output=True, text=True, env=env)
        else:
            proc = subprocess.run(msprof_cmd, check=False, capture_output=True, text=True, env=env)
    elif ASCEND_ENV_SH.exists():
        shell_cmd = (
            f"source {shlex.quote(str(ASCEND_ENV_SH))} >/dev/null 2>&1 && "
            f"export LD_LIBRARY_PATH={shlex.quote(env.get('LD_LIBRARY_PATH', ''))}:$LD_LIBRARY_PATH && "
            f"{' '.join(shlex.quote(part) for part in cmd)}"
        )
        proc = subprocess.run(["bash", "-lc", shell_cmd], check=False, capture_output=True, text=True, env=env)
    else:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)

    status["runner_python"] = str(runner_python)
    status["benchmark_root"] = str(benchmark_root)
    status["summary_path"] = str(summary_path)
    if proc.returncode != 0:
        status["diagnostics"].append(f"microbench failed with return code {proc.returncode}")
        if proc.stdout.strip():
            status["diagnostics"].append(proc.stdout.strip())
        if proc.stderr.strip():
            status["diagnostics"].append(proc.stderr.strip())
        return status

    if not summary_path.exists():
        status["diagnostics"].append("microbench finished without a summary JSON")
        if proc.stdout.strip():
            status["diagnostics"].append(proc.stdout.strip())
        return status

    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        status["diagnostics"].append(f"microbench summary has an unexpected type: {type(summary)!r}")
        return status

    msprof_summary = None
    if msprof_path is not None:
        msprof_summary = summarize_msprof_profile(msprof_root, summary)
        if msprof_summary is not None:
            for key in ("cube_peak_eff_gflops", "vector_peak_eff_gflops", "h2d_bw_gbps", "d2h_bw_gbps"):
                value = msprof_summary.get(key)
                if value is not None:
                    status[key] = value
            status["msprof_output_root"] = str(msprof_root)
            status["msprof_profile_dir"] = msprof_summary.get("profile_dir")
            status["msprof_op_summary_path"] = msprof_summary.get("op_summary_path")
            status["msprof_task_time_path"] = msprof_summary.get("task_time_path")
            status["diagnostics"].extend(msprof_summary.get("diagnostics", []))

    status.update(
        {
            "onnx_available": summary.get("onnx_available"),
            "onnxruntime_available": summary.get("onnxruntime_available"),
            "runner_python": summary.get("runner_python") or str(runner_python),
            "benchmark_root": summary.get("benchmark_root") or str(benchmark_root),
            "summary_path": str(summary_path),
            "cube_peak_eff_gflops": status["cube_peak_eff_gflops"] if status["cube_peak_eff_gflops"] is not None else summary.get("cube_peak_eff_gflops"),
            "vector_peak_eff_gflops": status["vector_peak_eff_gflops"] if status["vector_peak_eff_gflops"] is not None else summary.get("vector_peak_eff_gflops"),
            "h2d_bw_gbps": status["h2d_bw_gbps"] if status["h2d_bw_gbps"] is not None else summary.get("h2d_bw_gbps"),
            "d2h_bw_gbps": status["d2h_bw_gbps"] if status["d2h_bw_gbps"] is not None else summary.get("d2h_bw_gbps"),
            "cases": summary.get("cases", []),
            "resolved_provider": summary.get("resolved_provider"),
            "diagnostics": [
                *summary.get("diagnostics", []),
                *( [proc.stdout.strip()] if proc.stdout.strip() else [] ),
                *( [proc.stderr.strip()] if proc.stderr.strip() else [] ),
            ],
        }
    )
    status["available"] = bool(
        status["cube_peak_eff_gflops"] is not None
        and status["vector_peak_eff_gflops"] is not None
        and status["h2d_bw_gbps"] is not None
        and status["d2h_bw_gbps"] is not None
    )
    return status


def probe_hardware(device_id: int, output_dir: Path) -> dict[str, Any]:
    smi_info = parse_npu_smi_info()
    board_info = parse_npu_smi_board(device_id)
    common_info = parse_npu_smi_common(device_id)
    dmi_info = probe_ascend_dmi()
    microbench = probe_microbenchmarks(output_dir, device_id)

    ai_core_count = common_info.get("ai_core_count")
    cube_count = ai_core_count if ai_core_count is not None else DEFAULT_CUBE_COUNT
    vector_count = (int(ai_core_count) * 2) if ai_core_count is not None else DEFAULT_VECTOR_COUNT

    payload = {
        "case_id": "case_10_4_4_cann",
        "device_id": device_id,
        "device_name": smi_info.get("device_name") or "910B3",
        "device_count": smi_info.get("device_count"),
        "chip_count": board_info.get("chip_count"),
        "ai_core_count": ai_core_count,
        "cube_count": cube_count,
        "vector_count": vector_count,
        "frequency_mhz": common_info.get("frequency_mhz"),
        "current_frequency_mhz": common_info.get("current_frequency_mhz"),
        "cube_peak_eff_gflops": microbench.get("cube_peak_eff_gflops"),
        "vector_peak_eff_gflops": microbench.get("vector_peak_eff_gflops"),
        "h2d_bw_gbps": microbench.get("h2d_bw_gbps"),
        "d2h_bw_gbps": microbench.get("d2h_bw_gbps"),
        "microbench_summary_path": microbench.get("summary_path"),
        "source": {
            "device_name": "npu-smi info",
            "device_count": "npu-smi info",
            "chip_count": "npu-smi info -t board",
            "ai_core_count": "npu-smi info -t common",
            "cube_count": "npu-smi info -t common + official AIC/AIV layout",
            "vector_count": "npu-smi info -t common + official AIC/AIV layout",
            "frequency_mhz": "npu-smi info -t common",
            "current_frequency_mhz": "npu-smi info -t common",
            "cube_peak_eff_gflops": "benchmark" if microbench["available"] else None,
            "vector_peak_eff_gflops": "benchmark" if microbench["available"] else None,
            "h2d_bw_gbps": "benchmark" if microbench["available"] else None,
            "d2h_bw_gbps": "benchmark" if microbench["available"] else None,
        },
        "diagnostics": [
            *smi_info.get("diagnostics", []),
            *board_info.get("diagnostics", []),
            *common_info.get("diagnostics", []),
            *dmi_info.get("diagnostics", []),
            *microbench.get("diagnostics", []),
        ],
        "ascend_dmi": dmi_info,
        "microbench": microbench,
    }
    return payload


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    payload = probe_hardware(args.device_id, output_dir)
    output_path = output_dir / "hardware_profile_910b3.json"
    dump_json(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
