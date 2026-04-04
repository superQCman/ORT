from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from npu_sep_common import dump_json, ensure_dir


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


def probe_microbenchmarks() -> dict[str, Any]:
    onnx_available = importlib.util.find_spec("onnx") is not None
    onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None
    msprof_path = shutil.which("msprof")
    status = {
        "available": False,
        "onnx_available": onnx_available,
        "onnxruntime_available": onnxruntime_available,
        "msprof_path": msprof_path,
        "diagnostics": [],
        "cube_peak_eff_gflops": None,
        "vector_peak_eff_gflops": None,
        "h2d_bw_gbps": None,
        "d2h_bw_gbps": None,
    }
    missing: list[str] = []
    if not onnx_available:
        missing.append("onnx")
    if not onnxruntime_available:
        missing.append("onnxruntime")
    if missing:
        status["diagnostics"].append(f"microbench skipped: missing {', '.join(missing)}")
        return status

    status["diagnostics"].append(
        "microbench implementation is currently reserved for environments with a working ORT+CANN runtime stack"
    )
    return status


def probe_hardware(device_id: int) -> dict[str, Any]:
    smi_info = parse_npu_smi_info()
    board_info = parse_npu_smi_board(device_id)
    common_info = parse_npu_smi_common(device_id)
    dmi_info = probe_ascend_dmi()
    microbench = probe_microbenchmarks()

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
    payload = probe_hardware(args.device_id)
    output_path = output_dir / "hardware_profile_910b3.json"
    dump_json(output_path, payload)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
