from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from npu_sep_common import dump_json, ensure_dir, safe_float, safe_int


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DEVICE_ID = 0
DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASURE_RUNS = 8
DEFAULT_VECTOR_SHAPE = (8192, 4096)
DEFAULT_MATMUL_SHAPE = (4096, 4096, 4096)
DEFAULT_PROVIDER_NAME = "CANNExecutionProvider"
DEFAULT_CANN_OPTIONS = {
    "precision_mode": "force_fp32",
    "op_select_impl_mode": "high_performance",
    "arena_extend_strategy": "kNextPowerOfTwo",
    "enable_cann_graph": "0",
}
SUPPORTED_IR_VERSION = 11


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    kind: str
    op_name: str
    model_path: Path
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    units_per_run: float
    category: str
    attrs: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ascend NPU microbenchmarks for 910B3 hardware probing.")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive benchmark artifacts.")
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID, help="NPU device id to probe.")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS, help="Warmup iterations per case.")
    parser.add_argument("--measure-runs", type=int, default=DEFAULT_MEASURE_RUNS, help="Measured iterations per case.")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path for the microbench summary JSON. Defaults to <output-dir>/microbench_summary.json.",
    )
    return parser.parse_args()


def import_onnx_dependencies() -> tuple[Any, Any]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required to run the NPU microbenchmarks."
        ) from exc
    return onnx, ort


def resolve_cann_provider(ort: Any, device_id: int) -> tuple[list[Any], str]:
    available_providers = list(ort.get_available_providers())
    if DEFAULT_PROVIDER_NAME not in available_providers:
        raise RuntimeError(f"{DEFAULT_PROVIDER_NAME} is not available: {available_providers}")
    provider_chain: list[Any] = [
        (
            DEFAULT_PROVIDER_NAME,
            {
                "device_id": str(device_id),
                **DEFAULT_CANN_OPTIONS,
            },
        ),
        "CPUExecutionProvider",
    ]
    return provider_chain, DEFAULT_PROVIDER_NAME


def element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def make_case_dir(output_dir: Path, case_name: str) -> Path:
    return ensure_dir(output_dir / case_name)


def build_matmul_model(onnx: Any, case: BenchmarkCase) -> None:
    from onnx import TensorProto, helper

    m, k, n = case.attrs["m"], case.attrs["k"], case.attrs["n"]
    inputs = [
        helper.make_tensor_value_info(case.input_names[0], TensorProto.FLOAT, [m, k]),
        helper.make_tensor_value_info(case.input_names[1], TensorProto.FLOAT, [k, n]),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])]
    node = helper.make_node("MatMul", list(case.input_names), ["y"], name=case.op_name)
    graph = helper.make_graph([node], case.name, inputs, outputs)
    model = helper.make_model(graph, producer_name="npu_sep_modeling", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = SUPPORTED_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, case.model_path)


def build_add_model(onnx: Any, case: BenchmarkCase) -> None:
    from onnx import TensorProto, helper

    shape = list(case.input_shapes[0])
    inputs = [
        helper.make_tensor_value_info(case.input_names[0], TensorProto.FLOAT, shape),
        helper.make_tensor_value_info(case.input_names[1], TensorProto.FLOAT, shape),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
    node = helper.make_node("Add", list(case.input_names), ["y"], name=case.op_name)
    graph = helper.make_graph([node], case.name, inputs, outputs)
    model = helper.make_model(graph, producer_name="npu_sep_modeling", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = SUPPORTED_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, case.model_path)


def build_relu_model(onnx: Any, case: BenchmarkCase) -> None:
    from onnx import TensorProto, helper

    shape = list(case.input_shapes[0])
    inputs = [helper.make_tensor_value_info(case.input_names[0], TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)]
    node = helper.make_node("Relu", list(case.input_names), ["y"], name=case.op_name)
    graph = helper.make_graph([node], case.name, inputs, outputs)
    model = helper.make_model(graph, producer_name="npu_sep_modeling", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = SUPPORTED_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, case.model_path)


def build_transpose_model(onnx: Any, case: BenchmarkCase) -> None:
    from onnx import TensorProto, helper

    shape = list(case.input_shapes[0])
    perm = list(case.attrs.get("perm") or [1, 0])
    output_shape = [shape[index] for index in perm]
    inputs = [helper.make_tensor_value_info(case.input_names[0], TensorProto.FLOAT, shape)]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
    node = helper.make_node("Transpose", list(case.input_names), ["y"], name=case.op_name, perm=perm)
    graph = helper.make_graph([node], case.name, inputs, outputs)
    model = helper.make_model(graph, producer_name="npu_sep_modeling", opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = SUPPORTED_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, case.model_path)


def build_benchmark_case_models(onnx: Any, benchmark_root: Path) -> list[BenchmarkCase]:
    matmul_case = BenchmarkCase(
        name="cube_matmul",
        kind="matmul",
        op_name="MatMul",
        model_path=benchmark_root / "cube_matmul" / "model.onnx",
        input_names=("lhs", "rhs"),
        input_shapes=((DEFAULT_MATMUL_SHAPE[0], DEFAULT_MATMUL_SHAPE[1]), (DEFAULT_MATMUL_SHAPE[1], DEFAULT_MATMUL_SHAPE[2])),
        output_shape=(DEFAULT_MATMUL_SHAPE[0], DEFAULT_MATMUL_SHAPE[2]),
        units_per_run=2.0
        * float(DEFAULT_MATMUL_SHAPE[0])
        * float(DEFAULT_MATMUL_SHAPE[1])
        * float(DEFAULT_MATMUL_SHAPE[2]),
        category="cube",
        attrs={"m": DEFAULT_MATMUL_SHAPE[0], "k": DEFAULT_MATMUL_SHAPE[1], "n": DEFAULT_MATMUL_SHAPE[2]},
    )
    add_case = BenchmarkCase(
        name="vector_add",
        kind="add",
        op_name="Add",
        model_path=benchmark_root / "vector_add" / "model.onnx",
        input_names=("lhs", "rhs"),
        input_shapes=(DEFAULT_VECTOR_SHAPE, DEFAULT_VECTOR_SHAPE),
        output_shape=DEFAULT_VECTOR_SHAPE,
        units_per_run=float(element_count(DEFAULT_VECTOR_SHAPE)),
        category="vector",
        attrs={},
    )
    relu_case = BenchmarkCase(
        name="vector_relu",
        kind="relu",
        op_name="Relu",
        model_path=benchmark_root / "vector_relu" / "model.onnx",
        input_names=("x",),
        input_shapes=(DEFAULT_VECTOR_SHAPE,),
        output_shape=DEFAULT_VECTOR_SHAPE,
        units_per_run=float(element_count(DEFAULT_VECTOR_SHAPE)),
        category="vector",
        attrs={},
    )
    transfer_h2d_case = BenchmarkCase(
        name="transfer_h2d",
        kind="matmul",
        op_name="MatMul",
        model_path=benchmark_root / "transfer_h2d" / "model.onnx",
        input_names=("lhs", "rhs"),
        input_shapes=((8192, 4096), (4096, 1)),
        output_shape=(8192, 1),
        units_per_run=2.0 * 8192.0 * 4096.0 * 1.0,
        category="transfer",
        attrs={"m": 8192, "k": 4096, "n": 1, "direction": "h2d"},
    )
    transfer_d2h_case = BenchmarkCase(
        name="transfer_d2h",
        kind="matmul",
        op_name="MatMul",
        model_path=benchmark_root / "transfer_d2h" / "model.onnx",
        input_names=("lhs", "rhs"),
        input_shapes=((8192, 1), (1, 4096)),
        output_shape=(8192, 4096),
        units_per_run=2.0 * 8192.0 * 1.0 * 4096.0,
        category="transfer",
        attrs={"m": 8192, "k": 1, "n": 4096, "direction": "d2h"},
    )
    for case in (matmul_case, add_case, relu_case, transfer_h2d_case, transfer_d2h_case):
        ensure_dir(case.model_path.parent)
        if case.kind == "matmul":
            build_matmul_model(onnx, case)
        elif case.kind == "add":
            build_add_model(onnx, case)
        elif case.kind == "relu":
            build_relu_model(onnx, case)
        else:
            raise ValueError(f"Unsupported benchmark case kind: {case.kind}")
    return [matmul_case, add_case, relu_case, transfer_h2d_case, transfer_d2h_case]


def make_inputs(case: BenchmarkCase) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for name, shape in zip(case.input_names, case.input_shapes):
        inputs[name] = np.ones(shape, dtype=np.float32)
    return inputs


def create_session(ort: Any, model_path: Path, device_id: int, profiling: bool) -> Any:
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.enable_profiling = profiling
    if hasattr(session_options, "intra_op_num_threads"):
        session_options.intra_op_num_threads = 1
    if hasattr(session_options, "inter_op_num_threads"):
        session_options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers, _ = resolve_cann_provider(ort, device_id)
    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)


def run_session(session: Any, inputs: dict[str, np.ndarray], repeat: int) -> float:
    start = time.perf_counter()
    for _ in range(max(1, int(repeat))):
        session.run(None, inputs)
    end = time.perf_counter()
    return (end - start) * 1_000_000.0


def extract_bytes_from_event(event: dict[str, Any]) -> int:
    args = event.get("args") or {}
    candidates = [
        safe_int(args.get("output_size"), 0) or 0,
        safe_int(args.get("input_size"), 0) or 0,
        safe_int(args.get("activation_size"), 0) or 0,
        safe_int(args.get("parameter_size"), 0) or 0,
    ]
    return int(max(candidates))


def load_profile_events(profile_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = payload.get("traceEvents") or payload.get("events") or []
    else:
        raise ValueError(f"Unsupported ORT profile format in {profile_path}")
    out: list[dict[str, Any]] = []
    for event in events:
        if isinstance(event, dict):
            out.append(event)
    return out


def summarize_profile(profile_path: Path, provider_name: str, target_op_name: str) -> dict[str, Any]:
    events = load_profile_events(profile_path)
    target_op_time_us = 0.0
    target_op_count = 0
    h2d_time_us = 0.0
    h2d_bytes = 0
    d2h_time_us = 0.0
    d2h_bytes = 0
    provider_ops: list[str] = []
    for event in events:
        if str(event.get("cat")) != "Node":
            continue
        if str(event.get("ph")) != "X":
            continue
        args = event.get("args") or {}
        if str(args.get("provider")) != provider_name:
            continue
        op_name = str(args.get("op_name") or "")
        dur = float(safe_float(event.get("dur"), 0.0) or 0.0)
        provider_ops.append(op_name)
        if op_name == target_op_name:
            target_op_time_us += dur
            target_op_count += 1
        if op_name == "MemcpyFromHost":
            h2d_time_us += dur
            h2d_bytes += extract_bytes_from_event(event)
        elif op_name == "MemcpyToHost":
            d2h_time_us += dur
            d2h_bytes += extract_bytes_from_event(event)
    return {
        "profile_path": str(profile_path),
        "provider_name": provider_name,
        "target_op_name": target_op_name,
        "target_op_time_us": target_op_time_us,
        "target_op_count": target_op_count,
        "h2d_time_us": h2d_time_us,
        "h2d_bytes": h2d_bytes,
        "d2h_time_us": d2h_time_us,
        "d2h_bytes": d2h_bytes,
        "provider_ops": provider_ops,
    }


def effective_throughput_gflops(units_per_run: float, run_count: int, total_time_us: float) -> float | None:
    if run_count <= 0 or total_time_us <= 0.0:
        return None
    total_units = float(units_per_run) * float(run_count)
    return float(total_units / total_time_us / 1_000.0)


def bandwidth_gbps(total_bytes: int, total_time_us: float) -> float | None:
    if total_bytes <= 0 or total_time_us <= 0.0:
        return None
    return float(float(total_bytes) / total_time_us / 1_000.0)


def run_case(ort: Any, case: BenchmarkCase, device_id: int, warmup_runs: int, measure_runs: int) -> dict[str, Any]:
    case_dir = ensure_dir(case.model_path.parent)
    model_copy = case.model_path

    warmup_session = create_session(ort, model_copy, device_id, profiling=False)
    inputs = make_inputs(case)
    warmup_time_us = run_session(warmup_session, inputs, warmup_runs)

    profiled_session = create_session(ort, model_copy, device_id, profiling=True)
    measured_time_us = run_session(profiled_session, inputs, measure_runs)
    profile_path = Path(profiled_session.end_profiling()).resolve()
    profile_copy = case_dir / "ort_profile.json"
    shutil.copy2(profile_path, profile_copy)

    profile_summary = summarize_profile(profile_copy, DEFAULT_PROVIDER_NAME, case.op_name)
    target_op_time_us = float(profile_summary["target_op_time_us"])
    target_op_count = int(profile_summary["target_op_count"])
    h2d_time_us = float(profile_summary["h2d_time_us"])
    h2d_bytes = int(profile_summary["h2d_bytes"])
    d2h_time_us = float(profile_summary["d2h_time_us"])
    d2h_bytes = int(profile_summary["d2h_bytes"])
    throughput_gflops = effective_throughput_gflops(case.units_per_run, target_op_count, target_op_time_us)
    h2d_bw = bandwidth_gbps(h2d_bytes, h2d_time_us)
    d2h_bw = bandwidth_gbps(d2h_bytes, d2h_time_us)
    input_bytes_per_run = sum(element_count(shape) for shape in case.input_shapes) * 4
    output_bytes_per_run = element_count(case.output_shape) * 4

    case_summary = {
        "name": case.name,
        "kind": case.kind,
        "category": case.category,
        "op_name": case.op_name,
        "model_path": str(model_copy),
        "profile_path": str(profile_copy),
        "input_names": list(case.input_names),
        "input_shapes": [list(shape) for shape in case.input_shapes],
        "output_shape": list(case.output_shape),
        "units_per_run": case.units_per_run,
        "input_bytes_per_run": input_bytes_per_run,
        "output_bytes_per_run": output_bytes_per_run,
        "transfer_bytes_per_direction": output_bytes_per_run if case.category == "transfer" else None,
        "target_op_count": target_op_count,
        "target_op_time_us": target_op_time_us,
        "throughput_gflops": throughput_gflops,
        "h2d_bytes": h2d_bytes,
        "h2d_time_us": h2d_time_us,
        "h2d_bw_gbps": h2d_bw,
        "d2h_bytes": d2h_bytes,
        "d2h_time_us": d2h_time_us,
        "d2h_bw_gbps": d2h_bw,
        "warmup_time_us": warmup_time_us,
        "measured_time_us": measured_time_us,
        "measure_runs": measure_runs,
        "warmup_runs": warmup_runs,
        "provider_ops": profile_summary["provider_ops"],
    }
    dump_json(case_dir / "case_summary.json", case_summary)
    return case_summary


def aggregate_summary(case_summaries: list[dict[str, Any]], msprof_path: str | None) -> dict[str, Any]:
    cube_cases = [case for case in case_summaries if case.get("category") == "cube" and case.get("throughput_gflops") is not None]
    vector_cases = [case for case in case_summaries if case.get("category") == "vector" and case.get("throughput_gflops") is not None]
    h2d_cases = [case for case in case_summaries if case.get("h2d_bw_gbps") is not None]
    d2h_cases = [case for case in case_summaries if case.get("d2h_bw_gbps") is not None]

    cube_peak = cube_cases[0]["throughput_gflops"] if cube_cases else None
    vector_peak = median([case["throughput_gflops"] for case in vector_cases]) if vector_cases else None
    h2d_bw = (
        float(sum(case["h2d_bytes"] for case in h2d_cases) / sum(case["h2d_time_us"] for case in h2d_cases) / 1_000.0)
        if h2d_cases and sum(case["h2d_time_us"] for case in h2d_cases) > 0.0
        else None
    )
    d2h_bw = (
        float(sum(case["d2h_bytes"] for case in d2h_cases) / sum(case["d2h_time_us"] for case in d2h_cases) / 1_000.0)
        if d2h_cases and sum(case["d2h_time_us"] for case in d2h_cases) > 0.0
        else None
    )
    return {
        "available": bool(cube_peak is not None and vector_peak is not None and h2d_bw is not None and d2h_bw is not None),
        "runner_python": None,
        "onnx_available": True,
        "onnxruntime_available": True,
        "resolved_provider": DEFAULT_PROVIDER_NAME,
        "msprof_path": msprof_path,
        "cube_peak_eff_gflops": cube_peak,
        "vector_peak_eff_gflops": vector_peak,
        "h2d_bw_gbps": h2d_bw,
        "d2h_bw_gbps": d2h_bw,
        "cases": case_summaries,
        "diagnostics": [],
    }


def run_microbenchmarks(output_dir: Path, device_id: int, warmup_runs: int, measure_runs: int) -> dict[str, Any]:
    onnx, ort = import_onnx_dependencies()
    benchmark_root = ensure_dir(output_dir / "microbench")
    cases = build_benchmark_case_models(onnx, benchmark_root)
    case_summaries: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for case in cases:
        case_summaries.append(run_case(ort, case, device_id, warmup_runs, measure_runs))
    msprof_path = shutil.which("msprof")
    summary = aggregate_summary(case_summaries, msprof_path)
    summary["benchmark_root"] = str(benchmark_root)
    summary["diagnostics"] = diagnostics
    summary["runner_python"] = sys.executable
    summary["warmup_runs"] = warmup_runs
    summary["measure_runs"] = measure_runs
    summary["device_id"] = device_id
    summary["measured_at_unix_s"] = time.time()
    return summary


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    summary = run_microbenchmarks(output_dir, args.device_id, args.warmup_runs, args.measure_runs)
    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "microbench_summary.json"
    dump_json(summary_path, summary)
    print(f"Wrote {summary_path}")
    print(f"cube_peak_eff_gflops={summary.get('cube_peak_eff_gflops')}")
    print(f"vector_peak_eff_gflops={summary.get('vector_peak_eff_gflops')}")
    print(f"h2d_bw_gbps={summary.get('h2d_bw_gbps')}")
    print(f"d2h_bw_gbps={summary.get('d2h_bw_gbps')}")


if __name__ == "__main__":
    main()
