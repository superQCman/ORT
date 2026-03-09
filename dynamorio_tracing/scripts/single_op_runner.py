#!/usr/bin/env python3
"""
Run a single-operator ONNX model once (with warmup), intended to be launched under DynamoRIO.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort


def _elem_type_to_numpy(elem_type: int):
    import onnx

    m = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.BOOL: np.bool_,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.UINT64: np.uint64,
    }
    return m.get(elem_type, np.float32)


def _resolve_dynamic_dim(
    tensor_name: str,
    dim_index: int,
    dim: "onnx.TensorShapeProto.Dimension",
    batch_size: int,
    num_indices_per_lookup: int,
) -> int:
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)

    dim_param = dim.dim_param if dim.HasField("dim_param") else ""
    is_indices = tensor_name.startswith("indices") or tensor_name == "indices"
    if is_indices and num_indices_per_lookup <= 0:
        raise ValueError(
            f"Input {tensor_name!r} has a dynamic indices dimension; "
            "pass --num-indices-per-lookup to preserve the original shape."
        )

    if is_indices or dim_param == "indices_total":
        return batch_size * max(1, num_indices_per_lookup)
    if "batch" in dim_param.lower() or dim_index == 0:
        return batch_size
    return 1


def _resolve_value_info_shape(
    vi: onnx.ValueInfoProto,
    batch_size: int,
    num_indices_per_lookup: int,
) -> List[int]:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return []
    return [
        _resolve_dynamic_dim(vi.name, i, dim, batch_size, num_indices_per_lookup)
        for i, dim in enumerate(tt.shape.dim)
    ]


def _fallback_shape_for_single_node_input(
    model: onnx.ModelProto,
    input_name: str,
    batch_size: int,
    num_indices_per_lookup: int,
) -> List[int]:
    if len(model.graph.node) != 1:
        return []

    node = model.graph.node[0]
    if node.op_type != "Reshape" or not node.input or input_name != node.input[0]:
        return []

    output_name = node.output[0] if node.output else ""
    output_vi = next((out for out in model.graph.output if out.name == output_name), None)
    if output_vi is None:
        return []

    output_shape = _resolve_value_info_shape(
        output_vi, batch_size, num_indices_per_lookup
    )
    if not output_shape:
        return []

    total = int(np.prod(output_shape, dtype=np.int64))
    if len(output_shape) >= 2 and output_shape[-1] > 0 and total % output_shape[-1] == 0:
        return [total // output_shape[-1], output_shape[-1]]
    return [total]


def _shape_tensor_values_for_single_node(
    model: onnx.ModelProto,
    input_name: str,
    batch_size: int,
    num_indices_per_lookup: int,
) -> np.ndarray | None:
    if len(model.graph.node) != 1:
        return None

    node = model.graph.node[0]
    if len(node.input) < 2 or input_name != node.input[1]:
        return None
    if node.op_type not in ("Reshape", "Expand"):
        return None

    output_name = node.output[0] if node.output else ""
    output_vi = next((out for out in model.graph.output if out.name == output_name), None)
    if output_vi is None:
        return None

    target_shape = _resolve_value_info_shape(
        output_vi, batch_size, num_indices_per_lookup
    )
    return np.asarray(target_shape, dtype=np.int64)


def _gen_inputs(
    model: onnx.ModelProto,
    batch_size: int,
    seed: int,
    num_indices_per_lookup: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    init_names = {x.name for x in model.graph.initializer}
    feeds: Dict[str, np.ndarray] = {}

    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        elem_type = tt.elem_type
        shape = _resolve_value_info_shape(inp, batch_size, num_indices_per_lookup)
        if not shape:
            shape = _fallback_shape_for_single_node_input(
                model, inp.name, batch_size, num_indices_per_lookup
            )

        np_dtype = _elem_type_to_numpy(elem_type)
        shape_tensor_values = _shape_tensor_values_for_single_node(
            model, inp.name, batch_size, num_indices_per_lookup
        )

        if shape_tensor_values is not None:
            feeds[inp.name] = shape_tensor_values.astype(np_dtype, copy=False)
            continue

        if np_dtype in (np.float16, np.float32, np.float64):
            arr = rng.standard_normal(size=shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            arr = rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.bool_)
        else:
            arr = rng.integers(0, 32, size=shape, dtype=np_dtype)

        feeds[inp.name] = arr

    # Special case: Gather/GatherND indices must lie within the data tensor's
    # valid range on the gather axis.  Random int generation can easily produce
    # out-of-bound indices (e.g. idx=2 for a length-1 axis).
    if len(model.graph.node) == 1:
        node = model.graph.node[0]
        if node.op_type == "Gather" and len(node.input) >= 2:
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)
            data_name = node.input[0]
            idx_name = node.input[1]
            if data_name in feeds and idx_name in feeds:
                data_size = feeds[data_name].shape[axis] if feeds[data_name].ndim > axis else 1
                if data_size > 0:
                    feeds[idx_name] = feeds[idx_name] % data_size

    return feeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to single-op ONNX file")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--num-indices-per-lookup", type=int, default=0)
    ap.add_argument("--out-json", default="", help="Optional summary json path")
    ap.add_argument("--intra-threads", type=int, default=1,
                    help="ORT intra-op parallelism thread count")
    ap.add_argument("--inter-threads", type=int, default=1,
                    help="ORT inter-op parallelism thread count")
    args = ap.parse_args()

    model_path = Path(args.onnx)
    model = onnx.load(str(model_path))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = args.intra_threads
    sess_opts.inter_op_num_threads = args.inter_threads

    providers = [args.provider]
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)

    feeds = _gen_inputs(
        model, args.batch_size, args.seed, args.num_indices_per_lookup
    )

    for _ in range(max(0, args.warmup)):
        _ = sess.run(None, feeds)

    latencies_ms = []
    for _ in range(max(1, args.runs)):
        t0 = time.perf_counter()
        _ = sess.run(None, feeds)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "model": str(model_path),
        "provider": args.provider,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms": {
            "avg": float(np.mean(latencies_ms)),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "max": float(np.max(latencies_ms)),
        },
        "inputs": {k: list(v.shape) for k, v in feeds.items()},
    }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
