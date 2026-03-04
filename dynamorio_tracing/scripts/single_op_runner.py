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


def _normalize_shape(dims: List[int], batch_size: int) -> List[int]:
    out = []
    for i, d in enumerate(dims):
        if d > 0:
            out.append(d)
        else:
            out.append(batch_size if i == 0 else 1)
    return out


def _gen_inputs(model: onnx.ModelProto, batch_size: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    init_names = {x.name for x in model.graph.initializer}
    feeds: Dict[str, np.ndarray] = {}

    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        elem_type = tt.elem_type
        shape = _normalize_shape([
            d.dim_value if d.HasField("dim_value") else -1
            for d in tt.shape.dim
        ], batch_size)

        np_dtype = _elem_type_to_numpy(elem_type)

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
    ap.add_argument("--out-json", default="", help="Optional summary json path")
    args = ap.parse_args()

    model_path = Path(args.onnx)
    model = onnx.load(str(model_path))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [args.provider]
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)

    feeds = _gen_inputs(model, args.batch_size, args.seed)

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
