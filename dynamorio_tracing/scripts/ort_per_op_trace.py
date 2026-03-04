#!/usr/bin/env python3
"""
Extract per-op traces for an ONNX model by splitting each operator into a tiny ONNX model
and running each model under DynamoRIO drmemtrace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import helper, numpy_helper, shape_inference


@dataclass
class OpTask:
    idx: int
    op_type: str
    node_name: str
    onnx_path: Path
    trace_dir: Path
    run_json: Path
    log_path: Path


def _safe_name(s: str) -> str:
    bad = '/\\:<>"|?* '
    out = s
    for c in bad:
        out = out.replace(c, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unnamed"


def _value_info_map(model: onnx.ModelProto):
    mp: Dict[str, onnx.ValueInfoProto] = {}
    for x in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        mp[x.name] = x
    return mp


def _initializer_map(model: onnx.ModelProto):
    return {x.name: x for x in model.graph.initializer}


def _constant_tensor_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    """Return {output_name: TensorProto} for every Constant node in the graph.

    Constant node outputs should be embedded as initializers in sub-models
    rather than treated as dynamic graph inputs — otherwise single_op_runner
    generates random values that don't match the real shape/type constraints
    (e.g. the int64 shape tensor fed into Reshape).
    """
    mp: Dict[str, onnx.TensorProto] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        out_name = node.output[0]
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                t = onnx.TensorProto()
                t.CopyFrom(attr.t)
                t.name = out_name
                mp[out_name] = t
                break
            if attr.name == "value_int":
                t = helper.make_tensor(out_name, onnx.TensorProto.INT64, [], [attr.i])
                mp[out_name] = t
                break
            if attr.name == "value_ints":
                vals = list(attr.ints)
                t = helper.make_tensor(out_name, onnx.TensorProto.INT64, [len(vals)], vals)
                mp[out_name] = t
                break
            if attr.name == "value_float":
                t = helper.make_tensor(out_name, onnx.TensorProto.FLOAT, [], [attr.f])
                mp[out_name] = t
                break
            if attr.name == "value_floats":
                vals = list(attr.floats)
                t = helper.make_tensor(out_name, onnx.TensorProto.FLOAT, [len(vals)], vals)
                mp[out_name] = t
                break
    return mp


def _clone_vi(vi: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    out = onnx.ValueInfoProto()
    out.CopyFrom(vi)
    return out


def _build_single_op_model(model: onnx.ModelProto, node_idx: int) -> Optional[onnx.ModelProto]:
    g = model.graph
    if node_idx < 0 or node_idx >= len(g.node):
        return None

    node = g.node[node_idx]
    vi_map = _value_info_map(model)
    init_map = _initializer_map(model)
    const_tensor_map = _constant_tensor_map(model)

    # Collect required graph inputs for this node.
    new_inputs: List[onnx.ValueInfoProto] = []
    new_inits: List[onnx.TensorProto] = []

    for inp in node.input:
        if not inp:
            continue
        if inp in init_map:
            t = onnx.TensorProto()
            t.CopyFrom(init_map[inp])
            new_inits.append(t)
            continue
        # Constant node outputs must be baked in as initializers so the runner
        # does not generate incorrect random values for them (e.g. the int64
        # shape tensor fed into Reshape).
        if inp in const_tensor_map:
            new_inits.append(const_tensor_map[inp])
            continue
        if inp in vi_map:
            new_inputs.append(_clone_vi(vi_map[inp]))
            continue
        # Fallback: unspecified rank (None) to avoid axis-out-of-range errors
        # during shape inference when the actual rank is > 1.
        new_inputs.append(helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, None))

    new_outputs: List[onnx.ValueInfoProto] = []
    for out_name in node.output:
        if not out_name:
            continue
        if out_name in vi_map:
            vi = vi_map[out_name]
            # Keep elem_type but drop shape — let shape_inference fill in the
            # correct shape to avoid "MergeShapeInfo" conflicts at load time.
            elem_type = (
                vi.type.tensor_type.elem_type
                if vi.type.HasField("tensor_type")
                else onnx.TensorProto.FLOAT
            )
            new_outputs.append(helper.make_tensor_value_info(out_name, elem_type, None))
        else:
            new_outputs.append(helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None))

    new_node = onnx.NodeProto()
    new_node.CopyFrom(node)

    new_graph = helper.make_graph(
        nodes=[new_node],
        name=f"single_op_{node_idx}_{_safe_name(node.name or node.op_type)}",
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=new_inits,
    )

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version

    # Cap ai.onnx opset to 23 (ORT 1.23.x only guarantees support up to opset 23).
    _MAX_ONNX_OPSET = 23
    del new_model.opset_import[:]
    for op in model.opset_import:
        entry = new_model.opset_import.add()
        entry.domain = op.domain
        if op.domain in ("", "ai.onnx"):
            entry.version = min(op.version, _MAX_ONNX_OPSET)
        else:
            entry.version = op.version

    try:
        new_model = shape_inference.infer_shapes(new_model)
    except Exception:
        pass

    new_model = _patch_missing_shapes(new_model)
    new_model = _fixup_reshape_shapes(new_model)
    new_model = _fixup_expand_shape_input(new_model)
    new_model = _fixup_gather_data_size(new_model)

    return new_model


def _fixup_gather_data_size(new_model: onnx.ModelProto) -> onnx.ModelProto:
    """For single-Gather sub-models where indices is a constant initializer,
    ensure the data input's size on the gather axis is large enough to hold
    the maximum index value (e.g. constant idx=2 but data has only 1 element).
    """
    if len(new_model.graph.node) != 1:
        return new_model
    node = new_model.graph.node[0]
    if node.op_type != "Gather" or len(node.input) < 2:
        return new_model

    data_name = node.input[0]
    idx_name = node.input[1]
    init_map = {x.name: x for x in new_model.graph.initializer}
    input_map = {gi.name: gi for gi in new_model.graph.input}

    if idx_name not in init_map or data_name not in input_map:
        return new_model

    try:
        idx_vals = [int(v) for v in numpy_helper.to_array(init_map[idx_name]).flat]
    except Exception:
        return new_model
    if not idx_vals:
        return new_model

    max_idx = max(abs(v) for v in idx_vals)
    required = max_idx + 1

    axis = 0
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)

    data_gi = input_map[data_name]
    tt = data_gi.type.tensor_type
    if not tt.HasField("shape") or axis >= len(tt.shape.dim):
        return new_model

    dim = tt.shape.dim[axis]
    if dim.dim_value >= required:
        return new_model

    dim.ClearField("dim_param")
    dim.dim_value = required
    return new_model


def _patch_missing_shapes(new_model: onnx.ModelProto) -> onnx.ModelProto:
    """For single-op sub-models, assign concrete shapes to graph inputs that
    have no shape information.  When full-model shape inference could not
    propagate shapes (dynamic inputs, outputs absent from value_info), the
    runner's _normalize_shape returns [] causing rank-0 scalars—invalid for
    ops like Transpose, Gemm, MatMul, Flatten, Concat, Gather.
    """
    if len(new_model.graph.node) != 1:
        return new_model
    node = new_model.graph.node[0]
    op_type = node.op_type
    attrs = {a.name: a for a in node.attribute}
    init_map = {x.name: x for x in new_model.graph.initializer}
    input_map = {gi.name: gi for gi in new_model.graph.input}

    def _has_concrete_shape(gi) -> bool:
        tt = gi.type.tensor_type
        return tt.HasField("shape") and len(tt.shape.dim) > 0

    def _set_shape(gi, dims: List[int]):
        tt = gi.type.tensor_type
        tt.ClearField("shape")
        for d in dims:
            dim = tt.shape.dim.add()
            if d > 0:
                dim.dim_value = d
            else:
                dim.dim_param = "N"

    def _init_shape(name: str) -> List[int]:
        return list(init_map[name].dims) if name in init_map else []

    for inp_idx, inp_name in enumerate(node.input):
        if not inp_name or inp_name not in input_map:
            continue
        gi = input_map[inp_name]
        if _has_concrete_shape(gi):
            continue  # Already has shape; Reshape fixup handles remaining issues

        if op_type == "Transpose":
            perm = list(attrs["perm"].ints) if "perm" in attrs else [0, 1, 2]
            _set_shape(gi, [4] * len(perm))

        elif op_type == "Gemm":
            if inp_idx == 0:  # A: (M, K)
                b_name = node.input[1] if len(node.input) > 1 else ""
                b_shape = _init_shape(b_name)
                transB = bool(attrs["transB"].i) if "transB" in attrs else False
                K = (b_shape[0] if not transB else b_shape[1]) if b_shape else 4
                _set_shape(gi, [1, K])
            else:
                _set_shape(gi, [4, 4])

        elif op_type in ("MatMul", "BatchMatMul"):
            _set_shape(gi, [1, 4] if inp_idx == 0 else [4, 1])

        elif op_type == "Flatten":
            axis = int(attrs["axis"].i) if "axis" in attrs else 1
            _set_shape(gi, [4] * max(axis, 2))

        elif op_type == "Concat":
            axis = int(attrs["axis"].i) if "axis" in attrs else 0
            # Match non-concat-axis dims from a sibling input that has concrete shape,
            # so the runner doesn't generate mismatched dimensions.
            ref_dims = None
            for other_name in node.input:
                if not other_name or other_name == inp_name or other_name not in input_map:
                    continue
                other_gi = input_map[other_name]
                if _has_concrete_shape(other_gi):
                    ref_dims = []
                    for i, dim in enumerate(other_gi.type.tensor_type.shape.dim):
                        if i == axis:
                            ref_dims.append(4)   # concat axis: any positive value
                        elif dim.dim_value > 0:
                            ref_dims.append(dim.dim_value)
                        else:
                            ref_dims.append(1)   # dynamic non-axis dim → 1
                    break
            if ref_dims is None:
                ref_dims = [4] * (axis + 1)
            _set_shape(gi, ref_dims)

        elif op_type == "Gather":
            if inp_idx == 0:  # data must have rank > axis
                axis = int(attrs["axis"].i) if "axis" in attrs else 0
                _set_shape(gi, [4] * (axis + 1))
            else:  # indices
                _set_shape(gi, [1])

        elif op_type == "Expand":
            if inp_idx == 0:  # data
                _set_shape(gi, [1, 4])
            else:  # shape must be 1-D int64
                _set_shape(gi, [1])

        else:
            _set_shape(gi, [1, 4])

    return new_model


def _fixup_reshape_shapes(new_model: onnx.ModelProto) -> onnx.ModelProto:
    """For single-Reshape sub-models, ensure data and shape tensors are compatible
    so the runner can execute without element-count mismatches.

    Three cases:
      A. Shape is a constant initializer, data has dynamic dims → concrete-ify data.
      B. Shape is a dynamic graph input (e.g. output of Concat) → convert it to a
         flat-reshape initializer matching the data's static element count.
      C. Shape contains 0-copy dims → replace 0 with 1 in the initializer so that
         any input rank is acceptable (semantics differ slightly but op runs cleanly).
    """
    if len(new_model.graph.node) != 1:
        return new_model
    node = new_model.graph.node[0]
    if node.op_type != "Reshape" or len(node.input) < 2:
        return new_model

    data_name = node.input[0]
    shape_name = node.input[1]
    init_map = {x.name: x for x in new_model.graph.initializer}
    input_map = {gi.name: gi for gi in new_model.graph.input}
    data_gi = input_map.get(data_name)

    # Compute static element product and dynamic axes of data input.
    data_static_prod = 1
    data_dyn_axes: List[int] = []
    if data_gi:
        tt = data_gi.type.tensor_type
        if tt.HasField("shape"):
            for i, dim in enumerate(tt.shape.dim):
                if dim.dim_value > 0:
                    data_static_prod *= dim.dim_value
                else:
                    data_dyn_axes.append(i)

    # --- Case B: shape tensor is a dynamic graph input (not a constant init) ---
    if shape_name in input_map and shape_name not in init_map:
        total = max(data_static_prod, 1)
        flat_t = helper.make_tensor(shape_name, onnx.TensorProto.INT64, [1], [total])
        new_model.graph.initializer.append(flat_t)
        # Remove from graph inputs.
        new_model.graph.input.__delitem__(
            next(i for i, gi in enumerate(new_model.graph.input) if gi.name == shape_name)
        )
        # Concrete-ify any dynamic data dims.
        if data_gi:
            for ax in data_dyn_axes:
                data_gi.type.tensor_type.shape.dim[ax].ClearField("dim_param")
                data_gi.type.tensor_type.shape.dim[ax].dim_value = 1
        return new_model

    if shape_name not in init_map:
        return new_model

    try:
        target = [int(d) for d in numpy_helper.to_array(init_map[shape_name]).flat]
    except Exception:
        return new_model

    # --- Case C: replace 0-copy dims with 1 to avoid rank-mismatch errors ---
    if any(d == 0 for d in target):
        patched = [1 if d == 0 else d for d in target]
        flat_t = helper.make_tensor(
            shape_name, onnx.TensorProto.INT64, [len(patched)], patched
        )
        # Replace the existing initializer.
        for i, init in enumerate(new_model.graph.initializer):
            if init.name == shape_name:
                new_model.graph.initializer.__delitem__(i)
                break
        new_model.graph.initializer.append(flat_t)
        target = patched  # continue with patched target for element count

    # --- Case A: shape is a constant Init, fix data dynamic dims ---
    has_minus_one = any(d == -1 for d in target)
    total_target = 1
    for d in target:
        if d > 0:
            total_target *= d
    # total_target = product of all explicit positive dims

    if not data_gi:
        return new_model

    if data_static_prod > 0 and not data_dyn_axes:
        # Data already fully concrete; check element count matches.
        if not has_minus_one:
            if data_static_prod != total_target:
                # Element counts differ: reshape data to flat [total_target].
                tt = data_gi.type.tensor_type
                tt.ClearField("shape")
                tt.shape.dim.add().dim_value = total_target
        else:
            # Target like [-1, 100, 64]. If current input elem count is not divisible
            # by explicit dims product, force a compatible flat size.
            if total_target > 0 and (data_static_prod % total_target != 0):
                tt = data_gi.type.tensor_type
                tt.ClearField("shape")
                tt.shape.dim.add().dim_value = total_target
        return new_model

    if data_dyn_axes:
        if data_static_prod > 0 and total_target % data_static_prod == 0:
            # Dynamic dims absorb the batch factor regardless of whether shape has -1.
            batch = total_target // data_static_prod
            for ax in data_dyn_axes:
                data_gi.type.tensor_type.shape.dim[ax].ClearField("dim_param")
                data_gi.type.tensor_type.shape.dim[ax].dim_value = batch
            if has_minus_one:
                # Resolved data total = batch * data_static_prod = total_target.
                # Resolve -1 = total_target / total_target = 1.
                patched2 = [1 if d == -1 else d for d in target]
                flat_t2 = helper.make_tensor(
                    shape_name, onnx.TensorProto.INT64, [len(patched2)], patched2
                )
                for i, init in enumerate(new_model.graph.initializer):
                    if init.name == shape_name:
                        new_model.graph.initializer.__delitem__(i)
                        break
                new_model.graph.initializer.append(flat_t2)
        else:
            # Can't divide evenly: flatten data to [total_target].
            tt = data_gi.type.tensor_type
            tt.ClearField("shape")
            tt.shape.dim.add().dim_value = total_target
            if has_minus_one:
                # Data is [total_target] elements; resolve -1 = total_target / total_target = 1.
                patched2 = [1 if d == -1 else d for d in target]
                flat_t2 = helper.make_tensor(
                    shape_name, onnx.TensorProto.INT64, [len(patched2)], patched2
                )
                for i, init in enumerate(new_model.graph.initializer):
                    if init.name == shape_name:
                        new_model.graph.initializer.__delitem__(i)
                        break
                new_model.graph.initializer.append(flat_t2)

    return new_model


def _fixup_expand_shape_input(new_model: onnx.ModelProto) -> onnx.ModelProto:
    """For single-Expand sub-models, bake dynamic shape input as initializer.

    Expand's second input should be a 1-D int64 shape tensor. If left as a
    dynamic graph input, synthetic feeds may become invalid (e.g. zeros),
    causing runtime failures. We derive a stable shape from data input dims.
    """
    if len(new_model.graph.node) != 1:
        return new_model
    node = new_model.graph.node[0]
    if node.op_type != "Expand" or len(node.input) < 2:
        return new_model

    data_name = node.input[0]
    shape_name = node.input[1]

    init_map = {x.name: x for x in new_model.graph.initializer}
    input_map = {gi.name: gi for gi in new_model.graph.input}

    if shape_name in init_map:
        return new_model
    if shape_name not in input_map or data_name not in input_map:
        return new_model

    data_gi = input_map[data_name]
    tt = data_gi.type.tensor_type
    target: List[int] = []
    if tt.HasField("shape"):
        for dim in tt.shape.dim:
            if dim.dim_value > 0:
                target.append(int(dim.dim_value))
            else:
                target.append(1)
    if not target:
        target = [1]

    t = helper.make_tensor(shape_name, onnx.TensorProto.INT64, [len(target)], target)
    new_model.graph.initializer.append(t)

    for i, gi in enumerate(new_model.graph.input):
        if gi.name == shape_name:
            new_model.graph.input.__delitem__(i)
            break

    return new_model


def _trace_size_bytes(trace_dir: Path) -> int:
    total = 0
    for p in trace_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _run_cmd(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-dir", default="./per_op_drrio")
    ap.add_argument("--drrun", required=True, help="Path to DynamoRIO drrun")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-ops", type=int, default=0, help="0 means all ops")
    ap.add_argument("--start-op", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    # Run shape inference on the full model so that value_info contains
    # complete type/shape information for all intermediate tensors.  This
    # prevents the fallback path in _build_single_op_model from guessing
    # wrong elem_types (e.g. float instead of int64 for Constant outputs)
    # and wrong ranks (e.g. rank-1 for multi-dimensional tensors).
    try:
        model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    except Exception:
        pass

    out_dir = Path(args.out_dir).resolve()
    models_dir = out_dir / "models"
    traces_dir = out_dir / "traces"
    logs_dir = out_dir / "logs"
    run_json_dir = out_dir / "run_json"

    for d in [out_dir, models_dir, traces_dir, logs_dir, run_json_dir]:
        d.mkdir(parents=True, exist_ok=True)

    nodes = list(model.graph.node)
    start = max(0, args.start_op)
    stop = len(nodes)
    if args.max_ops > 0:
        stop = min(stop, start + args.max_ops)

    tasks: List[OpTask] = []
    for idx in range(start, stop):
        node = nodes[idx]
        sub = _build_single_op_model(model, idx)
        if sub is None:
            continue

        op_name = _safe_name(node.name or f"{node.op_type}_{idx}")
        file_base = f"{idx:05d}_{_safe_name(node.op_type)}_{op_name}"
        op_onnx = models_dir / f"{file_base}.onnx"
        trace_dir = traces_dir / file_base
        run_json = run_json_dir / f"{file_base}.json"
        log_path = logs_dir / f"{file_base}.log"

        onnx.save(sub, str(op_onnx))
        tasks.append(
            OpTask(
                idx=idx,
                op_type=node.op_type,
                node_name=node.name,
                onnx_path=op_onnx,
                trace_dir=trace_dir,
                run_json=run_json,
                log_path=log_path,
            )
        )

    summary_rows = []
    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "single_op_runner.py"

    for t in tasks:
        t.trace_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.drrun,
            "-t",
            "drmemtrace",
            "-offline",
            "-outdir",
            str(t.trace_dir),
            "--",
            args.python,
            str(runner),
            "--onnx",
            str(t.onnx_path),
            "--provider",
            args.provider,
            "--batch-size",
            str(args.batch_size),
            "--warmup",
            str(args.warmup),
            "--runs",
            str(args.runs),
            "--seed",
            str(args.seed),
            "--out-json",
            str(t.run_json),
        ]

        rc = _run_cmd(cmd, t.log_path, env=os.environ.copy())

        run_info = {}
        if t.run_json.exists():
            run_info = json.loads(t.run_json.read_text(encoding="utf-8"))

        summary_rows.append(
            {
                "op_idx": t.idx,
                "op_type": t.op_type,
                "node_name": t.node_name,
                "onnx": str(t.onnx_path),
                "trace_dir": str(t.trace_dir),
                "trace_bytes": _trace_size_bytes(t.trace_dir),
                "run_json": str(t.run_json),
                "log": str(t.log_path),
                "exit_code": rc,
                "latency_avg_ms": run_info.get("latency_ms", {}).get("avg", None),
            }
        )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "op_idx",
                "op_type",
                "node_name",
                "onnx",
                "trace_dir",
                "trace_bytes",
                "run_json",
                "log",
                "exit_code",
                "latency_avg_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    stats = {
        "onnx": str(Path(args.onnx).resolve()),
        "total_ops": len(nodes),
        "traced_ops": len(summary_rows),
        "csv": str(csv_path),
        "success_ops": sum(1 for r in summary_rows if r["exit_code"] == 0),
        "failed_ops": sum(1 for r in summary_rows if r["exit_code"] != 0),
        "total_trace_bytes": sum(int(r["trace_bytes"]) for r in summary_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
