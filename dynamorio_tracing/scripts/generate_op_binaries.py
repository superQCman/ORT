#!/usr/bin/env python3
"""
Generate one standalone ORT C executable per ONNX operator.

Workflow:
1. Reuse single-op ONNX extraction logic from ort_per_op_trace.py.
2. Emit one C source per op that embeds the ONNX bytes and runs it once.
3. Emit a CMakeLists.txt that builds all op binaries.

This is useful for gem5 simulation where each operator should be a dedicated binary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import onnx
from onnx import shape_inference

from ort_per_op_trace import (
    _build_single_op_model,
    _load_runtime_shape_overrides,
    _safe_name,
)


def _c_ident(s: str) -> str:
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"):
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out).strip("_")
    if not ident:
        ident = "x"
    if ident[0].isdigit():
        ident = "_" + ident
    return ident


def _onnx_elem_to_ort_enum(elem_type: int) -> str:
    mapping = {
        onnx.TensorProto.FLOAT: "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
        onnx.TensorProto.UINT8: "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8",
        onnx.TensorProto.INT8: "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8",
        onnx.TensorProto.UINT16: "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16",
        onnx.TensorProto.INT16: "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16",
        onnx.TensorProto.INT32: "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32",
        onnx.TensorProto.INT64: "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64",
        onnx.TensorProto.BOOL: "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL",
        onnx.TensorProto.FLOAT16: "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16",
        onnx.TensorProto.DOUBLE: "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE",
        onnx.TensorProto.UINT32: "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32",
        onnx.TensorProto.UINT64: "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64",
    }
    return mapping.get(elem_type, "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT")


def _c_str_literal(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


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
    sub_model: onnx.ModelProto,
    input_name: str,
    batch_size: int,
    num_indices_per_lookup: int,
) -> List[int]:
    if len(sub_model.graph.node) != 1:
        return []

    node = sub_model.graph.node[0]
    if node.op_type != "Reshape" or not node.input or input_name != node.input[0]:
        return []

    output_name = node.output[0] if node.output else ""
    output_vi = next((out for out in sub_model.graph.output if out.name == output_name), None)
    if output_vi is None:
        return []

    output_shape = _resolve_value_info_shape(
        output_vi, batch_size, num_indices_per_lookup
    )
    if not output_shape:
        return []

    total = 1
    for dim in output_shape:
        total *= dim
    if len(output_shape) >= 2 and output_shape[-1] > 0 and total % output_shape[-1] == 0:
        return [total // output_shape[-1], output_shape[-1]]
    return [total]


def _shape_tensor_values_for_single_node(
    sub_model: onnx.ModelProto,
    input_name: str,
    batch_size: int,
    num_indices_per_lookup: int,
) -> List[int] | None:
    if len(sub_model.graph.node) != 1:
        return None

    node = sub_model.graph.node[0]
    if len(node.input) < 2 or input_name != node.input[1]:
        return None
    if node.op_type not in ("Reshape", "Expand"):
        return None

    output_name = node.output[0] if node.output else ""
    output_vi = next((out for out in sub_model.graph.output if out.name == output_name), None)
    if output_vi is None:
        return None

    return _resolve_value_info_shape(output_vi, batch_size, num_indices_per_lookup)


def _input_infos(sub_model: onnx.ModelProto, batch_size: int, num_indices_per_lookup: int):
    init_names = {x.name for x in sub_model.graph.initializer}
    infos = []
    for inp in sub_model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        elem_type = tt.elem_type if tt.HasField("elem_type") else onnx.TensorProto.FLOAT

        dims = _resolve_value_info_shape(inp, batch_size, num_indices_per_lookup)
        if not dims:
            dims = _fallback_shape_for_single_node_input(
                sub_model, inp.name, batch_size, num_indices_per_lookup
            )
        literal_values = _shape_tensor_values_for_single_node(
            sub_model, inp.name, batch_size, num_indices_per_lookup
        )

        infos.append(
            {
                "name": inp.name,
                "elem_type": int(elem_type),
                "dims": dims,
                "literal_values": literal_values,
            }
        )
    return infos


def _output_names(sub_model: onnx.ModelProto):
    outs = []
    for out in sub_model.graph.output:
        outs.append(out.name)
    return outs


def _bytes_to_c_array(data: bytes, indent: str = "    ", width: int = 12) -> str:
    vals = [f"0x{b:02x}" for b in data]
    lines = []
    for i in range(0, len(vals), width):
        chunk = vals[i : i + width]
        suffix = "," if (i + width) < len(vals) else ""
        lines.append(indent + ", ".join(chunk) + suffix)
    return "\n".join(lines)


def _gen_c_source(
    file_base: str,
    op_idx: int,
    op_type: str,
    node_name: str,
    model_bytes: bytes,
    input_infos: List[Dict],
    output_names: List[str],
) -> str:
    model_var = _c_ident(f"model_{file_base}")
    input_count = len(input_infos)
    output_count = len(output_names)

    safe_op_type = _c_str_literal(op_type)
    safe_node_name = _c_str_literal(node_name or "")

    input_name_arr = ",\n".join(f'    "{_c_str_literal(x["name"])}"' for x in input_infos) or ""
    output_name_arr = ",\n".join(f'    "{_c_str_literal(x)}"' for x in output_names) or ""

    dims_blocks = []
    create_input_blocks = []
    release_input_blocks = []
    input_literal_blocks = []

    for i, info in enumerate(input_infos):
        dims = info["dims"]
        dims_name = f"input_{i}_dims"
        n_elem_expr = "1"
        if dims:
            n_elem_expr = " * ".join(str(int(d)) for d in dims)
        dims_literal = ", ".join(str(int(d)) for d in dims)
        dims_decl = f"static const int64_t {dims_name}[] = {{{dims_literal}}};" if dims else ""
        dims_blocks.append(dims_decl)

        literal_name = f"input_{i}_literal"
        literal_values = info.get("literal_values")
        if literal_values is not None:
            literal_vals = ", ".join(str(int(v)) for v in literal_values)
            input_literal_blocks.append(
                f"static const int64_t {literal_name}[] = {{{literal_vals}}};"
            )
        else:
            literal_name = ""

        ort_enum = _onnx_elem_to_ort_enum(int(info["elem_type"]))
        literal_init = ""
        if literal_values is not None:
            literal_init = f"""
    if (input_{i}_elem_count != (sizeof({literal_name}) / sizeof({literal_name}[0]))) {{
        fprintf(stderr, "input {i} literal size mismatch\\n");
        goto cleanup;
    }}
    memcpy(input_bufs[{i}], {literal_name}, sizeof({literal_name}));
"""
        create_input_blocks.append(
            f"""
    size_t input_{i}_elem_count = (size_t)({n_elem_expr});
    size_t input_{i}_bytes = input_{i}_elem_count * ort_elem_size({ort_enum});
    input_bufs[{i}] = calloc(input_{i}_bytes ? input_{i}_bytes : 1, 1);
    if (!input_bufs[{i}]) {{
        fprintf(stderr, "calloc failed for input {i}\\n");
        goto cleanup;
    }}
{literal_init}
    ORT_OK(ort->CreateTensorWithDataAsOrtValue(
        cpu_mem,
        input_bufs[{i}],
        input_{i}_bytes,
        {dims_name if dims else 'NULL'},
        {len(dims)},
        {ort_enum},
        &input_vals[{i}]
    ));
"""
        )
        release_input_blocks.append(
            f"""
    if (input_vals[{i}]) ort->ReleaseValue(input_vals[{i}]);
    if (input_bufs[{i}]) free(input_bufs[{i}]);
"""
        )

    if input_count > 0:
        input_names_decl = f"""static const char* kInputNames[] = {{
{input_name_arr}
}};
"""
    else:
        input_names_decl = ""

    if output_count > 0:
        output_names_decl = f"""static const char* kOutputNames[] = {{
{output_name_arr}
}};
"""
    else:
        output_names_decl = ""

    c = f"""#include <onnxruntime_c_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char {model_var}[] = {{
{_bytes_to_c_array(model_bytes)}
}};
static const size_t {model_var}_len = sizeof({model_var});

{input_names_decl}
{output_names_decl}

{chr(10).join(x for x in dims_blocks if x)}
{chr(10).join(input_literal_blocks)}

static size_t ort_elem_size(ONNXTensorElementDataType t) {{
    switch (t) {{
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return 8;
        default: return 4;
    }}
}}

#define ORT_OK(stmt) \\
    do {{ \\
        OrtStatus* _st = (stmt); \\
        if (_st) {{ \\
            fprintf(stderr, "ORT error: %s\\n", ort->GetErrorMessage(_st)); \\
            ort->ReleaseStatus(_st); \\
            goto cleanup; \\
        }} \\
    }} while (0)

int main(int argc, char** argv) {{
    int intra_threads = 1;
    int inter_threads = 1;
    for (int i = 1; i < argc; i++) {{
        if (strcmp(argv[i], "--intra-threads") == 0 && i + 1 < argc) {{
            intra_threads = atoi(argv[++i]);
        }} else if (strcmp(argv[i], "--inter-threads") == 0 && i + 1 < argc) {{
            inter_threads = atoi(argv[++i]);
        }}
    }}

    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;
    OrtSessionOptions* sess_opts = NULL;
    OrtSession* sess = NULL;
    OrtMemoryInfo* cpu_mem = NULL;

    OrtValue* input_vals[{input_count if input_count else 1}] = {{0}};
    void* input_bufs[{input_count if input_count else 1}] = {{0}};
    OrtValue* output_vals[{output_count if output_count else 1}] = {{0}};
    const char* const* input_names = {"kInputNames" if input_count > 0 else "NULL"};
    const char* const* output_names = {"kOutputNames" if output_count > 0 else "NULL"};

    int rc = 1;

    ORT_OK(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "op_runner", &env));
    ORT_OK(ort->CreateSessionOptions(&sess_opts));
    ORT_OK(ort->SetIntraOpNumThreads(sess_opts, intra_threads));
    ORT_OK(ort->SetInterOpNumThreads(sess_opts, inter_threads));
    ORT_OK(ort->CreateSessionFromArray(env, {model_var}, {model_var}_len, sess_opts, &sess));
    ORT_OK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_mem));

{''.join(create_input_blocks)}

    ORT_OK(ort->Run(
        sess,
        NULL,
        input_names,
        (const OrtValue* const*)input_vals,
        {input_count},
        output_names,
        {output_count},
        output_vals
    ));

    printf("OK op_idx={op_idx} op_type={safe_op_type} node={safe_node_name}\\n");
    rc = 0;

cleanup:
"""

    for i in range(output_count):
        c += f"    if (output_vals[{i}]) ort->ReleaseValue(output_vals[{i}]);\n"

    c += "\n"
    c += "".join(release_input_blocks)

    c += """
    if (cpu_mem) ort->ReleaseMemoryInfo(cpu_mem);
    if (sess) ort->ReleaseSession(sess);
    if (sess_opts) ort->ReleaseSessionOptions(sess_opts);
    if (env) ort->ReleaseEnv(env);
    return rc;
}
"""

    return c


def _gen_cmakelists(src_names: List[str]) -> str:
    add_targets = []
    for src in src_names:
        exe = Path(src).stem
        add_targets.append(f"add_executable({exe} src/{src})")
        add_targets.append(f"target_link_libraries({exe} PRIVATE onnxruntime)")

    return f"""cmake_minimum_required(VERSION 3.16)
project(ort_per_op_binaries C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Expect user to pass ORT_ROOT that contains include/ and lib/.
set(ORT_ROOT "" CACHE PATH "Path to ONNX Runtime install root")
if(NOT ORT_ROOT)
  message(FATAL_ERROR "Please provide -DORT_ROOT=/path/to/onnxruntime")
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_c_api.h
    HINTS ${{ORT_ROOT}}/include
    PATH_SUFFIXES . onnxruntime/core/session
    NO_DEFAULT_PATH)

find_library(ORT_LIBRARY onnxruntime
    HINTS ${{ORT_ROOT}}/lib ${{ORT_ROOT}}/lib64
    NO_DEFAULT_PATH)

if(NOT ORT_LIBRARY)
    find_file(ORT_LIBRARY_FILE
        NAMES libonnxruntime.so libonnxruntime.so.1 libonnxruntime.so.1.23.2
        HINTS ${{ORT_ROOT}}/lib ${{ORT_ROOT}}/lib64
        NO_DEFAULT_PATH)
    if(ORT_LIBRARY_FILE)
        set(ORT_LIBRARY "${{ORT_LIBRARY_FILE}}")
    endif()
endif()

if(NOT ORT_INCLUDE_DIR)
    message(FATAL_ERROR "Cannot find onnxruntime_c_api.h under ORT_ROOT/include (or include/onnxruntime/core/session)")
endif()
if(NOT ORT_LIBRARY)
  message(FATAL_ERROR "Cannot find libonnxruntime under ORT_ROOT/lib or lib64")
endif()

    add_library(onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${{ORT_LIBRARY}}"
  INTERFACE_INCLUDE_DIRECTORIES "${{ORT_INCLUDE_DIR}}")

{chr(10).join(add_targets)}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to source ONNX model")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--start-op", type=int, default=0)
    ap.add_argument("--max-ops", type=int, default=0, help="0 means all")
    ap.add_argument("--batch-size", type=int, default=1, help="Used for dynamic dims")
    ap.add_argument("--num-indices-per-lookup", type=int, default=0,
                    help="Concrete bag size for resolving dynamic indices_* dims")
    ap.add_argument("--allow-shape-fixups", action="store_true",
                    help="Allow legacy single-op shape coercions that may change input/output dims")
    ap.add_argument("--shape-csv", default="",
                    help="Optional runtime op-shape CSV used to restore missing intermediate tensor shapes")
    ap.add_argument("--intra-threads", type=int, default=1,
                    help="Default ORT intra-op thread count embedded in manifest (runtime override via CLI arg)")
    ap.add_argument("--inter-threads", type=int, default=1,
                    help="Default ORT inter-op thread count embedded in manifest (runtime override via CLI arg)")
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    try:
        model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
    except Exception:
        pass
    shape_overrides = _load_runtime_shape_overrides(args.shape_csv) if args.shape_csv else {}

    out_dir = Path(args.out_dir).resolve()
    models_dir = out_dir / "models"
    src_dir = out_dir / "src"
    for d in [out_dir, models_dir, src_dir]:
        d.mkdir(parents=True, exist_ok=True)

    nodes = list(model.graph.node)
    start = max(0, args.start_op)
    stop = len(nodes)
    if args.max_ops > 0:
        stop = min(stop, start + args.max_ops)

    manifest = []
    src_names = []

    for idx in range(start, stop):
        node = nodes[idx]
        sub = _build_single_op_model(
            model,
            idx,
            allow_shape_fixups=args.allow_shape_fixups,
            shape_overrides=shape_overrides,
        )
        if sub is None:
            continue

        op_name = _safe_name(node.name or f"{node.op_type}_{idx}")
        file_base = f"{idx:05d}_{_safe_name(node.op_type)}_{op_name}"

        op_onnx = models_dir / f"{file_base}.onnx"
        onnx.save(sub, str(op_onnx))

        model_bytes = op_onnx.read_bytes()
        inputs = _input_infos(sub, args.batch_size, args.num_indices_per_lookup)
        outputs = _output_names(sub)

        c_src = _gen_c_source(
            file_base=file_base,
            op_idx=idx,
            op_type=node.op_type,
            node_name=node.name,
            model_bytes=model_bytes,
            input_infos=inputs,
            output_names=outputs,
        )
        c_name = f"{file_base}.c"
        (src_dir / c_name).write_text(c_src, encoding="utf-8")
        src_names.append(c_name)

        manifest.append(
            {
                "op_idx": idx,
                "op_type": node.op_type,
                "node_name": node.name,
                "onnx": str(op_onnx),
                "c_source": str(src_dir / c_name),
                "exe_name": Path(c_name).stem,
                "batch_size": args.batch_size,
                "num_indices_per_lookup": args.num_indices_per_lookup,
                "allow_shape_fixups": args.allow_shape_fixups,
                "shape_csv": args.shape_csv,
                "default_intra_threads": args.intra_threads,
                "default_inter_threads": args.inter_threads,
            }
        )

    cmake_txt = _gen_cmakelists(src_names)
    (out_dir / "CMakeLists.txt").write_text(cmake_txt, encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "onnx": str(Path(args.onnx).resolve()),
        "total_ops": len(nodes),
        "generated_ops": len(manifest),
        "out_dir": str(out_dir),
        "cmake": str(out_dir / "CMakeLists.txt"),
        "manifest": str(out_dir / "manifest.json"),
        "batch_size": args.batch_size,
        "num_indices_per_lookup": args.num_indices_per_lookup,
        "allow_shape_fixups": args.allow_shape_fixups,
        "shape_csv": args.shape_csv,
        "default_intra_threads": args.intra_threads,
        "default_inter_threads": args.inter_threads,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
