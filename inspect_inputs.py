#!/usr/bin/env python3
"""
快速分析 ONNX 模型需要的输入参数。

用法：
  python inspect_inputs.py model.onnx
  python inspect_inputs.py model.onnx --ort          # 同时用 ORT Session 解析
  python inspect_inputs.py model.onnx --ort --cann   # 用 CANN provider

输出：
  - 真正需要外部传入的输入（排除 initializer/权重）
  - 每个输入的 dtype / shape（静态 + 符号维）
  - ORT session 视角的输入（与 onnx 解析对比）
"""

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# ONNX 静态解析
# ──────────────────────────────────────────────────────────────
_ELEM_TYPE_NAME = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
    5: "int16", 6: "int32", 7: "int64", 8: "string",
    9: "bool", 10: "float16", 11: "float64", 12: "uint32",
    13: "uint64", 16: "bfloat16",
}


def _fmt_shape(shape_proto) -> str:
    if not shape_proto.HasField("shape") if hasattr(shape_proto, "HasField") else False:
        return "unknown"
    dims = []
    for d in shape_proto.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param or "?")
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]" if dims else "scalar"


def analyze_onnx(onnx_path: str) -> None:
    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        print("[ERROR] onnx 未安装：pip install onnx")
        return

    print(f"\n{'='*60}")
    print(f"  ONNX 静态解析: {onnx_path}")
    print(f"{'='*60}")

    model = onnx.load(onnx_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  [WARN] shape_inference 失败（{e}），继续...")

    graph = model.graph

    # opset
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        print(f"  opset: {domain} v{opset.version}")

    # initializer 名集合（权重，不需要外部传入）
    init_names = {init.name for init in graph.initializer}

    # 真正需要外部传入的输入
    real_inputs = [inp for inp in graph.input if inp.name not in init_names]

    print(f"\n  需要外部传入的输入（共 {len(real_inputs)} 个）：")
    print(f"  {'#':<4} {'名称':<30} {'dtype':<12} {'shape'}")
    print(f"  {'-'*4} {'-'*30} {'-'*12} {'-'*25}")
    for i, inp in enumerate(real_inputs):
        tt = inp.type.tensor_type
        dtype = _ELEM_TYPE_NAME.get(tt.elem_type, f"type_{tt.elem_type}")
        try:
            shape_str = _fmt_shape(tt)
        except Exception:
            shape_str = "unknown"
        print(f"  {i:<4} {inp.name:<30} {dtype:<12} {shape_str}")

    print(f"\n  权重/常量 initializer（不需要外部传入）: {len(init_names)} 个")

    # 输出
    print(f"\n  模型输出（共 {len(graph.output)} 个）：")
    for out in graph.output:
        tt = out.type.tensor_type
        dtype = _ELEM_TYPE_NAME.get(tt.elem_type, f"type_{tt.elem_type}")
        try:
            shape_str = _fmt_shape(tt)
        except Exception:
            shape_str = "unknown"
        print(f"    {out.name:<30} {dtype:<12} {shape_str}")

    # 节点统计
    from collections import Counter
    op_counter = Counter(n.op_type for n in graph.node)
    print(f"\n  算子统计（共 {len(graph.node)} 个节点，top-10）：")
    for op, cnt in op_counter.most_common(10):
        print(f"    {op:<30} {cnt:>4} 个")

    return real_inputs


# ──────────────────────────────────────────────────────────────
# ORT Session 解析
# ──────────────────────────────────────────────────────────────
def analyze_ort(onnx_path: str, use_cann: bool = False) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime 未安装")
        return

    print(f"\n{'='*60}")
    print(f"  ORT Session 解析（use_cann={use_cann}）")
    print(f"{'='*60}")
    print(f"  ORT 版本: {ort.__version__}")
    print(f"  可用 providers: {ort.get_available_providers()}")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    if use_cann and "CANNExecutionProvider" in ort.get_available_providers():
        providers = [("CANNExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"]
    else:
        if use_cann:
            print("  [WARN] CANNExecutionProvider 不可用，回退到 CPU")
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    except Exception as e:
        print(f"  [ERROR] Session 创建失败: {e}")
        return

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"\n  输入（共 {len(inputs)} 个）：")
    print(f"  {'#':<4} {'名称':<30} {'dtype':<25} {'shape'}")
    print(f"  {'-'*4} {'-'*30} {'-'*25} {'-'*25}")
    for i, inp in enumerate(inputs):
        print(f"  {i:<4} {inp.name:<30} {inp.type:<25} {inp.shape}")

    print(f"\n  输出（共 {len(outputs)} 个）：")
    for out in outputs:
        print(f"    {out.name:<30} {out.type:<25} {out.shape}")

    # 生成 feed_dict 示例代码
    print(f"\n  ── 生成 feed_dict 示例代码 ──")
    print("  import numpy as np")
    print("  batch_size = 32")
    print("  feed = {}")
    for inp in inputs:
        # 将动态维度替换为 batch_size 或具体示例值
        shape_parts = []
        for d in inp.shape:
            if isinstance(d, int) and d > 0:
                shape_parts.append(str(d))
            else:
                shape_parts.append("batch_size")
        shape_str = ", ".join(shape_parts)
        if "float" in inp.type:
            print(f'  feed["{inp.name}"] = np.zeros(({shape_str},), dtype=np.float32)')
        elif "int64" in inp.type:
            print(f'  feed["{inp.name}"] = np.zeros(({shape_str},), dtype=np.int64)')
        else:
            print(f'  feed["{inp.name}"] = np.zeros(({shape_str},))  # dtype={inp.type}')
    out_names = [f'"{o.name}"' for o in outputs]
    print(f'  results = session.run([{", ".join(out_names)}], feed)')


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="快速分析 ONNX 模型输入参数")
    parser.add_argument("onnx_path", help="ONNX 模型路径")
    parser.add_argument("--ort", action="store_true", help="同时用 ORT Session 解析")
    parser.add_argument("--cann", action="store_true", help="ORT 使用 CANN provider")
    args = parser.parse_args()

    if not Path(args.onnx_path).exists():
        print(f"[ERROR] 文件不存在: {args.onnx_path}")
        sys.exit(1)

    analyze_onnx(args.onnx_path)

    if args.ort:
        analyze_ort(args.onnx_path, use_cann=args.cann)


if __name__ == "__main__":
    main()
