#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash run_per_op_trace.sh \
#     /data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx \
#     /data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun

ONNX_PATH=${1:-/data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx.cann_patched.onnx.loop_to_gather.onnx}
DRRUN=${2:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun}
OUT_DIR=${3:-/data/qc/dlrm/ORT/dynamorio_tracing/out_per_op}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

python3 "$SCRIPT_DIR/scripts/ort_per_op_trace.py" \
  --onnx "$ONNX_PATH" \
  --drrun "$DRRUN" \
  --out-dir "$OUT_DIR" \
  --provider CPUExecutionProvider \
  --batch-size 1 \
  --warmup 1 \
  --runs 1

python3 "$SCRIPT_DIR/scripts/analyze_op_traces.py" \
  --summary-csv "$OUT_DIR/summary.csv" \
  --topk 20
