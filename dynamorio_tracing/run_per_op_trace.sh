#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash run_per_op_trace.sh \
#     /data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx \
#     /data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun

ONNX_PATH=${1:-/data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx.cann_patched.onnx.loop_to_gather.onnx}
DRRUN=${2:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun}
OUT_DIR=${3:-/data/qc/dlrm/ORT/dynamorio_tracing/out_per_op_parallel_PHY}

# NUMA binding: set CPUNODEBIND/MEMBIND to a node number to enable numactl,
# or leave empty to disable. Defaults match run_ort.sh (node 0).
# Example: CPUNODEBIND=0 MEMBIND=0 bash run_per_op_trace.sh
CPUNODEBIND=${CPUNODEBIND:-0}
MEMBIND=${MEMBIND:-0}

# ORT thread counts for each single-op session (default: 1 thread each).
INTRA_THREADS=${INTRA_THREADS:-1}
INTER_THREADS=${INTER_THREADS:-1}
NUM_INDICES_PER_LOOKUP=${NUM_INDICES_PER_LOOKUP:-0}

# Build optional numactl flags.
NUMA_ARGS=()
if [[ -n "${CPUNODEBIND}" ]]; then
    NUMA_ARGS+=(--cpunodebind "${CPUNODEBIND}")
fi
if [[ -n "${MEMBIND}" ]]; then
    NUMA_ARGS+=(--membind "${MEMBIND}")
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DEFAULT_SHAPE_CSV="$SCRIPT_DIR/../op_shapes.csv"
SHAPE_CSV=${SHAPE_CSV:-}
if [[ -z "$SHAPE_CSV" && -f "$DEFAULT_SHAPE_CSV" ]]; then
    SHAPE_CSV="$DEFAULT_SHAPE_CSV"
fi

TRACE_ARGS=(
  --python "/data/qc/anaconda3/envs/ort/bin/python3"
  --onnx "$ONNX_PATH"
  --drrun "$DRRUN"
  --out-dir "$OUT_DIR"
  --provider CPUExecutionProvider
  --batch-size 1
  --num-indices-per-lookup "$NUM_INDICES_PER_LOOKUP"
  --warmup 1
  --runs 1
  --intra-threads "$INTRA_THREADS"
  --inter-threads "$INTER_THREADS"
  --use-physical
)
if [[ -n "$SHAPE_CSV" ]]; then
    TRACE_ARGS+=(--shape-csv "$SHAPE_CSV")
fi
python3 "$SCRIPT_DIR/scripts/ort_per_op_trace.py" "${TRACE_ARGS[@]}" "${NUMA_ARGS[@]}"

python3 "$SCRIPT_DIR/scripts/analyze_op_traces.py" \
  --summary-csv "$OUT_DIR/summary.csv" \
  --topk 20
