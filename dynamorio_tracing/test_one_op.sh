#!/bin/bash
# Test raw2trace on one operator and show all errors

DRRUN=/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun
TRACE_DIR=/data/qc/dlrm/ORT/dynamorio_tracing/drrio_traces_per_op_multithread/00001_Constant_emb_l0_ReshapeShape/drmemtrace.ld-linux-aarch64.so.1.3841282.9063.dir

# Find the drmemtrace.*.dir
DRMEM_DIR=$(find "$TRACE_DIR" -maxdepth 1 -name "drmemtrace.*.dir" -type d 2>/dev/null | head -1)
if [ -z "$DRMEM_DIR" ]; then
    echo "ERROR: no drmemtrace dir found in $TRACE_DIR"
    exit 1
fi
echo "drmemtrace dir: $DRMEM_DIR"

# Step 1: remove existing converted trace
echo "=== Removing existing trace/ dir ==="
rm -rf "$DRMEM_DIR/trace"

# Step 2: try basic_counts (no -use_physical)
echo "=== Running basic_counts ==="
"$DRRUN" -t drmemtrace -indir "$DRMEM_DIR" -tool basic_counts 2>&1
