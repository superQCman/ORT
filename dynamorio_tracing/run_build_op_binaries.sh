#!/usr/bin/env bash
set -euo pipefail

# Build one standalone executable per ONNX op for gem5 simulation.
#
# Example:
#   bash run_build_op_binaries.sh \
#     /data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx \
#     /path/to/onnxruntime/install \
#     /data/qc/dlrm/ORT/dynamorio_tracing/out_per_op_bins

ONNX_PATH=${1:-/data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx.cann_patched.onnx.loop_to_gather.onnx}
ORT_ROOT=${2:-$CONDA_PREFIX}
OUT_DIR=${3:-/data/qc/dlrm/ORT/dynamorio_tracing/out_per_op_bins}
MAX_OPS=${MAX_OPS:-0}
START_OP=${START_OP:-0}
BATCH_SIZE=${BATCH_SIZE:-1}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [[ -z "${ORT_ROOT:-}" ]]; then
  echo "ERROR: ORT_ROOT is empty. Activate conda env first or pass arg2 as ORT install root."
  exit 1
fi

_has_header=0
if [[ -f "$ORT_ROOT/include/onnxruntime_c_api.h" ]]; then
  _has_header=1
elif [[ -f "$ORT_ROOT/include/onnxruntime/core/session/onnxruntime_c_api.h" ]]; then
  _has_header=1
fi

_ort_lib=""
for cand in \
  "$ORT_ROOT/lib/libonnxruntime.so" \
  "$ORT_ROOT/lib64/libonnxruntime.so" \
  "$ORT_ROOT/lib/libonnxruntime.so."* \
  "$ORT_ROOT/lib64/libonnxruntime.so."*; do
  if [[ -f "$cand" ]]; then
    _ort_lib="$cand"
    break
  fi
done

if [[ -z "$_ort_lib" ]] && command -v python3 >/dev/null 2>&1; then
  _py_ort_lib=$(python3 - <<'PY'
import glob
import os
try:
    import onnxruntime as ort
except Exception:
    print("")
    raise SystemExit(0)

base = os.path.join(os.path.dirname(ort.__file__), "capi")
cands = sorted(glob.glob(os.path.join(base, "libonnxruntime.so*")))
print(cands[0] if cands else "")
PY
)
  if [[ -n "$_py_ort_lib" && -f "$_py_ort_lib" ]]; then
    _ort_lib="$_py_ort_lib"
  fi
fi

# If only python wheel lib exists (no dev headers), bootstrap a minimal ORT SDK.
if [[ $_has_header -eq 0 ]]; then
  if [[ -z "$_ort_lib" ]]; then
    echo "ERROR: Cannot find ORT headers or libonnxruntime.so* from ORT_ROOT=$ORT_ROOT"
    exit 1
  fi

  ORT_SDK_DIR="$OUT_DIR/ort_sdk"
  mkdir -p "$ORT_SDK_DIR/include" "$ORT_SDK_DIR/lib"

  if [[ ! -f "$ORT_SDK_DIR/include/onnxruntime_c_api.h" ]]; then
    curl -L -o "$ORT_SDK_DIR/include/onnxruntime_c_api.h" \
      https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/include/onnxruntime/core/session/onnxruntime_c_api.h
    curl -L -o "$ORT_SDK_DIR/include/onnxruntime_ep_c_api.h" \
      https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/include/onnxruntime/core/session/onnxruntime_ep_c_api.h
  fi

  cp -f "$_ort_lib" "$ORT_SDK_DIR/lib/"
  _copied_lib="$ORT_SDK_DIR/lib/$(basename "$_ort_lib")"
  ln -sfn "$(basename "$_copied_lib")" "$ORT_SDK_DIR/lib/libonnxruntime.so"

  ORT_ROOT="$ORT_SDK_DIR"
  echo "INFO: Bootstrapped ORT SDK at $ORT_ROOT"
fi

python3 "$SCRIPT_DIR/scripts/generate_op_binaries.py" \
  --onnx "$ONNX_PATH" \
  --out-dir "$OUT_DIR" \
  --start-op "$START_OP" \
  --max-ops "$MAX_OPS" \
  --batch-size "$BATCH_SIZE"

cmake -S "$OUT_DIR" -B "$OUT_DIR/build" -DORT_ROOT="$ORT_ROOT"
cmake --build "$OUT_DIR/build" -j"$(nproc)"

echo "Built binaries under: $OUT_DIR/build"
echo "You can list executables with: find $OUT_DIR/build -maxdepth 1 -type f -executable"
