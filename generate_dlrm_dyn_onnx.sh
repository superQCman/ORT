#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

ASCEND_ENV_SH=${ASCEND_ENV_SH:-/data/qc/Ascend/ascend-toolkit/set_env.sh}
DEFAULT_PYTHON_BIN=/data/qc/anaconda3/envs/ort/bin/python
DEFAULT_PYTHON_SITE_PACKAGES=/data/qc/anaconda3/envs/ort/lib/python3.11/site-packages
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
    PYTHON_BIN="$DEFAULT_PYTHON_BIN"
  else
    PYTHON_BIN=python3
  fi
fi
PYTHON_SITE_PACKAGES=${PYTHON_SITE_PACKAGES:-$DEFAULT_PYTHON_SITE_PACKAGES}
USE_ASCEND_ENV=${USE_ASCEND_ENV:-0}

if [[ "$USE_ASCEND_ENV" == "1" && -f "$ASCEND_ENV_SH" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_ENV_SH"
fi

DLRM_SCRIPT=${DLRM_SCRIPT:-"$REPO_ROOT/dlrm_msprof.py"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$SCRIPT_DIR/dlrm_onnx_dyn"}
ONNX_NAME=${ONNX_NAME:-dlrm_s_pytorch.onnx}
VARIANT_FILTER=${VARIANT_FILTER:-}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
NUM_BATCHES=${NUM_BATCHES:-1}
NUM_INDICES_PER_LOOKUP=${NUM_INDICES_PER_LOOKUP:-100}
NUM_WORKERS=${NUM_WORKERS:-1}
DATA_GENERATION=${DATA_GENERATION:-random}
DATA_SIZE=${DATA_SIZE:-1}
SAVE_CSV=${SAVE_CSV:-0}

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$DLRM_SCRIPT" ]]; then
  echo "ERROR: dlrm_msprof.py not found: $DLRM_SCRIPT" >&2
  exit 1
fi

MANIFEST_CSV="$OUTPUT_ROOT/manifest.csv"
echo "variant,arch_embedding_size,arch_mlp_bot,arch_mlp_top,onnx_path" > "$MANIFEST_CSV"

# CONFIGS=(
#   "case_1|3200-3200-3200-3200-3200-3200-3200-3200|400-300-200|400-200-1"
#   "case_2|3200-3200-3200-3200-3200-3200-3200-3200|800-800-400|1600-800-1"
#   "case_3|6400-6400-6400-6400-6400-6400-6400-6400|400-300-200|1600-800-1"
#   "case_4|6400-6400-6400-6400-6400-6400-6400-6400|800-800-400|1600-800-1"
#   "case_5|12800-12800-12800-12800-12800-12800-12800-12800|400-300-200|1600-800-1"
#   "case_6|12800-12800-12800-12800-12800-12800-12800-12800|800-800-400|1600-800-1"
#   "case_7|25600-25600-25600-25600-25600-25600-25600-25600|400-300-200|1600-800-1"
#   "case_8|25600-25600-25600-25600-25600-25600-25600-25600|800-800-400|1600-800-1"
#   "case_9|51200-51200-51200-51200-51200-51200-51200-51200|400-300-200|1600-800-1"
#   "case_10|51200-51200-51200-51200-51200-51200-51200-51200|800-800-400|1600-800-1"
# )

CONFIGS=(
  "case_11|512000-512000-512000-512000-512000-512000-512000-512000|8000-8000-40000|1600-800-1"
  "case_12|512000-512000-512000-512000-512000-512000-512000-512000|16000-16000-40000|1600-800-1"
  "case_13|1024000-1024000-1024000-1024000-1024000-1024000-1024000-1024000|8000-8000-40000|16000-8000-1"
)

variant_enabled() {
  local variant="$1"
  if [[ -z "$VARIANT_FILTER" ]]; then
    return 0
  fi

  local needle
  IFS=',' read -ra needles <<< "$VARIANT_FILTER"
  for needle in "${needles[@]}"; do
    needle="${needle//[[:space:]]/}"
    [[ -n "$needle" && "$needle" == "$variant" ]] && return 0
  done
  return 1
}

echo "[GEN] output_root=$OUTPUT_ROOT"
echo "[GEN] mini_batch_size=$MINI_BATCH_SIZE num_batches=$NUM_BATCHES num_indices_per_lookup=$NUM_INDICES_PER_LOOKUP num_workers=$NUM_WORKERS"
echo "[GEN] data_generation=$DATA_GENERATION data_size=$DATA_SIZE"
echo "[GEN] python_bin=$PYTHON_BIN"
echo "[GEN] python_site_packages=$PYTHON_SITE_PACKAGES"
echo "[GEN] ascend_env=${USE_ASCEND_ENV}"
echo "[GEN] variant_filter=${VARIANT_FILTER:-<unset>}"

for spec in "${CONFIGS[@]}"; do
  IFS='|' read -r variant emb bot top <<< "$spec"
  if ! variant_enabled "$variant"; then
    echo "[GEN] skip variant=$variant (filtered out)"
    continue
  fi
  variant_dir="$OUTPUT_ROOT/$variant"
  onnx_path="$variant_dir/$ONNX_NAME"
  mkdir -p "$variant_dir"

  echo "=================================================================="
  echo "[GEN] variant=$variant"
  echo "  emb=$emb"
  echo "  bot=$bot"
  echo "  top=$top"
  echo "  onnx=$onnx_path"

  sparse_feature_size="${bot##*-}"
  echo "  sparse_feature_size=$sparse_feature_size"

  (
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT${PYTHON_SITE_PACKAGES:+:$PYTHON_SITE_PACKAGES}${PYTHONPATH:+:$PYTHONPATH}" \
    # TORCH_DEVICE_BACKEND_AUTOLOAD=1 \
    # DLRM_ENABLE_TORCH_NPU=1 \
    "$PYTHON_BIN" -S "$DLRM_SCRIPT" \
      --use-gpu \
      --num-threads=8 \
      --arch-sparse-feature-size "$sparse_feature_size" \
      --arch-embedding-size "$emb" \
      --arch-mlp-bot "$bot" \
      --arch-mlp-top "$top" \
      --mini-batch-size "$MINI_BATCH_SIZE" \
      --num-batches "$NUM_BATCHES" \
      --num-indices-per-lookup "$NUM_INDICES_PER_LOOKUP" \
      --num-workers "$NUM_WORKERS" \
      --data-size "$DATA_SIZE" \
      --data-generation "$DATA_GENERATION" \
      --save-onnx \
      --num-indices-per-lookup-fixed=True \
      --onnx-output-root "$OUTPUT_ROOT" \
      --onnx-output-subdir "$variant" \
      --onnx-output-name "$ONNX_NAME"
  )

  if [[ ! -f "$onnx_path" ]]; then
    echo "ERROR: expected ONNX not found: $onnx_path" >&2
    exit 1
  fi

  printf '%s,%s,%s,%s,%s\n' \
    "$variant" "$emb" "$bot" "$top" "$onnx_path" >> "$MANIFEST_CSV"
  echo "[GEN] saved: $onnx_path"
done

echo "[GEN] manifest: $MANIFEST_CSV"
