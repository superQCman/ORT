#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRACE_DIR=${TRACE_DIR:-/data/qc/dlrm/ORT/dynamorio_tracing}
BUILD_ROOT=${BUILD_ROOT:-$TRACE_DIR/out_per_op_bins_sweep}
OUT_BASE=${OUT_BASE:-$SCRIPT_DIR/gem5_runs}
RUNNER=${RUNNER:-$TRACE_DIR/run_gem5_all_ops.sh}

COMBO_FILTER=${COMBO_FILTER:-}
START_COMBO_INDEX=${START_COMBO_INDEX:-0}
MAX_COMBOS=${MAX_COMBOS:-0}

# PARALLEL_JOBS=${PARALLEL_JOBS:-1}
START_IDX=${START_IDX:-0}
MAX_OPS=${MAX_OPS:-0}
RESUME=${RESUME:-1}

SIM_NUM_CPUS=${SIM_NUM_CPUS:-48}
# This should match the traced operator thread count.  In the current
# features_selected sweep, num_threads is consistently 4 across combos.
BMK_THREADS=${BMK_THREADS:-4}
BMK_CPU_AFFINITY=${BMK_CPU_AFFINITY:-}
CPU_CLOCK=${CPU_CLOCK:-3GHz}
MEM_SIZE0=${MEM_SIZE0:-32GiB}
MEM_SIZE1=${MEM_SIZE1:-32GiB}
LOCAL_MEM_DELAY=${LOCAL_MEM_DELAY:-20ns}
REMOTE_MEM_DELAY=${REMOTE_MEM_DELAY:-40ns}

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: gem5 runner not executable: $RUNNER"
  exit 1
fi
if [[ ! -d "$BUILD_ROOT" ]]; then
  echo "ERROR: build root not found: $BUILD_ROOT"
  exit 1
fi

mapfile -t combo_dirs < <(find "$BUILD_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'bs*_nip*' | sort)
if [[ -n "$COMBO_FILTER" ]]; then
  filtered=()
  for combo_dir in "${combo_dirs[@]}"; do
    combo=$(basename "$combo_dir")
    if [[ "$combo" == *"$COMBO_FILTER"* ]]; then
      filtered+=("$combo_dir")
    fi
  done
  combo_dirs=("${filtered[@]}")
fi

if [[ ${#combo_dirs[@]} -eq 0 ]]; then
  echo "ERROR: no combo build directories found under $BUILD_ROOT"
  exit 1
fi

count=0
for ((idx=START_COMBO_INDEX; idx<${#combo_dirs[@]}; idx++)); do
  if (( MAX_COMBOS > 0 && count >= MAX_COMBOS )); then
    break
  fi

  combo_dir=${combo_dirs[$idx]}
  combo=$(basename "$combo_dir")
  bin_dir="$combo_dir/build"
  out_root="$OUT_BASE/$combo"

  if [[ ! -d "$bin_dir" ]]; then
    echo "[SKIP] $combo missing build dir: $bin_dir"
    continue
  fi

  echo "[RUN ] combo=$combo"
  mkdir -p "$out_root"

  BIN_DIR="$bin_dir" \
  OUT_ROOT="$out_root" \
  START_IDX="$START_IDX" \
  MAX_OPS="$MAX_OPS" \
  RESUME="$RESUME" \
  SIM_NUM_CPUS="$SIM_NUM_CPUS" \
  BMK_THREADS="$BMK_THREADS" \
  BMK_CPU_AFFINITY="$BMK_CPU_AFFINITY" \
  CPU_CLOCK="$CPU_CLOCK" \
  MEM_SIZE0="$MEM_SIZE0" \
  MEM_SIZE1="$MEM_SIZE1" \
  LOCAL_MEM_DELAY="$LOCAL_MEM_DELAY" \
  REMOTE_MEM_DELAY="$REMOTE_MEM_DELAY" \
  bash "$RUNNER"

  ((count += 1))
done

echo "Completed combos: $count"
echo "gem5 outputs under: $OUT_BASE"
