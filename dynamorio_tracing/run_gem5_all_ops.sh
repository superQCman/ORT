#!/usr/bin/env bash
set -u

# Batch simulate all per-op binaries with gem5 (SE mode).
#
# Default usage:
#   bash run_gem5_all_ops.sh
#
# Common overrides:
#   GEM5_BIN=/data/qc/simulator/gem5/build/ALL/gem5.opt \
#   GEM5_CONFIG=/data/qc/dlrm/ops_profile/concorde/simulatoin/kunpeng920.py \
#   START_IDX=0 MAX_OPS=10 RESUME=1 SIM_NUM_CPUS=8 BMK_THREADS=8 BMK_CPU_AFFINITY=0-7 \
#   bash run_gem5_all_ops.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

GEM5_BIN=${GEM5_BIN:-/data/qc/simulator/gem5/build/ALL/gem5.opt}
GEM5_CONFIG=${GEM5_CONFIG:-/data/qc/dlrm/ops_profile/concorde/simulatoin/kunpeng920.py}
BIN_DIR=${BIN_DIR:-$SCRIPT_DIR/out_per_op_bins/build}
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/m5out_per_op_bins}

# Support segmented execution.
START_IDX=${START_IDX:-0}
MAX_OPS=${MAX_OPS:-0}    # 0 means all
RESUME=${RESUME:-1}      # 1: skip existing finished outdir

# Kunpeng config knobs.
CPU_CLOCK=${CPU_CLOCK:-2.6GHz}
MEM_SIZE0=${MEM_SIZE0:-8GiB}
MEM_SIZE1=${MEM_SIZE1:-8GiB}
LOCAL_MEM_DELAY=${LOCAL_MEM_DELAY:-20ns}
REMOTE_MEM_DELAY=${REMOTE_MEM_DELAY:-40ns}

# Single-simulation threading/affinity knobs.
# SIM_NUM_CPUS: number of simulated cores (used when BMK_CPU_AFFINITY is empty).
# BMK_THREADS: exported as OMP/MKL/OpenBLAS thread counts in SE workload env.
# BMK_CPU_AFFINITY: simulated core set, e.g. 0-7 or 0-3,8-11.
SIM_NUM_CPUS=${SIM_NUM_CPUS:-48}
BMK_THREADS=${BMK_THREADS:-0}
BMK_CPU_AFFINITY=${BMK_CPU_AFFINITY:-}

# Runtime linker helper for dynamically linked op binaries.
# If empty, run binary directly.
ORT_LIB_DIR=${ORT_LIB_DIR:-$SCRIPT_DIR/out_per_op_bins/ort_sdk/lib}
TARGET_LOADER=${TARGET_LOADER:-/lib/ld-linux-aarch64.so.1}

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "ERROR: GEM5_BIN is not executable: $GEM5_BIN"
  exit 1
fi
if [[ ! -f "$GEM5_CONFIG" ]]; then
  echo "ERROR: GEM5_CONFIG not found: $GEM5_CONFIG"
  exit 1
fi
if [[ ! -d "$BIN_DIR" ]]; then
  echo "ERROR: BIN_DIR not found: $BIN_DIR"
  exit 1
fi

# Ensure the SONAME symlink exists so the in-simulation dynamic linker can find the library.
if [[ -f "$ORT_LIB_DIR/libonnxruntime.so.1.23.2" && ! -e "$ORT_LIB_DIR/libonnxruntime.so.1" ]]; then
  ln -sfn libonnxruntime.so.1.23.2 "$ORT_LIB_DIR/libonnxruntime.so.1"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
FAIL_LIST="$OUT_ROOT/failed_ops.txt"

echo "op_idx,op_name,exit_code,outdir" > "$SUMMARY_CSV"
: > "$FAIL_LIST"

mapfile -t OP_BINS < <(find "$BIN_DIR" -maxdepth 1 -type f -executable -name '0*' | sort)
if [[ ${#OP_BINS[@]} -eq 0 ]]; then
  echo "ERROR: No executable op binaries found under $BIN_DIR"
  exit 1
fi

echo "Found ${#OP_BINS[@]} op binaries"
echo "Output root: $OUT_ROOT"
echo "Sim cores: ${BMK_CPU_AFFINITY:-0-$((SIM_NUM_CPUS-1))} | bmk threads: ${BMK_THREADS}"

count_started=0
count_ok=0
count_fail=0

for bin in "${OP_BINS[@]}"; do
  name=$(basename "$bin")
  op_idx_str=${name%%_*}
  op_idx=$((10#$op_idx_str))

  if (( op_idx < START_IDX )); then
    continue
  fi
  if (( MAX_OPS > 0 && count_started >= MAX_OPS )); then
    break
  fi

  outdir="$OUT_ROOT/$name"
  mkdir -p "$outdir"

  # Treat presence of stats.txt as finished (gem5 standard output artifact).
  if [[ "$RESUME" == "1" && -s "$outdir/stats.txt" ]]; then
    echo "[SKIP] $name (existing stats.txt)"
    continue
  fi

  # Build command for target process.
  target_cmd="$bin"
  target_opts=""
  if [[ -n "$ORT_LIB_DIR" && -d "$ORT_LIB_DIR" && -x "$TARGET_LOADER" ]]; then
    target_cmd="$TARGET_LOADER"
    target_opts="--library-path $ORT_LIB_DIR $bin"
  fi

  echo "[RUN ] idx=$op_idx name=$name"
  ((count_started++))

  affinity_args=()
  if [[ -n "$BMK_CPU_AFFINITY" ]]; then
    affinity_args+=(--cpu-affinity="$BMK_CPU_AFFINITY")
  else
    affinity_args+=(--num-cpus="$SIM_NUM_CPUS")
  fi

  "$GEM5_BIN" \
    --outdir="$outdir" \
    "$GEM5_CONFIG" \
    --cmd="$target_cmd" \
    --options="$target_opts" \
    --benchmark-threads="$BMK_THREADS" \
    "${affinity_args[@]}" \
    --cpu-clock="$CPU_CLOCK" \
    --mem-size0="$MEM_SIZE0" \
    --mem-size1="$MEM_SIZE1" \
    --local-mem-delay="$LOCAL_MEM_DELAY" \
    --remote-mem-delay="$REMOTE_MEM_DELAY" \
    > "$outdir/gem5_stdout.log" 2>&1
  rc=$?

  echo "$op_idx,$name,$rc,$outdir" >> "$SUMMARY_CSV"

  if [[ $rc -eq 0 ]]; then
    ((count_ok++))
    echo "[ OK ] $name"
  else
    ((count_fail++))
    echo "$name" >> "$FAIL_LIST"
    echo "[FAIL] $name (rc=$rc)"
  fi
done

echo
echo "Done. started=$count_started ok=$count_ok fail=$count_fail"
echo "Summary: $SUMMARY_CSV"
if [[ $count_fail -gt 0 ]]; then
  echo "Failed list: $FAIL_LIST"
  exit 1
fi
