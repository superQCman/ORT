#!/usr/bin/env bash
set -u

# Batch simulate all per-op binaries with gem5 (SE mode) — PARALLEL version.
#
# Launches up to PARALLEL_JOBS gem5 processes simultaneously, each simulating
# a different operator binary.  A semaphore-based slot manager ensures at most
# PARALLEL_JOBS jobs run at the same time; the main loop waits for a free slot
# before spawning the next child.
#
# Default usage:
#   bash run_gem5_all_ops_parallel.sh
#
# Common overrides:
#   GEM5_BIN=/data/qc/simulator/gem5/build/ALL/gem5.opt \
#   GEM5_CONFIG=/data/qc/dlrm/ops_profile/concorde/simulatoin/kunpeng920.py \
#   PARALLEL_JOBS=4 START_IDX=0 MAX_OPS=20 RESUME=1 \
#   SIM_NUM_CPUS=8 BMK_THREADS=8 BMK_CPU_AFFINITY=0-7 \
#   bash run_gem5_all_ops_parallel.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
GEM5_BIN=${GEM5_BIN:-/data/qc/simulator/gem5/build/ALL/gem5.opt}
GEM5_CONFIG=${GEM5_CONFIG:-/data/qc/dlrm/ops_profile/concorde/simulatoin/kunpeng920.py}
BIN_DIR=${BIN_DIR:-$SCRIPT_DIR/out_per_op_bins/build}
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/m5out_per_op_bins}

# Parallelism: how many gem5 instances run concurrently.
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

# Segmented execution.
START_IDX=${START_IDX:-0}
MAX_OPS=${MAX_OPS:-0}    # 0 = all
RESUME=${RESUME:-1}      # 1 = skip ops whose outdir already contains stats.txt

# Kunpeng config knobs.
CPU_CLOCK=${CPU_CLOCK:-2.6GHz}
MEM_SIZE0=${MEM_SIZE0:-8GiB}
MEM_SIZE1=${MEM_SIZE1:-8GiB}
LOCAL_MEM_DELAY=${LOCAL_MEM_DELAY:-20ns}
REMOTE_MEM_DELAY=${REMOTE_MEM_DELAY:-40ns}

# Per-simulation threading / affinity knobs.
# SIM_NUM_CPUS: simulated core count (used when BMK_CPU_AFFINITY is empty).
# BMK_THREADS:  OMP/MKL/OpenBLAS thread count exported into the SE workload env.
# BMK_CPU_AFFINITY: explicit simulated core set, e.g. "0-7" or "0-3,8-11".
SIM_NUM_CPUS=${SIM_NUM_CPUS:-48}
BMK_THREADS=${BMK_THREADS:-0}
BMK_CPU_AFFINITY=${BMK_CPU_AFFINITY:-}

# Runtime linker for dynamically linked op binaries.  Leave ORT_LIB_DIR empty
# to run binaries directly.
ORT_LIB_DIR=${ORT_LIB_DIR:-$SCRIPT_DIR/out_per_op_bins/ort_sdk/lib}
TARGET_LOADER=${TARGET_LOADER:-/lib/ld-linux-aarch64.so.1}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
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

# Ensure the SONAME symlink exists so the in-simulation dynamic linker can
# find the library.
if [[ -f "$ORT_LIB_DIR/libonnxruntime.so.1.23.2" && \
      ! -e "$ORT_LIB_DIR/libonnxruntime.so.1" ]]; then
  ln -sfn libonnxruntime.so.1.23.2 "$ORT_LIB_DIR/libonnxruntime.so.1"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
FAIL_LIST="$OUT_ROOT/failed_ops.txt"

echo "op_idx,op_name,exit_code,outdir" > "$SUMMARY_CSV"
: > "$FAIL_LIST"

# ---------------------------------------------------------------------------
# Shared-state files for child → parent result reporting.
# Each child appends one line atomically (single write via printf).
# ---------------------------------------------------------------------------
RESULT_FIFO="$OUT_ROOT/.result_fifo"
[[ -p "$RESULT_FIFO" ]] && rm -f "$RESULT_FIFO"
mkfifo "$RESULT_FIFO"

# ---------------------------------------------------------------------------
# Collect op binaries
# ---------------------------------------------------------------------------
mapfile -t OP_BINS < <(find "$BIN_DIR" -maxdepth 1 -type f -executable -name '0*' | sort)
if [[ ${#OP_BINS[@]} -eq 0 ]]; then
  echo "ERROR: No executable op binaries found under $BIN_DIR"
  rm -f "$RESULT_FIFO"
  exit 1
fi

echo "Found ${#OP_BINS[@]} op binaries"
echo "Output root : $OUT_ROOT"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Sim cores   : ${BMK_CPU_AFFINITY:-0-$((SIM_NUM_CPUS-1))} | bmk threads: $BMK_THREADS"
echo

# ---------------------------------------------------------------------------
# Background result-reader: drains the FIFO and writes CSV / fail-list.
# Each line from a child: "<op_idx> <name> <rc> <outdir>"
# ---------------------------------------------------------------------------
count_ok=0
count_fail=0

reader_pid=""
(
  while IFS= read -r line; do
    [[ "$line" == "DONE_SENTINEL" ]] && break
    read -r r_idx r_name r_rc r_dir <<< "$line"
    printf '%s,%s,%s,%s\n' "$r_idx" "$r_name" "$r_rc" "$r_dir" >> "$SUMMARY_CSV"
    if [[ "$r_rc" == "0" ]]; then
      echo "[ OK ] $r_name"
    else
      echo "[FAIL] $r_name (rc=$r_rc)"
      echo "$r_name" >> "$FAIL_LIST"
    fi
  done < "$RESULT_FIFO"
) &
reader_pid=$!

# ---------------------------------------------------------------------------
# Worker function — runs in a subshell, reports result via FIFO.
# ---------------------------------------------------------------------------
run_one_op() {
  local bin="$1"
  local name op_idx outdir target_cmd target_opts

  name=$(basename "$bin")
  op_idx_str=${name%%_*}
  op_idx=$((10#$op_idx_str))
  outdir="$OUT_ROOT/$name"

  mkdir -p "$outdir"

  # Build target command.
  target_cmd="$bin"
  target_opts=""
  if [[ -n "$ORT_LIB_DIR" && -d "$ORT_LIB_DIR" && -x "$TARGET_LOADER" ]]; then
    target_cmd="$TARGET_LOADER"
    target_opts="--library-path $ORT_LIB_DIR $bin"
  fi

  local affinity_args=()
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
  local rc=$?

  # Atomic single-write to FIFO.
  printf '%s %s %s %s\n' "$op_idx" "$name" "$rc" "$outdir" > "$RESULT_FIFO"
}

# ---------------------------------------------------------------------------
# Main dispatch loop with a job-slot semaphore.
# pids[] maps slot index → child PID (0 = free).
# ---------------------------------------------------------------------------
declare -a slot_pids
for (( s=0; s<PARALLEL_JOBS; s++ )); do
  slot_pids[$s]=0
done

# Returns the index of a free slot, blocking until one is available.
wait_for_slot() {
  while true; do
    for (( s=0; s<PARALLEL_JOBS; s++ )); do
      local p="${slot_pids[$s]}"
      if [[ "$p" == "0" ]]; then
        echo "$s"; return
      fi
      # Check if the child with this PID has finished.
      if ! kill -0 "$p" 2>/dev/null; then
        # Reap it (may already be reaped by a previous wait).
        wait "$p" 2>/dev/null || true
        slot_pids[$s]=0
        echo "$s"; return
      fi
    done
    sleep 0.2
  done
}

count_started=0

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

  # Skip already-finished ops.
  if [[ "$RESUME" == "1" && -s "$outdir/stats.txt" ]]; then
    echo "[SKIP] $name (existing stats.txt)"
    continue
  fi

  slot=$(wait_for_slot)
  echo "[RUN ] idx=$op_idx name=$name  (slot $slot)"
  (( count_started++ ))

  # Launch worker in background; store its PID in the slot.
  run_one_op "$bin" &
  slot_pids[$slot]=$!
done

# ---------------------------------------------------------------------------
# Wait for all in-flight workers to finish.
# ---------------------------------------------------------------------------
echo
echo "All ops dispatched ($count_started started). Waiting for remaining jobs…"
for (( s=0; s<PARALLEL_JOBS; s++ )); do
  p="${slot_pids[$s]}"
  if [[ "$p" != "0" ]]; then
    wait "$p" 2>/dev/null || true
  fi
done

# Signal the reader to exit, then wait for it.
printf 'DONE_SENTINEL\n' > "$RESULT_FIFO"
wait "$reader_pid" 2>/dev/null || true
rm -f "$RESULT_FIFO"

# ---------------------------------------------------------------------------
# Tally results from summary CSV (skip header).
# ---------------------------------------------------------------------------
count_ok=0
count_fail=0
while IFS=, read -r _ _ rc _; do
  [[ "$rc" == "exit_code" ]] && continue
  if [[ "$rc" == "0" ]]; then
    (( count_ok++ ))
  else
    (( count_fail++ ))
  fi
done < "$SUMMARY_CSV"

echo
echo "======================================================================"
echo "Done.  started=$count_started  ok=$count_ok  fail=$count_fail"
echo "Summary : $SUMMARY_CSV"
if (( count_fail > 0 )); then
  echo "Failures: $FAIL_LIST"
  exit 1
fi
