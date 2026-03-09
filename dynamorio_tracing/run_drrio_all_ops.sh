#!/usr/bin/env bash
set -u

# Collect DynamoRIO drmemtrace offline traces for all per-op binaries —
# sequential version.
#
# The resulting raw trace directories land under
# OUT_ROOT/<op_name>/drmemtrace.*.dir and can be post-processed with:
#   drrun -t drmemtrace -indir <dir> -tool basic_counts
#
# Default usage:
#   bash run_drrio_all_ops_parallel.sh
#
# Common overrides:
#   DRRUN=/path/to/drrun \
#   USE_PHYSICAL=1 \
#   START_IDX=0 MAX_OPS=20 RESUME=1 \
#   NUM_INDICES_PER_LOOKUP=100 \
#   INTRA_THREADS=4 INTER_THREADS=1 \
#   bash run_drrio_all_ops_parallel.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DRRUN=${DRRUN:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun}
BIN_DIR=${BIN_DIR:-$SCRIPT_DIR/out_per_op_bins/build}
ORT_LIB_DIR=${ORT_LIB_DIR:-$SCRIPT_DIR/out_per_op_bins/ort_sdk/lib}
MANIFEST_JSON=${MANIFEST_JSON:-$SCRIPT_DIR/out_per_op_bins/manifest.json}
TARGET_LOADER=${TARGET_LOADER:-/lib/ld-linux-aarch64.so.1}
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/drrio_traces_per_op_multithread_64_batch_size}

# Segmented execution.
START_IDX=${START_IDX:-0}
MAX_OPS=${MAX_OPS:-0}   # 0 = all
RESUME=${RESUME:-1}     # 1 = skip ops whose trace dir already exists and is non-empty

# ORT thread counts forwarded to each op binary at runtime.
INTRA_THREADS=${INTRA_THREADS:-1}
INTER_THREADS=${INTER_THREADS:-1}

# Informational / validation knob: single-op binaries have bag_size baked in
# at build time. This value is checked against manifest.json when available.
NUM_INDICES_PER_LOOKUP=${NUM_INDICES_PER_LOOKUP:-0}

# Enable physical-address tracing (requires root / sudo).
# When set to 1, "-use_physical" is passed to drmemtrace and drrun is
# invoked via sudo.
USE_PHYSICAL=${USE_PHYSICAL:-0}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [[ ! -x "$DRRUN" ]]; then
  echo "ERROR: DRRUN is not executable: $DRRUN"
  exit 1
fi
if [[ ! -d "$BIN_DIR" ]]; then
  echo "ERROR: BIN_DIR not found: $BIN_DIR"
  exit 1
fi
if [[ ! -d "$ORT_LIB_DIR" ]]; then
  echo "ERROR: ORT_LIB_DIR not found: $ORT_LIB_DIR"
  exit 1
fi
if [[ ! -x "$TARGET_LOADER" ]]; then
  echo "ERROR: TARGET_LOADER not executable: $TARGET_LOADER"
  exit 1
fi

if [[ -f "$MANIFEST_JSON" ]] && command -v python3 >/dev/null 2>&1; then
  manifest_num_indices=$(python3 - "$MANIFEST_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

value = None
if isinstance(data, list) and data:
    value = data[0].get("num_indices_per_lookup")
print("" if value is None else value)
PY
)
  if [[ -n "${manifest_num_indices}" ]]; then
    if [[ "$NUM_INDICES_PER_LOOKUP" == "0" ]]; then
      NUM_INDICES_PER_LOOKUP="$manifest_num_indices"
    elif [[ "$NUM_INDICES_PER_LOOKUP" != "$manifest_num_indices" ]]; then
      echo "ERROR: NUM_INDICES_PER_LOOKUP=$NUM_INDICES_PER_LOOKUP but binaries were built with $manifest_num_indices according to $MANIFEST_JSON"
      exit 1
    fi
  fi
fi

if [[ "$USE_PHYSICAL" == "1" ]]; then
  if ! sudo -n true 2>/dev/null; then
    echo "ERROR: USE_PHYSICAL=1 requires passwordless sudo. Run 'sudo -v' first."
    exit 1
  fi
fi

# Ensure SONAME symlink so the in-process dynamic linker finds the library.
if [[ -f "$ORT_LIB_DIR/libonnxruntime.so.1.23.2" && \
      ! -e "$ORT_LIB_DIR/libonnxruntime.so.1" ]]; then
  ln -sfn libonnxruntime.so.1.23.2 "$ORT_LIB_DIR/libonnxruntime.so.1"
fi

mkdir -p "$OUT_ROOT"
SUMMARY_CSV="$OUT_ROOT/summary.csv"
FAIL_LIST="$OUT_ROOT/failed_ops.txt"

echo "op_idx,op_name,exit_code,trace_dir" > "$SUMMARY_CSV"
: > "$FAIL_LIST"

# ---------------------------------------------------------------------------
# Collect op binaries
# ---------------------------------------------------------------------------
mapfile -t OP_BINS < <(find "$BIN_DIR" -maxdepth 1 -type f -executable -name '0*' | sort)
if [[ ${#OP_BINS[@]} -eq 0 ]]; then
  echo "ERROR: No executable op binaries found under $BIN_DIR"
  exit 1
fi

echo "Found ${#OP_BINS[@]} op binaries"
echo "Output root  : $OUT_ROOT"
echo "USE_PHYSICAL : $USE_PHYSICAL"
echo "NUM_INDICES_PER_LOOKUP (baked into binaries): $NUM_INDICES_PER_LOOKUP"
echo

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------
run_one_op() {
  local bin="$1"
  local name op_idx trace_out

  name=$(basename "$bin")
  op_idx_str=${name%%_*}
  op_idx=$((10#$op_idx_str))
  trace_out="$OUT_ROOT/$name"

  mkdir -p "$trace_out"

  # Build drrun command.
  local drrun_cmd=()

  if [[ "$USE_PHYSICAL" == "1" ]]; then
    drrun_cmd+=(sudo)
  fi

  drrun_cmd+=(
    "$DRRUN"
    -t drmemtrace
    -offline
    -outdir "$trace_out"
  )

  if [[ "$USE_PHYSICAL" == "1" ]]; then
    drrun_cmd+=(-use_physical)
  fi

  # Separator and target: use the ARM dynamic linker to load the binary so
  # it can find libonnxruntime.so at the correct path.
  drrun_cmd+=(
    --
    "$TARGET_LOADER"
    --library-path "$ORT_LIB_DIR"
    "$bin"
    --intra-threads "$INTRA_THREADS"
    --inter-threads "$INTER_THREADS"
  )

  "${drrun_cmd[@]}" > "$trace_out/drrio_stdout.log" 2>&1
  local rc=$?

  # Write result directly to CSV.
  printf '%s,%s,%s,%s\n' "$op_idx" "$name" "$rc" "$trace_out" >> "$SUMMARY_CSV"
  if [[ "$rc" == "0" ]]; then
    echo "[ OK ] $name"
  else
    echo "[FAIL] $name (rc=$rc)"
    echo "$name" >> "$FAIL_LIST"
  fi
}

# ---------------------------------------------------------------------------
# Main dispatch loop (sequential)
# ---------------------------------------------------------------------------
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

  trace_out="$OUT_ROOT/$name"

  # Resume: skip if a non-empty drmemtrace raw directory already exists.
  if [[ "$RESUME" == "1" ]] && \
     [[ -n "$(find "$trace_out" -maxdepth 2 -name 'raw' -type d 2>/dev/null | head -1)" ]]; then
    echo "[SKIP] $name (existing trace raw/)"
    continue
  fi

  echo "[RUN ] idx=$op_idx name=$name"
  (( count_started++ ))

  run_one_op "$bin"
done

echo

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
