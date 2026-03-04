#!/usr/bin/env bash
set -u

# Run all per-op binaries natively on host and summarize pass/fail.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN_DIR=${BIN_DIR:-$SCRIPT_DIR/out_per_op_bins/build}
ORT_LIB_DIR=${ORT_LIB_DIR:-$SCRIPT_DIR/out_per_op_bins/ort_sdk/lib}
TARGET_LOADER=${TARGET_LOADER:-/lib/ld-linux-aarch64.so.1}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/host_run_per_op}

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

# Ensure SONAME link exists.
if [[ -f "$ORT_LIB_DIR/libonnxruntime.so.1.23.2" && ! -e "$ORT_LIB_DIR/libonnxruntime.so.1" ]]; then
  ln -sfn libonnxruntime.so.1.23.2 "$ORT_LIB_DIR/libonnxruntime.so.1"
fi

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
FAILS="$OUT_DIR/failed_ops.txt"

: > "$FAILS"
echo "op_idx,op_name,exit_code" > "$SUMMARY"

set +e
ok=0
fail=0
total=0

while IFS= read -r bin; do
  name=$(basename "$bin")
  idx=${name%%_*}
  "$TARGET_LOADER" --library-path "$ORT_LIB_DIR" "$bin" > "$OUT_DIR/${name}.log" 2>&1
  rc=$?
  echo "$((10#$idx)),$name,$rc" >> "$SUMMARY"
  total=$((total+1))
  if [[ $rc -eq 0 ]]; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    echo "$name" >> "$FAILS"
  fi
done < <(find "$BIN_DIR" -maxdepth 1 -type f -executable -name '0*' | sort)

echo "TOTAL=$total OK=$ok FAIL=$fail"
echo "SUMMARY=$SUMMARY"
if [[ -s "$FAILS" ]]; then
  echo "FAILED_LIST=$FAILS"
  exit 1
fi
