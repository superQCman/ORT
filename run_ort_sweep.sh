#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRACE_DIR="$SCRIPT_DIR/dynamorio_tracing"
ANALYSIS_DIR="$SCRIPT_DIR/onnx_operator_analysis"

ASCEND_ENV_SH=${ASCEND_ENV_SH:-/data/qc/Ascend/ascend-toolkit/set_env.sh}
ONNX_PATH=${ONNX_PATH:-/data/qc/dlrm/dlrm_onnx/dlrm_s_pytorch.onnx}

BATCH_START=${BATCH_START:-32}
BATCH_END=${BATCH_END:-2048}
BATCH_STEP=${BATCH_STEP:-32}

NUM_INDICES_START=${NUM_INDICES_START:-100}
NUM_INDICES_END=${NUM_INDICES_END:-1000}
NUM_INDICES_STEP=${NUM_INDICES_STEP:-50}

NUM_BATCHES=${NUM_BATCHES:-3}
WARMUP_BATCHES=${WARMUP_BATCHES:-2}
INTRA_THREADS=${INTRA_THREADS:-4}
INTER_THREADS=${INTER_THREADS:-1}
DEVICE_ID=${DEVICE_ID:-0}
USE_CANN=${USE_CANN:-0}
NO_REPLACE_LOOP=${NO_REPLACE_LOOP:-0}
PROFILE_WARMUP=${PROFILE_WARMUP:-0}
DISABLE_GRAPH_OPTIMIZATIONS=${DISABLE_GRAPH_OPTIMIZATIONS:-1}

USE_NUMACTL=${USE_NUMACTL:-1}
NUMA_NODE=${NUMA_NODE:-0}

OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/sweep_runs}
OP_SHAPES_DIR=${OP_SHAPES_DIR:-$OUT_ROOT/op_shapes}
PROFILE_ROOT=${PROFILE_ROOT:-$OUT_ROOT/onnx_profiles}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/logs}
BUILD_ROOT=${BUILD_ROOT:-$TRACE_DIR/out_per_op_bins_sweep}
TRACE_ROOT=${TRACE_ROOT:-$TRACE_DIR/drrio_traces_sweep}
FEATURE_ROOT=${FEATURE_ROOT:-$TRACE_DIR/trace_features_sweep}
FEATURE_DATASET_ROOT=${FEATURE_DATASET_ROOT:-$SCRIPT_DIR/features}
FEATURE_DATASET_MERGED_CSV=${FEATURE_DATASET_MERGED_CSV:-$FEATURE_DATASET_ROOT/all_features.csv}
GENERATED_ONNX_ROOT=${GENERATED_ONNX_ROOT:-$OUT_ROOT/generated_onnx}
SUMMARY_CSV=${SUMMARY_CSV:-$OUT_ROOT/sweep_summary.csv}

ORT_ROOT=${ORT_ROOT:-${CONDA_PREFIX:-}}
DRRUN=${DRRUN:-/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun}
TARGET_LOADER=${TARGET_LOADER:-/lib/ld-linux-aarch64.so.1}
USE_PHYSICAL=${USE_PHYSICAL:-0}
TRACE_RESUME=${TRACE_RESUME:-1}
START_IDX=${START_IDX:-0}
MAX_OPS=${MAX_OPS:-0}
MAX_COMBOS=${MAX_COMBOS:-0}
RESUME=${RESUME:-1}
EXTRACT_FEATURES=${EXTRACT_FEATURES:-1}
FEATURE_JOBS=${FEATURE_JOBS:-64}
CACHE_CONF=${CACHE_CONF:-/data/qc/dlrm/ops_profile/concorde/test/cache_nl.conf}

mkdir -p "$OUT_ROOT" "$OP_SHAPES_DIR" "$PROFILE_ROOT" "$LOG_ROOT" "$BUILD_ROOT" "$TRACE_ROOT" "$FEATURE_ROOT" "$FEATURE_DATASET_ROOT" "$GENERATED_ONNX_ROOT"

if [[ -f "$ASCEND_ENV_SH" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_ENV_SH"
fi

if [[ "$BATCH_STEP" -le 0 || "$NUM_INDICES_STEP" -le 0 ]]; then
  echo "ERROR: BATCH_STEP and NUM_INDICES_STEP must be > 0"
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: ONNX model not found: $ONNX_PATH"
  exit 1
fi

echo "timestamp,batch_size,num_indices_per_lookup,status,failed_stage,shape_csv,profile_dir,build_dir,trace_dir,trace_feature_csv,cpu_thread_detail_csv,training_feature_csv,log_dir" > "$SUMMARY_CSV"

completed=0
failed=0
combo_count=0

has_profile_json() {
  local dir="$1"
  compgen -G "$dir/ort_cann_profile*.json" > /dev/null
}

find_latest_profile_json() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'ort_cann_profile*.json' | sort | tail -n 1
}

stage_completed() {
  local stage="$1"
  local shape_csv="$2"
  local profile_dir="$3"
  local build_dir="$4"
  local trace_dir="$5"
  local feature_csv="$6"

  case "$stage" in
    infer)
      [[ -f "$shape_csv" ]] && has_profile_json "$profile_dir"
      ;;
    build)
      [[ -f "$build_dir/manifest.json" ]] && [[ -n "$(find "$build_dir/build" -maxdepth 1 -type f -executable -name '0*' 2>/dev/null | head -n 1)" ]]
      ;;
    trace)
      [[ -f "$trace_dir/summary.csv" ]] && [[ ! -s "$trace_dir/failed_ops.txt" ]]
      ;;
    features)
      [[ "$EXTRACT_FEATURES" != "1" ]] || [[ -s "$feature_csv" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_ort_lib_dir() {
  local build_dir="$1"
  if [[ -d "$build_dir/ort_sdk/lib" ]]; then
    printf '%s\n' "$build_dir/ort_sdk/lib"
    return 0
  fi
  if [[ -n "$ORT_ROOT" && -d "$ORT_ROOT/lib" ]]; then
    printf '%s\n' "$ORT_ROOT/lib"
    return 0
  fi
  if [[ -n "$ORT_ROOT" && -d "$ORT_ROOT/lib64" ]]; then
    printf '%s\n' "$ORT_ROOT/lib64"
    return 0
  fi
  return 1
}

append_summary() {
  local batch_size="$1"
  local num_indices="$2"
  local status="$3"
  local failed_stage="$4"
  local shape_csv="$5"
  local profile_dir="$6"
  local build_dir="$7"
  local trace_dir="$8"
  local feature_csv="$9"
  local cpu_detail_csv="${10}"
  local training_feature_csv="${11}"
  local log_dir="${12}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date '+%F %T')" \
    "$batch_size" \
    "$num_indices" \
    "$status" \
    "$failed_stage" \
    "$shape_csv" \
    "$profile_dir" \
    "$build_dir" \
    "$trace_dir" \
    "$feature_csv" \
    "$cpu_detail_csv" \
    "$training_feature_csv" \
    "$log_dir" >> "$SUMMARY_CSV"
}

run_inference_stage() {
  local onnx_path="$1"
  local batch_size="$2"
  local num_indices="$3"
  local shape_csv="$4"
  local profile_dir="$5"
  local log_path="$6"

  local cmd=()
  if [[ "$USE_NUMACTL" == "1" ]] && command -v numactl > /dev/null 2>&1; then
    cmd+=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
  fi
  cmd+=(
    python3 "$SCRIPT_DIR/run_ort_dlrm.py"
    --onnx-path "$onnx_path"
    --batch-size "$batch_size"
    --num-batches "$NUM_BATCHES"
    --warmup-batches "$WARMUP_BATCHES"
    --shape-csv "$shape_csv"
    --enable-profiling
    --profile-dir "$profile_dir"
    --intra-threads "$INTRA_THREADS"
    --inter-threads "$INTER_THREADS"
    --num-indices-per-lookup "$num_indices"
  )
  if [[ "$USE_CANN" == "1" ]]; then
    cmd+=(--use-cann --device-id "$DEVICE_ID")
  fi
  if [[ "$NO_REPLACE_LOOP" == "1" ]]; then
    cmd+=(--no-replace-loop)
  fi
  if [[ "$PROFILE_WARMUP" == "1" ]]; then
    cmd+=(--profile-warmup)
  fi
  if [[ "$DISABLE_GRAPH_OPTIMIZATIONS" == "1" ]]; then
    cmd+=(--disable-graph-optimizations)
  fi

  "${cmd[@]}" > "$log_path" 2>&1
}

run_build_stage() {
  local build_onnx_path="$1"
  local batch_size="$2"
  local num_indices="$3"
  local shape_csv="$4"
  local build_dir="$5"
  local log_path="$6"

  local cmd=(bash "$TRACE_DIR/run_build_op_binaries.sh" "$build_onnx_path")
  if [[ -n "$ORT_ROOT" ]]; then
    cmd+=("$ORT_ROOT")
  fi

  (
    cd "$TRACE_DIR" || exit 1
    OUT_DIR="$build_dir" \
    BATCH_SIZE="$batch_size" \
    NUM_INDICES_PER_LOOKUP="$num_indices" \
    INTRA_THREADS="$INTRA_THREADS" \
    INTER_THREADS="$INTER_THREADS" \
    SHAPE_CSV="$shape_csv" \
    "${cmd[@]}"
  ) > "$log_path" 2>&1
}

prepare_combo_onnx() {
  local combo_source_onnx="$1"
  mkdir -p "$(dirname "$combo_source_onnx")"
  cp -f "$ONNX_PATH" "$combo_source_onnx"
}

resolve_effective_onnx_path() {
  local combo_source_onnx="$1"
  local infer_log="$2"
  local parsed=""

  if [[ -f "$infer_log" ]]; then
    parsed=$(awk -F': ' '/^\[ORT\] 加载模型:/ {path=$2} END{print path}' "$infer_log")
    if [[ -n "$parsed" && -f "$parsed" ]]; then
      printf '%s\n' "$parsed"
      return 0
    fi
  fi

  for candidate in \
    "$combo_source_onnx.cann_patched.onnx.loop_to_gather.onnx.cpu_ops.onnx" \
    "$combo_source_onnx.cann_patched.onnx.loop_to_gather.onnx" \
    "$combo_source_onnx.loop_to_gather.onnx.cpu_ops.onnx" \
    "$combo_source_onnx.loop_to_gather.onnx" \
    "$combo_source_onnx.cann_patched.onnx" \
    "$combo_source_onnx"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

build_log_matches_onnx() {
  local build_log="$1"
  local effective_onnx_path="$2"
  [[ -f "$build_log" ]] && grep -F "\"onnx\": \"$effective_onnx_path\"" "$build_log" > /dev/null
}

run_trace_stage() {
  local num_indices="$1"
  local build_dir="$2"
  local trace_dir="$3"
  local log_path="$4"
  local ort_lib_dir

  if ! ort_lib_dir=$(resolve_ort_lib_dir "$build_dir"); then
    echo "ERROR: cannot resolve ORT_LIB_DIR for build dir $build_dir" > "$log_path"
    return 1
  fi

  (
    cd "$TRACE_DIR" || exit 1
    DRRUN="$DRRUN" \
    TARGET_LOADER="$TARGET_LOADER" \
    BIN_DIR="$build_dir/build" \
    ORT_LIB_DIR="$ort_lib_dir" \
    MANIFEST_JSON="$build_dir/manifest.json" \
    OUT_ROOT="$trace_dir" \
    INTRA_THREADS="$INTRA_THREADS" \
    INTER_THREADS="$INTER_THREADS" \
    NUM_INDICES_PER_LOOKUP="$num_indices" \
    USE_PHYSICAL="$USE_PHYSICAL" \
    RESUME="$TRACE_RESUME" \
    START_IDX="$START_IDX" \
    MAX_OPS="$MAX_OPS" \
    bash "$TRACE_DIR/run_drrio_all_ops.sh"
  ) > "$log_path" 2>&1
}

run_feature_stage() {
  local trace_dir="$1"
  local feature_csv="$2"
  local log_path="$3"

  local cmd=(
    python3 -u "$TRACE_DIR/extract_trace_features.py"
    --ops_dir "$trace_dir"
    --out "$feature_csv"
    --jobs "$FEATURE_JOBS"
    --cache_conf "$CACHE_CONF"
    --drrun "$DRRUN"
  )
  if [[ "$USE_PHYSICAL" == "1" ]]; then
    cmd+=(--use_physical)
  fi

  "${cmd[@]}" > "$log_path" 2>&1
}

cpu_parse_completed() {
  local cpu_detail_csv="$1"
  local cpu_agg_csv="$2"
  [[ -s "$cpu_detail_csv" ]] && [[ -s "$cpu_agg_csv" ]]
}

dataset_completed() {
  local final_feature_csv="$1"
  local aligned_cpu_detail_csv="$2"
  [[ -s "$final_feature_csv" ]] && [[ -s "$aligned_cpu_detail_csv" ]]
}

run_cpu_parse_stage() {
  local profile_json="$1"
  local profile_dir="$2"
  local log_path="$3"

  python3 -u "$ANALYSIS_DIR/extract_cpu_thread_usage.py" \
    "$profile_json" \
    --out-dir "$profile_dir" > "$log_path" 2>&1
}

run_dataset_stage() {
  local batch_size="$1"
  local num_indices="$2"
  local shape_csv="$3"
  local cpu_detail_csv="$4"
  local trace_feature_csv="$5"
  local aligned_cpu_detail_csv="$6"
  local cpu_node_agg_csv="$7"
  local cpu_unmatched_csv="$8"
  local final_feature_csv="$9"
  local log_path="${10}"

  python3 -u "$ANALYSIS_DIR/build_training_features.py" \
    --op-shapes "$shape_csv" \
    --cpu-detail "$cpu_detail_csv" \
    --trace-features "$trace_feature_csv" \
    --aligned-cpu-detail-out "$aligned_cpu_detail_csv" \
    --cpu-agg-out "$cpu_node_agg_csv" \
    --unmatched-out "$cpu_unmatched_csv" \
    --out "$final_feature_csv" \
    --batch-size "$batch_size" \
    --num-indices-per-lookup "$num_indices" > "$log_path" 2>&1
}

merge_feature_datasets() {
  local feature_root="$1"
  local merged_csv="$2"

  python3 - "$feature_root" "$merged_csv" <<'PY'
import csv
import sys
from pathlib import Path

feature_root = Path(sys.argv[1])
merged_csv = Path(sys.argv[2])

csv_paths = sorted(
    path for path in feature_root.glob("*.csv")
    if path.name != merged_csv.name and path.is_file()
)

if not csv_paths:
    print(f"[WARN] no feature CSVs found under {feature_root}")
    sys.exit(0)

header = None
rows_written = 0

merged_csv.parent.mkdir(parents=True, exist_ok=True)
with merged_csv.open("w", encoding="utf-8", newline="") as out_f:
    writer = None
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as in_f:
            reader = csv.reader(in_f)
            current_header = next(reader, None)
            if not current_header:
                continue
            if header is None:
                header = current_header
                writer = csv.writer(out_f)
                writer.writerow(header)
            elif current_header != header:
                raise SystemExit(
                    f"ERROR: header mismatch while merging {path}. "
                    f"expected {header}, got {current_header}"
                )

            for row in reader:
                writer.writerow(row)
                rows_written += 1

print(f"[ OK ] merged {len(csv_paths)} feature CSVs into {merged_csv} ({rows_written} rows)")
PY
}

for ((batch_size = BATCH_START; batch_size <= BATCH_END; batch_size += BATCH_STEP)); do
  for ((num_indices = NUM_INDICES_START; num_indices <= NUM_INDICES_END; num_indices += NUM_INDICES_STEP)); do
    ((combo_count += 1))
    if (( MAX_COMBOS > 0 && combo_count > MAX_COMBOS )); then
      break 2
    fi

    combo_tag="bs${batch_size}_nip${num_indices}"
    shape_csv="$OP_SHAPES_DIR/op_shapes_${batch_size}_${num_indices}.csv"
    profile_dir="$PROFILE_ROOT/$combo_tag"
    build_dir="$BUILD_ROOT/$combo_tag"
    trace_dir="$TRACE_ROOT/$combo_tag"
    feature_csv="$FEATURE_ROOT/${combo_tag}.csv"
    final_feature_csv="$FEATURE_DATASET_ROOT/${combo_tag}.csv"
    combo_onnx_dir="$GENERATED_ONNX_ROOT/$combo_tag"
    combo_source_onnx="$combo_onnx_dir/$(basename "$ONNX_PATH")"
    log_dir="$LOG_ROOT/$combo_tag"

    infer_log="$log_dir/run_ort.log"
    cpu_parse_log="$log_dir/extract_cpu_threads.log"
    build_log="$log_dir/build_ops.log"
    trace_log="$log_dir/drrio.log"
    feature_log="$log_dir/extract_features.log"
    dataset_log="$log_dir/build_training_features.log"

    mkdir -p "$profile_dir" "$build_dir" "$trace_dir" "$combo_onnx_dir" "$log_dir"
    prepare_combo_onnx "$combo_source_onnx"

    echo "=================================================================="
    echo "[COMBO] $combo_tag"
    echo "  shape_csv : $shape_csv"
    echo "  profile   : $profile_dir"
    echo "  build_dir : $build_dir"
    echo "  trace_dir : $trace_dir"
    echo "  dataset   : $final_feature_csv"

    infer_ran=0
    if [[ "$RESUME" != "1" ]] || ! stage_completed infer "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv"; then
      echo "[RUN ] inference + profiling"
      if ! run_inference_stage "$combo_source_onnx" "$batch_size" "$num_indices" "$shape_csv" "$profile_dir" "$infer_log"; then
        echo "[FAIL] inference stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "inference" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "" "$final_feature_csv" "$log_dir"
        ((failed += 1))
        continue
      fi
      infer_ran=1
    else
      echo "[SKIP] inference already completed"
    fi

    profile_json=$(find_latest_profile_json "$profile_dir")
    if [[ -z "$profile_json" || ! -f "$profile_json" ]]; then
      echo "[FAIL] cannot resolve profile JSON: $combo_tag"
      append_summary "$batch_size" "$num_indices" "FAILED" "profile_json" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "" "$final_feature_csv" "$log_dir"
      ((failed += 1))
      continue
    fi

    profile_stem=$(basename "${profile_json%.json}")
    cpu_detail_csv="$profile_dir/${profile_stem}_cpu_thread_detail.csv"
    cpu_agg_csv="$profile_dir/${profile_stem}_cpu_thread_aggregated.csv"
    aligned_cpu_detail_csv="$profile_dir/${profile_stem}_cpu_thread_detail_aligned.csv"
    cpu_node_agg_csv="$profile_dir/${profile_stem}_cpu_thread_node_aggregated.csv"
    cpu_unmatched_csv="$profile_dir/${profile_stem}_cpu_thread_unmatched.csv"

    cpu_parse_ran=0
    if [[ "$RESUME" != "1" ]] || [[ "$infer_ran" == "1" ]] || ! cpu_parse_completed "$cpu_detail_csv" "$cpu_agg_csv"; then
      echo "[RUN ] extract CPU thread usage"
      if ! run_cpu_parse_stage "$profile_json" "$profile_dir" "$cpu_parse_log"; then
        echo "[FAIL] cpu profile parse stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "cpu_profile_parse" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        ((failed += 1))
        continue
      fi
      cpu_parse_ran=1
    else
      echo "[SKIP] CPU thread extraction already completed"
    fi

    if ! effective_onnx_path=$(resolve_effective_onnx_path "$combo_source_onnx" "$infer_log"); then
      echo "[FAIL] cannot resolve rewritten ONNX path: $combo_tag"
      append_summary "$batch_size" "$num_indices" "FAILED" "resolve_onnx" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
      ((failed += 1))
      continue
    fi

    build_ran=0
    if [[ "$RESUME" != "1" ]] || ! stage_completed build "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" || ! build_log_matches_onnx "$build_log" "$effective_onnx_path"; then
      echo "[RUN ] build per-op binaries"
      if ! run_build_stage "$effective_onnx_path" "$batch_size" "$num_indices" "$shape_csv" "$build_dir" "$build_log"; then
        echo "[FAIL] build stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "build" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        ((failed += 1))
        continue
      fi
      build_ran=1
    else
      echo "[SKIP] build already completed"
    fi

    trace_ran=0
    if [[ "$RESUME" != "1" ]] || [[ "$build_ran" == "1" ]] || ! stage_completed trace "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv"; then
      echo "[RUN ] DynamoRIO trace"
      if ! run_trace_stage "$num_indices" "$build_dir" "$trace_dir" "$trace_log"; then
        echo "[FAIL] trace stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "trace" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        ((failed += 1))
        continue
      fi
      trace_ran=1
    else
      echo "[SKIP] DynamoRIO trace already completed"
    fi

    if [[ "$EXTRACT_FEATURES" == "1" ]]; then
      if [[ "$RESUME" != "1" ]] || [[ "$trace_ran" == "1" ]] || [[ "$build_ran" == "1" ]] || ! stage_completed features "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv"; then
        echo "[RUN ] extract trace features"
        if ! run_feature_stage "$trace_dir" "$feature_csv" "$feature_log"; then
          echo "[FAIL] feature extraction stage: $combo_tag"
          append_summary "$batch_size" "$num_indices" "FAILED" "features" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
          ((failed += 1))
          continue
        fi
      else
        echo "[SKIP] feature extraction already completed"
      fi
    else
      echo "[FAIL] training feature dataset requires EXTRACT_FEATURES=1"
      append_summary "$batch_size" "$num_indices" "FAILED" "trace_features_disabled" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
      ((failed += 1))
      continue
    fi

    if [[ "$RESUME" != "1" ]] || [[ "$cpu_parse_ran" == "1" ]] || [[ "$trace_ran" == "1" ]] || [[ "$build_ran" == "1" ]] || ! dataset_completed "$final_feature_csv" "$aligned_cpu_detail_csv"; then
      echo "[RUN ] build training features"
      if ! run_dataset_stage "$batch_size" "$num_indices" "$shape_csv" "$cpu_detail_csv" "$feature_csv" "$aligned_cpu_detail_csv" "$cpu_node_agg_csv" "$cpu_unmatched_csv" "$final_feature_csv" "$dataset_log"; then
        echo "[FAIL] training feature stage: $combo_tag"
        append_summary "$batch_size" "$num_indices" "FAILED" "training_features" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
        ((failed += 1))
        continue
      fi
    else
      echo "[SKIP] training feature dataset already completed"
    fi

    echo "[ OK ] $combo_tag"
    append_summary "$batch_size" "$num_indices" "OK" "" "$shape_csv" "$profile_dir" "$build_dir" "$trace_dir" "$feature_csv" "$cpu_detail_csv" "$final_feature_csv" "$log_dir"
    ((completed += 1))
  done
done

echo "=================================================================="
echo "Sweep finished"
echo "  completed : $completed"
echo "  failed    : $failed"
echo "  summary   : $SUMMARY_CSV"

if ! merge_feature_datasets "$FEATURE_DATASET_ROOT" "$FEATURE_DATASET_MERGED_CSV"; then
  echo "ERROR: failed to merge feature datasets"
  exit 1
fi
echo "  merged    : $FEATURE_DATASET_MERGED_CSV"

if (( failed > 0 )); then
  exit 1
fi
