#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

FEATURES_ROOT=${FEATURES_ROOT:-/data/qc/dlrm/ORT/features_selected}
HW_PROFILE=${HW_PROFILE:-$SCRIPT_DIR/hardware_profiles/kunpeng920_gem5.yaml}
HARDWARE_NAME=${HARDWARE_NAME:-kunpeng920_gem5}
GEM5_ROOT=${GEM5_ROOT:-}
GEM5_DEFAULT_COMBO=${GEM5_DEFAULT_COMBO:-}
TARGET=${TARGET:-label_real_dur_us}
OUTPUT_BASE=${OUTPUT_BASE:-$SCRIPT_DIR/artifacts/${HARDWARE_NAME}_${TARGET}}
DATASET_CSV=${DATASET_CSV:-$OUTPUT_BASE/dataset.csv}
MODEL_DIR=${MODEL_DIR:-$OUTPUT_BASE/model}

mkdir -p "$OUTPUT_BASE" "$MODEL_DIR"

build_args=(
  --features-root "$FEATURES_ROOT"
  --hw-profile "$HW_PROFILE"
  --hardware-name "$HARDWARE_NAME"
  --output-csv "$DATASET_CSV"
)

if [[ -n "$GEM5_ROOT" ]]; then
  IFS=: read -r -a gem5_roots <<< "$GEM5_ROOT"
  for root in "${gem5_roots[@]}"; do
    build_args+=(--gem5-root "$root")
  done
fi
if [[ -n "$GEM5_DEFAULT_COMBO" ]]; then
  build_args+=(--gem5-default-combo "$GEM5_DEFAULT_COMBO")
fi

python3 "$SCRIPT_DIR/build_operator_dataset.py" "${build_args[@]}"
python3 "$SCRIPT_DIR/train_operator_model.py" \
  --dataset-csv "$DATASET_CSV" \
  --target "$TARGET" \
  --output-dir "$MODEL_DIR"

echo "dataset: $DATASET_CSV"
echo "artifacts: $MODEL_DIR"
