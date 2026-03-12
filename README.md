# ORT DLRM Sweep And Training Features

Chinese version: [README.zh-CN.md](./README.zh-CN.md)

This directory contains the current ONNX Runtime based DLRM workflow for:

- running full-model inference with profiling
- generating batch-specific `op_shapes` metadata
- building and tracing per-op binaries with DynamoRIO
- extracting per-op trace features
- merging ORT CPU-thread profiling with trace features into a training dataset
- extracting a compact selected-feature dataset for downstream modeling

## Main Entry Points

Use [run_ort_sweep.sh](./run_ort_sweep.sh) for the canonical end-to-end workflow.

For each `(batch_size, num_indices_per_lookup)` combination it will:

1. run [run_ort_dlrm.py](./run_ort_dlrm.py) with ORT profiling enabled
2. write a combo-specific `op_shapes_{batch}_{nip}.csv`
3. write a combo-specific ORT profiling JSON under `sweep_runs/onnx_profiles/`
4. parse CPU thread usage from the profiling JSON
5. build per-op binaries from the rewritten ONNX actually used in inference
6. run DynamoRIO tracing on all per-op binaries
7. extract per-op trace features
8. merge `op_shapes + cpu_thread_detail + trace_features` into a final training CSV under `features/`
9. extract a compact selected-feature CSV under `features_selected/`

Use [run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh) when you want the same downstream sweep pipeline but need a different inference frontend such as `branch_parallel`.

Use [run_ort.sh](./run_ort.sh) for ad-hoc local runs. It supports:

- the standard full-model runner [run_ort_dlrm.py](./run_ort_dlrm.py)
- the manual branch-parallel runner [run_ort_dlrm_branch_parallel.py](./run_ort_dlrm_branch_parallel.py) when `MANUAL_BRANCH_PARALLEL=1`

The default sweep range is:

- `batch_size`: `32..2048`, step `32`
- `num_indices_per_lookup`: `100..1000`, step `50`

## Key Scripts

- [run_ort_sweep.sh](./run_ort_sweep.sh)
  End-to-end batch sweep driver.

- [run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh)
  Extensible sweep driver with `RUNNER_MODE=standard|branch_parallel`.

- [run_ort_dlrm.py](./run_ort_dlrm.py)
  Runs DLRM inference with ORT, exports `op_shapes.csv`, and writes ORT profiling JSON.

- [run_ort_dlrm_branch_parallel.py](./run_ort_dlrm_branch_parallel.py)
  Splits the rewritten DLRM graph into `bottom`, `emb_l0..emb_l7`, and `tail`, then runs branch tasks concurrently via separate ORT sessions.

- [extract_cpu_thread_usage.py](./onnx_operator_analysis/extract_cpu_thread_usage.py)
  Parses ORT profiling JSON and extracts CPU thread scheduling statistics.

- [extract_trace_features.py](./dynamorio_tracing/extract_trace_features.py)
  Parses DynamoRIO trace output and produces per-op trace feature CSVs.

- [build_training_features.py](./onnx_operator_analysis/build_training_features.py)
  Aligns ORT profile nodes to `op_shapes`, aggregates CPU-thread metrics per node, and merges them with trace features.

- [select_feature_subset.py](./onnx_operator_analysis/select_feature_subset.py)
  Extracts a compact selected-feature CSV from the merged training dataset and backfills shape fields from aligned CPU profiling CSVs when needed.

- [visualize_ort_profile_timeline.py](./onnx_operator_analysis/visualize_ort_profile_timeline.py)
  Visualizes ORT `Node` events as lane-based timelines and summarizes operator concurrency.

## Overall Directory Structure

```text
ORT/
├── README.md
├── run_ort_dlrm.py
├── run_ort_dlrm_branch_parallel.py
├── run_ort.sh
├── run_ort_sweep.sh
├── run_ort_sweep_extensible.sh
├── op_shapes.csv
├── features/
│   └── bs*_nip*.csv
├── features_selected/
│   └── bs*_nip*.csv
├── features_extensible/
│   └── bs*_nip*.csv
├── features_extensible_selected/
│   └── bs*_nip*.csv
├── sweep_runs/
│   ├── generated_onnx/
│   │   └── bs*_nip*/
│   ├── logs/
│   │   └── bs*_nip*/
│   ├── onnx_profiles/
│   │   └── bs*_nip*/
│   ├── op_shapes/
│   │   └── op_shapes_<batch>_<num_indices>.csv
│   └── sweep_summary.csv
├── sweep_runs_extensible/
│   ├── generated_onnx/
│   │   └── bs*_nip*/
│   ├── logs/
│   │   └── bs*_nip*/
│   ├── onnx_profiles/
│   │   └── bs*_nip*/
│   ├── op_shapes/
│   │   └── op_shapes_<batch>_<num_indices>.csv
│   └── sweep_summary.csv
├── onnx_operator_analysis/
│   ├── build_training_features.py
│   ├── extract_cpu_thread_usage.py
│   ├── visualize_ort_profile_timeline.py
│   └── *.md
└── dynamorio_tracing/
    ├── extract_trace_features.py
    ├── run_build_op_binaries.sh
    ├── run_drrio_all_ops.sh
    ├── out_per_op_bins_sweep/
    │   └── bs*_nip*/
    ├── drrio_traces_sweep/
    │   └── bs*_nip*/
    ├── trace_features_sweep/
    │   └── bs*_nip*.csv
    ├── out_per_op_bins_sweep_extensible/
    │   └── bs*_nip*/
    ├── drrio_traces_sweep_extensible/
    │   └── bs*_nip*/
    ├── trace_features_sweep_extensible/
    │   └── bs*_nip*.csv
    └── scripts/
        ├── generate_op_binaries.py
        ├── ort_per_op_trace.py
        ├── ort_with_markers.py
        └── single_op_runner.py
```

## Directory Layout

- `sweep_runs/op_shapes/`
  Per-combo `op_shapes_{batch}_{nip}.csv`

- `sweep_runs/onnx_profiles/<combo>/`
  Full-model ORT profiling JSON and derived CPU-thread CSVs

- `dynamorio_tracing/out_per_op_bins_sweep/<combo>/`
  Generated per-op binaries and manifest

- `dynamorio_tracing/drrio_traces_sweep/<combo>/`
  Raw DynamoRIO trace outputs

- `dynamorio_tracing/trace_features_sweep/<combo>.csv`
  Per-op trace features extracted from DynamoRIO traces

- `features/<combo>.csv`
  Final training dataset, one row per ONNX node

- `features_selected/<combo>.csv`
  Compact selected-feature dataset extracted from `features/<combo>.csv`

- `sweep_runs/sweep_summary.csv`
  Summary row per combo, including key output paths

- `sweep_runs_extensible/onnx_profiles/<combo>/`
  Extensible-sweep ORT profiles, including branch-parallel merged `ort_cann_profile_*.json`

- `features_extensible/<combo>.csv`
  Final training dataset produced by the extensible sweep

- `features_extensible_selected/<combo>.csv`
  Compact selected-feature dataset produced from the extensible sweep outputs

- `sweep_runs_extensible/sweep_summary.csv`
  Extensible-sweep summary row per combo, including `runner_mode`, `profile_json`, and `effective_onnx_path`

## Prerequisites

The workflow assumes:

- a working Python environment with `onnxruntime`, `numpy`, and `onnx`
- Ascend CANN available if `USE_CANN=1`
- DynamoRIO available at the configured `DRRUN` path
- access to the DLRM ONNX model at `ONNX_PATH`

By default:

- `ASCEND_ENV_SH=/data/qc/Ascend/ascend-toolkit/set_env.sh`
- `ONNX_PATH=<ORT PATH>/dlrm_onnx/dlrm_s_pytorch.onnx`
- `DRRUN=/data/qc/simulator/DynamoRIO-AArch64-Linux-11.3.0-1/bin64/drrun`

## Quick Start

Run the full sweep:

```bash
cd /data/qc/dlrm/ORT
bash run_ort_sweep.sh
```

Run a single combo for validation:

```bash
cd /data/qc/dlrm/ORT
BATCH_START=32 BATCH_END=32 \
NUM_INDICES_START=100 NUM_INDICES_END=100 \
MAX_COMBOS=1 \
bash run_ort_sweep.sh
```

Resume an interrupted sweep:

```bash
cd /data/qc/dlrm/ORT
RESUME=1 bash run_ort_sweep.sh
```

Run one combo through the extensible sweep with the branch-parallel runner:

```bash
cd /data/qc/dlrm/ORT
RUNNER_MODE=branch_parallel \
PYTHON_BIN=/data/qc/anaconda3/envs/ort/bin/python \
BATCH_START=32 BATCH_END=32 \
NUM_INDICES_START=100 NUM_INDICES_END=100 \
MAX_COMBOS=1 \
INTRA_THREADS=4 \
INTER_THREADS=4 \
PARALLEL_BRANCHES=2 \
bash run_ort_sweep_extensible.sh
```

Run a single local branch-parallel profile via `run_ort.sh`:

```bash
cd /data/qc/dlrm/ORT
MANUAL_BRANCH_PARALLEL=1 bash run_ort.sh
```

## Important Configuration

Common environment variables supported by [run_ort_sweep.sh](./run_ort_sweep.sh):

- `NUM_BATCHES`, `WARMUP_BATCHES`
- `INTRA_THREADS`, `INTER_THREADS`
- `USE_CANN`, `DEVICE_ID`
- `NO_REPLACE_LOOP`
- `DISABLE_GRAPH_OPTIMIZATIONS`
- `USE_NUMACTL`, `NUMA_NODE`
- `RESUME`
- `EXTRACT_FEATURES`
- `SELECT_FEATURE_SUBSET`
- `SELECT_FEATURE_SUBSET_DUR_SOURCE=avg|min|max|sum`
- `START_IDX`, `MAX_OPS`

Additional environment variables supported by [run_ort_sweep_extensible.sh](./run_ort_sweep_extensible.sh):

- `RUNNER_MODE=standard|branch_parallel`
- `PARALLEL_BRANCHES`
- `TAIL_INTRA_THREADS`
- `VERIFY_FULL_OUTPUT`
- `BRANCH_SUBMODEL_ROOT`
- `FORCE_CPU_OPS`
- `OUT_ROOT`, `PROFILE_ROOT`, `FEATURE_DATASET_ROOT`
- `FEATURE_SUBSET_ROOT`

Important defaults in the current workflow:

- `NO_REPLACE_LOOP=0`
  Loop replacement stays enabled.

- `DISABLE_GRAPH_OPTIMIZATIONS=1`
  Graph optimizations are disabled during full-model profiling so runtime node names stay closer to `op_shapes`.

- `EXTRACT_FEATURES=1`
  The final training dataset depends on DynamoRIO trace features.

- `SELECT_FEATURE_SUBSET=1`
  After merging `features/*.csv`, the sweep also writes compact selected-feature CSVs under `features_selected/` or `features_extensible_selected/`.

For `RUNNER_MODE=branch_parallel`:

- `INTER_THREADS` remains the generic sweep-level concurrency knob.
- `PARALLEL_BRANCHES` is the branch-runner-specific override.
- Effective branch concurrency is `PARALLEL_BRANCHES > 0 ? PARALLEL_BRANCHES : INTER_THREADS`.

## Standalone Usage

Run full-model ORT inference and export `op_shapes` plus profiling JSON:

```bash
cd /data/qc/dlrm/ORT
python3 run_ort_dlrm.py \
  --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx \
  --batch-size 32 \
  --num-batches 3 \
  --warmup-batches 2 \
  --num-indices-per-lookup 100 \
  --shape-csv ./op_shapes.csv \
  --enable-profiling \
  --profile-dir ./onnx_operator_analysis \
  --disable-graph-optimizations
```

Run manual branch-parallel inference and export both split profiles and a merged ORT profile:

```bash
cd /data/qc/dlrm/ORT
python3 run_ort_dlrm_branch_parallel.py \
  --onnx-path ./dlrm_onnx/dlrm_s_pytorch.onnx \
  --batch-size 32 \
  --num-batches 3 \
  --warmup-batches 2 \
  --num-indices-per-lookup 100 \
  --shape-csv ./op_shapes.csv \
  --enable-profiling \
  --profile-dir ./onnx_operator_analysis/branch_parallel \
  --intra-threads 4 \
  --inter-threads 4 \
  --parallel-branches 2 \
  --disable-graph-optimizations
```

Parse CPU thread statistics from one ORT profile JSON:

```bash
cd /data/qc/dlrm/ORT
PROFILE_DIR=./sweep_runs/onnx_profiles/bs32_nip100
PROFILE_JSON=$(find "$PROFILE_DIR" -maxdepth 1 -name 'ort_cann_profile*.json' | sort | tail -n 1)

python3 onnx_operator_analysis/extract_cpu_thread_usage.py \
  "$PROFILE_JSON" \
  --out-dir "$PROFILE_DIR"
```

Visualize operator overlap from one ORT profile JSON:

```bash
cd /data/qc/dlrm/ORT
python3 onnx_operator_analysis/visualize_ort_profile_timeline.py \
  ./onnx_operator_analysis/ort_cann_profile_2026-03-11_15-32-04.json
```

For branch-parallel runs, the profile directory can additionally contain:

- split per-submodel JSON files such as `bottom_profile_*.json` and `emb_l*_profile_*.json`
- a merged `ort_cann_profile_*.json` for compatibility with `extract_cpu_thread_usage.py`
- `branch_parallel_timeline.csv/html`
- `branch_parallel_op_timeline.csv/html`
- `branch_parallel_concurrency_segments.csv`
- `branch_parallel_op_concurrency_segments.csv`

Build the final merged training CSV from existing artifacts:

```bash
cd /data/qc/dlrm/ORT
BATCH_SIZE=32
NUM_INDICES=100
COMBO=bs${BATCH_SIZE}_nip${NUM_INDICES}
PROFILE_DIR=./sweep_runs/onnx_profiles/$COMBO
PROFILE_STEM=$(basename "$(find "$PROFILE_DIR" -maxdepth 1 -name 'ort_cann_profile*.json' | sort | tail -n 1)" .json)
SHAPE_CSV=$(find ./sweep_runs/op_shapes -maxdepth 1 -name "op_shapes_${BATCH_SIZE}_${NUM_INDICES}.csv" | head -n 1)
TRACE_FEATURES=./dynamorio_tracing/trace_features_sweep/$COMBO.csv

python3 onnx_operator_analysis/build_training_features.py \
  --op-shapes "$SHAPE_CSV" \
  --cpu-detail "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_detail.csv" \
  --trace-features "$TRACE_FEATURES" \
  --aligned-cpu-detail-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_detail_aligned.csv" \
  --cpu-agg-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_node_aggregated.csv" \
  --unmatched-out "$PROFILE_DIR/${PROFILE_STEM}_cpu_thread_unmatched.csv" \
  --out ./features/$COMBO.csv \
  --batch-size "$BATCH_SIZE" \
  --num-indices-per-lookup "$NUM_INDICES"
```

Extract the compact selected-feature CSV from an existing merged training CSV:

```bash
cd /data/qc/dlrm/ORT
python3 onnx_operator_analysis/select_feature_subset.py \
  --input ./features/$COMBO.csv \
  --output ./features_selected/$COMBO.csv
```

## Output Semantics

The final `features/<combo>.csv` keeps one row per ONNX node from `op_shapes`.

The final `features_selected/<combo>.csv` keeps one row per ONNX node as well, but only retains the compact feature subset used for downstream experiments. By default `dur_us` is sourced from `cpu_dur_us_avg`, and `input_type_shape` / `output_type_shape` are backfilled from `*_cpu_thread_detail_aligned.csv` when they are not already present in the merged dataset.

## Compact Selected Features

The sweep can additionally produce a compact dataset under `features_selected/` and `features_extensible_selected/` via [select_feature_subset.py](./onnx_operator_analysis/select_feature_subset.py).

For each combo, the sweep writes:

- `features_selected/<combo>.csv`
- `features_selected/all_features.csv`
- `features_extensible_selected/<combo>.csv`
- `features_extensible_selected/all_features.csv`

The selected dataset keeps one row per ONNX node and currently includes:

- metadata: `batch_size`, `num_indices_per_lookup`, `node_name`, `op_type`, `trace_op_name`
- operator shape and size fields: `input_type_shape`, `output_type_shape`, `output_size`, `activation_size`, `parameter_size`
- trace instruction and memory counts: `total_instructions`, `total_loads`, `total_stores`, `load_store_ratio`, `num_threads`
- reuse-time summary: `reuse_time_mean` and all available `reuse_time_bin_<n>_pct` columns
- reuse-distance summary: `reuse_distance_mean`, `reuse_distance_median`, `reuse_distance_std`, `reuse_distance_unique_cache_lines_per_k_accesses`, `reuse_distance_instruction_accesses`, `reuse_distance_data_accesses`
- opcode-mix features: `opc_branch_ratio`, `opc_fp_convert`, `opc_fp_load_simd`, `opc_fp_math`, `opc_fp_move`, `opc_fp_store_simd`, `opc_math`, `opc_simd`
- duration: `dur_us`

Field mapping rules:

- `dur_us` defaults to `cpu_dur_us_avg`; change it with `SELECT_FEATURE_SUBSET_DUR_SOURCE=avg|min|max|sum`
- `output_size`, `activation_size`, and `parameter_size` are taken from the merged dataset's CPU-aggregated size columns
- `input_type_shape` and `output_type_shape` are backfilled from `*_cpu_thread_detail_aligned.csv` when needed
- reuse-time bins are discovered dynamically from the merged feature header, so the exact number of `reuse_time_bin_<n>_pct` columns can expand if the upstream extractor adds more bins

Each row may include:

- canonical `node_idx`, `node_name`, `op_type`
- trace features from DynamoRIO
- aggregated CPU-thread profiling features from ORT
- `has_trace_features`
- `has_cpu_profile`
- `cpu_profile_missing_reason`

`Constant` and other lightweight nodes often remain in the dataset even when they do not produce CPU thread scheduling stats in ORT profiling. Those rows usually have:

- `has_trace_features=1`
- `has_cpu_profile=0`
- `cpu_profile_missing_reason=constant_or_no_thread_stats`

## Alignment Rules

Correct results depend on keeping these aligned for the same combo:

- `batch_size`
- `num_indices_per_lookup`
- `op_shapes_{batch}_{nip}.csv`
- the rewritten ONNX actually used in ORT inference
- the per-op binaries built from that rewritten ONNX

If these drift, the pipeline may still finish, but the final dataset will not be strictly shape-equivalent to the target model run.

## Troubleshooting

- `extract_features.log` stays empty while running
  Python output may be buffered unless unbuffered mode is used. The current sweep uses `python3 -u` for feature extraction.

- full-model profiling node count does not match per-op binary count
  ORT profiling reflects runtime execution events, while per-op binaries come from ONNX graph nodes. `Constant` nodes and optimized or fused runtime nodes can cause differences.

- some rows have `has_cpu_profile=0`
  This is expected for nodes that do not emit `thread_scheduling_stats` in ORT profiling, especially `Constant` nodes.

- per-op build uses the wrong ONNX
  The current sweep resolves the effective rewritten ONNX from the inference stage and passes that path into per-op binary generation.

## Notes

- `run_ort_sweep.sh` is the preferred entry point for reproducible dataset generation.
- `run_ort_sweep_extensible.sh` is the preferred place to add new inference frontends without destabilizing active jobs that still use `run_ort_sweep.sh`.
- The sweep writes combo-specific outputs to avoid overwriting artifacts across different `(batch_size, num_indices_per_lookup)` pairs.
