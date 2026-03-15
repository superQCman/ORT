# ORT Operator Modeling

`ORT/model` is an isolated workspace for DLRM CPU performance modeling. It does not change the existing ORT feature-extraction or tracing flow; it only consumes the outputs that already exist under `ORT/features_selected/`, `ORT/dynamorio_tracing/`, and the gem5 config in `ops_profile/concorde/simulatoin/kunpeng920.py`.

## What it builds

This directory focuses on the operator-level stage of the workflow:

1. Load the compact per-op feature table from `features_selected/`.
2. Add derived shape, memory-footprint, and hardware-parameter features.
3. Optionally merge gem5 labels parsed from `stats.txt`.
4. Train and compare several regressors.

The default model inputs include:

- ORT/DynamoRIO features already extracted in `features_selected/*.csv`
- engineered tensor-shape features from `input_type_shape` and `output_type_shape`
- hardware-profile fields from `hardware_profiles/kunpeng920_gem5.yaml`
- operator-aware hardware context such as `hw_core_active_cores` and active-cache-byte features derived from `num_threads`

The default labels include:

- `label_real_dur_us`
- `label_real_ipc_proxy`
- `label_gem5_sim_us`
- `label_gem5_weighted_ipc`

`label_real_ipc_proxy` is a quick debugging target derived from:

`total_instructions / (dur_us * cpu_clock_ghz * 1e3 * num_threads)`

It is useful before full gem5 labels are ready, but the architecture-sensitive training target should eventually switch to gem5 labels such as `label_gem5_weighted_ipc` or `label_gem5_sim_us`.
When interpreting hardware features, treat `hw_core_total_cores` as the machine upper bound and `hw_core_active_cores` as the per-operator concurrency actually exposed to the model.

## Files

- `sync_kunpeng920_profile.py`
  Extracts the current gem5 configuration into a reusable YAML hardware profile.
- `build_operator_dataset.py`
  Merges `features_selected` with hardware features and optional gem5 stats.
- `train_operator_model.py`
  Trains and compares `ridge`, `random_forest`, and `hist_gbdt` regressors.
- `run_collect_gem5_labels.sh`
  Wraps the existing per-combo gem5 runner and writes outputs under `ORT/model/gem5_runs/`.
- `run_train_operator_models.sh`
  End-to-end wrapper that builds a dataset and trains the models.

## Recommended workflow

### 1. Refresh the hardware profile after editing gem5 config

```bash
python3 ORT/model/sync_kunpeng920_profile.py
```

### 2. Quick baseline using real-machine labels

```bash
bash ORT/model/run_train_operator_models.sh
```

This uses `label_real_dur_us` by default and is the fastest way to validate the feature/model pipeline.

### 3. Collect gem5 labels for one or more sweep combos

```bash
COMBO_FILTER=bs32_nip100 \
MAX_COMBOS=1 \
MAX_OPS=8 \
PARALLEL_JOBS=2 \
bash ORT/model/run_collect_gem5_labels.sh
```

For a full tmux run, remove `MAX_OPS` and broaden the combo selection. Outputs will land under:

`ORT/model/gem5_runs/<combo>/`

Because the output path contains `bs*_nip*`, `build_operator_dataset.py` can merge the gem5 stats back to the correct feature rows without touching the existing ORT pipeline.

### 4. Train on gem5 labels

```bash
GEM5_ROOT=/data/qc/dlrm/ORT/model/gem5_runs \
TARGET=label_gem5_weighted_ipc \
bash ORT/model/run_train_operator_models.sh
```

You can also predict latency directly:

```bash
GEM5_ROOT=/data/qc/dlrm/ORT/model/gem5_runs \
TARGET=label_gem5_sim_us \
bash ORT/model/run_train_operator_models.sh
```

## Model choices

Three model families are included by default:

- `ridge_log`
  A simple linear baseline after scaling. Good for sanity checks and feature-audit work.
- `random_forest_log`
  Robust to mixed nonlinear interactions and usually a strong latency baseline on tabular data.
- `hist_gbdt_log`
  A gradient-boosted tree model for heavier nonlinear structure. This is the best default candidate when the label is IPC or simulated latency.

Validation uses grouped folds by `sample_group` by default, which maps to one `bs*_nip*` combo. This avoids leaking the same workload shape configuration into both train and test folds.

## Hardware-source notes

The default YAML profile tracks the current gem5 approximation:

- 48 cores, 24 cores per die
- 4-wide frontend/backend widths
- 64 KiB L1I, 64 KiB L1D, 512 KiB private L2
- 32 MB L3 per die
- 128 ROB / 64 LQ / 48 SQ
- DDR4_2400_16x4 baseline

The Huawei paper in [鲲鹏920.pdf](/data/qc/dlrm/ORT/鲲鹏920.pdf) is used as a cross-check reference. It confirms the 64 KiB L1I/L1D, 512 KiB L2, and 4-wide pipeline assumptions, but also documents silicon-level details such as 64 MB total LLC and up to DDR4-2933 that are not identical to the current gem5 approximation. For modeling, the YAML profile should follow the gem5 config you actually simulate.

One important distinction is whole-chip capacity versus per-operator active resources. The YAML profile keeps full-machine limits such as `hw_core_total_cores=48`, but the dataset builder now also derives operator-aware features from `num_threads`, for example:

- `hw_core_active_cores = min(num_threads, hw_core_total_cores)`
- `hw_cache_l1d_active_bytes = hw_cache_l1d_size * hw_core_active_cores`
- `hw_cache_l2_active_bytes = hw_cache_l2_size * hw_core_active_cores`
- `hw_cache_l3_active_bytes = hw_cache_l3_per_die_size * ceil(hw_core_active_cores / hw_core_cores_per_die)`

For gem5 collection, `BMK_THREADS` should match the thread count used by the traced operator binaries. In the current selected-feature dataset, `num_threads` is consistently `4`, so the default `BMK_THREADS=4` remains aligned with the existing traces.
