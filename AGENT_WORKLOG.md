# Agent Worklog

This file is the persistent handoff and change record for `/data/qc/dlrm/ORT/single_op_stage1_mlp`.

Future agents working in this directory must read this file before changing code.

## Project Snapshot

### Purpose

This project is a self-contained single-operator latency modeling pipeline for ORT DLRM. It builds train/val/test tables from all `features_extensible_case_*` cases, engineers stage-1-style features, and trains an MLP to predict per-operator latency.

### Main Components

- `dataset_builder.py`
  Builds `dataset_full.csv`, `train.csv`, `val.csv`, `test.csv`, and `dataset_summary.json`.
- `feature_contract.py`
  Defines the training feature contract, metadata columns, and target column.
- `feature_engineering.py`
  Reconstructs shape-derived and operator-specific numeric features locally.
- `train_mlp.py`
  Trains a PyTorch MLP, exports artifacts, saves preprocessing state, and supports `torch_npu`.
- `run_pipeline.py`
  Runs dataset building plus training in one command.
- `infer_mlp_onnx.py`
  Uses ONNX Runtime for exported-model inference, including NPU-oriented provider selection.
- `analyze_feature_correlation.py`
  Produces feature-feature and feature-target correlation CSVs and heatmaps.
- `plot_loss_curve.py`
  Draws loss curves from saved training history.

### Current Data and Label Conventions

- Split ratio: train/val/test = `7:2:1`
- Grouped split key: `sample_group`
- Label column: `label_operator_actual_dur_us`
- Profile label policy:
  - drop the earliest profile batch by default
  - use the mean of the remaining batches as the label
  - optionally filter unstable rows using profile-derived volatility metrics
- Default instability filter:
  - metric: `last2_range_ratio`
  - threshold: `0.20`

### Current Feature Conventions

- Categorical features remain:
  - `op_type`
  - `node_scope`
  - `node_name_normalized`
  - `arch_embedding_size`
  - `arch_mlp_bot`
  - `arch_mlp_top`
- Numeric features were pruned using `artifacts/latest/correlation_analysis/numeric_feature_target_correlation.csv`.
- Any numeric feature with absolute target correlation below `0.1` was removed from the active training contract.
- A first batch of stage-2-style candidate features is now supported.
- Promoted into the active contract after candidate correlation screening:
  - `hw_ratio_working_set_to_l1d_active_bytes`
  - `hw_ratio_working_set_to_l2_active_bytes`
  - `hw_ratio_working_set_to_l3_active_bytes`
  - `local_ctx_same_op_overlap_ratio_mean`
  - `comp_feat_pressure_ws_to_l2_ratio`
  - `comp_feat_pressure_ws_to_l3_ratio`
- Lower-correlation stage-2 candidates remain analysis-only until explicitly approved.

### Training Conventions

- Numeric features: median imputation + standardization
- Categorical features: one-hot encoding
- Target transform: `log1p` by default unless disabled
- Loss: `MSELoss`
- Training backend: PyTorch
- NPU training path: `torch_npu`
- Deployment path: ONNX Runtime

### Operational Rules

- Keep this project self-contained. Do not add dependencies on scripts from other directories for core pipeline steps.
- When changing project behavior, update this file in the same task.
- After each completed modification in this directory, create a git commit in the independent repository rooted at `/data/qc/dlrm/ORT/single_op_stage1_mlp`.
- Do not commit project changes into the parent `ORT` repository.

## Change History

### 2026-03-26 - Initial project snapshot and workflow guardrails

Request summary:
- Add persistent agent instructions, a project-wide worklog, and a repo-local skill so future agent work in this directory always starts from shared context and ends with a git save.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENTS.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/.codex/skills/ort-single-op-stage1-mlp/SKILL.md`

Project state at this point:
- Built a self-contained dataset pipeline from ORT case CSVs and op-shape/profile artifacts.
- Added manual case selection support.
- Switched training to PyTorch MLP with preprocessing state persistence.
- Added `torch_npu` initialization for NPU training and ONNX export/inference helpers.
- Added correlation analysis outputs and top-10 target-correlation bar charts.
- Added profile-batch cleaning:
  - drop first profile batch
  - filter unstable samples using configurable profile metrics
- Pruned low-correlation numeric features from the active feature contract.

Validation run:
- File creation only for this task. No runtime validation required beyond git status review before commit.

Open risks:
- The repo worktree contains many unrelated untracked/modified files, so future commits must stage only task-relevant files.

### 2026-03-26 - Switch guardrails to the independent project repo

Request summary:
- Do not commit governance changes into the parent `ORT` repository; manage this project through its own standalone git repository.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENTS.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/.codex/skills/ort-single-op-stage1-mlp/SKILL.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/.gitignore`

Behavior changes:
- Governance instructions now explicitly require using the independent git repository rooted at this directory.
- The local skill now lives inside this project so the rule set can be versioned with the project itself.
- Artifacts and cache files are ignored in the independent repository.

Validation run:
- Verified that `/data/qc/dlrm/ORT/single_op_stage1_mlp` is already an independent git repository.

Open risks:
- This independent repository currently starts from an empty history, so the first commit should be treated as the initial project snapshot.

### 2026-03-26 - Add stage-2-style candidate feature screening

Request summary:
- Evaluate whether hardware-ratio features and stage-2 concurrency/context features can help the single-op MLP.
- First run target-correlation analysis, then automatically keep the higher-correlation candidates and leave weaker ones for manual review.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_contract.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_feature_correlation.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`

Behavior changes:
- The dataset builder can now reconstruct stage-2-style candidate columns locally from:
  - hardware profile YAML
  - `branch_parallel_op_timeline.csv`
  - `branch_parallel_concurrency_segments.csv`
  - `branch_parallel_op_concurrency_segments.csv`
- Candidate hardware/context/concurrency features are exported in the dataset even when they are not active training inputs.
- Correlation analysis now emits `stage2_candidate_feature_target_correlation.csv` and a keep/review split using `|corr| >= 0.1`.
- Promoted into the active training contract:
  - `hw_ratio_working_set_to_l1d_active_bytes`
  - `hw_ratio_working_set_to_l2_active_bytes`
  - `hw_ratio_working_set_to_l3_active_bytes`
  - `local_ctx_same_op_overlap_ratio_mean`
  - `comp_feat_pressure_ws_to_l2_ratio`
  - `comp_feat_pressure_ws_to_l3_ratio`
- Left in analysis-only review:
  - `hw_ratio_threads_to_total_cores`
  - `local_ctx_overlap_ratio_mean`
  - `local_ctx_cross_task_overlap_ratio_mean`
  - `local_ctx_mean_other_active_mean`
  - `local_ctx_mean_other_tasks_mean`
  - `combo_task_parallel_fraction`
  - `combo_task_weighted_mean_parallel_concurrency`
  - `combo_op_parallel_fraction`
  - `combo_op_weighted_mean_parallel_concurrency`
  - `comp_feat_pressure_threads`

Validation run:
- `python3 -m py_compile ORT/single_op_stage1_mlp/*.py`
- `python3 ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_stage2_candidate_smoke --selected-cases case_1_1_1 --max-files-per-case 2`
- `python3 ORT/single_op_stage1_mlp/dataset_builder.py --output-dir ORT/single_op_stage1_mlp/artifacts/latest/dataset_stage2_candidates_sample --max-files-per-case 20`
- `python3 ORT/single_op_stage1_mlp/analyze_feature_correlation.py --input-csv ORT/single_op_stage1_mlp/artifacts/latest/dataset_stage2_candidates_sample/dataset_full.csv --output-dir ORT/single_op_stage1_mlp/artifacts/latest/correlation_analysis_stage2_candidates_sample`

Open risks:
- The keep/review decision is currently based on a representative all-case sample (`max-files-per-case=20`) rather than the full dataset, because the full rebuild with the new stage-2 candidate extraction path was too slow for this turn.
- The promoted hardware ratio features are partially collinear with existing working-set / size features, so downstream retraining should confirm whether they provide incremental gain.

### 2026-03-27 - Add per-op-type val/test metric analysis

Request summary:
- Add a local script that analyzes validation and test prediction quality by `op_type`.
- Keep the workflow self-contained inside this project and make it easy to run from an existing model artifact directory.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_op_type_metrics.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added `analyze_op_type_metrics.py` to load `predictions_val.csv` and `predictions_test.csv` from a model directory.
- The script automatically resolves `data_dir` from `metrics.json` unless `--data-dir` is passed explicitly.
- If prediction CSVs do not already contain `op_type`, the script joins them back to the matching `val.csv` / `test.csv` rows using `row_uid`.
- The script now exports, for each analyzed split:
  - a full per-`op_type` metric table with `row_count`, `mae_us`, `rmse_us`, `r2`, `mape`, `median_ape`, `p90_ape`, target/prediction mean and median, and mean bias
  - a ranked table filtered by a configurable minimum sample count
  - a Top-N worst-op-type bar chart using a configurable ranking metric
- README now documents the new analysis entry point and output files.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_op_type_metrics.py --model-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_clean_2_stage_2`

Open risks:
- The script assumes `row_uid` remains a stable one-to-one key between prediction CSVs and dataset split CSVs.
- Worst-op-type ranking can be dominated by low-support operator types if `--min-count-for-ranking` is set too low.

### 2026-03-27 - Add a separate trace-feature proxy MLP pipeline

Request summary:
- Add a separate PyTorch MLP path that predicts trace-derived features from non-trace inputs.
- Keep this implementation separate from the main latency MLP so the contracts, scripts, and artifacts do not get mixed together.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/trace_feature_contract.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/train_trace_feature_mlp.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_trace_feature_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added `trace_feature_contract.py` to define a dedicated non-trace input feature contract and a dedicated trace-target contract for the proxy model.
- Added `train_trace_feature_mlp.py` as a self-contained multi-output PyTorch MLP trainer that:
  - uses only non-trace features as inputs
  - predicts a separate vector of trace-derived targets
  - supports `torch_npu`
  - saves a separate model checkpoint, preprocessing state, per-target metrics, and prediction CSVs
- Added `run_trace_feature_pipeline.py` as a one-command dataset-build plus trace-proxy-training entry point.
- README now documents the new independent trace-feature proxy path, its targets, its outputs, and the fact that raw counters such as `total_instructions` are not in the current default dataset export.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/train_trace_feature_mlp.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/run_trace_feature_pipeline.py --help`
- Verified that the default trace targets and non-trace input columns are present in `artifacts/latest/dataset_all_clean_2_stage_2/train.csv`

Open risks:
- This environment currently does not have `torch` installed, so the new trace-feature proxy MLP could not be run end-to-end in this turn.
- The current prepared dataset export does not include raw DynamoRIO count columns such as `total_instructions`, `total_loads`, and `total_stores`, so those are only supported as optional user-selected targets for future dataset variants that include them.

### 2026-03-27 - Support both trace and no-trace dataset dialects

Request summary:
- Make `single_op_stage1_mlp` properly support both trace and no-trace feature CSV dialects.
- For no-trace inputs, do not silently keep trace-only features in the training contract.
- Recover `num_threads` explicitly from the sweep configuration logs when the source CSV does not provide it.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_contract.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_trace_feature_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added explicit dataset dialect support with `trace`, `no_trace`, and `auto`.
- `feature_contract.py` now separates shared numeric features from trace-only numeric features and can build the active training contract per dialect.
- `dataset_builder.py` now:
  - detects each source CSV's observed dialect from its columns
  - resolves a dataset-wide dialect and writes it into `feature_columns.json` and `dataset_summary.json`
  - removes trace-only features from the active export contract when the resolved dialect is `no_trace`
  - recovers missing `num_threads` from `sweep_runs_extensible_case_*/logs/<combo>/run_ort.log` by parsing the configured `Intra threads`
  - skips non-canonical `features_extensible_case_*` directories such as suffix variants instead of failing discovery
- `train_mlp.py` now reads `feature_columns.json` and trains on the dataset's emitted feature contract instead of always assuming the trace feature list.
- `run_pipeline.py` and `run_trace_feature_pipeline.py` now pass through `--feature-dialect`.
- README now documents both dialects, the explicit `num_threads` recovery path, and the dataset manifest fields that record dialect resolution.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_case9_no_trace_smoke --selected-cases case_9_4_4 --max-files-per-case 2 --feature-dialect auto`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_case2_trace_smoke --selected-cases case_2_4_4 --max-files-per-case 2 --feature-dialect auto`
- Verified from the generated manifests and CSVs that:
  - `case_9_4_4` resolves to `feature_dialect=no_trace`
  - `case_2_4_4` resolves to `feature_dialect=trace`
  - `num_threads` is recovered as `4` for the no-trace smoke dataset
  - `load_store_ratio` is present only in the trace dialect manifest

Open risks:
- The separate trace-feature proxy training path still requires trace-derived target columns to exist in the prepared dataset; using `--feature-dialect no_trace` there will not produce trainable trace targets by itself.
- Mixed trace and no-trace source sets currently resolve to `no_trace` in `auto` mode so the shared contract stays safe, which is conservative but may hide available trace columns when users intentionally mix both sources.

### 2026-03-27 - Add configurable hardware profile selection

Request summary:
- Add a `--hardware-profile` parameter so the single-op dataset pipeline can switch hardware architecture profiles without editing source code.
- Confirm whether the current built-in Kunpeng hardware profile already contains the requested cache, memory, frequency, instruction-width, and pipeline parameters.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_trace_feature_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `dataset_builder.py` now accepts `--hardware-profile /path/to/profile.yaml` and forwards it to the local hardware-aware feature reconstruction path.
- `run_pipeline.py` and `run_trace_feature_pipeline.py` now pass the same parameter through to dataset building.
- `dataset_summary.json` now records the resolved `hardware_profile_path` used to build the dataset.
- README now documents how to switch hardware architectures with a custom profile YAML instead of editing the default path in code.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_hw_profile_smoke --selected-cases case_9_4_4 --max-files-per-case 1 --feature-dialect auto --hardware-profile /data/qc/dlrm/ORT/model/hardware_profiles/kunpeng920_gem5.yaml`
- Verified that `/tmp/single_op_stage1_hw_profile_smoke/dataset_summary.json` records `/data/qc/dlrm/ORT/model/hardware_profiles/kunpeng920_gem5.yaml` under `hardware_profile_path`

Open risks:
- The hardware profile parameter currently only affects the stage-2-style hardware/context features derived during dataset building; it does not automatically validate that a custom YAML contains every field needed for all downstream derived ratios.
- The built-in YAML still lacks explicit instruction bit-width and memory bandwidth fields, so switching profiles currently changes cache/core topology more directly than ISA throughput assumptions.

### 2026-03-27 - Move the default hardware profile into the project and seed it with host-plus-paper values

Request summary:
- Create a dedicated hardware-profile directory inside `single_op_stage1_mlp`.
- Add a YAML profile that includes cache size/latency, memory bandwidth/latency, instruction width/latency, CPU frequency, and pipeline width.
- Make dataset extraction use the project-local profile by default.
- Check the local machine's actual architecture parameters and compare them with the cited Kunpeng papers.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/hardware_profile/kunpeng920_host_4socket.yaml`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The default hardware profile path now points to the project-local file `hardware_profile/kunpeng920_host_4socket.yaml` instead of the external `ORT/model/hardware_profiles` directory.
- The new YAML keeps host-observable values for topology/cache sizes and adds paper-reference values for:
  - memory bandwidth
  - memory latency
  - SIMD width
  - FP instruction latency/throughput
  - pipeline width
- `load_hardware_features()` now ignores both `paper_cross_check` and `host_cross_check` sections so those audit values can be kept in the YAML without polluting model features.
- `dataset_summary.json` now records the effective hardware profile path even when the default project-local profile is used implicitly.

Validation run:
- `lscpu`
- `lscpu -C`
- `numactl -H`
- `grep -m 2 -E 'model name|cpu MHz|Features' /proc/cpuinfo`
- `python3 - <<'PY' ... from feature_engineering import HARDWARE_PROFILE_PATH, load_hardware_features ... PY`
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_internal_hw_default2 --selected-cases case_9_4_4 --max-files-per-case 1 --feature-dialect auto`
- Verified that `/tmp/single_op_stage1_internal_hw_default2/dataset_summary.json` records `/data/qc/dlrm/ORT/single_op_stage1_mlp/hardware_profile/kunpeng920_host_4socket.yaml`

Observed host-versus-paper findings:
- The local machine is genuinely `aarch64 / HiSilicon / Kunpeng-920`.
- Directly observed local topology:
  - `4 sockets`
  - `48 cores per socket`
  - `8 NUMA nodes`
  - `64KiB L1I`, `64KiB L1D`, `512KiB L2`
  - `24MiB L3` shared by each `24-core` NUMA group
- These observations are only partially consistent with the cited papers:
  - cache sizes at L1/L2 and the 4-wide pipeline story are consistent
  - the local host topology does not match the paper's single-socket / 2-socket examples exactly
  - the local host exposes `48 cores per socket` and `24MiB L3 per 24-core NUMA group`, while the papers/reference materials describe different socket-level totals

Open risks:
- CPU frequency could not be read reliably from the local kernel interfaces on this host, so the YAML keeps `2.6GHz` as a paper/reference value rather than a direct machine readout.
- Cache latency and FP instruction latency are still reference values from papers or prior approximations, not locally measured hardware counters on this machine.

### Entry Template

Use this format for future updates:

```text
### YYYY-MM-DD - Short title

Request summary:
- ...

Files changed:
- ...

Behavior changes:
- ...

Validation run:
- ...

Open risks:
- ...
```
