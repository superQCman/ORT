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

### 2026-03-27 - Switch the default modeling scope to a single NUMA domain

Request summary:
- Because current DLRM workloads are pinned within a single NUMA node, switch the default hardware profile from whole-host scope to single-NUMA scope.
- Re-anchor core count, L3 sharing domain, memory channels, and local bandwidth to the execution-local NUMA domain instead of the 4-socket machine total.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/hardware_profile/kunpeng920_single_numa.yaml`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The default hardware profile path now points to `hardware_profile/kunpeng920_single_numa.yaml`.
- The new default profile models one local NUMA domain with:
  - `24` local cores
  - `24MiB` shared local L3
  - `4` local memory channels
  - `100 GB/s` local memory bandwidth approximation
- The previous `kunpeng920_host_4socket.yaml` remains in the repo as a whole-host audit/reference profile, but it is no longer the default modeling scope.
- README now explains that the default profile is single-NUMA because the current DLRM runs do not span the full host.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/*.py`
- `python3 - <<'PY' ... from feature_engineering import HARDWARE_PROFILE_PATH, load_hardware_features ... PY`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_stage1_single_numa_smoke --selected-cases case_9_4_4 --max-files-per-case 1 --feature-dialect auto`
- Verified that the loaded default hardware values include:
  - `hw_core_total_cores = 24`
  - `hw_cache_l3_per_die_size = 24MiB`
  - `hw_memory_channels_total = 4`
  - `hw_memory_bandwidth_gb_s_total = 100`
- Verified that `/tmp/single_op_stage1_single_numa_smoke/dataset_summary.json` records `/data/qc/dlrm/ORT/single_op_stage1_mlp/hardware_profile/kunpeng920_single_numa.yaml`

Open risks:
- The single-NUMA bandwidth value is still a derived approximation from the paper's socket-level bandwidth, not a locally measured NUMA-specific STREAM result on this host.
- Some field names such as `cores_per_die` and `l3_per_die` are retained for backward compatibility with the existing feature-engineering code even though the current modeling scope is really a NUMA-local L3 sharing domain.

### 2026-03-27 - Add cache-tier analytical features and residual-target training mode

Request summary:
- Implement the soft/hardware fusion refinement plan inside `single_op_stage1_mlp` only.
- Add a minimal analytical feature family centered on cache-tier fit and expected latency.
- Keep the direct MLP conservative by promoting only `ana_cache_fit_level` and `ana_expected_latency_ns` into the training contract.
- Export the broader `ana_*` family for analysis/residual support, and add an `analytical_residual` training mode built on `ana_base_us`.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_contract.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/infer_mlp_onnx.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added two direct analytical features to both `trace` and `no_trace` training contracts:
  - `ana_cache_fit_level`
  - `ana_expected_latency_ns`
- Added exported analytical support columns:
  - `ana_compute_ops`
  - `ana_roofline_base_us`
  - `ana_base_us`
  - `ana_mem_bw_time_us`
  - `ana_latency_proxy_us`
  - `ana_ridge_gap`
- The analytical feature family now follows fixed soft/hardware rules:
  - cache fit is determined by `working_set / active_cache_capacity`
  - expected latency is selected from `L1/L2/L3 response latency` or `local_mem_delay_ns`
  - compute baseline uses `vector_sp_fma` throughput, SIMD width, CPU frequency, and active cores
  - `Gemm` and `MatMul` now share the same gemm-like shape recovery path for analytical compute estimation
- Dataset export now includes:
  - `label_operator_residual_log = log(label_operator_actual_dur_us / ana_base_us)`
  - manifest fields for `analytical_base_column`, `residual_target_column`, and analytical feature columns
- `train_mlp.py` now supports:
  - `--target-mode direct_us`
  - `--target-mode analytical_residual`
- In analytical residual mode:
  - the model trains on `label_operator_residual_log`
  - predictions are reconstructed as `ana_base_us * exp(pred_residual_log)`
  - predictions CSVs now carry `ana_base_us` and residual-target columns when applicable
- ONNX inference now mirrors the same reconstruction logic, so residual-trained exports remain deployable.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/feature_contract.py /data/qc/dlrm/ORT/single_op_stage1_mlp/feature_engineering.py /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py /data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py /data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py /data/qc/dlrm/ORT/single_op_stage1_mlp/infer_mlp_onnx.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/train_mlp.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/run_pipeline.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/infer_mlp_onnx.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_ana_smoke_trace --selected-cases case_1_1_1 --max-files-per-case 1`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/dataset_builder.py --output-dir /tmp/single_op_ana_smoke_notrace --selected-cases case_9_4_4 --feature-dialect no_trace --max-files-per-case 1`
- Verified both smoke datasets contain the new `ana_*` columns, `label_operator_residual_log`, and manifest analytical metadata.
- Verified residual reconstruction helper with `ana_base_us=[10, 100]` and residual logs `[0, log(2)]` produces `[10, 200]`.

Open risks:
- This environment still does not have `torch` installed, so a full end-to-end MLP retraining run could not be executed in this turn.
- The newly exported analysis-only `ana_*` columns are intentionally not yet part of the direct feature contract; future ablations should confirm whether any of them deserve promotion beyond the current cache-tier pair.

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

### 2026-03-27 - Add a dedicated Analytical Model V2 design document

Request summary:
- Add a new standalone document that explains Analytical Model V2 in detail.
- Base the explanation on ORT CPU kernel behavior for `Gather`, `ReduceSum`, `Gemm`, `MatMul`, `Transpose`, and `Concat`.
- Include modeling rationale, pseudocode, new-feature definitions, parameter availability in the current dataset, flowcharts, and the overall implementation/validation plan.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added `ANALYTICAL_MODEL_V2.md` as the project-local design document for the next-generation soft/hardware analytical feature stack.
- The document now:
  - explains why the current global `ana_*` family is insufficient
  - maps ORT CPU kernel behavior to family-specific analytical models
  - includes pseudocode for the six target operator families
  - defines the proposed V2 analytical features and what each one means
  - records, for each required parameter, whether it already exists in `dataset_full.csv`, can be reconstructed, or requires new export
  - includes multiple Mermaid flowcharts for cache-tier decision logic, overall feature flow, and analytical-feature gating
  - documents the recommended rollout and validation order
- README now links directly to the new design document near the top-level project overview.

Validation run:
- `sed -n '1,260p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `cd /data/qc/dlrm/ORT/single_op_stage1_mlp && git diff --check`
- `python3 - <<'PY' ... verify mermaid block count, pseudocode block count, and ORT source-link count in ANALYTICAL_MODEL_V2.md ... PY`
- `python3 - <<'PY' ... inspect /tmp/single_op_ana_smoke_trace/feature_columns.json and dataset_full.csv columns to confirm which parameters are currently exported ... PY`

Open risks:
- The document is intentionally a design/spec artifact; it does not yet implement the proposed V2 family-specific analytical features in code.
- Some parameters referenced by the document, such as explicit `M/N/K`, transpose regime labels, and concat input-byte summaries, are currently reconstructible from existing inputs but are not exported as dedicated dataset columns yet.

### 2026-03-28 - Clarify the ReduceSum explanation in Analytical Model V2

Request summary:
- Tighten the `ReduceSum` explanation in the Analytical Model V2 document.
- Merge the overlapping ideas of “reduce axis continuity” and “streaming vs stride-heavy input reads” into one clearer point.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The `ReduceSum` summary table now describes the main cache-sensitive factor as whether the input being accumulated is contiguous in memory.
- The detailed `ReduceSum` rationale now lists two primary factors instead of three:
  - whether the accumulated input is contiguous
  - whether partial sums can stay resident in L1/L2
- This removes wording overlap while preserving the meaning of `ana_reduce_strided_flag` as a stride-sensitive regime indicator.

Validation run:
- `rg -n "ReduceSum|ana_reduce_strided_flag|reduce axis 是否连续|streaming|stride-heavy" /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `sed -n '100,320p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `cd /data/qc/dlrm/ORT/single_op_stage1_mlp && git diff --check`

Open risks:
- This is a documentation-only clarification; it does not yet revise any code or feature extraction logic for `ana_reduce_strided_flag`.

### 2026-03-28 - Converge Analytical Model V2 to a small set of primary features

Request summary:
- Reduce the perceived clutter in the per-operator Analytical Model V2 feature lists.
- Make the document explicitly separate primary performance-oriented analytical features from internal intermediate variables.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added a new “feature convergence” section that states the V2 contract rule: each operator family should keep at most `2~3` primary analytical features in the direct contract.
- The document now explicitly distinguishes:
  - primary features such as `base_us`, effective throughput, and one key regime/bottleneck indicator
  - intermediate variables such as `stream_bytes`, `unique_rows_est`, `effective_weight_bytes`, and `dispatch_penalty_us`
- For each operator family (`Gather`, `ReduceSum`, `Gemm`, `MatMul`, `Transpose`, `Concat`), the document now lists the final recommended primary analytical features that should survive feature contraction.

Validation run:
- `rg -n "新特征|base_us|throughput|吞吐|中间变量|direct analytical|analysis-only" /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `sed -n '140,560p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `cd /data/qc/dlrm/ORT/single_op_stage1_mlp && git diff --check`

Open risks:
- This change refines the design contract in documentation only; the current code path still exports the older, more verbose analytical candidate columns.

### 2026-03-28 - Add explicit explanations for analytical formulas

Request summary:
- Make the formulas in `ANALYTICAL_MODEL_V2.md` easier to understand.
- Add direct explanations for the shared formulas and for each operator family's key analytical formulas.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added a new shared-formula explanation section covering:
  - `T = min(num_threads, hw_core_total_cores)`
  - `lat(level) = response_latency_cycles / cpu_clock`
  - `BW = hw_memory_bandwidth_gb_s_total * 1e3`
  - `peak_fp32_ops_per_us`
  - `fit(bytes)`
  - the repeated `max(..., 1)` numerical-safety pattern
- Added per-operator `公式解释` sections for:
  - `Gather`
  - `ReduceSum`
  - `Gemm`
  - `MatMul`
  - `Transpose`
  - `Concat`
- Each section now explains not only what the formula computes, but also what hardware or kernel behavior it is trying to approximate, for example:
  - why `Gather` uses an occupancy-style unique-row estimate
  - why `ReduceSum` adds a stride penalty instead of only using total bytes
  - why `Gemm` uses `max(compute_us, stream_us)` rather than summing both terms
  - why `Transpose` and `Concat` add dispatch/latency-like terms on top of streaming copy time

Validation run:
- `rg -n "### 4\\.1\\.1|### 5\\.1\\.4|### 5\\.2\\.4|### 5\\.3\\.4|### 5\\.4\\.4|### 5\\.5\\.4|### 5\\.6\\.4|公式解释" /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `sed -n '140,430p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `cd /data/qc/dlrm/ORT/single_op_stage1_mlp && git diff --check`

Open risks:
- This is still a documentation-only refinement; the code path does not yet emit every family-specific V2 analytical feature described in the document.
- The document now explains the current formulas much more explicitly, but some formulas remain approximation-level by design because they summarize ORT kernel behavior rather than replicate MLAS or cache hardware exactly.

### 2026-03-28 - Add a shared CPU submodel to Analytical Model V2 documentation

Request summary:
- Extend `ANALYTICAL_MODEL_V2.md` so the CPU model explicitly covers instruction width/latency, CPU frequency, and pipeline width.
- Document that these hardware quantities should first enter a shared CPU submodel and then be absorbed into a small set of family-specific primary analytical features, rather than being exposed as raw direct features.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added CPU-related shared symbols to the document:
  - `W_simd`
  - `f_cpu`
  - `L_fma / L_add`
  - `Th_fma / Th_add`
  - `W_pipe`
  - `cacheline_bytes`
- Added a dedicated `CPU submodel` section under the shared modeling framework.
  - The document now defines three CPU-side ceilings/floors:
    - `throughput ceiling`
    - `issue ceiling`
    - `dependency latency floor`
  - It includes explicit formulas for:
    - `ana_cpu_effective_pipeline_width`
    - `ana_cpu_peak_issue_slots_per_us`
    - `ana_cpu_peak_vec_fma_ops_per_us`
    - `ana_cpu_peak_vec_add_ops_per_us`
    - `ana_cpu_fp_fma_latency_us`
    - `ana_cpu_fp_add_latency_us`
  - It also includes a new Mermaid diagram showing how CPU ceilings and memory/cache terms merge into family-specific `base_us`.
- Added a CPU parameter source table explaining that the required CPU quantities already exist in `kunpeng920_single_numa.yaml`, even though they are not exported as standalone `dataset_full.csv` columns.
- Added a `CPU 融合方式` subsection for all six operator families:
  - `Gather`
  - `ReduceSum`
  - `Gemm`
  - `MatMul`
  - `Transpose`
  - `Concat`
- Rewrote the operator-family primary-feature definitions so that CPU effects are absorbed into:
  - `base_us`
  - effective throughput
  - one family-specific bottleneck/regime feature
- Inserted a new section explaining why raw CPU constants should not be fed directly into MLP under the current single-hardware dataset regime, and added a table mapping:
  - raw CPU quantity
  - shared CPU intermediate
  - affected primary analytical feature
- Renumbered downstream sections accordingly.

Validation run:
- `rg -n "CPU submodel|effective_pipeline_width|peak_issue_slots_per_us|dependency_latency_us|compute_share|CPU 融合方式|cpu_clock|simd_width_bits|pipeline width" /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `rg -n "^## |^### " /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `sed -n '130,760p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V2.md`
- `cd /data/qc/dlrm/ORT/single_op_stage1_mlp && git diff --check`

Open risks:
- This is still a documentation-only update; the code path has not yet been changed to emit the newly described CPU-derived analytical intermediates.
- The issue-side formulas such as `copy_issue_us` and `issue_us` are intentionally coarse analytical proxies; they summarize pipeline pressure rather than reproducing microarchitectural instruction scheduling exactly.

## 2026-03-29

Summary:
- Added a new paper-style analysis document:
  - [ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md)
- The new document formalizes two analytical-model families for the heavy-op DLRM slice:
  - an explainable calibrated family with physically interpretable parameters
  - a pure no-fit analytical family used to probe the accuracy ceiling of zero-calibration formulas
- It fixes the evaluation scope to heavy operators from:
  - `case_9_4_4`
  - `case_10_2_1`
  - `case_10_4_4`
  - combos `bs1024_nip1500`, `bs1440_nip1700`, `bs1888_nip1800`
- It documents the heavy-op sample counts:
  - `Gather`: 59
  - `ReduceSum`: 59
  - `Gemm`: 35
  - `MatMul`: 8
  - `Transpose`: 8
  - `Concat`: 9
- It records the final explainable calibrated models and interpretable parameter meanings for:
  - `Concat`
  - `ReduceSum`
  - `Gather`
  - `Gemm`
  - `MatMul`
  - `Transpose`
- It records the calibrated evaluation results:
  - `Gather`: `7.89%`
  - `ReduceSum`: `11.82%`
  - `Gemm`: `13.22%`
  - `MatMul`: `15.29%`
  - `Transpose`: `27.09%`
  - `Concat`: `14.66%`
  - family macro `MAPE = 14.99%`
  - weighted overall `MAPE = 11.78%`
- It also defines the pure analytical upper-bound group and reports the best no-fit structure per operator family:
  - `Gather`: `req_lat_t`
  - `ReduceSum`: `stream_plus_blocks_lat_mem`
  - `Gemm`: `roofline+dependency`
  - `MatMul`: `dep_batch_over_t`
  - `Transpose`: `prefix_lat_outfit`
  - `Concat`: `memory_plus_chunk_lat`
- It records the pure analytical upper-bound results:
  - `Gather`: `85.88%`
  - `ReduceSum`: `34.97%`
  - `Gemm`: `56.63%`
  - `MatMul`: `91.15%`
  - `Transpose`: `52.23%`
  - `Concat`: `78.83%`
  - family macro `MAPE = 66.62%`
  - weighted overall `MAPE = 61.62%`

Validation run:
- `sed -n '1,260p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`
- `rg -n "MAPE|纯 Analytical|可解释校准|Gather|ReduceSum|Gemm|MatMul|Transpose|Concat" /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md AGENT_WORKLOG.md`

Open risks:
- This is still a documentation-only update; no feature builder or training data path has been changed yet.
- The calibrated formulas are currently recorded as design-level analytical models rather than executable repo code, so the next implementation step must translate them into `feature_engineering.py` carefully and keep the parameter semantics intact.

## 2026-03-29

Summary:
- Added a reproducible held-out calibration/generalization evaluator:
  - [evaluate_analytical_generalization.py](/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py)
- Added a new results document focused on out-of-fold behavior:
  - [ANALYTICAL_MODEL_V4_GENERALIZATION.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md)
- The new script calibrates the explainable analytical model family on training folds only, then evaluates held-out `MAPE` under:
  - `leave_one_case_out`
  - `leave_one_combo_out`
- The evaluation scope is the same heavy-op slice documented in V3:
  - `case_9_4_4`
  - `case_10_2_1`
  - `case_10_4_4`
  - combos `bs1024_nip1500`, `bs1440_nip1700`, `bs1888_nip1800`
  - total heavy-op rows `178`
- The script writes reproducible outputs under:
  - `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization/`
  - including `summary.md`, `summary.json`, `fold_parameters.csv`, `fold_family_metrics.csv`, and `heavy_op_eval_slice.csv`

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The project now has a repo-local analytical evaluation script that:
  - filters the heavy-op slice directly from `dataset_full.csv`
  - reconstructs family-specific shape semantics from `input_type_shape` / `output_type_shape`
  - calibrates the explainable analytical parameters on training folds only using bounded coordinate descent over interpretable grids
  - reports train/test per-family MAPE and calibrated parameter values for each fold
- The new V4 document records the first genuine held-out results for the explainable analytical model:
  - `leave_one_case_out`: mean macro `MAPE = 21.20%`, weighted family `MAPE = 16.96%`
  - `leave_one_combo_out`: mean macro `MAPE = 15.75%`, weighted family `MAPE = 11.66%`
- The held-out results show:
  - `Gather / ReduceSum / Gemm / Concat` generalize relatively well
  - `MatMul` is acceptable but still weaker on cross-case transfer
  - `Transpose` remains the largest generalization gap

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `sed -n '1,260p' /data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- evaluate_analytical_generalization.py ANALYTICAL_MODEL_V4_GENERALIZATION.md`

Open risks:
- The current evaluator calibrates only the explainable parameter family recorded in V3; it does not yet compare against alternative model structures such as an explicit `C_contention` term during held-out evaluation.
- The largest remaining held-out gap is concentrated in `Transpose`, and secondarily `MatMul`, which suggests that the next model revision should separate kernel base time from concurrency-induced wall-time inflation more explicitly.

## 2026-03-29

Summary:
- Replaced the copy-like heavy-op parameterization in the held-out evaluator from explicit `B50_*` half-saturation parameters to the equivalent white-box form:
  - `BW_eff = BW_inf * s / (BW_inf * tau_start + s)`
- Kept the same heavy-op slice and the same held-out protocols, then reran the full generalization evaluation to check whether the more explicit `tau_start_*` formulation changes out-of-fold behavior.
- Updated the V4 results document to reflect the new parameter semantics and the refreshed held-out metrics.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `Concat`, `ReduceSum`, and `Gather` now calibrate:
  - `tau_copy_start`
  - `tau_reduce_start`
  - `tau_gather_row_start`
  instead of `B50_copy`, `B50_reduce`, and `B50_gather_row`.
- The evaluator now computes the effective bandwidth for these families through:
  - `BW_eff = BW_inf * s / (BW_inf * tau_start + s)`
  rather than exposing `B50 = BW_inf * tau_start` as the primary fitted parameter.
- Held-out generalization remains essentially unchanged after the rewrite:
  - `leave_one_case_out`: mean macro `MAPE = 20.89%`, weighted family `MAPE = 16.70%`
  - `leave_one_combo_out`: mean macro `MAPE = 15.79%`, weighted family `MAPE = 11.73%`
- The small metric movement relative to the earlier `B50_*` version indicates that this change improves interpretability much more than it changes predictive behavior.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- evaluate_analytical_generalization.py ANALYTICAL_MODEL_V4_GENERALIZATION.md AGENT_WORKLOG.md`

Open risks:
- The `tau_start_*` rewrite is mathematically cleaner, but it is still evaluated on the same heavy-op slice; broader regime coverage is still needed before treating the calibrated startup times as globally transferable.
- `Transpose` and `MatMul` remain the dominant held-out error sources, so the next structural gain still likely comes from explicit contention/context terms rather than further reparameterizing the copy-like bandwidth curve.

## 2026-03-29

Summary:
- Updated the V3 analytical design document so its main copy-like formulas now match the newer explicit startup-time form:
  - `BW_eff = BW_inf * s / (BW_inf * tau_start + s)`
- Replaced the V3 tables and operator-family explanations that previously exposed `B50_copy`, `B50_reduce`, and `B50_gather_row` as primary calibrated parameters.
- Preserved the half-saturation interpretation as an equivalent secondary quantity:
  - `B50 = BW_inf * tau_start`

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The V3 document now treats:
  - `tau_copy_start`
  - `tau_reduce_start`
  - `tau_gather_row_start`
  as the primary interpretable startup parameters for copy-like families.
- `B50_*` is no longer presented as the main fitted object; it is explicitly documented as an equivalent derived quantity used only when half-saturation intuition is helpful.
- The operator sections for `Concat`, `ReduceSum`, and `Gather` now explain startup overhead in time units first, then map back to the older `B50` view only as a derived interpretation.

Validation run:
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md AGENT_WORKLOG.md`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff -- ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`

Open risks:
- This update only synchronizes the design document wording; the pure-analytical comparison sections still keep some half-saturation terminology for `M50/N50/K50`, which is intentional because those GEMM occupancy parameters are still best understood as saturation scales rather than startup times.
- The V3 narrative is now aligned with the newer evaluator, but a future cleanup pass could further unify wording across V3 and V4 so the two documents read as a single evolution path.

## 2026-03-29

Summary:
- Added an optional GEMM analytical variant to the held-out evaluation script:
  - `m50_saturation`
  - `alpha_shape_penalty`
- The new `alpha_shape_penalty` form keeps the same high-level roofline structure, but replaces `M50/N50/K50` saturation with:
  - `T_compute = flops / (PeakFMA * rho_fma_inf) + alpha_M / M + alpha_N / N + alpha_K / K`
- Ran a dedicated held-out generalization test for the new GEMM form in a separate artifact directory so the existing V4 baseline outputs were not overwritten.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `evaluate_analytical_generalization.py` now accepts:
  - `--gemm-model m50_saturation`
  - `--gemm-model alpha_shape_penalty`
- The `alpha_shape_penalty` variant exposes interpretable GEMM shape-penalty parameters:
  - `alpha_m`
  - `alpha_n`
  - `alpha_k`
- On the current heavy-op held-out slice, this new GEMM formulation generalized worse than the existing `M50/N50/K50` saturation form:
  - baseline `m50_saturation`
    - `leave-one-case-out`: mean macro `20.89%`, weighted family `16.70%`
    - `leave-one-combo-out`: mean macro `15.79%`, weighted family `11.73%`
  - new `alpha_shape_penalty`
    - `leave-one-case-out`: mean macro `22.41%`, weighted family `18.63%`
    - `leave-one-combo-out`: mean macro `19.23%`, weighted family `16.23%`
- The degradation is concentrated in `Gemm` itself:
  - `leave-one-case-out` GEMM mean test `MAPE = 25.73%`
  - `leave-one-combo-out` GEMM mean test `MAPE = 31.83%`

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py --gemm-model alpha_shape_penalty --output-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization_gemm_alpha`

Open risks:
- The current `alpha_M/M + alpha_N/N + alpha_K/K` penalty is interpretable but appears less stable under held-out combo transfer than the saturation form, likely because it does not constrain shape effects to remain bounded as cleanly as the `M50/N50/K50` parameterization.
- If we want a more interpretable GEMM model without losing generalization, the next best candidate is probably a hybrid form: keep bounded saturation for `M/N`, but rewrite only the `K` direction as an explicit packing/startup term.

## 2026-03-29

Summary:
- Reframed the V3 `Gemm` section so `M50/N50/K50` are no longer introduced as unexplained constants.
- The document now derives the saturation form from a simpler base-time-plus-shape-penalty model:
  - `T_base = flops / (PeakFMA * rho_fma_inf)`
  - `T_shape = T_base * (M50/M + N50/N + K50/K)`
  - then rewrite to the equivalent bounded utilization form
    `rho_fma_eff = rho_fma_inf * M/(M+M50) * N/(N+N50) * K/(K+K50)`
- Removed the temporary `alpha_shape_penalty` GEMM branch from the held-out evaluator so the project returns to a single canonical GEMM analytical form aligned with the V3/V4 docs.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The V3 document now explains `M50/N50/K50` as equivalent half-saturation scales induced by shape-dependent penalties, rather than presenting them as standalone fitted constants.
- `evaluate_analytical_generalization.py` is back to a single GEMM form:
  - bounded `M50/N50/K50` saturation
  - no optional `--gemm-model` switch
- This keeps the repo’s official held-out evaluation path aligned with the better-generalizing GEMM parameterization.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md evaluate_analytical_generalization.py AGENT_WORKLOG.md`

Open risks:
- The V3 derivation is intentionally compact and does not descend into kernel-internal tile or packing metadata, because those signals are not explicitly available in the current dataset; this keeps the explanation interpretable, but still approximate.
- The reverted evaluator restores the better empirical baseline, but the longer-term hybrid GEMM idea (`M/N` saturation + explicit `K` packing term) remains open for future testing.
