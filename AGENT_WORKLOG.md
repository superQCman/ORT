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
- `roofline_op_type_analysis/analyze_roofline_op_types.py`
  Builds row-level and op-type-level Roofline summaries plus visualization artifacts.

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

### 2026-03-29 - Add isolated analytical_calibrated and classed_op_mlp pipelines

Request summary:
- Implement the new three-class single-op modeling plan on top of `dataset_all_no_trace`.
- Keep the work isolated from the existing main single-MLP pipeline.
- Add a separate calibrated analytical pipeline, a separate classed-MLP pipeline, feature-definition summaries, analytical generalization evaluation entry points, and baseline comparison entry points against `model_all_no_trace`.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/contracts.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/build_analytical_features.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/evaluate_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/compare_against_baseline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added `analytical_calibrated/` as a new isolated subproject that:
  - rebuilds local `feat_*` / `hw_*` / legacy `ana_*` columns from `dataset_all_no_trace`
  - calibrates heavy-family analytical parameters on the full dataset
  - exports row-level `ana_calib_total_us`, `ana_calib_mem_us`, `ana_calib_compute_us`, `ana_calib_overhead_us`, `ana_calib_family`, and `op_class`
  - provides a separate held-out generalization evaluation entry point for `leave_one_case_out` and `leave_one_combo_out`
- Added `classed_op_mlp/` as a new isolated subproject that:
  - statically routes `op_type` into `memory_pure`, `mixed_balanced`, and `compute_dominant`
  - documents the per-class feature meanings in its own README
  - merges `ana_calib_*` features with a local `feat_gemm_*` rebuild to create class-specific datasets
  - trains one MLP per class through the existing project-local `train_mlp.py`
  - recombines class predictions into full split-level prediction tables
  - adds a baseline-comparison entry point against `artifacts/latest/model_all_no_trace`
- Kept all new code and documentation inside the two new subdirectories so the existing single-MLP path is unchanged.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/*.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/*.py`
- `python3 - <<'PY' ... import analytical_calibrated.build_analytical_features ... import analytical_calibrated.evaluate_generalization ... import classed_op_mlp.build_classed_dataset ... PY`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/build_analytical_features.py --input-csv /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace/dataset_full.csv --output-dir /tmp/single_op_analytical_calibrated_smoke`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --input-data-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/dataset_all_no_trace --analytical-dir /tmp/single_op_analytical_calibrated_smoke --output-dir /tmp/single_op_classed_dataset_smoke`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/compare_against_baseline.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py --help`

Open risks:
- This environment does not currently have `torch` installed, so the new classed MLP training path could not be executed end-to-end in this turn.
- The full held-out analytical generalization run on the entire `dataset_all_no_trace` surface is implemented, but the full fold sweep was not allowed to finish in this turn; only the feature-export path and dataset-build path were completed end-to-end.
- The independent repo already has unrelated user modifications in `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md` and `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, so this task intentionally did not edit those files.
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

## 2026-03-29

Summary:
- Added a duration-aware held-out metric to the analytical generalization evaluator so long-running operators contribute more to the overall error summary.
- The new metric is:
  - `duration-weighted relative error = sum(|pred-actual|) / sum(actual)`
- Re-ran the default held-out evaluation and updated the V4 document to report this metric alongside the existing macro `MAPE` and row-count-weighted family `MAPE`.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `evaluate_analytical_generalization.py` now exports, per family and per fold:
  - `dwre`
  - `actual_sum_us`
  - `abs_error_sum_us`
- Scheme summaries now include:
  - `duration_weighted_relative_error`
- Default held-out results now report:
  - `leave-one-case-out`: duration-weighted relative error `14.44%`
  - `leave-one-combo-out`: duration-weighted relative error `10.31%`
- The V4 document now explains the new metric and shows that, when weighting by actual operator time, the overall held-out error is lower than the plain sample-average `MAPE`.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- evaluate_analytical_generalization.py ANALYTICAL_MODEL_V4_GENERALIZATION.md AGENT_WORKLOG.md`

Open risks:
- The duration-weighted metric improves deployment relevance, but it is still computed only on the current heavy-op slice; if the evaluation scope broadens to include many short operators, the balance between macro `MAPE` and duration-weighted error will become even more important to interpret jointly.
- Some family-level means of the duration-weighted metric can exceed the corresponding mean `MAPE`, because the former upweights longer-running examples inside the family rather than averaging all rows equally.

## 2026-03-29

Summary:
- Added a new overall metric to `analyze_op_type_metrics.py` that weights op-type errors by the total runtime of each `combo + op_type` bucket.
- The goal is to better reflect both:
  - how expensive an operator family is per invocation
  - how often it appears inside a combo
- The metric is exported into `overall_metrics` in the summary JSON as:
  - `combo_op_type_total_duration_weighted_mape`

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_op_type_metrics.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Metadata enrichment for prediction tables now resolves both:
  - `op_type`
  - `combo`
- The new metric is computed by:
  1. grouping rows by `combo` and `op_type`
  2. computing each group’s mean APE
  3. weighting those group-level APEs by the group’s total `target_us`
- This lets the overall score reflect both operator duration and operator count within each combo, instead of averaging all rows equally.
- Validation on the existing sample model output produced:
  - `val`: `combo_op_type_total_duration_weighted_mape = 0.27198992277246287`
  - `test`: `combo_op_type_total_duration_weighted_mape = 0.31297357553663835`

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_op_type_metrics.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analyze_op_type_metrics.py --model-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_clean_2_stage_2`
- Verified `combo_op_type_total_duration_weighted_mape` is present in:
  - `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/model_all_clean_2_stage_2/op_type_metrics/op_type_metrics_summary.json`

Open risks:
- The new metric is currently summary-only; it is not yet exposed as a ranking metric or as a per-op-type CSV column because the user request was specifically for one overall aggregate.
- README documentation was intentionally left unchanged in this task because the current worktree already contains unrelated README edits that should not be mixed into this scoped commit.

## 2026-03-29

Summary:
- Added a new independent `roofline_op_type_analysis/` directory for Roofline-based operator-type analysis.
- The new path classifies rows and aggregated operator groups into `memory_bound`, `near_ridge`, or `compute_bound`, then exports both CSV summaries and plots.
- README now documents the new analysis entry point and its default outputs.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added a dedicated CLI entry point:
  - `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- The script defaults to:
  - input: `artifacts/latest/dataset_all_no_trace/dataset_full.csv`
  - output: `artifacts/latest/roofline_op_type_analysis`
  - hardware profile: `hardware_profile/kunpeng920_single_numa.yaml`
- The script now:
  - backfills or recomputes required engineered / analytical columns when needed
  - derives row-level Roofline metrics including arithmetic intensity, achieved performance, ridge point, and ridge gap
  - emits:
    - `row_level_roofline.csv`
    - `op_type_thread_summary.csv`
    - `op_type_summary.csv`
    - `roofline_by_threads.png`
    - `op_type_bound_share.png`
    - `op_type_ridge_gap_heatmap.png`
    - `roofline_summary.json`
- The main Roofline figure is thread-faceted instead of mixing all `num_threads` values into one plot, so each subplot keeps a consistent hardware ceiling and ridge point.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`

Open risks:
- The compute proxy comes from existing `ana_compute_ops`, so copy-like operators with negligible modeled FLOPs are intentionally pushed toward the memory-bound side.
- The ridge-gap heatmap log-compresses very large values for readability, so it should be read as a relative comparison view rather than a raw-magnitude report.

## 2026-03-29

Summary:
- Improved the readability of `roofline_by_threads.png` by removing the label pile-up in the lower-left corner.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- The Roofline scatter plot now detects dense low-intensity / low-performance operator clusters in each thread subplot.
- Instead of annotating every point in that corner individually, the plot keeps the points in place and renders a compact `Low-intensity cluster` textbox listing the affected `op_type` values.
- Non-clustered operators such as `Gemm`, `MatMul`, `ReduceSum`, `Sigmoid`, `Add`, and `Mul` still keep direct point labels.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/analyze_roofline_op_types.py`
- Visually checked `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/roofline_op_type_analysis/roofline_by_threads.png`

Open risks:
- The cluster textbox improves readability but intentionally stops distinguishing the exact point-to-label correspondence inside that lower-left pile-up; the exact operator identities remain available in `op_type_thread_summary.csv`.

## 2026-03-29

Summary:
- Implemented the planned explicit hardware-parameterized analytical evaluation variants in `evaluate_analytical_generalization.py`.
- Added two non-baseline variants:
  - `explicit_no_reuse`: keeps the current held-out behavior but expands the shared hardware submodel explicitly
  - `explicit_unique_reuse`: additionally switches `Gather` source-miss counting to a unique-row-aware approximation
- Kept documentation unchanged in this round on purpose; this task was limited to script-side implementation and validation.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- Added a `--variant` flag to the generalization evaluator:
  - `baseline`
  - `explicit_no_reuse`
  - `explicit_unique_reuse`
- Expanded the shared hardware submodel into explicit helper functions:
  - `fit_level(...)`
  - `latency_from_level(...)`
  - `peak_add_ops_per_us(...)`
  - `peak_fma_ops_per_us(...)`
  - `issue_slots_per_us(...)`
  - `add_latency_us(...)`
- Reworked heavy-slice preprocessing to emit explicit intermediate hardware-aware columns, including:
  - `gather_table_rows`
  - `gather_unique_rows_est`
  - `gather_src_unique_bytes`
  - `gather_src_fit_level`
  - `gather_src_latency_us`
  - `reduce_acc_bytes_per_thread`
  - `reduce_acc_fit_level`
  - `reduce_acc_latency_us`
  - `transpose_suffix_block_bytes`
  - `transpose_suffix_fit_level`
  - `transpose_stride_latency_us`
  - `issue_slots_per_us`
- `Gather` prediction now supports a unique-row-aware source path:
  - `unique_rows_est = table_rows * (1 - exp(-request_rows / table_rows))`
  - `T_src = unique_rows_est * cachelines_per_row * lat_src_us / (T * m_gather)`
- `Concat`, `ReduceSum`, and `Transpose` can now evaluate through explicit hardware-expanded paths without introducing new reuse parameters.
- Summary markdown now records the analytical variant used for each run.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py --variant baseline --output-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization_variant_baseline`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py --variant explicit_no_reuse --output-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization_variant_explicit_no_reuse`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/evaluate_analytical_generalization.py --variant explicit_unique_reuse --output-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization_variant_explicit_unique_reuse`

Key results:
- `explicit_no_reuse` preserved the prior held-out aggregate metrics on the current heavy-op slice:
  - leave-one-case-out: macro `20.89%`, duration-weighted RE `14.44%`
  - leave-one-combo-out: macro `15.79%`, duration-weighted RE `10.31%`
- `explicit_unique_reuse` did not improve the heavy-op held-out results:
  - leave-one-case-out: macro `21.54%`, duration-weighted RE `15.78%`
  - leave-one-combo-out: macro `15.79%`, duration-weighted RE `10.31%`
- The new `Gather` reuse approximation is numerically well-formed:
  - `unique_rows_est <= request_rows`
  - `unique_rows_est <= table_rows`
  - all heavy-op `Gather` rows received non-null `gather_table_rows`, `gather_unique_rows_est`, and `gather_src_fit_level`

Open risks:
- On the current heavy-op DLRM slice, `Gather` requests are so large relative to table size that the occupancy estimate saturates to essentially the full table (`unique_rows_est / table_rows ~= 1.0` across the slice). This means the unique-row correction does not reduce the modeled source working set in the regime we currently evaluate.
- Because the evaluated heavy-op `Gather` rows all remain memory-tier (`gather_src_fit_level = 4`), the explicit reuse correction currently changes the row-count term more than the residency tier, and that shift slightly worsens leave-one-case-out error.
- V3/V4 documentation was intentionally not updated in this task; if the user wants the explicit hardware submodel written back into the docs, that should be a follow-up scoped change after reviewing these held-out results.

## 2026-03-29

Summary:
- Wrote the accepted conclusion back into the analytical design docs:
  - keep the explicit shared hardware submodel
  - do not adopt the `Gather` unique-row reuse correction into the formal model for the current heavy-op regime

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/ANALYTICAL_MODEL_V4_GENERALIZATION.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- V3 now explicitly documents the shared hardware submodel:
  - `fit(bytes)`
  - `lat(level)`
  - `PeakAdd(T)`
  - `PeakFMA(T)`
  - `IssueSlots(T)`
- V3 `Gather` now explicitly states that:
  - `L1/L2/L3 size` and `L1/L2/L3/MEM latency` enter through `fit(src_working_set_bytes)` and `lat(src_fit)`
  - the formal model still uses the conservative `request_rows` source-miss path
  - the occupancy-style `unique_rows_est` correction was evaluated but not adopted for the current heavy-op DLRM slice
- V4 now records the post-V4 variant comparison:
  - `explicit_no_reuse` keeps held-out metrics equal to baseline while improving formula transparency
  - `explicit_unique_reuse` does not improve the current heavy-op held-out results and is therefore kept only as an explored candidate branch

Validation run:
- `git -C /data/qc/dlrm/ORT/single_op_stage1_mlp diff --check -- ANALYTICAL_MODEL_V3_CALIBRATED_VS_PURE.md ANALYTICAL_MODEL_V4_GENERALIZATION.md`

Open risks:
- The non-adoption of the unique-row correction is specific to the current heavy-op slice, where `request_rows / table_rows` is already large enough to saturate the occupancy estimate.
- If later evaluation expands to lower-lookup or more skewed-index regimes, the `unique_rows_est` branch may become useful again and should be re-tested there rather than treated as permanently invalid.
- `README.md` and `roofline_op_type_analysis/README.md` were intentionally left out of this task because they already contain unrelated worktree changes.

## 2026-03-29

Summary:
- Added a dedicated correlation-analysis utility for the new `analytical_calibrated` pipeline and ran it against the current `artifacts/latest/analytical_calibrated` outputs to explain why the relative-error metrics are so large.
- Documented the new analysis entrypoint and generated a markdown summary artifact so the worst families and strongest feature/error correlations can be inspected without opening multiple CSV files.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/analyze_correlation.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `analyze_correlation.py` now exports two correlation views:
  - `raw_feature_*`: correlations between original software/data features and `target_us`, `abs_error_us`, `APE`, and signed relative error
  - `analytical_component_*`: correlations between `ana_calib_mem_us` / `ana_calib_compute_us` / `ana_calib_overhead_us` and the same error targets
- The script now writes `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated/correlation_analysis/correlation_summary.md` in addition to the CSV/JSON outputs.
- The summary makes the current failure mode explicit:
  - `generic_memory` dominates the inflated relative-error metrics
  - `Reshape` is the worst offender because the proxy predicts copy-like memory cost while the measured runtime is still only tens of microseconds
  - `Gather` has a long-tail small-target issue, so mean MAPE is much worse than median APE or duration-weighted RE

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/analyze_correlation.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/analyze_correlation.py`

Key findings:
- Full-data `memory_pure` duration-weighted relative error is about `40.18%`, but its mean MAPE is inflated to about `689.69x` because `generic_memory` contributes about `1476.36x` mean MAPE by itself.
- `Reshape` is the dominant structural mismatch:
  - median actual latency is about `19.5 us`
  - median analytical prediction is about `50,585.6 us`
- `Gather` mean MAPE is about `26.74x`, but median APE is only about `0.56x` and duration-weighted relative error is about `30.70%`, which points to distribution skew instead of a universally bad fit.
- The strongest raw-feature correlation with APE is `feat_io_bytes_sum` (`r ~= 0.775`), and the strongest raw-feature correlations with absolute error are also memory-volume-related terms, which is consistent with the main failure mode being the memory-side proxy design.

Open risks:
- The current `generic_memory` proxy is too coarse for metadata/view operators like `Reshape` and `Unsqueeze`; if these operators remain in-scope for analytical supervision, they likely need separate near-zero-overhead formulas instead of a shared bandwidth proxy.
- The correlation analysis is descriptive, not causal. It helps localize the dominant failure modes, but it does not by itself choose the next best formula update.
- `README.md`, `roofline_op_type_analysis/README.md`, `classed_op_mlp/train_class_models.py`, `data.sh`, and `model.sh` still contain unrelated worktree changes and were intentionally not touched in this task.

## 2026-03-29

Summary:
- Added a branchable feature-contract layer for `classed_op_mlp` so the classification pipeline can run in a new `no_analytical` mode that completely removes `ana_calib_*` from the class MLP inputs.
- Aligned the new `no_analytical` branch to the existing `dataset_all_no_trace` contract and selected a compact set of per-class features from the original single-MLP `all_features` list, explicitly excluding `hw_ratio_*`, `local_ctx_*`, `comp_feat_*`, and old analytical columns.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `classed_op_mlp` now supports two explicit feature branches:
  - `with_analytical`: keeps the previous `ana_calib_*` feature contract
  - `no_analytical`: removes analytical proxy inputs entirely and routes rows into classes using static `op_type -> op_class`
- `build_classed_dataset.py` now supports `--feature-branch` and branch-specific default output roots:
  - `with_analytical` defaults to `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp`
  - `no_analytical` defaults to `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp/no_analytical`
- `run_pipeline.py` now supports `--feature-branch no_analytical`; in that mode it:
  - keeps the input dataset aligned to `dataset_all_no_trace`
  - skips analytical feature construction and analytical generalization
  - still builds the classed dataset, trains per-class models, and compares against `model_all_no_trace`
- `train_class_models.py` now supports `--feature-branch` for default data-root resolution and tolerates merged datasets that do not carry `ana_calib_family`.
- The new `no_analytical` numeric feature subsets are:
  - `memory_pure`: `batch_size`, `num_indices_per_lookup`, `num_threads`, `output_size`, `activation_size`, `parameter_size`, `feat_io_bytes_sum`, `feat_output_input_bytes_ratio`, `feat_lookup_count`, `feat_output_elements_per_lookup`, `feat_output_elements_per_batch`
  - `mixed_balanced`: `batch_size`, `num_threads`, `output_size`, `activation_size`, `feat_io_bytes_sum`, `feat_output_input_bytes_ratio`, `feat_output_elements_per_batch`, `feat_activation_elements_per_batch`, `feat_reduction_axes_count`, `feat_reduction_axes_product`, `feat_reduction_input_rank`, `feat_reduction_output_rank`, `feat_reduction_work_items`
  - `compute_dominant`: `batch_size`, `num_threads`, `output_size`, `activation_size`, `parameter_size`, `feat_io_bytes_sum`, `feat_output_input_bytes_ratio`, `feat_gemm_m`, `feat_gemm_n`, `feat_gemm_k`, `feat_gemm_mac_count`, `feat_gemm_bytes_per_mac`

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch no_analytical --output-dir /tmp/classed_op_mlp_no_analytical_smoke`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch with_analytical --analytical-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated --output-dir /tmp/classed_op_mlp_with_analytical_smoke`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py --feature-branch no_analytical --data-root /tmp/classed_op_mlp_no_analytical_smoke --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py --help`

Open risks:
- This task only established the branchable dataset/training contract; it did not run end-to-end model training on the current machine because `torch` is not installed in this environment.
- The chosen `no_analytical` feature subsets are intentionally compact and mechanism-driven; they are a strong starting point for the ablation, but they are not yet validated against the final test metrics.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py` default hidden-layer edit, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` contained pre-existing or user-driven worktree changes; this task preserved them and only layered the needed branch support on top.

## 2026-03-29

Summary:
- Refined the `no_analytical` classed-MLP branch so `memory_pure` is no longer trained as one heterogeneous bucket.
- The `no_analytical` training route now uses five model groups:
  - `gather`
  - `layout_move`
  - `view_meta`
  - `mixed_balanced`
  - `compute_dominant`

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `no_analytical` now uses static `op_type -> model_group` routing:
  - `Gather -> gather`
  - `Concat/Transpose -> layout_move`
  - `Reshape/Shape/Unsqueeze/Flatten -> view_meta`
  - `ReduceSum/Sigmoid/Relu/Add/Mul -> mixed_balanced`
  - `Gemm/MatMul -> compute_dominant`
- The original `op_class` label is still preserved in the merged dataset and combined predictions, so three-class summaries remain available.
- `build_classed_dataset.py` now exports:
  - `model_group`
  - `model_group_order`
  - `op_type_model_group_map`
  - `per_model_group_numeric_features`
- `train_class_models.py` now trains and combines predictions by `model_group` instead of assuming exactly three datasets keyed by `op_class`.
- The new `no_analytical` memory-side feature subsets are:
  - `gather`: keep the richer lookup-centric feature set
  - `layout_move`: focus on bytes, output/input ratio, and batch-scale movement features
  - `view_meta`: keep a compact low-overhead shape/view feature set

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch no_analytical --output-dir /tmp/classed_op_mlp_no_analytical_split_memory_smoke`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/train_class_models.py --feature-branch no_analytical --data-root /tmp/classed_op_mlp_no_analytical_split_memory_smoke --help`

Key results:
- The new `no_analytical` smoke dataset exports the expected five-group order:
  - `gather`: `21359` rows
  - `layout_move`: `13368` rows
  - `view_meta`: `29714` rows
  - `mixed_balanced`: `23677` rows
  - `compute_dominant`: `9556` rows
- Group-local feature manifests were successfully generated for `gather`, `layout_move`, and `view_meta`, confirming the finer routing and per-group feature contracts are wired through to training.

Open risks:
- I validated dataset generation and training-entry wiring, but did not run the full new five-group training/evaluation loop on this machine because `torch` is not installed in the current environment.
- `view_meta` still groups four semantically light operators together; if error remains high after this change, the next split candidate is likely `Shape` vs. the pure view operators.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` still contain unrelated worktree changes and were intentionally left untouched.

## 2026-03-29

Summary:
- Added a targeted ablation to the `classed_op_mlp` `no_analytical` branch so it regains only the two high-cardinality node identity features from the baseline:
  - `node_scope`
  - `node_name_normalized`
- Kept all numeric feature subsets unchanged so this experiment isolates whether these two categorical features are enough to recover the memory-side groups.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `classed_op_mlp/contracts.py` now resolves categorical features per feature branch:
  - `with_analytical`: unchanged 4-way categorical contract
  - `no_analytical`: `op_type`, `node_scope`, `node_name_normalized`, `arch_embedding_size`, `arch_mlp_bot`, `arch_mlp_top`
- `build_classed_dataset.py` now writes branch-specific categorical feature manifests and summaries instead of assuming a single shared categorical feature list for every branch.
- The generated `feature_columns.json` for `no_analytical` datasets now includes descriptions for:
  - `node_scope`
  - `node_name_normalized`
- README now documents this as a controlled node-identity ablation intended to test whether `gather`, `layout_move`, and `view_meta` can be pulled closer to the baseline without reintroducing analytical or hardware/context features.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch no_analytical --output-dir /tmp/classed_op_mlp_no_analytical_node_id_smoke`
- Inspected `/tmp/classed_op_mlp_no_analytical_node_id_smoke/datasets/gather/feature_columns.json` to confirm:
  - `categorical_features` contains both `node_scope` and `node_name_normalized`
  - `feature_descriptions` contains explicit meanings for both fields

Open risks:
- This task only updated the data/manifest contract; it did not run the full training/evaluation loop after reintroducing the two node identity features.
- The high-cardinality categorical features may improve `gather/layout_move/view_meta`, but they also increase encoded input dimensionality and may overfit specific node identities if the held-out split distribution changes.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` still contain unrelated worktree changes and were intentionally left untouched.

## 2026-03-29

Summary:
- Added a fast validation path for the analytical and classed-MLP pipelines so `with_analytical` experiments can reuse `ana_calib_*` without waiting for the full held-out analytical generalization sweep.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `analytical_calibrated/run_pipeline.py` now supports:
  - `--skip-generalization`
  - `--schemes ...`
- `classed_op_mlp/run_pipeline.py` now supports:
  - `--skip-analytical-generalization`
  - `--reuse-analytical-features`
  - `--analytical-schemes ...`
- This makes a fast `with_analytical` validation path possible:
  - build or reuse `analytical_features_full.csv`
  - skip the slow fold-level analytical generalization
  - continue directly into grouped dataset build, training, and baseline comparison
- README examples now document:
  - fast analytical feature-only export
  - lightweight case-only analytical evaluation
  - fast `with_analytical` classed-MLP validation using `--skip-analytical-generalization`
  - reuse of an existing `analytical_features_full.csv` via `--reuse-analytical-features`

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/analytical_calibrated/run_pipeline.py --help`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py --help`

Open risks:
- The new fast path is intended for iteration speed; it deliberately skips or reduces analytical generalization, so it should not replace the full held-out evaluation when producing final analytical-model claims.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py` already had a user/local hidden-layer default change before this task; this update preserved that local default while layering the new fast-run flags on top.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` still contain unrelated worktree changes and were intentionally left untouched.

## 2026-03-29

Summary:
- Fixed a `with_analytical` dataset-build bug that made the classed pipeline report all analytical features as missing, even when `analytical_features_full.csv` was complete.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `build_classed_dataset.py` now attaches analytical columns through a dedicated helper that drops any pre-existing analytical columns before merging.
- This avoids pandas suffix collisions such as `op_class_x/op_class_y`, which previously caused the post-merge missing-value check to think all `97674` rows were missing analytical annotations.
- `with_analytical` runs can now correctly reuse:
  - `op_class`
  - `ana_calib_family`
  - `ana_calib_total_us`
  - `ana_calib_mem_us`
  - `ana_calib_compute_us`
  - `ana_calib_overhead_us`

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch with_analytical --analytical-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated --output-dir /tmp/classed_op_mlp_with_analytical_merge_fix_smoke`

Key results:
- The smoke build completed successfully instead of failing with `analytical_calibrated features are missing for 97674 rows`.
- The merged dataset now contains valid `op_class`, `ana_calib_family`, and `model_group` values.
- The with-analytical smoke summary resolved to the expected three groups:
  - `memory_pure`: `64441` rows
  - `mixed_balanced`: `23677` rows
  - `compute_dominant`: `9556` rows

Open risks:
- This fix covered dataset construction and merge integrity; it did not rerun the full training loop on NPU in this environment.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py` still carries a separate user/local hidden-layer default change in the working tree, which was intentionally preserved.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` still contain unrelated worktree changes and were intentionally left untouched.

## 2026-03-29

Summary:
- Upgraded the `with_analytical` branch so it now matches `classed_op_mlp_test` as a true 5-way augmentation experiment instead of using the older 3-way contract.

Files changed:
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/README.md`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Behavior changes:
- `with_analytical` now uses the same 5-way static routing as the strong `no_analytical` experiment:
  - `gather`
  - `layout_move`
  - `view_meta`
  - `mixed_balanced`
  - `compute_dominant`
- `with_analytical` now uses the same shared categorical features as `classed_op_mlp_test`:
  - `op_type`
  - `node_scope`
  - `node_name_normalized`
  - `arch_embedding_size`
  - `arch_mlp_bot`
  - `arch_mlp_top`
- `with_analytical` now keeps the same raw numeric features as `classed_op_mlp_test`, then appends only the relevant analytical features per group:
  - `gather`: adds `ana_calib_total_us`, `ana_calib_mem_us`
  - `layout_move`: adds `ana_calib_total_us`, `ana_calib_mem_us`
  - `view_meta`: adds `ana_calib_total_us`, `ana_calib_mem_us`
  - `mixed_balanced`: adds `ana_calib_total_us`, `ana_calib_mem_us`, `ana_calib_compute_us`
  - `compute_dominant`: adds `ana_calib_total_us`, `ana_calib_compute_us`
- `build_classed_dataset.py` now keeps `model_group` routing based on `op_type` for both branches, so `with_analytical` no longer collapses back to `op_class`.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/contracts.py /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py`
- `python3 /data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/build_classed_dataset.py --feature-branch with_analytical --analytical-dir /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/analytical_calibrated --output-dir /tmp/classed_op_mlp_with_analytical_5way_smoke`

Key results:
- The with-analytical smoke dataset now reports:
  - `model_group_order = ['gather', 'layout_move', 'view_meta', 'mixed_balanced', 'compute_dominant']`
  - shared categorical features include `node_scope` and `node_name_normalized`
  - `gather/layout_move/view_meta` row counts match the strong `no_analytical` experiment exactly
- `with_analytical` is now an apples-to-apples augmentation baseline instead of a separate 3-way contract.

Open risks:
- This task aligned the dataset and feature contracts, but did not rerun the full NPU training/evaluation loop after the contract change.
- The current analytical additions are intentionally compact; if the next run regresses, the likely next step is to ablate which `ana_calib_*` terms help each 5-way group.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/classed_op_mlp/run_pipeline.py` still carries a separate user/local hidden-layer default change in the working tree, which was intentionally preserved.
- `/data/qc/dlrm/ORT/single_op_stage1_mlp/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/roofline_op_type_analysis/README.md`, `/data/qc/dlrm/ORT/single_op_stage1_mlp/data.sh`, and `/data/qc/dlrm/ORT/single_op_stage1_mlp/model.sh` still contain unrelated worktree changes and were intentionally left untouched.
