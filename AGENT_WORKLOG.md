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
