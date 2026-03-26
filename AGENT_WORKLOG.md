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
