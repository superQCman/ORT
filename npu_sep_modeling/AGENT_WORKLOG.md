# Agent Worklog

This file is the persistent handoff and change record for `/data/qc/dlrm/ORT/npu_sep_modeling`.

Future agents working in this directory must read this file before changing code.

## Project Snapshot

### Purpose

This project is a self-contained NPU single-operator modeling pipeline for Ascend 910B3 separated architecture inside the ORT workspace. V1 focuses on parsing the CANN provider trace, constructing a compact combo/node dataset, fitting a small-parameter analytical model, and exporting calibration artifacts for later reuse.

### Main Components

- `README.md`
  Describes scope, data source, lane mapping, and run commands.
- `AGENTS.md`
  Directory guardrail that forces future agents to read the root and local worklogs before editing.
- `build_npu_dataset.py`
  Builds `dataset_full.csv`, `train.csv`, `val.csv`, `test.csv`, `dataset_summary.json`, and `feature_columns.json`.
- `hardware_probe.py`
  Collects locally observable 910B3 hardware inputs and best-effort peak estimates.
- `fit_sep_analytical_model.py`
  Fits the calibrated analytical model on the train split and emits `calibration.json` plus metrics.
- `evaluate_sep_analytical_model.py`
  Compares baseline vs calibrated predictions on val/test.

### Data and Label Conventions

- Dataset source:
  - `ORT/features_extensible_case_10_4_4_cann`
  - `ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles`
- Truth source:
  - `ort_cann_profile_*.json`
  - only `args.provider == "CANNExecutionProvider"`
- Aggregation key:
  - `(combo, node_index, op_name, node_name)`
- Label policy:
  - drop the first call by default
  - aggregate remaining calls with `mean/min/max/std/count`
  - label column is `label_npu_dur_us`
- Lane mapping:
  - `cube` = `MatMul`
  - `vector` = `Transpose` / `Add` / `Relu`
  - `transfer` = `MemcpyFromHost` / `MemcpyToHost`

### Workflow Conventions

- This directory is guarded by its own `AGENTS.md`, but the parent `ORT/AGENTS.md` and `ORT/AGENT_WORKLOG.md` remain the top-level source of truth.
- Future edits must append a dated entry here before the task is considered complete.
- Parent-repo commits must stay scoped to the files touched by the task.

## Change History

### 2026-04-04 - Scaffold NPU separation modeling project and guardrails

Request summary:
- Create a new `ORT/npu_sep_modeling` subproject for Ascend 910B3 separated-architecture NPU modeling.
- Add local guardrails plus a repo skill that forces future agents to read the parent and local worklogs before editing.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/README.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENTS.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/.codex/skills/ort-npu-sep-modeling/SKILL.md`

Behavior changes:
- Established the new project directory and documented its v1 scope, data source, and lane mapping.
- Added a local workflow guardrail that requires reading both the parent ORT worklog and the project worklog before future edits.
- Captured the initial project snapshot for the NPU modeling subproject.

Validation run:
- Documentation-only scaffold. Verified the new directory and files were created successfully.

Open risks:
- The current ORT data tree appears to expose 30 parseable NPU combo directories with one additional combo directory missing a JSON profile; if future work needs to enforce a 24-combo subset, we will need an explicit selector list.
- The hardware microbenchmark path still needs implementation and may depend on optional Ascend tooling or runtime packages that are not currently installed in this environment.

### 2026-04-04 - Build the NPU dataset from case_10_4_4_cann provider traces

Request summary:
- Implement the NPU dataset builder for `case_10_4_4_cann`.
- Parse only `CANNExecutionProvider` node events from the `ort_cann_profile_*.json` truth source.
- Preserve the source catalog CSV as metadata, while adding the new NPU lane, label, and shape/size columns.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`

Behavior changes:
- Added reusable helpers for combo parsing, shape/byte accounting, lane inference, split assignment, and regression metrics.
- Built `dataset_full.csv`, `train.csv`, `val.csv`, `test.csv`, `dataset_summary.json`, and `feature_columns.json` from the NPU provider trace truth source.
- The current tree now builds 360 aggregated rows across 30 combos, with a 7:2:1 combo split and no `CPUExecutionProvider` rows.
- `MemcpyFromHost` / `MemcpyToHost` are retained as a separate `transfer` lane with `h2d` / `d2h` direction tags.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py --case-id case_10_4_4_cann --output-dir /tmp/npu_sep_modeling_dataset_test --drop-first-call true`
- Verified the generated dataset contains 360 rows, 30 combos, and only `CANNExecutionProvider` rows.

Open risks:
- One combo directory under `onnx_profiles` still lacks a JSON profile, so the builder records the omission instead of fabricating labels.
- The current observed combo count is 30 rather than the earlier 24-combo planning assumption, so any downstream fixed-count tests should use the dataset summary instead of a hardcoded row total.
