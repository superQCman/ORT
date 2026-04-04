# Agent Worklog

This file is the persistent handoff and change record for `/data/qc/dlrm/ORT`.

Future agents working in this repository must read this file before changing code.

## Project Snapshot

### Purpose

This repository is the parent ORT workspace for the DLRM profiling, sweep, trace, and feature-generation pipeline. It also hosts two nested independent git repositories:

- `single_op_stage1_mlp`
- `static_pipeline_eval`

The parent repo coordinates the full-model sweep, branch-parallel execution, dynamic ONNX generation, per-op tracing, merged training feature creation, and the accompanying documentation.

### Main Components

- `run_ort_dlrm.py`
  Runs DLRM inference with ORT, applies CANN-oriented graph patching, exports `op_shapes.csv`, and writes profiling JSON.
- `run_ort_dlrm_branch_parallel.py`
  Splits the rewritten graph into `bottom`, `emb_l0..emb_l7`, and `tail`, then runs branch tasks concurrently via separate ORT sessions.
- `run_ort_sweep.sh`
  Canonical batch sweep driver that runs inference, tracing, feature extraction, dataset merging, and selected-feature extraction.
- `run_ort_sweep_extensible.sh`
  Extensible sweep driver that can switch to alternative inference frontends such as `branch_parallel` and carry model metadata from `dlrm_onnx_dyn/manifest.csv`.
- `generate_dlrm_dyn_onnx.sh`
  Builds a manifest of dynamic ONNX variants together with `arch_embedding_size`, `arch_mlp_bot`, and `arch_mlp_top`.
- `onnx_operator_analysis/build_training_features.py`
  Merges `op_shapes`, CPU-thread profiling, and trace features into the final training CSV.
- `features/README.md`
  Documents the schema of the merged `features/*.csv` outputs.
- `README.md` and `README.zh-CN.md`
  High-level workflow documentation for the parent ORT workspace.

### Current Metadata and Dataset Conventions

- Final merged feature rows keep one row per ONNX node from `op_shapes`.
- When the sweep ONNX is resolved from `dlrm_onnx_dyn/manifest.csv`, the merged dataset also carries:
  - `arch_embedding_size`
  - `arch_mlp_bot`
  - `arch_mlp_top`
- The selected-feature dataset preserves the same metadata columns.
- The extensible sweep path already reads manifest-backed architecture metadata and passes it into the training feature builder.

### Workflow Conventions

- Root-repo edits must be logged in this file after each completed change.
- Nested repositories keep their own workflow files, worklogs, and commits.
- The parent `ORT` repository should commit only parent-repo files; nested repositories should not be swept into the same git commit.

## Change History

### 2026-04-04 - Add root ORT workflow guardrail and align parent docs with metadata flow

Request summary:
- Add a root-level agent workflow guardrail for `ORT`.
- Review the ORT README files for any additional documentation updates.
- Keep the already-updated `arch_*` metadata flow and pipeline-script changes together in the parent repository.

Files changed:
- `/data/qc/dlrm/ORT/AGENTS.md`
- `/data/qc/dlrm/ORT/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/.codex/skills/ort-root-workflow/SKILL.md`
- `/data/qc/dlrm/ORT/README.md`
- `/data/qc/dlrm/ORT/README.zh-CN.md`
- `/data/qc/dlrm/ORT/features/README.md`
- `/data/qc/dlrm/ORT/generate_dlrm_dyn_onnx.sh`
- `/data/qc/dlrm/ORT/run_ort_dlrm.py`
- `/data/qc/dlrm/ORT/run_ort_dlrm_branch_parallel.py`
- `/data/qc/dlrm/ORT/run_ort_sweep.sh`

Behavior changes:
- Established a root-level workflow rule: future agents must read the root worklog before editing, append a dated entry after each completed change, and commit in the parent `ORT` repository.
- Documented the parent repo's `arch_*` metadata flow and the merged-feature schema in the README files.
- Aligned the `features/*.csv` field dictionary with the new metadata columns.
- Kept the parent sweep pipeline consistent with the manifest-backed dynamic ONNX metadata and the updated merged-header handling.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/run_ort_dlrm.py /data/qc/dlrm/ORT/run_ort_dlrm_branch_parallel.py /data/qc/dlrm/ORT/onnx_operator_analysis/build_training_features.py`
- `bash -n /data/qc/dlrm/ORT/run_ort_sweep.sh /data/qc/dlrm/ORT/generate_dlrm_dyn_onnx.sh /data/qc/dlrm/ORT/run_ort_sweep_extensible.sh`

Open risks:
- The root workflow guardrail depends on future agents honoring the new root `AGENTS.md`; existing nested-repo guardrails still apply independently.
- The parent worktree still contains unrelated dirty files outside this change set, so commit hygiene must stay strict.

### 2026-04-04 - Add 910B3 separated-architecture NPU modeling subproject

Request summary:
- Create the new `ORT/npu_sep_modeling` subproject for Ascend 910B3 separated-architecture NPU single-op modeling.
- Add the local guardrail files plus the dedicated skill that forces future agents to read the parent and local worklogs before editing.
- Implement the NPU dataset builder, hardware probe, analytical calibration, and evaluation report workflow.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/README.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENTS.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py`
- `/data/qc/dlrm/ORT/.codex/skills/ort-npu-sep-modeling/SKILL.md`
- `/data/qc/dlrm/ORT/AGENT_WORKLOG.md`

Behavior changes:
- Established a self-contained NPU modeling workflow with explicit lane separation for `cube`, `vector`, and `transfer`.
- Captured the current data reality for `case_10_4_4_cann`: 30 parseable combos and 360 aggregated rows from `CANNExecutionProvider` trace events.
- Added a best-effort hardware probe and a small-parameter calibration flow that exports reusable JSON artifacts for downstream evaluation.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py --case-id case_10_4_4_cann --output-dir /tmp/npu_sep_modeling_dataset_test --drop-first-call true`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py --output-dir /tmp/npu_sep_modeling_hw_test3`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test3/hardware_profile_910b3.json --output-dir /tmp/npu_sep_modeling_fit_test3`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test3/hardware_profile_910b3.json --calibration /tmp/npu_sep_modeling_fit_test3/calibration.json --output-dir /tmp/npu_sep_modeling_eval_test3`

Open risks:
- The hardware probe still emits null peak fields because the current environment does not provide the ORT/CANN microbenchmark runtime stack needed to measure them directly.
- The current data tree resolves to 30 combos instead of the earlier planning assumption of 24, so downstream assertions should key off the dataset summary.
