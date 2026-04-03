# Agent Worklog

This file is the persistent handoff and change record for `/data/qc/dlrm/ORT/static_pipeline_eval`.

Future agents working in this directory must read this file before changing code.

## Project Snapshot

### Purpose

This project is a self-contained static pipeline evaluator for ORT DLRM branch-parallel execution. It reuses saved per-op predictions from `single_op_stage1_mlp`, reconstructs combo-level graph structure from `op_shapes`, applies a static scheduler for the 8 embedding branches, and compares predicted makespan against actual whole-graph timing from branch-parallel timeline traces.

### Main Components

- `run_static_pipeline_eval.py`
  Public CLI entrypoint for loading artifact inputs, running combo-level scheduling, and writing reports.
- `static_pipeline_eval/artifact_loader.py`
  Reads the classed-op artifact, sweep metadata, op-shapes graphs, and branch-parallel timeline traces.
- `static_pipeline_eval/graph_contract.py`
  Defines typed records for combo specs, op nodes, coverage summaries, timing summaries, and schedule results.
- `static_pipeline_eval/schedule_engine.py`
  Contracts embedding chains into branch tasks, reconstructs task dependencies, and performs static FIFO slot scheduling controlled by `inter_threads`.
- `tests/`
  Unit and integration coverage for graph parsing, scheduler behavior, timeline truth extraction, and end-to-end artifact handling.

### Current Scheduling Conventions

- `v1` only targets the existing test artifact:
  - `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_200_iter_quick`
- Default per-op prediction source:
  - `models/combined/combined_predictions_test.csv`
- Default ground-truth source for combo-level makespan:
  - `branch_parallel_op_timeline.csv`
- Whole-graph truth label policy:
  - drop the earliest batch
  - use the mean of the remaining batch spans
- Embedding branch rule set:
  - launch order is fixed FIFO `0 -> 7`
  - first-wave maximum gather concurrency equals `inter_threads`
  - one branch holds a slot from its `Gather` start until its `ReduceSum` end
  - later branches enter when the earliest occupied slot frees

### Current Report Conventions

- Full-graph combo:
  - all 60 modeled non-Constant ops from `op_shapes` are present in the saved prediction rows
- Partial combo:
  - raw sweep/profile exists, but some nodes were filtered out earlier by `single_op_stage1_mlp` label-stability rules
- Reports are split into:
  - full-graph E2E metrics
  - partial observed-subgraph diagnostics
- Partial metrics must never be mixed into the primary E2E summary.

## Change History

### 2026-04-03 - Bootstrap static pipeline evaluation project

Request summary:
- Create a new independent project under `ORT/static_pipeline_eval`.
- Implement the initial static pipeline evaluator, project workflow files, local skill, tests, and a nested git workflow.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENTS.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`

Behavior changes:
- Established the project scope, workflow guardrails, and initial scheduling/report assumptions for the static pipeline evaluator.

Validation run:
- Pending implementation.

Open risks:
- Pending implementation.

### 2026-04-03 - Implement v1 static scheduler, reports, tests, and nested git workflow

Request summary:
- Implement the full `ORT/static_pipeline_eval` v1 project.
- Reuse saved per-op predictions from `single_op_stage1_mlp`, reconstruct combo DAGs from `op_shapes`, model embedding branch FIFO slot scheduling, split outputs into full/partial reports, identify concrete calibration points, and save the work in an independent nested git repository.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/.gitignore`
- `/data/qc/dlrm/ORT/static_pipeline_eval/.codex/skills/ort-static-pipeline-eval/SKILL.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/pyproject.toml`
- `/data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/__init__.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/artifact_loader.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/graph_contract.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/schedule_engine.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/tests/conftest.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/tests/test_schedule_engine.py`

Behavior changes:
- Initialized an independent git repository rooted at `/data/qc/dlrm/ORT/static_pipeline_eval`.
- Added project guardrails through `AGENTS.md`, `AGENT_WORKLOG.md`, and local skill `ort-static-pipeline-eval`.
- Implemented `artifact_loader.py` to read:
  - `classed_dataset_full.csv`
  - `models/combined/combined_predictions_test.csv`
  - case `sweep_summary.csv`
  - combo `op_shapes_*.csv`
  - combo `branch_parallel_op_timeline.csv`
- Implemented typed contracts for combo metadata, op DAG nodes, branch tasks, coverage, timing summaries, and schedule results.
- Implemented the v1 scheduler with these fixed semantics:
  - `bottom` and embedding branches start together
  - embedding branches are collapsed into `/emb_lX/{Gather,Reshape,ReduceSum}` composite tasks
  - launch order is fixed FIFO `0 -> 7`
  - `inter_threads` controls available branch slots
  - each branch holds a slot from `Gather` start through `ReduceSum` end
  - `tail` is modeled as a separate barriered session that starts after branch pool / bottom completion, then follows exact `op_shapes` DAG dependencies internally
- Implemented `run_static_pipeline_eval.py` to emit:
  - `summary.json`
  - `full_combo_metrics.csv`
  - `partial_combo_metrics.csv`
  - `embedding_order_check.csv`
  - `calibration_candidates.md`
- Explicitly labeled partial metrics as `observed_subgraph_non_e2e` inside `partial_combo_metrics.csv`.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/graph_contract.py /data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/artifact_loader.py /data/qc/dlrm/ORT/static_pipeline_eval/static_pipeline_eval/schedule_engine.py /data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py`
- `pytest -q`
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py --run-name v1_validation`
- Output directory:
  - `/data/qc/dlrm/ORT/static_pipeline_eval/artifacts/latest/v1_validation`
- Validated result snapshot:
  - total test combos: `331`
  - full combos: `49`
  - partial combos: `282`
  - full-graph MAPE: `0.041985`
  - partial observed-subgraph MAPE: `0.171456`
  - all combos recovered embedding launch order `0 -> 7`
  - all combos recovered max gather concurrency equal to `inter_threads`

Open risks:
- Full-graph error is already moderate, but worst full combo still reaches about `14.7%` APE; branch-level calibration is still worth reserving.
- Partial observed-subgraph error remains large because missing nodes are zero-duration placeholders in v1, so this report is diagnostic only and must not be read as E2E quality.
- Embedding handoff gaps cluster in the low hundreds of microseconds, but rare outliers still appear in some combos; a later black-box correction should probably use robust features/statistics instead of raw maxima.
