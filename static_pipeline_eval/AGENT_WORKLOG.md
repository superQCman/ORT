# Agent Worklog

This file is the persistent handoff and change record for `/data/qc/dlrm/ORT/static_pipeline_eval`.

Future agents working in this directory must read this file before changing code.

## Project Snapshot

### Purpose

This project is a self-contained static pipeline evaluator for ORT DLRM branch-parallel execution within the ORT monorepo. It reuses saved per-op predictions from `single_op_stage1_mlp`, reconstructs combo-level graph structure from `op_shapes`, applies a static scheduler for the 8 embedding branches, and compares predicted makespan against actual whole-graph timing from branch-parallel timeline traces.

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

- Repository layout:
  - this project now lives inside the ORT monorepo via subtree migration
  - the artifact paths and combo naming conventions remain unchanged
- `v1` defaults to the existing test artifact:
  - `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_200_iter_quick`
- The CLI can override `--artifact-root` for schema-compatible artifacts from `single_op_stage1_mlp`.
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

### 2026-04-05 - Add unified Chapter 4 experiment directory and one-click runner

Request summary:
- Reorganize Chapter 4 into a single `static_pipeline_eval/chapter4_experiments` directory.
- Add one top-level runner that can rebuild the Chapter 4 tables, figures, manifests, and chapter draft in one command.
- Keep the existing `single_op_stage1_mlp` and `static_pipeline_eval` artifacts as the data sources, but move the control surface into `static_pipeline_eval`.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/README.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_cpu_experiments_draft.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/README.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/__init__.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/chapter4_config.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/chapter4_shared.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_all_chapter4_experiments.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_single_op_core_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_single_op_ood_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_single_op_ablation_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_e2e_core_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_e2e_sum_baseline.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/export_timeline_cases.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/build_chapter4_figures.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/write_chapter4_draft.py`

Behavior changes:
- Added a unified Chapter 4 control plane under `chapter4_experiments/` with shared configuration and helper code.
- Added `run_all_chapter4_experiments.py` as the single entry point for:
  - platform statistics
  - single-op core evaluation
  - single-op OOD evaluation
  - single-op ablation evaluation
  - E2E static aggregation
  - E2E simple-sum baseline
  - timeline and critical-path export
  - figure catalog generation
  - chapter draft generation
- Standardized the Chapter 4 output root to `artifacts/latest/chapter4_cpu`.
- Standardized the draft output path to `chapter4_cpu_experiments_draft.md`.
- Kept the Chapter 4 scripts configurable through explicit artifact-root arguments so they can reuse existing `single_op_stage1_mlp` and `static_pipeline_eval` outputs without hard-coded local paths.

Validation run:
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_all_chapter4_experiments.py --help`
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_all_chapter4_experiments.py --only single_op --skip-ood --skip-ablation`
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_all_chapter4_experiments.py --only e2e --skip-timelines`
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/run_all_chapter4_experiments.py --only all`
- `python3 -m py_compile /data/qc/dlrm/ORT/static_pipeline_eval/chapter4_experiments/*.py`

Open risks:
- The generated Chapter 4 draft is intentionally auto-assembled and still reads like a structured technical draft rather than a publication-polished manuscript.
- The output tree under `artifacts/latest/chapter4_cpu` is reproducible locally, but it is not yet clear whether every generated table/figure should be checked into git or kept as derived output only.
- If upstream artifact schemas change, the chapter runner will need another compatibility pass to keep the one-click workflow stable.

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

### 2026-04-03 - Validate v1 on the 300-iter nodrop artifact and harden empty-report handling

Request summary:
- Run the static pipeline evaluator on `/data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop`.
- If needed, fix compatibility issues and save the result in the nested git repository.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py`
- `/data/qc/dlrm/ORT/static_pipeline_eval/tests/test_run_static_pipeline_eval.py`

Behavior changes:
- Hardened `run_static_pipeline_eval.py` so `full_rows` or `partial_rows` may be empty without crashing.
- Empty reports now still emit CSV files with stable headers.
- Confirmed the CLI works on the schema-compatible `300_iter_quick_nodrop` artifact via `--artifact-root`.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py`
- `pytest -q`
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py --artifact-root /data/qc/dlrm/ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop --run-name v1_300_iter_quick_nodrop`
- Output directory:
  - `/data/qc/dlrm/ORT/static_pipeline_eval/artifacts/latest/v1_300_iter_quick_nodrop`
- Validated result snapshot:
  - total test combos: `331`
  - full combos: `331`
  - partial combos: `0`
  - full-graph MAPE: `0.063821`
  - full-graph p95 APE: `0.180852`
  - worst full combo: `case_8_1_1 / bs2048_nip2000`, APE `0.223484`
  - all combos recovered embedding launch order `0 -> 7`
  - all combos recovered max gather concurrency equal to `inter_threads`
- Comparison against the earlier `200_iter_quick` run:
  - full combos increased from `49` to `331`
  - partial combos dropped from `282` to `0`
  - full-graph MAPE increased from `0.041985` to `0.063821`
  - full-graph p95 APE increased from `0.121682` to `0.180852`

Open risks:
- `nodrop` removes the partial-coverage blind spot, but it also exposes more hard samples, so aggregate full-graph error is worse than the filtered artifact.
- The worst nodrop errors are still dominated by embedding branch residuals, so branch-level calibration remains the most valuable next step.

### 2026-04-03 - Add project README and paper-style scheduler document

Request summary:
- Add a `README.md` under `ORT/static_pipeline_eval` to explain the static pipeline scheduling flow and how to use the project.
- Add a second document in academic-paper style to formalize the scheduling method.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/README.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/STATIC_PIPELINE_SCHEDULER_PAPER.md`

Behavior changes:
- Added `README.md` as the engineering-facing entry document for:
  - project goal
  - scheduling flow
  - directory structure
  - CLI usage
  - output artifact interpretation
  - current validated runs
- Added `STATIC_PIPELINE_SCHEDULER_PAPER.md` as a Chinese paper-style technical description covering:
  - problem formulation
  - graph reconstruction
  - branch contraction
  - FIFO slot scheduling with `inter_threads`
  - truth extraction
  - coverage regimes
  - residual interpretation
  - limitations

Validation run:
- `python3 /data/qc/dlrm/ORT/static_pipeline_eval/run_static_pipeline_eval.py --help`
- Manual inspection of:
  - `/data/qc/dlrm/ORT/static_pipeline_eval/README.md`
  - `/data/qc/dlrm/ORT/static_pipeline_eval/STATIC_PIPELINE_SCHEDULER_PAPER.md`

Open risks:
- The paper-style document is currently a technical-method note in论文写法, not a polished publication draft with formal experiments, citations, or figure/table numbering.
- If this document is later turned into an external paper, the evaluation and related-work sections will need to be expanded substantially.

### 2026-04-03 - Rewrite scheduler paper into a more academic manuscript style

Request summary:
- Rewrite `STATIC_PIPELINE_SCHEDULER_PAPER.md` so it reads like an academic paper rather than an engineering note.
- Remove repository/file-path-centric narration, avoid code variable names, describe the method in prose, and replace the pseudocode with a more standard paper-style algorithm block.

Files changed:
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/STATIC_PIPELINE_SCHEDULER_PAPER.md`

Behavior changes:
- Reframed the document around formal problem formulation, execution semantics, task-graph construction, scheduling equations, and residual interpretation.
- Removed explicit repository paths, CLI/file-level descriptions, and most engineering-facing workflow language from the paper document.
- Replaced code-like feature names with mathematical symbols or full semantic descriptions.
- Rewrote the scheduling pseudocode into a standard `Require/Ensure` academic algorithm format with numbered lines.

Validation run:
- `sed -n '1,260p' /data/qc/dlrm/ORT/static_pipeline_eval/STATIC_PIPELINE_SCHEDULER_PAPER.md`
- Manual inspection for:
  - removal of engineering file references
  - prose-style methodological description
  - standard paper-style pseudocode formatting

Open risks:
- The document is now much more manuscript-like, but it is still a methods paper draft rather than a submission-ready paper with citations, related work, theorem statements, or formatted tables/figures.
- If this is later adapted to a formal venue template, equation numbering, algorithm environment styling, references, and experimental sections will still need another pass.

### 2026-04-04 - Migrate static_pipeline_eval into the ORT monorepo via subtree

Request summary:
- Import the standalone `static_pipeline_eval` repository into the parent `ORT` monorepo while preserving commit history.
- Update the project docs and agent instructions so they describe the subtree-imported monorepo layout instead of a separate nested repository.

Files changed:
- `/data/qc/dlrm/ORT/README.md`
- `/data/qc/dlrm/ORT/README.zh-CN.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENTS.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/static_pipeline_eval/README.md`

Behavior changes:
- Preserved the static pipeline evaluator history by importing it into the ORT monorepo with `git subtree add` from a bare clone.
- Reworded the project README so it now describes `static_pipeline_eval` as an ORT monorepo subproject rather than an independent repository.
- Updated agent instructions and commit guidance to point at the parent ORT repository instead of a nested git root.
- Added a root ORT README section that links the subtree-imported `single_op_stage1_mlp` and `static_pipeline_eval` subprojects.

Validation run:
- `git -C /tmp/ORT_monorepo_merge subtree add --prefix=static_pipeline_eval /tmp/ort_subtree_sources/static_pipeline_eval.git master`
- Manual review of:
  - `/tmp/ORT_monorepo_merge/README.md`
  - `/tmp/ORT_monorepo_merge/README.zh-CN.md`
  - `/tmp/ORT_monorepo_merge/static_pipeline_eval/AGENTS.md`
  - `/tmp/ORT_monorepo_merge/static_pipeline_eval/README.md`

Open risks:
- The original workspace had pre-existing dirty state, so this migration was carried out in a temporary clean worktree and then mirrored back into the visible workspace.
- The subtree import preserved history, but future edits should continue to avoid reintroducing a nested `.git` directory under `static_pipeline_eval`.
