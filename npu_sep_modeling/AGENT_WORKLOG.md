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

### 2026-04-04 - Add hardware probe, analytical calibration, and evaluation report

Request summary:
- Implement the hardware probe that reads `npu-smi` and emits `hardware_profile_910b3.json`.
- Fit the separated analytical model with small-parameter calibration.
- Generate a baseline-vs-calibrated comparison report and document the final workflow in the README.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/README.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`

Behavior changes:
- `hardware_probe.py` now emits an 8-device 910B3 hardware JSON with `ai_core_count=20` and explicit `null` peaks when benchmark dependencies are missing.
- `fit_sep_analytical_model.py` fits `scale + bias_us` calibration parameters for `op_name` and `npu_lane`, using built-in fallback roofline defaults when microbench peaks are absent.
- `evaluate_sep_analytical_model.py` now writes a Markdown comparison report plus a JSON summary for baseline vs calibrated metrics.
- The README now documents the lane split, the `op_name` vs `op_type` distinction, the current 30-combo / 360-row dataset size, and the output artifacts.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py /data/qc/dlrm/ORT/npu_sep_modeling/build_npu_dataset.py /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py --output-dir /tmp/npu_sep_modeling_hw_test3`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test3/hardware_profile_910b3.json --output-dir /tmp/npu_sep_modeling_fit_test3`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test3/hardware_profile_910b3.json --calibration /tmp/npu_sep_modeling_fit_test3/calibration.json --output-dir /tmp/npu_sep_modeling_eval_test3`
- Verified outputs:
  - hardware JSON: `device_count=8`, `chip_count=1`, `ai_core_count=20`, `frequency_mhz=1800`
  - fit calibration: `hardware_profile_effective.ai_core_count=20`
  - overall metrics:
    - train baseline MAE `3090.198` -> calibrated MAE `249.288`
    - val baseline MAE `3145.618` -> calibrated MAE `261.388`
    - test baseline MAE `3310.722` -> calibrated MAE `444.670`

Open risks:
- Peak values remain `null` in the hardware JSON because this environment does not yet have the ORT/CANN microbenchmark runtime stack (`onnx` / `onnxruntime`) needed to measure them directly.
- The calibration therefore uses built-in fallback roofline defaults for `cube`, `vector`, and `transfer`; if a future environment provides real microbench values, the reported coefficients should be refreshed.
- The current data tree still resolves to 30 combos rather than the earlier planning assumption of 24, so any downstream test assertions should key off `dataset_summary.json` instead of a fixed row count.

### 2026-04-04 - Tighten 910B3 hardware inputs and document the analytical model

Request summary:
- Remove `board_name` from the exported hardware profile.
- Confirm the 910B3 Cube/Vector core counts and keep them as explicit hardware inputs.
- Expand the README so the analytical model construction is clear and reproducible.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/README.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py`

Behavior changes:
- `hardware_probe.py` no longer emits `board_name`; it now exports `cube_count=20` and `vector_count=40` for the 910B3 separated architecture input set.
- `fit_sep_analytical_model.py` now carries those count fields into `hardware_profile_effective` alongside the peak and bandwidth defaults.
- The README now explains the baseline construction flow, the per-lane roofline formulas, and the small-parameter calibration strategy in plain terms.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/hardware_probe.py --output-dir /tmp/npu_sep_modeling_hw_test4`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test4/hardware_profile_910b3.json --output-dir /tmp/npu_sep_modeling_fit_test4`

Observed results:
- hardware JSON now contains `cube_count=20` and `vector_count=40`, while `board_name` is absent.
- calibration JSON now includes `hardware_profile_effective.cube_count=20` and `hardware_profile_effective.vector_count=40`.

Open risks:
- The cube/vector count split is inferred from the separated AI Core layout and the observed `Aicore Count=20`; if future official data revises the mapping, the profile should be updated in one place.

### 2026-04-04 - Add subset calibration and conditional generalization reporting

Request summary:
- Allow the calibration stage to fit on only a stratified fraction of the training rows.
- Expose an internal train holdout so the remaining rows can be used to check generalization.
- Document the fact that generalization needs assumptions, and provide a conditional proof sketch instead of an unconditional claim.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/README.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`
- `/data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`

Behavior changes:
- `fit_sep_analytical_model.py` now accepts `--calibration-fit-fraction` and `--calibration-seed`.
- The calibration subset is sampled stratified by `op_name`; the unused rows are preserved as an internal holdout.
- `calibration.json` now records the subset metadata and `metrics_summary.json` includes `train_fit` and `train_heldout` metrics.
- `evaluate_sep_analytical_model.py` now renders the internal train holdout in the Markdown comparison report when subset calibration is used.
- The README now states the conditional nature of the generalization argument and adds a proof sketch with the required assumptions.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py /data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test4/hardware_profile_910b3.json --calibration-fit-fraction 0.3 --calibration-seed 42 --output-dir /tmp/npu_sep_modeling_fit_subset_test`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test4/hardware_profile_910b3.json --calibration /tmp/npu_sep_modeling_fit_subset_test/calibration.json --output-dir /tmp/npu_sep_modeling_eval_subset_test`

Observed results:
- The 252-row train split was divided into 78 fit rows and 174 internal holdout rows at `calibration-fit-fraction=0.3`.
- Holdout `calibrated` MAPE stayed around `9.49%`, which is close to the fitted subset's `7.16%` and well below the baseline `100%`.

Open risks:
- The conditional generalization argument still depends on the i.i.d. / bounded-noise assumptions stated in the README.
- If a future data split has very small per-op groups, stratified sampling may collapse to one row per group, which keeps coverage but weakens the statistical bound.

### 2026-04-04 - Fix baseline numeric parsing and refresh error estimates

Request summary:
- Investigate why the baseline MAPE stayed at 100% after the subset calibration work.
- Fix the baseline numeric parsing bug so the analytical roofline is evaluated with the real shape/byte inputs.
- Recompute the current baseline and calibrated error levels.

Files changed:
- `/data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py`
- `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`

Behavior changes:
- `safe_float` now correctly parses numeric scalars and strings instead of returning `None` for every non-`None` input.
- The baseline roofline now uses the actual per-row `matmul_m/k/n`, bytes, and vector element counts, so its error metrics are meaningful again.

Validation run:
- `python3 -m py_compile /data/qc/dlrm/ORT/npu_sep_modeling/npu_sep_common.py /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/fit_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test4/hardware_profile_910b3.json --calibration-fit-fraction 0.3 --calibration-seed 42 --output-dir /tmp/npu_sep_modeling_fit_subset_test2`
- `python3 /data/qc/dlrm/ORT/npu_sep_modeling/evaluate_sep_analytical_model.py --data-dir /tmp/npu_sep_modeling_dataset_test --hardware-profile /tmp/npu_sep_modeling_hw_test4/hardware_profile_910b3.json --calibration /tmp/npu_sep_modeling_fit_subset_test2/calibration.json --output-dir /tmp/npu_sep_modeling_eval_subset_test2`

Observed results:
- Baseline MAPE dropped from the broken `100%` artifact to about `77%` on train/val/test.
- The calibrated model remained in the `~5%` to `~12%` MAPE range, and the internal train holdout stayed close to the fit subset.

Open risks:
- Even with the parsing bug fixed, the baseline is still only a coarse roofline approximation, so it is not expected to reach `20%` MAPE without better hardware peaks or richer per-op correction.
