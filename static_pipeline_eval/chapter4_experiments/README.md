# Chapter 4 Experiments

This directory contains the unified Chapter 4 experiment orchestrator and the
section-specific entry points that feed it.

## Layout

```text
chapter4_experiments/
├── chapter4_config.py
├── run_all_chapter4_experiments.py
├── run_single_op_core_eval.py
├── run_single_op_ood_eval.py
├── run_single_op_ablation_eval.py
├── run_e2e_core_eval.py
├── run_e2e_sum_baseline.py
├── export_timeline_cases.py
├── build_chapter4_figures.py
└── write_chapter4_draft.py
```

## Defaults

- Output root: `ORT/static_pipeline_eval/artifacts/latest/chapter4_cpu`
- Draft: `ORT/static_pipeline_eval/chapter4_cpu_experiments_draft.md`
- Single-op source: `ORT/single_op_stage1_mlp/artifacts/latest/classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop`
- E2E source: `ORT/static_pipeline_eval/artifacts/latest/v1_300_iter_quick_nodrop`
- OOD source: `ORT/single_op_stage1_mlp/artifacts/latest/analytical_generalization`
- Ablation source: `ORT/single_op_stage1_mlp/artifacts/latest/feature_ablation/classed_op_mlp_test_2_analytical_5_200_iter`

## Usage

```bash
cd /data/qc/dlrm/ORT/static_pipeline_eval
python3 chapter4_experiments/run_all_chapter4_experiments.py
python3 chapter4_experiments/run_all_chapter4_experiments.py --only single_op
python3 chapter4_experiments/run_all_chapter4_experiments.py --only e2e --skip-timelines
```
