# Chapter 4 Experiments

This directory contains the unified Chapter 4 experiment orchestrator, the
section-specific entry points, and the paper-oriented report builders used to
refresh the CPU experiment chapter under `ORT/static_pipeline_eval`.

## Layout

```text
chapter4_experiments/
├── chapter4_config.py
├── run_all_chapter4_experiments.py
├── run_single_op_fair_baseline.py
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

## What The Runner Produces

- `Table 4-1` to `Table 4-7` in CSV/Markdown form
- `Figure 4-1` to `Figure 4-19` under `artifacts/latest/chapter4_cpu/figures/`
- a regenerated Chapter 4 draft at `chapter4_cpu_experiments_draft.md`

The current Chapter 4 workflow keeps the Concorde-style main composition chain
and adds a fair single-MLP comparator:

1. `Analytical + simple add`
2. `Analytical + fair single MLP + simple add`
3. `Analytical + grouped MLP + simple add`
4. `Analytical + grouped MLP + static pipeline`

This keeps the paper-style ablation in Section 4.4 reproducible from one
command while also making the single-vs-grouped MLP comparison apples-to-apples.

For fair single-op comparisons, `run_single_op_fair_baseline.py` rebuilds a
single-MLP baseline on the exact same `classed_dataset_full.csv` split used by
the grouped artifact and uses the union of grouped numeric features plus the
shared categorical features as its input contract.

## Usage

```bash
cd /data/qc/dlrm/ORT/static_pipeline_eval
python3 chapter4_experiments/run_all_chapter4_experiments.py
python3 chapter4_experiments/run_single_op_fair_baseline.py --force-retrain
python3 chapter4_experiments/run_all_chapter4_experiments.py --only single_op
python3 chapter4_experiments/run_all_chapter4_experiments.py --only e2e --skip-timelines
```
