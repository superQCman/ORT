from __future__ import annotations

from pathlib import Path

from static_pipeline_eval.artifact_loader import DEFAULT_ORT_ROOT


CHAPTER4_OUTPUT_ROOT = (
    DEFAULT_ORT_ROOT / "static_pipeline_eval" / "artifacts" / "latest" / "chapter4_cpu"
)
CHAPTER4_DRAFT_PATH = DEFAULT_ORT_ROOT / "static_pipeline_eval" / "chapter4_cpu_experiments_draft.md"
CHAPTER4_SINGLE_ONLY_OUTPUT_ROOT = (
    DEFAULT_ORT_ROOT / "static_pipeline_eval" / "artifacts" / "latest" / "chapter4_cpu_single_only"
)
CHAPTER4_SINGLE_ONLY_DRAFT_PATH = (
    DEFAULT_ORT_ROOT / "static_pipeline_eval" / "chapter4_cpu_single_only_experiments_draft.md"
)

SINGLE_OP_ARTIFACT_ROOT = (
    DEFAULT_ORT_ROOT
    / "single_op_stage1_mlp"
    / "artifacts"
    / "latest"
    / "classed_op_mlp_test_78910_analytical_5_300_iter_quick_nodrop"
)
E2E_ARTIFACT_ROOT = (
    DEFAULT_ORT_ROOT
    / "static_pipeline_eval"
    / "artifacts"
    / "latest"
    / "v1_300_iter_quick_nodrop"
)
OOD_ARTIFACT_ROOT = (
    DEFAULT_ORT_ROOT
    / "single_op_stage1_mlp"
    / "artifacts"
    / "latest"
    / "analytical_generalization"
)
ABLATION_ARTIFACT_ROOT = (
    DEFAULT_ORT_ROOT
    / "single_op_stage1_mlp"
    / "artifacts"
    / "latest"
    / "feature_ablation"
    / "classed_op_mlp_test_2_analytical_5_200_iter"
)
BASELINE_MODEL_ROOT = DEFAULT_ORT_ROOT / "single_op_stage1_mlp" / "artifacts" / "latest" / "model_all_no_trace"
FAIR_SINGLE_MLP_HIDDEN_LAYERS = (128,128,128,128,128,128)
FAIR_SINGLE_MLP_MAX_ITER = 200
FAIR_SINGLE_MLP_BATCH_SIZE = 1024
FAIR_SINGLE_MLP_ALPHA = 1e-4
FAIR_SINGLE_MLP_LEARNING_RATE_INIT = 1e-3
FAIR_SINGLE_MLP_SEED = 42

# Global font-size control for Chapter 4 figures.
# Increase or decrease this single scale factor to adjust figure text uniformly.
CHAPTER4_FIGURE_FONT_SCALE = 1.0

OOD_BATCH_HOLDS = (1856, 1920, 1984, 2016, 2048)
OOD_NUM_THREADS_HOLD = 3

REPRESENTATIVE_OP_TYPES = ("Gather", "ReduceSum", "Transpose", "Concat", "Gemm")

TIMELINE_CASES: tuple[tuple[str, str], ...] = (
    ("case_10_3_3", "bs2048_nip2000"),
    ("case_10_4_6", "bs1856_nip1700"),
    ("case_8_1_1", "bs2048_nip2000"),
)

MODEL_GROUP_ORDER = ("gather", "layout_move", "view_meta", "mixed_balanced", "compute_dominant")

SECTION_ORDER = (
    "platform",
    "single_op_core",
    "single_op_ood",
    "single_op_ablation",
    "e2e_core",
    "e2e_sum_baseline",
    "timelines",
    "figures",
    "draft",
)

ONLY_CHOICES = ("all", "single_op", "e2e")

TABLE_FILENAMES = {
    "4-1": "table_4_1_platform_dataset.csv",
    "4-2": "table_4_2_single_op_group_metrics.csv",
    "4-3": "table_4_3_single_op_optype_metrics.csv",
    "4-4": "table_4_4_single_op_representative_ops.csv",
    "4-5": "table_4_5_e2e_static_summary.csv",
    "4-6": "table_4_6_e2e_sum_baseline.csv",
    "4-7": "table_4_7_single_op_ablation_summary.csv",
}

FIGURE_FILENAMES = {
    "4-1": "fig_4_1_platform_dataset_overview.png",
    "4-2": "fig_4_2_single_op_group_metrics.png",
    "4-3": "fig_4_3_single_op_group_baseline_compare.png",
    "4-4": "fig_4_4_single_op_optype_metrics.png",
    "4-5": "fig_4_5_single_op_representative_graph.png",
    "4-6": "fig_4_6_single_op_prediction_scatter.png",
    "4-7": "fig_4_7_single_op_training_history.png",
    "4-8": "fig_4_8_single_op_residual_distribution.png",
    "4-9": "fig_4_9_single_op_ood_slices.png",
    "4-10": "fig_4_10_single_op_ood_generalization.png",
    "4-11": "fig_4_11_e2e_predicted_vs_actual.png",
    "4-12": "fig_4_12_e2e_error_heatmap.png",
    "4-13": "fig_4_13_e2e_error_distribution.png",
    "4-14": "fig_4_14_timeline_cases.png",
    "4-15": "fig_4_15_critical_path_breakdown.png",
    "4-16": "fig_4_16_ablation_delta_bars.png",
    "4-17": "fig_4_17_ablation_delta_heatmap.png",
    "4-18": "fig_4_18_sum_baseline_compare.png",
    "4-19": "fig_4_19_best_ablation_summary.png",
}

REPRESENTATIVE_PANEL_FILENAME = "fig_4_5_single_op_representative_panel.png"


def output_subdir(root: Path, name: str) -> Path:
    return Path(root) / name
