from __future__ import annotations

from pathlib import Path

from analytical_calibrated.contracts import (
    ANALYTICAL_FEATURE_DESCRIPTIONS,
    BASELINE_COMPARE_DIR,
    DEFAULT_INPUT_DATASET_DIR,
    FEATURE_DESCRIPTIONS as BASE_FEATURE_DESCRIPTIONS,
    OP_CLASS_ORDER,
    OP_TYPE_CLASS_MAP,
    SHARED_CATEGORICAL_FEATURES,
    resolve_op_class,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "latest" / "classed_op_mlp"

FEATURE_BRANCH_WITH_ANALYTICAL = "with_analytical"
FEATURE_BRANCH_NO_ANALYTICAL = "no_analytical"
SUPPORTED_FEATURE_BRANCHES = (
    FEATURE_BRANCH_WITH_ANALYTICAL,
    FEATURE_BRANCH_NO_ANALYTICAL,
)
DEFAULT_FEATURE_BRANCH = FEATURE_BRANCH_WITH_ANALYTICAL

EXTRA_FEATURE_DESCRIPTIONS = {
    "batch_size": "该样本对应的 DLRM batch size。",
    "num_indices_per_lookup": "每次 embedding lookup 的索引数配置。",
    "output_size": "算子输出张量字节量。",
    "activation_size": "算子输入激活张量字节量。",
    "parameter_size": "算子参数张量字节量。",
    "feat_activation_elements_per_batch": "每个 batch 对应的输入激活元素规模。",
    "feat_reduction_axes_count": "被归约轴的数量。",
    "feat_reduction_input_rank": "归约前输入张量 rank。",
    "feat_reduction_output_rank": "归约后输出张量 rank。",
}

CLASSED_FEATURE_DESCRIPTIONS = {
    **BASE_FEATURE_DESCRIPTIONS,
    **EXTRA_FEATURE_DESCRIPTIONS,
    **ANALYTICAL_FEATURE_DESCRIPTIONS,
}

WITH_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES = {
    "memory_pure": (
        "num_threads",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_lookup_count",
        "feat_output_elements_per_lookup",
        "ana_calib_total_us",
        "ana_calib_mem_us",
    ),
    "mixed_balanced": (
        "num_threads",
        "feat_io_bytes_sum",
        "feat_output_elements_per_batch",
        "feat_output_input_bytes_ratio",
        "feat_reduction_work_items",
        "feat_reduction_axes_product",
        "ana_calib_total_us",
        "ana_calib_mem_us",
        "ana_calib_compute_us",
    ),
    "compute_dominant": (
        "num_threads",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
        "ana_calib_total_us",
        "ana_calib_compute_us",
    ),
}

NO_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES = {
    "memory_pure": (
        "batch_size",
        "num_indices_per_lookup",
        "num_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_lookup_count",
        "feat_output_elements_per_lookup",
        "feat_output_elements_per_batch",
    ),
    "mixed_balanced": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_output_elements_per_batch",
        "feat_activation_elements_per_batch",
        "feat_reduction_axes_count",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "feat_reduction_work_items",
    ),
    "compute_dominant": (
        "batch_size",
        "num_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "feat_io_bytes_sum",
        "feat_output_input_bytes_ratio",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
    ),
}

PER_BRANCH_NUMERIC_FEATURES = {
    FEATURE_BRANCH_WITH_ANALYTICAL: WITH_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES,
    FEATURE_BRANCH_NO_ANALYTICAL: NO_ANALYTICAL_PER_CLASS_NUMERIC_FEATURES,
}


def resolve_output_dir(output_dir: str | Path | None, feature_branch: str) -> Path:
    if output_dir:
        return Path(output_dir)
    if feature_branch == FEATURE_BRANCH_WITH_ANALYTICAL:
        return DEFAULT_OUTPUT_ROOT
    return DEFAULT_OUTPUT_ROOT / feature_branch


def resolve_branch_features(feature_branch: str) -> dict[str, tuple[str, ...]]:
    if feature_branch not in PER_BRANCH_NUMERIC_FEATURES:
        raise ValueError(f"Unsupported feature branch: {feature_branch}")
    return PER_BRANCH_NUMERIC_FEATURES[feature_branch]
