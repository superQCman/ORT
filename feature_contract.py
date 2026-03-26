from __future__ import annotations

TARGET_COLUMN = "label_operator_actual_dur_us"
GROUP_COLUMN = "sample_group"
PROFILE_INSTABILITY_METRICS = (
    "last2_range_ratio",
    "last2_cv",
)

BASELINE_CATEGORICAL_FEATURES = (
    "op_type",
    "node_scope",
    "node_name_normalized",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
)

BASELINE_NUMERIC_FEATURES = (
    "batch_size",
    "num_indices_per_lookup",
    "num_threads",
    "output_size",
    "activation_size",
    "parameter_size",
    "load_store_ratio",
    "feat_io_bytes_sum",
    "feat_output_input_bytes_ratio",
    "feat_memops_per_inst",
    "feat_insts_per_thread",
    "feat_lookup_count",
    "feat_output_elements_per_lookup",
    "feat_output_elements_per_batch",
    "feat_activation_elements_per_batch",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_reduction_input_rank",
    "feat_reduction_output_rank",
    "feat_reduction_work_items",
    "reuse_time_mean",
    "reuse_distance_mean",
    "reuse_distance_unique_cache_lines_per_k_accesses",
    "opc_branch_ratio",
    "opc_fp_math_ratio",
    "opc_load_ratio",
    "opc_math_ratio",
    "opc_simd_ratio",
    "opc_store_ratio",
)

FEATURE_COLUMNS = list(BASELINE_CATEGORICAL_FEATURES + BASELINE_NUMERIC_FEATURES)

METADATA_COLUMNS = (
    "row_uid",
    "case_id",
    "source_name",
    "source_mode",
    "combo",
    "sample_group",
    "split",
    "op_idx",
    "node_name",
    "trace_op_name",
    "has_cpu_profile",
    "input_type_shape",
    "output_type_shape",
    "profile_batch_count_total",
    "profile_batch_count_kept",
    "profile_dropped_batch_indices",
    "profile_kept_batch_indices",
    "profile_label_all_batch_mean_us",
    "profile_label_kept_batch_mean_us",
    "profile_last2_abs_diff_us",
    "profile_last2_range_ratio",
    "profile_last2_cv",
)

DEFAULT_SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}
