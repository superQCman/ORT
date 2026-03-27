from __future__ import annotations

from feature_contract import METADATA_COLUMNS


TRACE_PROXY_CATEGORICAL_FEATURES = (
    "op_type",
    "node_scope",
    "node_name_normalized",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
)

TRACE_PROXY_NUMERIC_FEATURES = (
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
    "feat_activation_elements_per_batch",
    "feat_reduction_axes_count",
    "feat_reduction_axes_product",
    "feat_reduction_input_rank",
    "feat_reduction_output_rank",
    "feat_reduction_work_items",
    "hw_ratio_working_set_to_l1d_active_bytes",
    "hw_ratio_working_set_to_l2_active_bytes",
    "hw_ratio_working_set_to_l3_active_bytes",
    "local_ctx_same_op_overlap_ratio_mean",
    "comp_feat_pressure_ws_to_l2_ratio",
    "comp_feat_pressure_ws_to_l3_ratio",
    "hw_ratio_threads_to_total_cores",
    "local_ctx_overlap_ratio_mean",
    "local_ctx_cross_task_overlap_ratio_mean",
    "local_ctx_mean_other_active_mean",
    "local_ctx_mean_other_tasks_mean",
    "combo_task_parallel_fraction",
    "combo_task_weighted_mean_parallel_concurrency",
    "combo_op_parallel_fraction",
    "combo_op_weighted_mean_parallel_concurrency",
    "comp_feat_pressure_threads",
)

# These are the trace-derived columns currently exported in the prepared dataset tables.
DEFAULT_TRACE_PROXY_TARGET_COLUMNS = (
    "load_store_ratio",
    "feat_memops_per_inst",
    "feat_insts_per_thread",
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

# These raw DynamoRIO counters are not present in the current dataset export by default,
# but the training script can consume them later if a dataset variant includes them.
OPTIONAL_TRACE_PROXY_TARGET_COLUMNS = (
    "total_instructions",
    "total_loads",
    "total_stores",
)

ALL_TRACE_PROXY_TARGET_COLUMNS = tuple(
    list(OPTIONAL_TRACE_PROXY_TARGET_COLUMNS) + list(DEFAULT_TRACE_PROXY_TARGET_COLUMNS)
)

TRACE_PROXY_LOG_SCALE_TARGET_COLUMNS = (
    "total_instructions",
    "total_loads",
    "total_stores",
    "load_store_ratio",
    "feat_memops_per_inst",
    "feat_insts_per_thread",
    "reuse_time_mean",
    "reuse_distance_mean",
    "reuse_distance_unique_cache_lines_per_k_accesses",
)

TRACE_PROXY_INPUT_COLUMNS = list(TRACE_PROXY_CATEGORICAL_FEATURES + TRACE_PROXY_NUMERIC_FEATURES)
TRACE_PROXY_METADATA_COLUMNS = list(METADATA_COLUMNS)
