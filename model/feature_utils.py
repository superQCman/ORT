from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_utils import infer_combo_from_text, parse_trace_op_idx, safe_float


DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "uint16": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "double": 8,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}


def load_selected_feature_rows(features_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(features_root.glob("bs*_nip*.csv")):
        combo = path.stem
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["combo"] = combo
        df["feature_csv"] = str(path)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No feature CSVs found under {features_root}")

    df = pd.concat(rows, ignore_index=True)
    df["op_idx"] = df["trace_op_name"].map(parse_trace_op_idx)
    df["op_type"] = df["op_type"].fillna("Unknown").astype(str)
    df["node_name"] = df["node_name"].fillna("").astype(str)
    df["node_scope"] = df["node_name"].map(extract_node_scope)
    df["combo"] = df["combo"].astype(str)
    return df


def extract_node_scope(node_name: str) -> str:
    if not node_name:
        return "unknown"
    stripped = node_name.strip("/")
    if not stripped:
        return "root"
    return stripped.split("/", 1)[0]


def _shape_entries(value: str | float | int | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    out = []
    for item in parsed:
        if not isinstance(item, dict) or not item:
            continue
        dtype, dims = next(iter(item.items()))
        if not isinstance(dims, list):
            continue
        clean_dims = []
        dynamic_dims = 0
        for dim in dims:
            if isinstance(dim, int) and dim > 0:
                clean_dims.append(dim)
            elif isinstance(dim, float) and dim.is_integer() and dim > 0:
                clean_dims.append(int(dim))
            else:
                dynamic_dims += 1
        out.append(
            {
                "dtype": str(dtype),
                "dims": clean_dims,
                "rank": len(clean_dims),
                "dynamic_dims": dynamic_dims,
            }
        )
    return out


def _entry_num_elements(entry: dict[str, Any]) -> float:
    dims = entry.get("dims", [])
    if not dims:
        return 0.0
    product = 1
    for dim in dims:
        product *= dim
    return float(product)


def _shape_features(shape_text: str | float | int | None, prefix: str) -> dict[str, float]:
    entries = _shape_entries(shape_text)
    if not entries:
        return {
            f"feat_{prefix}_tensor_count": 0.0,
            f"feat_{prefix}_rank_sum": 0.0,
            f"feat_{prefix}_rank_max": 0.0,
            f"feat_{prefix}_elements_sum": 0.0,
            f"feat_{prefix}_elements_max": 0.0,
            f"feat_{prefix}_bytes_sum": 0.0,
            f"feat_{prefix}_dynamic_dims": 0.0,
        }

    ranks = [float(entry["rank"]) for entry in entries]
    elems = [_entry_num_elements(entry) for entry in entries]
    bytes_list = [
        elem_count * DTYPE_SIZES.get(str(entry["dtype"]).lower(), 4)
        for entry, elem_count in zip(entries, elems)
    ]
    dynamic_dims = [float(entry["dynamic_dims"]) for entry in entries]
    return {
        f"feat_{prefix}_tensor_count": float(len(entries)),
        f"feat_{prefix}_rank_sum": float(sum(ranks)),
        f"feat_{prefix}_rank_max": float(max(ranks)),
        f"feat_{prefix}_elements_sum": float(sum(elems)),
        f"feat_{prefix}_elements_max": float(max(elems)),
        f"feat_{prefix}_bytes_sum": float(sum(bytes_list)),
        f"feat_{prefix}_dynamic_dims": float(sum(dynamic_dims)),
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_candidates = [
        "batch_size",
        "num_indices_per_lookup",
        "total_instructions",
        "total_loads",
        "total_stores",
        "load_store_ratio",
        "num_threads",
        "output_size",
        "activation_size",
        "parameter_size",
        "reuse_time_mean",
        "reuse_distance_mean",
        "reuse_distance_median",
        "reuse_distance_std",
        "reuse_distance_unique_cache_lines_per_k_accesses",
        "reuse_distance_instruction_accesses",
        "reuse_distance_data_accesses",
        "opc_branch_ratio",
        "opc_fp_convert_ratio",
        "opc_fp_load_simd_ratio",
        "opc_fp_math_ratio",
        "opc_fp_move_ratio",
        "opc_fp_store_simd_ratio",
        "opc_load_ratio",
        "opc_math_ratio",
        "opc_simd_ratio",
        "opc_store_ratio",
        "opc_fp_convert",
        "opc_fp_load_simd",
        "opc_fp_math",
        "opc_fp_move",
        "opc_fp_store_simd",
        "opc_math",
        "opc_simd",
        "dur_us",
    ]

    for column in out.columns:
        if column.startswith("reuse_time_bin_"):
            numeric_candidates.append(column)

    for column in sorted(set(numeric_candidates)):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    input_shape_rows = out["input_type_shape"].map(lambda value: _shape_features(value, "input"))
    output_shape_rows = out["output_type_shape"].map(lambda value: _shape_features(value, "output"))

    input_shape_df = pd.DataFrame(list(input_shape_rows))
    output_shape_df = pd.DataFrame(list(output_shape_rows))
    out = pd.concat([out, input_shape_df, output_shape_df], axis=1)

    out["feat_batch_lookup_product"] = (
        out["batch_size"].fillna(0) * out["num_indices_per_lookup"].fillna(0)
    )
    out["feat_memory_ops"] = out["total_loads"].fillna(0) + out["total_stores"].fillna(0)
    out["feat_working_set_bytes"] = (
        out["output_size"].fillna(0)
        + out["activation_size"].fillna(0)
        + out["parameter_size"].fillna(0)
    )
    out["feat_threads_effective"] = out["num_threads"].fillna(1).clip(lower=1)
    out["feat_insts_per_thread"] = out["total_instructions"].fillna(0) / out["feat_threads_effective"]
    out["feat_memops_per_inst"] = (
        out["feat_memory_ops"].fillna(0) / out["total_instructions"].clip(lower=1)
    )
    out["feat_output_to_working_set_ratio"] = (
        out["output_size"].fillna(0) / out["feat_working_set_bytes"].clip(lower=1)
    )
    out["feat_inst_per_working_set_byte"] = (
        out["total_instructions"].fillna(0) / out["feat_working_set_bytes"].clip(lower=1)
    )
    out["feat_output_input_elements_ratio"] = (
        out["feat_output_elements_sum"].fillna(0) / out["feat_input_elements_sum"].clip(lower=1)
    )

    return out


def add_operator_hardware_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    threads = pd.to_numeric(
        out["feat_threads_effective"] if "feat_threads_effective" in out.columns else out.get("num_threads", 1),
        errors="coerce",
    ).fillna(1.0).clip(lower=1.0)

    if "hw_core_total_cores" in out.columns:
        total_cores = pd.to_numeric(out["hw_core_total_cores"], errors="coerce").fillna(threads).clip(lower=1.0)
    else:
        total_cores = threads

    active_cores = np.minimum(threads.to_numpy(dtype=float), total_cores.to_numpy(dtype=float))
    out["hw_core_active_cores"] = active_cores
    out["hw_core_active_core_fraction"] = active_cores / total_cores.clip(lower=1.0)

    for cache_column, out_column in [
        ("hw_cache_l1i_size", "hw_cache_l1i_active_bytes"),
        ("hw_cache_l1d_size", "hw_cache_l1d_active_bytes"),
        ("hw_cache_l2_size", "hw_cache_l2_active_bytes"),
    ]:
        if cache_column in out.columns:
            cache_size = pd.to_numeric(out[cache_column], errors="coerce").fillna(0.0)
            out[out_column] = cache_size * active_cores

    if "hw_cache_l3_per_die_size" in out.columns:
        l3_size = pd.to_numeric(out["hw_cache_l3_per_die_size"], errors="coerce").fillna(0.0)
        if "hw_core_cores_per_die" in out.columns:
            cores_per_die = pd.to_numeric(out["hw_core_cores_per_die"], errors="coerce").fillna(total_cores).clip(lower=1.0)
            total_dies = np.ceil(total_cores.to_numpy(dtype=float) / cores_per_die.to_numpy(dtype=float))
            active_dies = np.ceil(active_cores / cores_per_die.to_numpy(dtype=float))
            active_dies = np.minimum(active_dies, total_dies)
        else:
            active_dies = np.ones(len(out), dtype=float)
        out["hw_cache_l3_active_dies"] = active_dies
        out["hw_cache_l3_active_bytes"] = l3_size * active_dies

    return out


def add_real_targets(df: pd.DataFrame, hw_clock_ghz: float | None) -> pd.DataFrame:
    out = df.copy()
    out["label_real_dur_us"] = pd.to_numeric(out["dur_us"], errors="coerce")
    if hw_clock_ghz is None or hw_clock_ghz <= 0:
        return out

    elapsed_cycles = out["label_real_dur_us"] * hw_clock_ghz * 1e3
    thread_scaled_cycles = elapsed_cycles * out["feat_threads_effective"].fillna(1)

    out["label_real_elapsed_cycles"] = elapsed_cycles
    out["label_real_thread_scaled_cycles"] = thread_scaled_cycles
    out["label_real_ipc_proxy"] = out["total_instructions"] / thread_scaled_cycles.clip(lower=1)
    out["label_real_ips"] = out["total_instructions"] / (
        out["label_real_dur_us"].clip(lower=1e-9) * 1e-6
    )
    return out


def feature_columns_for_training(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    exclude = {
        "input_type_shape",
        "output_type_shape",
        "trace_op_name",
        "feature_csv",
        "dur_us",
        target_column,
    }
    exclude.update({column for column in df.columns if column.startswith("label_") and column != target_column})

    categorical = [column for column in ["op_type", "node_scope"] if column in df.columns]
    numeric = []
    for column in df.columns:
        if column in exclude or column in categorical:
            continue
        if column in {"combo", "node_name"}:
            continue
        if column.startswith("feat_") or column.startswith("hw_"):
            numeric.append(column)
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
    return sorted(set(numeric)), categorical
