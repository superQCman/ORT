from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COMBO_RE = re.compile(r"bs(?P<batch>\d+)_nip(?P<nip>\d+)")
TRACE_NODE_RE = re.compile(r"_kernel_time$")
TRACE_OP_RE = re.compile(r"^(?P<op_idx>\d+)")

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


def normalize_node_name(node_name: str | float | int | None) -> str:
    text = "" if node_name is None else str(node_name).strip()
    if not text or text.lower() == "nan":
        return ""
    return TRACE_NODE_RE.sub("", text)


def extract_node_scope(node_name: str | float | int | None) -> str:
    text = normalize_node_name(node_name)
    if not text:
        return "unknown"
    stripped = text.strip("/")
    if not stripped:
        return "root"
    return stripped.split("/", 1)[0]


def split_combo(combo: str | None) -> tuple[int | None, int | None]:
    if not combo:
        return None, None
    match = COMBO_RE.search(str(combo))
    if not match:
        return None, None
    return int(match.group("batch")), int(match.group("nip"))


def parse_trace_op_idx(trace_op_name: str | float | int | None) -> int | None:
    if trace_op_name is None:
        return None
    match = TRACE_OP_RE.search(str(trace_op_name).strip())
    if not match:
        return None
    return int(match.group("op_idx"))


def parse_shape_dims(value: str | float | int | None) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []

    dims: list[int] = []
    for dim in parsed:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        elif isinstance(dim, float) and dim.is_integer() and dim > 0:
            dims.append(int(dim))
    return dims


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

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or not item:
            continue
        dtype, dims = next(iter(item.items()))
        if not isinstance(dims, list):
            continue
        clean_dims: list[int] = []
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
        product *= int(dim)
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


def _safe_ratio_series(numerator: pd.Series, denominator: pd.Series, floor: float = 1.0) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0).clip(lower=floor)
    return num / den


def _product(values: list[int]) -> float:
    if not values:
        return 0.0
    product = 1
    for value in values:
        product *= int(value)
    return float(product)


def _infer_gemm_mnk(
    input_entries: list[dict[str, Any]],
    output_entries: list[dict[str, Any]],
) -> tuple[float, float, float]:
    if not input_entries or not output_entries:
        return 0.0, 0.0, 0.0

    a_dims = [int(dim) for dim in input_entries[0].get("dims", [])]
    b_dims = [int(dim) for dim in input_entries[1].get("dims", [])] if len(input_entries) > 1 else []
    c_dims = [int(dim) for dim in output_entries[0].get("dims", [])]
    if not c_dims:
        return 0.0, 0.0, 0.0

    n_dim = float(c_dims[-1])
    m_dim = float(int(_product(c_dims[:-1])) if len(c_dims) > 1 else 1)
    k_dim = float(a_dims[-1]) if a_dims else 0.0

    if len(b_dims) >= 2:
        if float(b_dims[0]) == n_dim:
            k_dim = float(b_dims[1])
        elif float(b_dims[-1]) == n_dim:
            k_dim = float(b_dims[-2])

    return m_dim, n_dim, k_dim


def _infer_reduction_dims(input_dims: list[int], output_dims: list[int]) -> list[int]:
    if not input_dims or input_dims == output_dims:
        return []

    reduced: list[int] = []
    out_idx = 0
    for dim in input_dims:
        if out_idx < len(output_dims) and dim == output_dims[out_idx]:
            out_idx += 1
        else:
            reduced.append(int(dim))
    if out_idx == len(output_dims):
        return reduced

    return [int(dim) for idx, dim in enumerate(input_dims) if idx >= len(output_dims) or output_dims[idx] != dim]


def _infer_reduction_axis_positions(input_dims: list[int], output_dims: list[int]) -> list[int]:
    if not input_dims or input_dims == output_dims:
        return []

    positions: list[int] = []
    out_idx = 0
    for idx, dim in enumerate(input_dims):
        if out_idx < len(output_dims) and dim == output_dims[out_idx]:
            out_idx += 1
        else:
            positions.append(idx)
    if out_idx == len(output_dims):
        return positions

    return [idx for idx, dim in enumerate(input_dims) if idx >= len(output_dims) or output_dims[idx] != dim]


def serialize_shape_entries(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""

    work = rows.copy()
    work["port_idx"] = pd.to_numeric(work["port_idx"], errors="coerce")
    work = work.sort_values(["port_idx", "tensor_name"], kind="stable")
    entries: list[dict[str, list[Any]]] = []
    for _, row in work.iterrows():
        entries.append(
            {
                str(row.get("dtype", "")).strip(): parse_shape_dims(row.get("shape")),
            }
        )
    return repr(entries) if entries else ""


def load_op_shapes_for_combo(op_shapes_csv: Path) -> pd.DataFrame:
    if not op_shapes_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(op_shapes_csv)
    required = {"node_idx", "node_name", "op_type", "tensor_role", "port_idx", "dtype", "shape"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["node_idx", "node_name", "op_type"], dropna=False):
        node_idx, node_name, op_type = keys
        input_rows = group[group["tensor_role"].astype(str).str.strip().eq("input")]
        output_rows = group[group["tensor_role"].astype(str).str.strip().eq("output")]
        rows.append(
            {
                "op_idx": pd.to_numeric(node_idx, errors="coerce"),
                "node_name_normalized": normalize_node_name(node_name),
                "op_type": "" if pd.isna(op_type) else str(op_type),
                "input_type_shape_op_shapes": serialize_shape_entries(input_rows),
                "output_type_shape_op_shapes": serialize_shape_entries(output_rows),
            }
        )
    return pd.DataFrame(rows)


def merge_op_shapes_into_feature_rows(df: pd.DataFrame, op_shapes_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or op_shapes_df.empty:
        out = df.copy()
        out["input_type_shape"] = ""
        out["output_type_shape"] = ""
        return out

    out = df.copy()
    idx_rows = op_shapes_df.dropna(subset=["op_idx"]).copy()
    if not idx_rows.empty:
        idx_rows["op_idx"] = pd.to_numeric(idx_rows["op_idx"], errors="coerce").astype("Int64")
        out = out.merge(
            idx_rows[["op_idx", "op_type", "input_type_shape_op_shapes", "output_type_shape_op_shapes"]],
            on=["op_idx", "op_type"],
            how="left",
        )
    else:
        out["input_type_shape_op_shapes"] = ""
        out["output_type_shape_op_shapes"] = ""

    name_rows = op_shapes_df[
        ["node_name_normalized", "op_type", "input_type_shape_op_shapes", "output_type_shape_op_shapes"]
    ].drop_duplicates(subset=["node_name_normalized", "op_type"], keep="first")
    out = out.merge(
        name_rows.rename(
            columns={
                "input_type_shape_op_shapes": "input_type_shape_op_shapes_by_name",
                "output_type_shape_op_shapes": "output_type_shape_op_shapes_by_name",
            }
        ),
        on=["node_name_normalized", "op_type"],
        how="left",
    )

    for target in ["input_type_shape", "output_type_shape"]:
        idx_col = f"{target}_op_shapes"
        name_col = f"{target}_op_shapes_by_name"
        existing = out[target].fillna("").astype(str) if target in out.columns else pd.Series("", index=out.index)
        existing = existing.where(~existing.str.lower().eq("nan"), "")
        idx_values = out[idx_col].fillna("").astype(str)
        name_values = out[name_col].fillna("").astype(str)
        merged = existing.where(existing.str.len() > 0, idx_values)
        out[target] = merged.where(merged.str.len() > 0, name_values)

    drop_columns = [
        "input_type_shape_op_shapes",
        "output_type_shape_op_shapes",
        "input_type_shape_op_shapes_by_name",
        "output_type_shape_op_shapes_by_name",
    ]
    return out.drop(columns=[column for column in drop_columns if column in out.columns])


def ensure_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "batch_size": 0,
        "num_indices_per_lookup": 0,
        "total_instructions": 0,
        "total_loads": 0,
        "total_stores": 0,
        "num_threads": 1,
        "dur_us": np.nan,
        "output_size": 0.0,
        "activation_size": 0.0,
        "parameter_size": 0.0,
    }
    source_map = {
        "dur_us": "cpu_dur_us_avg",
        "output_size": "cpu_output_size_avg",
        "activation_size": "cpu_activation_size_avg",
        "parameter_size": "cpu_parameter_size_avg",
    }
    for target, source in source_map.items():
        if target not in out.columns or out[target].isna().all():
            if source in out.columns:
                out[target] = out[source]
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    numeric_columns = [
        "batch_size",
        "num_indices_per_lookup",
        "total_instructions",
        "total_loads",
        "total_stores",
        "num_threads",
        "dur_us",
        "output_size",
        "activation_size",
        "parameter_size",
        "reuse_time_mean",
        "reuse_distance_mean",
        "reuse_distance_unique_cache_lines_per_k_accesses",
        "opc_branch_ratio",
        "opc_fp_math_ratio",
        "opc_load_ratio",
        "opc_math_ratio",
        "opc_simd_ratio",
        "opc_store_ratio",
        "cpu_dur_us_avg",
        "cpu_output_size_avg",
        "cpu_activation_size_avg",
        "cpu_parameter_size_avg",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_runtime_columns(df)

    input_shape_rows = out["input_type_shape"].map(lambda value: _shape_features(value, "input"))
    output_shape_rows = out["output_type_shape"].map(lambda value: _shape_features(value, "output"))
    input_shape_entries = out["input_type_shape"].map(_shape_entries)
    output_shape_entries = out["output_type_shape"].map(_shape_entries)

    input_shape_df = pd.DataFrame(list(input_shape_rows))
    output_shape_df = pd.DataFrame(list(output_shape_rows))
    out = pd.concat([out.reset_index(drop=True), input_shape_df, output_shape_df], axis=1)

    batch_size = out["batch_size"].fillna(0).clip(lower=0)
    batch_size_safe = batch_size.clip(lower=1)
    output_bytes = out["output_size"].fillna(0).clip(lower=0)
    activation_bytes = out["activation_size"].fillna(0).clip(lower=0)
    parameter_bytes = out["parameter_size"].fillna(0).clip(lower=0)
    op_type = out["op_type"].fillna("").astype(str)
    gemm_mask = op_type.eq("Gemm")
    reduction_mask = op_type.eq("ReduceSum")
    gather_mask = op_type.eq("Gather")

    output_elements = output_bytes / 4.0
    activation_elements = activation_bytes / 4.0
    parameter_elements = parameter_bytes / 4.0
    output_elements_per_batch = output_elements / batch_size_safe
    activation_elements_per_batch = activation_elements / batch_size_safe
    lookup_count = batch_size * out["num_indices_per_lookup"].fillna(0).clip(lower=0)
    lookup_count_safe = lookup_count.clip(lower=1)
    io_bytes = output_bytes + activation_bytes + parameter_bytes

    gemm_shape_rows = [
        _infer_gemm_mnk(inp, outp) for inp, outp in zip(input_shape_entries, output_shape_entries)
    ]
    gemm_shape_df = pd.DataFrame(gemm_shape_rows, columns=["feat_gemm_m", "feat_gemm_n", "feat_gemm_k"]).fillna(0.0)

    reduction_rows: list[dict[str, float]] = []
    for inp, outp in zip(input_shape_entries, output_shape_entries):
        input_dims = [int(dim) for dim in inp[0].get("dims", [])] if inp else []
        output_dims = [int(dim) for dim in outp[0].get("dims", [])] if outp else []
        reduced_dims = _infer_reduction_dims(input_dims, output_dims)
        reduced_positions = _infer_reduction_axis_positions(input_dims, output_dims)
        trailing_positions: list[int] = []
        reduced_position_set = set(reduced_positions)
        for idx in range(len(input_dims) - 1, -1, -1):
            if idx in reduced_position_set:
                trailing_positions.append(idx)
            else:
                break
        reduction_rows.append(
            {
                "feat_reduction_axes_count": float(len(reduced_dims)),
                "feat_reduction_axes_product": _product(reduced_dims) if reduced_dims else 0.0,
                "feat_reduction_input_rank": float(len(input_dims)),
                "feat_reduction_output_rank": float(len(output_dims)),
                "feat_reduction_trailing_fraction": (
                    float(len(trailing_positions)) / float(len(reduced_positions))
                    if reduced_positions
                    else 0.0
                ),
            }
        )
    reduction_df = pd.DataFrame(reduction_rows).fillna(0.0)
    shape_features_df = pd.concat([gemm_shape_df, reduction_df], axis=1)

    for column in ["feat_gemm_m", "feat_gemm_n", "feat_gemm_k"]:
        shape_features_df[column] = np.where(gemm_mask, shape_features_df[column].fillna(0.0), 0.0)
    for column in [
        "feat_reduction_axes_count",
        "feat_reduction_axes_product",
        "feat_reduction_input_rank",
        "feat_reduction_output_rank",
        "feat_reduction_trailing_fraction",
    ]:
        shape_features_df[column] = np.where(reduction_mask, shape_features_df[column].fillna(0.0), 0.0)

    out = pd.concat([out.reset_index(drop=True), shape_features_df.reset_index(drop=True)], axis=1)

    gemm_mac_count = out["feat_gemm_m"].fillna(0.0) * out["feat_gemm_n"].fillna(0.0) * out["feat_gemm_k"].fillna(0.0)
    gemm_mac_count = gemm_mac_count.where(gemm_mac_count > 0.0, output_elements * activation_elements_per_batch)

    reduction_factor = _safe_ratio_series(activation_elements, output_elements)
    reduction_factor = reduction_factor.where(
        out["feat_reduction_axes_product"].fillna(0.0).le(0.0),
        out["feat_reduction_axes_product"],
    )
    reduction_work_items = (activation_elements - output_elements).clip(lower=0.0)
    reduction_work_items = reduction_work_items.where(
        out["feat_reduction_axes_product"].fillna(0.0).le(0.0),
        output_elements * np.clip(out["feat_reduction_axes_product"].fillna(0.0) - 1.0, a_min=0.0, a_max=None),
    )

    feat_threads_effective = out["num_threads"].fillna(1).clip(lower=1)
    feat_memory_ops = out["total_loads"].fillna(0) + out["total_stores"].fillna(0)
    feat_working_set_bytes = output_bytes + activation_bytes + parameter_bytes
    engineered_df = pd.DataFrame(
        {
            "feat_memory_ops": feat_memory_ops,
            "feat_working_set_bytes": feat_working_set_bytes,
            "feat_threads_effective": feat_threads_effective,
            "feat_insts_per_thread": out["total_instructions"].fillna(0) / feat_threads_effective,
            "feat_memops_per_inst": feat_memory_ops.fillna(0) / out["total_instructions"].clip(lower=1),
            "feat_output_elements": output_elements,
            "feat_activation_elements": activation_elements,
            "feat_parameter_elements": parameter_elements,
            "feat_lookup_count": np.where(gather_mask, lookup_count, 0.0),
            "feat_output_elements_per_lookup": np.where(
                gather_mask,
                _safe_ratio_series(output_elements, lookup_count_safe),
                0.0,
            ),
            "feat_output_elements_per_batch": output_elements_per_batch,
            "feat_activation_elements_per_batch": activation_elements_per_batch,
            "feat_activation_per_output_element": _safe_ratio_series(activation_elements, output_elements),
            "feat_gemm_mac_count": np.where(gemm_mask, gemm_mac_count, 0.0),
            "feat_gemm_bytes_per_mac": np.where(
                gemm_mask,
                _safe_ratio_series(io_bytes, gemm_mac_count),
                0.0,
            ),
            "feat_reduction_factor": np.where(reduction_mask, reduction_factor, 0.0),
            "feat_reduction_work_items": np.where(reduction_mask, reduction_work_items, 0.0),
            "feat_io_bytes_sum": io_bytes,
            "feat_output_input_bytes_ratio": _safe_ratio_series(
                out["feat_output_bytes_sum"].fillna(output_bytes),
                out["feat_input_bytes_sum"].fillna(0.0),
            ),
        },
        index=out.index,
    )
    out = pd.concat([out.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1)
    return out

