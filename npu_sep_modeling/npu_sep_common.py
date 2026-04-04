from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ORT_ROOT = PROJECT_DIR.parent
CASE_ID = "case_10_4_4_cann"
FEATURE_ROOT = ORT_ROOT / f"features_extensible_{CASE_ID}"
PROFILE_ROOT = ORT_ROOT / f"sweep_runs_extensible_{CASE_ID}" / "onnx_profiles"
PROFILE_GLOB = "ort_cann_profile_*.json"

DEFAULT_CUBE_PEAK_EFF_GFLOPS = 20000.0
DEFAULT_VECTOR_PEAK_EFF_GFLOPS = 8000.0
DEFAULT_MEMORY_BW_GBPS = 50.0
DEFAULT_TRANSFER_BW_GBPS = 50.0


DTYPE_SIZES: dict[str, int] = {
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

LANE_BY_OP: dict[str, str] = {
    "MatMul": "cube",
    "Transpose": "vector",
    "Add": "vector",
    "Relu": "vector",
    "MemcpyFromHost": "transfer",
    "MemcpyToHost": "transfer",
}

TRANSFER_DIRECTION_BY_OP: dict[str, str] = {
    "MemcpyFromHost": "h2d",
    "MemcpyToHost": "d2h",
}

QUEUE_PROXY_FIELDS: tuple[str, ...] = (
    "cpu_main_Wait_avg",
    "cpu_main_DistributionEnqueue_avg",
)


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")


def load_hardware_profile(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in hardware profile: {path}")
    return payload


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return default
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def hardware_value(profile: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not profile:
        return default
    value = profile.get(key, default)
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def parse_combo_name(combo: str) -> tuple[int, int]:
    match = re.fullmatch(r"bs(?P<bs>\d+)_nip(?P<nip>\d+)", combo.strip())
    if match is None:
        raise ValueError(f"Unsupported combo name: {combo!r}")
    return int(match.group("bs")), int(match.group("nip"))


def combo_sort_key(combo: str) -> tuple[int, int]:
    return parse_combo_name(combo)


def combo_to_paths(combo: str) -> tuple[Path, Path]:
    feature_csv = FEATURE_ROOT / f"{combo}.csv"
    profile_dir = PROFILE_ROOT / combo
    return feature_csv, profile_dir


def list_profile_combos(profile_root: Path = PROFILE_ROOT) -> list[str]:
    combos: list[str] = []
    if not profile_root.exists():
        return combos
    for combo_dir in sorted(profile_root.iterdir(), key=lambda p: combo_sort_key(p.name) if p.is_dir() else (10**9, 10**9)):
        if not combo_dir.is_dir():
            continue
        if list(combo_dir.glob(PROFILE_GLOB)):
            combos.append(combo_dir.name)
    return combos


def load_feature_csv(combo: str, feature_root: Path = FEATURE_ROOT) -> pd.DataFrame:
    path = feature_root / f"{combo}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature CSV for combo {combo}: {path}")
    df = pd.read_csv(path)
    if "node_idx" not in df.columns:
        raise ValueError(f"Feature CSV is missing node_idx column: {path}")
    df = df.copy()
    df["combo"] = combo
    df["source_csv"] = str(path)
    df["node_idx"] = pd.to_numeric(df["node_idx"], errors="coerce").astype("Int64")
    return df


def load_latest_profile_json(combo: str, profile_root: Path = PROFILE_ROOT) -> Path:
    profile_dir = profile_root / combo
    if not profile_dir.exists():
        raise FileNotFoundError(f"Missing profile directory for combo {combo}: {profile_dir}")
    candidates = sorted(profile_dir.glob(PROFILE_GLOB))
    if not candidates:
        raise FileNotFoundError(f"No profile JSON found for combo {combo}: {profile_dir}")
    return candidates[-1]


def tensor_entries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, dict):
        parsed = [value]
    else:
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
        for dim in dims:
            if isinstance(dim, bool):
                continue
            if isinstance(dim, int) and dim > 0:
                clean_dims.append(dim)
            elif isinstance(dim, float) and dim.is_integer() and dim > 0:
                clean_dims.append(int(dim))
        out.append({"dtype": str(dtype), "dims": clean_dims})
    return out


def tensor_numel(entry: dict[str, Any]) -> int:
    dims = entry.get("dims", [])
    if not dims:
        return 0
    product = 1
    for dim in dims:
        product *= int(dim)
    return int(product)


def tensor_bytes(entry: dict[str, Any]) -> int:
    dtype = str(entry.get("dtype", "float")).lower()
    size = DTYPE_SIZES.get(dtype, 4)
    return tensor_numel(entry) * size


def entries_total_bytes(entries: Iterable[dict[str, Any]]) -> int:
    return int(sum(tensor_bytes(entry) for entry in entries))


def entries_total_numel(entries: Iterable[dict[str, Any]]) -> int:
    return int(sum(tensor_numel(entry) for entry in entries))


def shape_entry_signature(entry: dict[str, Any]) -> str:
    dims = entry.get("dims", [])
    dims_text = "x".join(str(dim) for dim in dims) if dims else "?"
    return f"{str(entry.get('dtype', 'unknown')).lower()}:{dims_text}"


def shape_signature(input_entries: Iterable[dict[str, Any]], output_entries: Iterable[dict[str, Any]]) -> str:
    input_text = "|".join(shape_entry_signature(entry) for entry in input_entries) or "none"
    output_text = "|".join(shape_entry_signature(entry) for entry in output_entries) or "none"
    return f"in={input_text};out={output_text}"


def event_name_without_kernel_suffix(name: str | None) -> str:
    text = (name or "").strip()
    if text.endswith("_kernel_time"):
        text = text[: -len("_kernel_time")]
    return text


def infer_transfer_direction(op_name: str) -> str:
    return TRANSFER_DIRECTION_BY_OP.get(op_name, "")


def queue_proxy_components(row: dict[str, Any] | pd.Series) -> dict[str, float]:
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)

    wait_us = float(safe_float(row_dict.get("cpu_main_Wait_avg"), 0.0) or 0.0) / 1000.0
    enqueue_us = float(safe_float(row_dict.get("cpu_main_DistributionEnqueue_avg"), 0.0) or 0.0) / 1000.0
    wait_us = max(wait_us, 0.0)
    enqueue_us = max(enqueue_us, 0.0)
    return {
        "queue_wait_proxy_us": wait_us,
        "queue_enqueue_proxy_us": enqueue_us,
        "queue_proxy_us": wait_us + enqueue_us,
    }


def lane_baseline_components(row: dict[str, Any] | pd.Series, hardware_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)

    lane = str(row_dict.get("npu_lane") or infer_lane(str(row_dict.get("op_name") or "")))
    op_name = str(row_dict.get("op_name") or "")
    input_bytes = float(safe_float(row_dict.get("input_bytes"), 0.0) or 0.0)
    output_bytes = float(safe_float(row_dict.get("output_bytes"), 0.0) or 0.0)
    activation_bytes = float(safe_float(row_dict.get("activation_bytes"), 0.0) or 0.0)
    parameter_bytes = float(safe_float(row_dict.get("parameter_bytes"), 0.0) or 0.0)

    cube_peak = float(
        safe_float(
            hardware_value(hardware_profile, "cube_peak_eff_gflops", DEFAULT_CUBE_PEAK_EFF_GFLOPS),
            DEFAULT_CUBE_PEAK_EFF_GFLOPS,
        )
        or DEFAULT_CUBE_PEAK_EFF_GFLOPS
    )
    vector_peak = float(
        safe_float(
            hardware_value(hardware_profile, "vector_peak_eff_gflops", DEFAULT_VECTOR_PEAK_EFF_GFLOPS),
            DEFAULT_VECTOR_PEAK_EFF_GFLOPS,
        )
        or DEFAULT_VECTOR_PEAK_EFF_GFLOPS
    )
    memory_bw = float(
        safe_float(
            hardware_value(hardware_profile, "memory_bw_gbps", DEFAULT_MEMORY_BW_GBPS),
            DEFAULT_MEMORY_BW_GBPS,
        )
        or DEFAULT_MEMORY_BW_GBPS
    )
    h2d_bw = float(
        safe_float(
            hardware_value(hardware_profile, "h2d_bw_gbps", DEFAULT_TRANSFER_BW_GBPS),
            DEFAULT_TRANSFER_BW_GBPS,
        )
        or DEFAULT_TRANSFER_BW_GBPS
    )
    d2h_bw = float(
        safe_float(
            hardware_value(hardware_profile, "d2h_bw_gbps", DEFAULT_TRANSFER_BW_GBPS),
            DEFAULT_TRANSFER_BW_GBPS,
        )
        or DEFAULT_TRANSFER_BW_GBPS
    )

    if lane == "cube":
        m = float(safe_float(row_dict.get("matmul_m"), 0.0) or 0.0)
        k = float(safe_float(row_dict.get("matmul_k"), 0.0) or 0.0)
        n = float(safe_float(row_dict.get("matmul_n"), 0.0) or 0.0)
        flops = 2.0 * m * k * n
        compute_us = flops / max(cube_peak, 1e-9) / 1000.0
        data_bytes = input_bytes + output_bytes + activation_bytes + parameter_bytes
        memory_us = data_bytes / max(memory_bw, 1e-9) / 1000.0
        baseline_us = max(compute_us, memory_us)
        return {
            "lane": lane,
            "op_name": op_name,
            "baseline_us": baseline_us,
            "compute_us": compute_us,
            "memory_us": memory_us,
            "data_bytes": data_bytes,
            "peak_gflops": cube_peak,
            "bw_gbps": memory_bw,
            "formula": "cube_roofline",
        }

    if lane == "vector":
        elem_count = float(safe_float(row_dict.get("vector_elem_count"), 0.0) or 0.0)
        if elem_count <= 0.0:
            elem_count = max(output_bytes / 4.0, input_bytes / 4.0, 0.0)
        compute_us = elem_count / max(vector_peak, 1e-9) / 1000.0
        data_bytes = input_bytes + output_bytes + activation_bytes
        memory_us = data_bytes / max(memory_bw, 1e-9) / 1000.0
        baseline_us = max(compute_us, memory_us)
        return {
            "lane": lane,
            "op_name": op_name,
            "baseline_us": baseline_us,
            "compute_us": compute_us,
            "memory_us": memory_us,
            "data_bytes": data_bytes,
            "peak_gflops": vector_peak,
            "bw_gbps": memory_bw,
            "formula": "vector_roofline",
        }

    if lane == "transfer":
        transfer_bytes = max(input_bytes, output_bytes, activation_bytes, parameter_bytes)
        bw = h2d_bw if infer_transfer_direction(op_name) == "h2d" else d2h_bw
        baseline_us = transfer_bytes / max(bw, 1e-9) / 1000.0
        return {
            "lane": lane,
            "op_name": op_name,
            "baseline_us": baseline_us,
            "compute_us": 0.0,
            "memory_us": baseline_us,
            "data_bytes": transfer_bytes,
            "peak_gflops": None,
            "bw_gbps": bw,
            "formula": "transfer_roofline",
        }

    data_bytes = input_bytes + output_bytes + activation_bytes + parameter_bytes
    baseline_us = data_bytes / max(memory_bw, 1e-9) / 1000.0
    return {
        "lane": lane,
        "op_name": op_name,
        "baseline_us": baseline_us,
        "compute_us": 0.0,
        "memory_us": baseline_us,
        "data_bytes": data_bytes,
        "peak_gflops": None,
        "bw_gbps": memory_bw,
        "formula": "fallback_bytes_roofline",
    }


def baseline_prediction(row: dict[str, Any] | pd.Series, hardware_profile: dict[str, Any] | None = None) -> float:
    return float(lane_baseline_components(row, hardware_profile)["baseline_us"])


def infer_lane(op_name: str) -> str:
    return LANE_BY_OP.get(op_name, "unknown")


def infer_matmul_mkn(input_entries: list[dict[str, Any]], output_entries: list[dict[str, Any]]) -> tuple[int | None, int | None, int | None]:
    if len(input_entries) < 2:
        return None, None, None
    a_dims = [int(dim) for dim in input_entries[0].get("dims", [])]
    b_dims = [int(dim) for dim in input_entries[1].get("dims", [])]
    c_dims = [int(dim) for dim in output_entries[0].get("dims", [])] if output_entries else []
    if not a_dims:
        return None, None, None
    m = int(np.prod(a_dims[:-1])) if len(a_dims) > 1 else 1
    k = int(a_dims[-1])
    if c_dims:
        n = int(c_dims[-1])
    elif b_dims:
        n = int(b_dims[-1])
    else:
        n = None
    if len(b_dims) >= 2 and b_dims[0] == k:
        n = int(b_dims[1])
    elif len(b_dims) >= 2 and b_dims[-1] == n:
        k = int(b_dims[-2])
    return m, k, n


def vector_elem_count(output_entries: list[dict[str, Any]], input_entries: list[dict[str, Any]]) -> int:
    if output_entries:
        return entries_total_numel(output_entries)
    return entries_total_numel(input_entries)


def extract_event_measurement(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args", {}) or {}
    input_entries = tensor_entries(args.get("input_type_shape"))
    output_entries = tensor_entries(args.get("output_type_shape"))
    input_bytes = entries_total_bytes(input_entries)
    output_bytes = entries_total_bytes(output_entries)
    activation_bytes = safe_int(args.get("activation_size"), 0) or 0
    parameter_bytes = safe_int(args.get("parameter_size"), 0) or 0
    op_name = str(args.get("op_name") or "").strip()
    node_index = safe_int(args.get("node_index"))
    return {
        "node_index": node_index,
        "op_name": op_name,
        "event_name": event_name_without_kernel_suffix(event.get("name")),
        "raw_event_name": str(event.get("name") or ""),
        "duration_us": float(safe_float(event.get("dur"), 0.0) or 0.0),
        "ts": int(safe_int(event.get("ts"), 0) or 0),
        "tid": int(safe_int(event.get("tid"), 0) or 0),
        "pid": int(safe_int(event.get("pid"), 0) or 0),
        "provider": str(args.get("provider") or ""),
        "input_entries": input_entries,
        "output_entries": output_entries,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "activation_bytes": activation_bytes,
        "parameter_bytes": parameter_bytes,
        "shape_signature": shape_signature(input_entries, output_entries),
        "matmul_m": None,
        "matmul_k": None,
        "matmul_n": None,
        "vector_elem_count": None,
        "transfer_direction": infer_transfer_direction(op_name),
    }


def enrich_event_measurement(measurement: dict[str, Any]) -> dict[str, Any]:
    op_name = measurement.get("op_name", "")
    input_entries = measurement.get("input_entries", [])
    output_entries = measurement.get("output_entries", [])
    if op_name == "MatMul":
        m, k, n = infer_matmul_mkn(input_entries, output_entries)
        measurement["matmul_m"] = m
        measurement["matmul_k"] = k
        measurement["matmul_n"] = n
    elif infer_lane(op_name) == "vector":
        measurement["vector_elem_count"] = vector_elem_count(output_entries, input_entries)
    return measurement


def aggregate_duration_stats(durations: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(durations), dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def split_items_by_ratio(
    items: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if not items:
        return {"train": [], "val": [], "test": []}

    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    ratios = [value / ratio_sum for value in ratios]

    ordered = list(items)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    raw = [len(ordered) * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = len(ordered) - sum(counts)
    fractions = sorted(
        ((raw[idx] - counts[idx], idx) for idx in range(len(counts))),
        reverse=True,
    )
    for _, idx in fractions[:remainder]:
        counts[idx] += 1

    # Keep the split sizes stable even when rounding gives an empty bucket.
    if len(ordered) >= 3:
        for idx in range(len(counts)):
            if counts[idx] == 0:
                donor = max(range(len(counts)), key=lambda j: counts[j])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    train_count, val_count, test_count = counts
    train_items = ordered[:train_count]
    val_items = ordered[train_count : train_count + val_count]
    test_items = ordered[train_count + val_count : train_count + val_count + test_count]
    return {"train": train_items, "val": val_items, "test": test_items}


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size == 0:
        return {"mae": float("nan"), "mape": float("nan"), "rmse": float("nan")}
    abs_err = np.abs(true - pred)
    denom = np.clip(np.abs(true), 1e-9, None)
    return {
        "mae": float(abs_err.mean()),
        "mape": float((abs_err / denom).mean() * 100.0),
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


def fit_scale_bias(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.size == 0:
        return 1.0, 0.0
    if x_arr.size == 1 or np.allclose(x_arr, x_arr[0]):
        return 1.0, float(y_arr.mean() - x_arr.mean())
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    scale = float(coef[0])
    bias = float(coef[1])
    return scale, bias


def apply_scale_bias(x: Iterable[float], scale: float, bias: float) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float)
    return x_arr * float(scale) + float(bias)


def group_metrics_frame(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    group_col: str,
    op_col: str = "op_name",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group_df in df.groupby(group_col, dropna=False):
        metrics = compute_regression_metrics(group_df[label_col], group_df[pred_col])
        row: dict[str, Any] = {"group": group_name, **metrics, "count": int(len(group_df))}
        if op_col in group_df.columns:
            row["op_types"] = sorted({str(value) for value in group_df[op_col].dropna().unique()})
        rows.append(row)
    return rows
