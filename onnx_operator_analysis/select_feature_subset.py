#!/usr/bin/env python3
"""
Select a compact feature subset from merged ORT training feature CSVs.

Examples:
  python ORT/onnx_operator_analysis/select_feature_subset.py \
      --input ORT/features/bs32_nip1000.csv \
      --output /tmp/bs32_nip1000_selected.csv

  python ORT/onnx_operator_analysis/select_feature_subset.py \
      --input ORT/features \
      --output ORT/features_selected
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REUSE_TIME_BIN_PATTERN = re.compile(r"^reuse_time_bin_(\d+)_pct$")

FIXED_BEFORE_BINS: List[Tuple[str, str]] = [
    ("batch_size", "batch_size"),
    ("num_indices_per_lookup", "num_indices_per_lookup"),
    ("node_name", "node_name"),
    ("op_type", "op_type"),
    ("input_type_shape", "input_type_shape"),
    ("output_type_shape", "output_type_shape"),
    ("trace_op_name", "trace_op_name"),
    ("total_instructions", "total_instructions"),
    ("total_loads", "total_loads"),
    ("total_stores", "total_stores"),
    ("load_store_ratio", "load_store_ratio"),
    ("num_threads", "num_threads"),
    ("output_size", "cpu_output_size_avg"),
    ("activation_size", "cpu_activation_size_avg"),
    ("parameter_size", "cpu_parameter_size_avg"),
    ("reuse_time_mean", "reuse_time_mean"),
]

FIXED_AFTER_BINS: List[Tuple[str, str]] = [
    ("reuse_distance_mean", "reuse_distance_mean"),
    ("reuse_distance_median", "reuse_distance_median"),
    ("reuse_distance_std", "reuse_distance_std"),
    (
        "reuse_distance_unique_cache_lines_per_k_accesses",
        "reuse_distance_unique_cache_lines_per_k_accesses",
    ),
    ("reuse_distance_instruction_accesses", "reuse_distance_instruction_accesses"),
    ("reuse_distance_data_accesses", "reuse_distance_data_accesses"),
    ("opc_branch_ratio", "opc_branch_ratio"),
    ("opc_fp_convert", "opc_fp_convert"),
    ("opc_fp_load_simd", "opc_fp_load_simd"),
    ("opc_fp_math", "opc_fp_math"),
    ("opc_fp_move", "opc_fp_move"),
    ("opc_fp_store_simd", "opc_fp_store_simd"),
    ("opc_math", "opc_math"),
    ("opc_simd", "opc_simd"),
]

DUR_SOURCE_BY_MODE = {
    "avg": "cpu_dur_us_avg",
    "min": "cpu_dur_us_min",
    "max": "cpu_dur_us_max",
    "sum": "cpu_dur_us_sum",
}

SHAPE_FIELDS = ("input_type_shape", "output_type_shape")
SIZE_FIELDS = ("output_size", "activation_size", "parameter_size")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compact feature subset from merged ORT feature CSVs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input feature CSV file or directory containing bs*_nip*.csv files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file (for single-file input) or output directory (for directory input).",
    )
    parser.add_argument(
        "--pattern",
        default="bs*_nip*.csv",
        help="Glob used when --input is a directory. Default: %(default)s",
    )
    parser.add_argument(
        "--dur-source",
        choices=sorted(DUR_SOURCE_BY_MODE),
        default="avg",
        help="Which aggregated CPU duration column to expose as dur_us. Default: %(default)s",
    )
    return parser.parse_args()


def read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def find_reuse_time_bin_columns(fieldnames: Iterable[str]) -> List[str]:
    matched = []
    for name in fieldnames:
        match = REUSE_TIME_BIN_PATTERN.match(name)
        if match:
            matched.append((int(match.group(1)), name))
    matched.sort()
    return [name for _, name in matched]


def build_column_plan(fieldnames: Sequence[str], dur_source: str) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    plan.extend(FIXED_BEFORE_BINS)
    for bin_name in find_reuse_time_bin_columns(fieldnames):
        plan.append((bin_name, bin_name))
    plan.extend(FIXED_AFTER_BINS)
    plan.append(("dur_us", DUR_SOURCE_BY_MODE[dur_source]))
    return plan


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def choose_profile_roots(input_path: Path) -> List[Path]:
    ort_root = get_repo_root() / "ORT"
    regular = ort_root / "sweep_runs" / "onnx_profiles"
    extensible = ort_root / "sweep_runs_extensible" / "onnx_profiles"

    if "features_extensible" in input_path.parts:
        return [extensible, regular]
    return [regular, extensible]


def find_aux_detail_csv(input_path: Path) -> Path | None:
    combo = input_path.stem
    for root in choose_profile_roots(input_path):
        combo_dir = root / combo
        if not combo_dir.is_dir():
            continue
        aligned = sorted(combo_dir.glob("*_cpu_thread_detail_aligned.csv"))
        if aligned:
            return aligned[-1]
        detail = sorted(combo_dir.glob("*_cpu_thread_detail.csv"))
        if detail:
            return detail[-1]
    return None


def collect_distinct_values(rows: Iterable[Dict[str, str]], field: str) -> List[str]:
    values: List[str] = []
    seen = set()
    for row in rows:
        value = str(row.get(field, "")).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def build_aux_feature_lookup(input_path: Path) -> Tuple[Dict[str, Dict[str, str]], Path | None]:
    detail_path = find_aux_detail_csv(input_path)
    if detail_path is None:
        return {}, None

    with detail_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        node_idx = str(row.get("node_index", "")).strip()
        if node_idx:
            grouped_rows[node_idx].append(row)

    lookup: Dict[str, Dict[str, str]] = {}
    for node_idx, node_rows in grouped_rows.items():
        out: Dict[str, str] = {}
        for field in SHAPE_FIELDS:
            values = collect_distinct_values(node_rows, field)
            if values:
                out[field] = values[0] if len(values) == 1 else " || ".join(values)

        for field in SIZE_FIELDS:
            numeric_values = [
                value
                for value in (parse_float(row.get(field, "")) for row in node_rows)
                if value is not None
            ]
            if numeric_values:
                out[field] = format_number(sum(numeric_values) / len(numeric_values))

        if out:
            lookup[node_idx] = out

    return lookup, detail_path


def resolve_selected_value(
    row: Dict[str, str],
    aux_row: Dict[str, str],
    output_name: str,
    source_name: str,
) -> str:
    direct_value = str(row.get(source_name, "")).strip()
    if direct_value:
        return direct_value

    if source_name != output_name:
        fallback_direct = str(row.get(output_name, "")).strip()
        if fallback_direct:
            return fallback_direct

    return str(aux_row.get(output_name, "")).strip()


def write_selected_csv(input_path: Path, output_path: Path, dur_source: str) -> Tuple[int, List[str]]:
    fieldnames = read_header(input_path)
    if not fieldnames:
        raise ValueError(f"Input CSV is empty: {input_path}")

    column_plan = build_column_plan(fieldnames, dur_source)
    output_fieldnames = [output_name for output_name, _ in column_plan]
    aux_lookup, aux_path = build_aux_feature_lookup(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=output_fieldnames)
        writer.writeheader()

        for row in reader:
            node_idx = str(row.get("node_idx", "")).strip()
            aux_row = aux_lookup.get(node_idx, {})
            selected_row = {
                output_name: resolve_selected_value(row, aux_row, output_name, source_name)
                for output_name, source_name in column_plan
            }
            writer.writerow(selected_row)
            row_count += 1

    if aux_path is not None:
        print(f"  aux_profile={aux_path}")
    else:
        print("  aux_profile=not_found")

    return row_count, output_fieldnames


def resolve_input_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob(pattern) if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_output_path(input_root: Path, output_root: Path, input_file: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / input_file.name


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_files = resolve_input_files(input_path, args.pattern)
    if not input_files:
        raise FileNotFoundError(f"No input CSV files matched under {input_path} with pattern {args.pattern!r}")

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(input_files)} input file(s)")
    for input_file in input_files:
        target_path = resolve_output_path(input_path, output_path, input_file)
        row_count, output_fieldnames = write_selected_csv(input_file, target_path, args.dur_source)
        print(f"{input_file} -> {target_path}")
        print(f"  rows={row_count}, columns={len(output_fieldnames)}")


if __name__ == "__main__":
    main()
