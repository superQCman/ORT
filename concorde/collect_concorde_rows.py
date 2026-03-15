#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_META_COLUMNS = [
    "hardware_name",
    "combo",
    "op_name",
    "op_idx",
    "trace_dir",
    "view_log",
    "config_path",
    "shared_llc",
    "target_tid",
    "instruction_count",
    "window_size",
]


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def ordered_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    first_row_order = list(rows[0].keys())
    ordered = [name for name in DEFAULT_META_COLUMNS if name in first_row_order]
    ordered.extend(name for name in first_row_order if name not in ordered)

    for row in rows[1:]:
        for name in row.keys():
            if name not in ordered:
                ordered.append(name)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-op Concorde CSV rows into a single merge-ready CSV.")
    parser.add_argument(
        "--artifacts-root",
        default="/data/qc/dlrm/ORT/concorde/artifacts",
        help="Root directory containing per-op Concorde artifact folders",
    )
    parser.add_argument(
        "--row-file",
        default="performance_distribution_row.csv",
        help="Row filename to collect from each per-op artifact directory",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the merged output CSV",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    row_paths = sorted(artifacts_root.rglob(args.row_file))
    if not row_paths:
        raise FileNotFoundError(f"No {args.row_file} files found under {artifacts_root}")

    rows = [read_single_row(path) for path in row_paths]
    rows.sort(key=lambda row: (row.get("hardware_name", ""), row.get("combo", ""), int(row.get("op_idx", "-1"))))

    fieldnames = ordered_fieldnames(rows)
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(output_csv)


if __name__ == "__main__":
    main()
