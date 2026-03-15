#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONCORDE_METADATA_COLUMNS = {
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
}

KNOWN_DUPLICATE_CDF_PREFIXES = [
    "concorde_static_fetch_width_",
    "concorde_static_decode_width_",
    "concorde_static_rename_width_",
    "concorde_static_commit_width_",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def concorde_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("hardware_name", ""), row.get("combo", ""), str(row.get("op_idx", "")))


def should_drop_feature(name: str, drop_known_duplicate_cdf: bool) -> bool:
    if name in CONCORDE_METADATA_COLUMNS:
        return True
    if not drop_known_duplicate_cdf:
        return False
    return any(name.startswith(prefix) for prefix in KNOWN_DUPLICATE_CDF_PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Concorde CDF features into the existing operator dataset.")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--concorde-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--keep-known-duplicate-cdf",
        action="store_true",
        help="Keep static global-width CDF columns even though they duplicate hw_core_* widths",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional report JSON summarizing matches, drops, and appended features",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_csv).resolve()
    concorde_path = Path(args.concorde_csv).resolve()
    output_path = Path(args.output_csv).resolve()

    dataset_rows = read_csv_rows(dataset_path)
    concorde_rows = read_csv_rows(concorde_path)
    concorde_map = {concorde_key(row): row for row in concorde_rows}

    if not dataset_rows:
        raise ValueError(f"No rows found in dataset CSV: {dataset_path}")
    if not concorde_rows:
        raise ValueError(f"No rows found in Concorde CSV: {concorde_path}")

    sample_concorde = concorde_rows[0]
    concorde_feature_columns = [
        name
        for name in sample_concorde.keys()
        if not should_drop_feature(name, drop_known_duplicate_cdf=not args.keep_known_duplicate_cdf)
    ]

    merged_rows = []
    matched = 0
    for row in dataset_rows:
        key = concorde_key(row)
        merged = dict(row)
        concorde_row = concorde_map.get(key)
        if concorde_row is not None:
            matched += 1
            for name in concorde_feature_columns:
                merged[name] = concorde_row.get(name, "")
        else:
            for name in concorde_feature_columns:
                merged[name] = ""
        merged_rows.append(merged)

    fieldnames = list(dataset_rows[0].keys()) + [name for name in concorde_feature_columns if name not in dataset_rows[0]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    report = {
        "dataset_rows": len(dataset_rows),
        "concorde_rows": len(concorde_rows),
        "matched_rows": matched,
        "unmatched_rows": len(dataset_rows) - matched,
        "appended_concorde_feature_count": len(concorde_feature_columns),
        "dropped_duplicate_prefixes": [] if args.keep_known_duplicate_cdf else KNOWN_DUPLICATE_CDF_PREFIXES,
    }
    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report_path)

    print(output_path)


if __name__ == "__main__":
    main()
