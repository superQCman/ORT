from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import pandas as pd

from npu_sep_common import (
    CASE_ID,
    FEATURE_ROOT,
    LANE_BY_OP,
    ORT_ROOT,
    PROFILE_GLOB,
    PROFILE_ROOT,
    aggregate_duration_stats,
    combo_sort_key,
    compute_regression_metrics,
    dump_json,
    ensure_dir,
    enrich_event_measurement,
    extract_event_measurement,
    infer_lane,
    load_feature_csv,
    load_json,
    load_latest_profile_json,
    safe_int,
    split_items_by_ratio,
    str2bool,
)


PROJECT_DIR = Path(__file__).resolve().parent
NEW_DATASET_COLUMNS = [
    "case_id",
    "combo",
    "profile_dir",
    "profile_json",
    "provider",
    "op_name",
    "npu_lane",
    "transfer_direction",
    "node_index",
    "event_name",
    "raw_event_name",
    "call_count_raw",
    "call_count_after_drop",
    "label_npu_dur_us",
    "label_npu_dur_us_min",
    "label_npu_dur_us_max",
    "label_npu_dur_us_std",
    "label_npu_dur_us_count",
    "input_bytes",
    "output_bytes",
    "activation_bytes",
    "parameter_bytes",
    "shape_signature",
    "matmul_m",
    "matmul_k",
    "matmul_n",
    "vector_elem_count",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Ascend 910B3 NPU dataset for case_10_4_4_cann.")
    parser.add_argument("--case-id", default=CASE_ID, choices=[CASE_ID], help="Case id to build.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSVs and summaries.")
    parser.add_argument("--drop-first-call", type=str2bool, default=True, help="Drop the first call of each repeated node.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_profile_events(profile_json: Path) -> list[dict[str, Any]]:
    payload = load_json(profile_json)
    if not isinstance(payload, list):
        raise ValueError(f"Expected trace event list in {profile_json}")
    events: list[dict[str, Any]] = []
    for idx, event in enumerate(payload):
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args", {}) or {}
        if str(args.get("provider") or "") != "CANNExecutionProvider":
            continue
        measurement = extract_event_measurement(event)
        measurement["event_order"] = idx
        measurement["provider"] = str(args.get("provider") or "")
        events.append(enrich_event_measurement(measurement))
    return events


def aggregate_event_group(events: list[dict[str, Any]], drop_first_call: bool) -> dict[str, Any] | None:
    if not events:
        return None

    ordered = sorted(events, key=lambda item: (item["ts"], item["event_order"]))
    retained = ordered[1:] if drop_first_call and len(ordered) > 1 else ordered
    if not retained:
        retained = ordered

    durations = [float(item["duration_us"]) for item in retained]
    stats = aggregate_duration_stats(durations)
    first = retained[0]
    return {
        "provider": first["provider"],
        "op_name": first["op_name"],
        "event_name": first["event_name"],
        "raw_event_name": first["raw_event_name"],
        "node_index": int(first["node_index"]),
        "call_count_raw": len(ordered),
        "call_count_after_drop": len(retained),
        "label_npu_dur_us": float(stats["mean"]),
        "label_npu_dur_us_min": float(stats["min"]),
        "label_npu_dur_us_max": float(stats["max"]),
        "label_npu_dur_us_std": float(stats["std"]),
        "label_npu_dur_us_count": int(stats["count"]),
        "input_bytes": int(round(sum(float(item["input_bytes"]) for item in retained) / len(retained))),
        "output_bytes": int(round(sum(float(item["output_bytes"]) for item in retained) / len(retained))),
        "activation_bytes": int(round(sum(float(item["activation_bytes"]) for item in retained) / len(retained))),
        "parameter_bytes": int(round(sum(float(item["parameter_bytes"]) for item in retained) / len(retained))),
        "shape_signature": first["shape_signature"],
        "matmul_m": first.get("matmul_m"),
        "matmul_k": first.get("matmul_k"),
        "matmul_n": first.get("matmul_n"),
        "vector_elem_count": first.get("vector_elem_count"),
        "transfer_direction": first.get("transfer_direction", ""),
    }


def build_dataset(case_id: str, output_dir: Path, drop_first_call: bool, ratios: tuple[float, float, float], seed: int) -> dict[str, Any]:
    feature_root = ORT_ROOT / f"features_extensible_{case_id}"
    profile_root = ORT_ROOT / f"sweep_runs_extensible_{case_id}" / "onnx_profiles"
    if not feature_root.exists():
        raise FileNotFoundError(f"Missing feature root: {feature_root}")
    if not profile_root.exists():
        raise FileNotFoundError(f"Missing profile root: {profile_root}")

    combo_dirs = sorted([path for path in profile_root.iterdir() if path.is_dir()], key=lambda p: combo_sort_key(p.name))
    rows: list[dict[str, Any]] = []
    missing_feature_combos: list[str] = []
    missing_profile_combos: list[str] = []
    combo_event_counts: dict[str, int] = {}
    combo_row_counts: dict[str, int] = {}
    op_type_counts = collections.Counter()
    lane_counts = collections.Counter()
    provider_counts = collections.Counter()

    for combo_dir in combo_dirs:
        combo = combo_dir.name
        feature_csv = feature_root / f"{combo}.csv"
        try:
            profile_json = load_latest_profile_json(combo, profile_root)
        except FileNotFoundError:
            missing_profile_combos.append(combo)
            continue
        if not feature_csv.exists():
            missing_feature_combos.append(combo)
            continue

        feature_df = load_feature_csv(combo, feature_root)
        events = iter_profile_events(profile_json)
        combo_event_counts[combo] = len(events)
        if not events:
            missing_profile_combos.append(combo)
            continue

        catalog_by_node_index = {int(row.node_idx): row for row in feature_df.itertuples(index=False)}
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for event in events:
            key = (int(event["node_index"]), str(event["op_name"]))
            grouped[key].append(event)

        combo_rows = 0
        for (node_index, op_name), node_events in sorted(grouped.items(), key=lambda item: item[0]):
            base_row = catalog_by_node_index.get(node_index)
            if base_row is None:
                # The row is still valuable, but keep the structure explicit.
                row_dict: dict[str, Any] = {
                    "node_idx": node_index,
                    "node_name": node_events[0]["event_name"],
                    "op_type": op_name,
                    "batch_size": None,
                    "num_indices_per_lookup": None,
                    "arch_embedding_size": None,
                    "arch_mlp_bot": None,
                    "arch_mlp_top": None,
                    "source_csv": str(feature_csv),
                    "combo": combo,
                }
            else:
                row_dict = dict(base_row._asdict())

            agg = aggregate_event_group(node_events, drop_first_call)
            if agg is None:
                continue

            row_dict.update(
                {
                    "case_id": case_id,
                    "combo": combo,
                    "profile_dir": str(combo_dir),
                    "profile_json": str(profile_json),
                    "provider": agg["provider"],
                    "op_name": agg["op_name"],
                    "npu_lane": infer_lane(agg["op_name"]),
                    "transfer_direction": agg["transfer_direction"],
                    "node_index": node_index,
                    "event_name": agg["event_name"],
                    "raw_event_name": agg["raw_event_name"],
                    "call_count_raw": agg["call_count_raw"],
                    "call_count_after_drop": agg["call_count_after_drop"],
                    "label_npu_dur_us": agg["label_npu_dur_us"],
                    "label_npu_dur_us_min": agg["label_npu_dur_us_min"],
                    "label_npu_dur_us_max": agg["label_npu_dur_us_max"],
                    "label_npu_dur_us_std": agg["label_npu_dur_us_std"],
                    "label_npu_dur_us_count": agg["label_npu_dur_us_count"],
                    "input_bytes": agg["input_bytes"],
                    "output_bytes": agg["output_bytes"],
                    "activation_bytes": agg["activation_bytes"],
                    "parameter_bytes": agg["parameter_bytes"],
                    "shape_signature": agg["shape_signature"],
                    "matmul_m": agg["matmul_m"],
                    "matmul_k": agg["matmul_k"],
                    "matmul_n": agg["matmul_n"],
                    "vector_elem_count": agg["vector_elem_count"],
                }
            )
            row_dict.setdefault("node_idx", node_index)
            row_dict.setdefault("node_name", row_dict.get("node_name", ""))
            row_dict.setdefault("op_type", row_dict.get("op_type", op_name))
            row_dict.setdefault("source_csv", str(feature_csv))
            row_dict.setdefault("feature_csv", str(feature_csv))
            rows.append(row_dict)
            combo_rows += 1
            op_type_counts[str(row_dict.get("op_type") or "unknown")] += 1
            lane_counts[str(row_dict.get("npu_lane") or "unknown")] += 1
            provider_counts[str(row_dict.get("provider") or "unknown")] += 1

        combo_row_counts[combo] = combo_rows

    if not rows:
        raise RuntimeError("No NPU rows were parsed from the case_10_4_4_cann data sources")

    df = pd.DataFrame(rows)
    df = df.sort_values(["combo", "node_index", "op_name", "node_name"], kind="stable").reset_index(drop=True)

    split_map = split_items_by_ratio(sorted(df["combo"].dropna().unique(), key=combo_sort_key), *ratios, seed=seed)
    combo_to_split = {combo: split for split, combos in split_map.items() for combo in combos}
    df["split"] = df["combo"].map(combo_to_split)
    if df["split"].isna().any():
        raise RuntimeError("Some rows were not assigned a split")

    ensure_dir(output_dir)
    dataset_full_path = output_dir / "dataset_full.csv"
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "dataset_summary.json"
    feature_columns_path = output_dir / "feature_columns.json"

    df.to_csv(dataset_full_path, index=False)
    df[df["split"] == "train"].to_csv(train_path, index=False)
    df[df["split"] == "val"].to_csv(val_path, index=False)
    df[df["split"] == "test"].to_csv(test_path, index=False)

    source_columns = [column for column in pd.read_csv(feature_root / f"{df.iloc[0]['combo']}.csv", nrows=0).columns]
    added_columns = [column for column in df.columns if column not in source_columns]
    numeric_columns = [
        "batch_size",
        "num_indices_per_lookup",
        "input_bytes",
        "output_bytes",
        "activation_bytes",
        "parameter_bytes",
        "call_count_after_drop",
        "call_count_raw",
        "label_npu_dur_us_std",
        "label_npu_dur_us_count",
        "matmul_m",
        "matmul_k",
        "matmul_n",
        "vector_elem_count",
    ]
    categorical_columns = [
        "combo",
        "provider",
        "op_name",
        "op_type",
        "npu_lane",
        "transfer_direction",
        "node_name",
        "shape_signature",
        "split",
    ]
    metadata_columns = [
        "case_id",
        "node_idx",
        "profile_dir",
        "profile_json",
        "event_name",
        "raw_event_name",
        "source_csv",
    ]
    label_columns = ["label_npu_dur_us"]

    feature_columns_payload = {
        "case_id": case_id,
        "source_columns": source_columns,
        "added_columns": added_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "metadata_columns": metadata_columns,
        "label_columns": label_columns,
        "all_columns": list(df.columns),
    }
    dump_json(feature_columns_path, feature_columns_payload)

    summary = {
        "case_id": case_id,
        "feature_root": str(feature_root),
        "profile_root": str(profile_root),
        "drop_first_call": bool(drop_first_call),
        "train_ratio": float(ratios[0]),
        "val_ratio": float(ratios[1]),
        "test_ratio": float(ratios[2]),
        "seed": int(seed),
        "combo_count": int(df["combo"].nunique()),
        "combo_dirs_seen": len(combo_dirs),
        "row_count": int(len(df)),
        "rows_per_combo": {combo: int(count) for combo, count in combo_row_counts.items()},
        "events_per_combo": {combo: int(count) for combo, count in combo_event_counts.items()},
        "split_counts": df["split"].value_counts().to_dict(),
        "lane_counts": dict(lane_counts),
        "provider_counts": dict(provider_counts),
        "op_type_counts": dict(op_type_counts),
        "missing_feature_combos": missing_feature_combos,
        "missing_profile_combos": missing_profile_combos,
        "columns": list(df.columns),
        "new_columns": NEW_DATASET_COLUMNS,
    }
    dump_json(summary_path, summary)

    return summary


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    summary = build_dataset(args.case_id, Path(args.output_dir), args.drop_first_call, ratios, args.seed)
    print(
        f"Built {summary['row_count']} rows across {summary['combo_count']} combos "
        f"into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
