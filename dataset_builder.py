from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_contract import (
    ANALYSIS_NUMERIC_FEATURES,
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    DATASET_NUMERIC_COLUMNS,
    DEFAULT_SPLIT_RATIOS,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    METADATA_COLUMNS,
    PROFILE_INSTABILITY_METRICS,
    TARGET_COLUMN,
)
from feature_engineering import (
    add_engineered_features,
    build_stage2_candidate_feature_frame,
    extract_node_scope,
    load_op_shapes_for_combo,
    merge_op_shapes_into_feature_rows,
    normalize_node_name,
    parse_trace_op_idx,
    split_combo,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ORT_ROOT = SCRIPT_DIR.parent
CASE_DIR_RE = re.compile(r"^features_extensible_case_(?P<case>\d+)_(?P<run>\d+)_(?P<tag>\d+)$")
CASE_ID_RE = re.compile(r"^case_(?P<case>\d+)_(?P<run>\d+)_(?P<tag>\d+)$")
SOURCE_NAME_RE = re.compile(r"^case_(?P<case>\d+)_run_(?P<run>\d+)_(?P<tag>\d+)$")
CASE_TRIPLE_RE = re.compile(r"^(?P<case>\d+)_(?P<run>\d+)_(?P<tag>\d+)$")

RAW_COLUMNS = {
    "batch_size",
    "num_indices_per_lookup",
    "arch_embedding_size",
    "arch_mlp_bot",
    "arch_mlp_top",
    "node_idx",
    "node_name",
    "op_type",
    "trace_op_name",
    "total_instructions",
    "total_loads",
    "total_stores",
    "load_store_ratio",
    "num_threads",
    "reuse_time_mean",
    "reuse_distance_mean",
    "reuse_distance_unique_cache_lines_per_k_accesses",
    "opc_branch_ratio",
    "opc_fp_math_ratio",
    "opc_load_ratio",
    "opc_math_ratio",
    "opc_simd_ratio",
    "opc_store_ratio",
    "has_cpu_profile",
    "cpu_profile_missing_reason",
    "cpu_dur_us_avg",
    "cpu_output_size_avg",
    "cpu_activation_size_avg",
    "cpu_parameter_size_avg",
    "dur_us",
    "output_size",
    "activation_size",
    "parameter_size",
}


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-case single-op regression tables with stage-1-style features.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for dataset_full.csv, train.csv, val.csv, test.csv, and summaries.",
    )
    parser.add_argument(
        "--case-pattern",
        default="features_extensible_case_*",
        help="Glob-style pattern applied to feature case directories.",
    )
    parser.add_argument(
        "--selected-cases",
        nargs="*",
        default=None,
        help=(
            "Optional manual case list. Supports space-separated or comma-separated "
            "values like case_1_1_1, features_extensible_case_1_1_1, or case_1_run_1_1."
        ),
    )
    parser.add_argument(
        "--selected-cases-file",
        default="",
        help="Optional text/JSON file containing the manually selected cases.",
    )
    parser.add_argument(
        "--max-files-per-case",
        type=int,
        default=0,
        help="Optional debug limit for the number of combo CSVs loaded per case.",
    )
    parser.add_argument(
        "--group-column",
        default=GROUP_COLUMN,
        help="Grouping column used for the 7:2:1 split.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-first-profile-batch",
        type=str2bool,
        default=True,
        help="Whether to exclude the earliest profile batch before computing labels and stability metrics.",
    )
    parser.add_argument(
        "--profile-instability-metric",
        choices=PROFILE_INSTABILITY_METRICS,
        default="last2_range_ratio",
        help="Metric used to filter unstable operator samples after dropping the first profile batch.",
    )
    parser.add_argument(
        "--profile-instability-threshold",
        type=float,
        default=0.20,
        help="Drop samples whose profile instability metric is above this threshold.",
    )
    parser.add_argument(
        "--disable-profile-stability-filter",
        action="store_true",
        help="Keep all samples after first-batch removal, even if their remaining batches are still unstable.",
    )
    return parser.parse_args()


def infer_case_metadata(case_dir: Path) -> dict[str, Any]:
    match = CASE_DIR_RE.fullmatch(case_dir.name)
    if match is None:
        raise ValueError(f"Unsupported case directory name: {case_dir.name}")

    case_id = f"case_{match.group('case')}_{match.group('run')}_{match.group('tag')}"
    source_name = f"case_{match.group('case')}_run_{match.group('run')}_{match.group('tag')}"
    source_mode = "serial_control" if match.group("tag") == "1" else "concurrent"
    sweep_dir = ORT_ROOT / f"sweep_runs_extensible_{case_id}"
    return {
        "case_id": case_id,
        "source_name": source_name,
        "source_mode": source_mode,
        "sweep_dir": sweep_dir,
    }


def split_case_selector_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", value.strip()) if token]


def canonicalize_case_selector(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Empty case selector is not allowed")

    for pattern in (CASE_ID_RE, CASE_DIR_RE, SOURCE_NAME_RE, CASE_TRIPLE_RE):
        match = pattern.fullmatch(text)
        if match is not None:
            return f"case_{match.group('case')}_{match.group('run')}_{match.group('tag')}"

    raise ValueError(
        "Unsupported case selector "
        f"{value!r}; expected case_1_1_1, features_extensible_case_1_1_1, case_1_run_1_1, or 1_1_1"
    )


def load_case_selectors_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Selected case file does not exist: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            return [str(item) for item in payload["cases"]]
        raise ValueError("JSON selected case file must be a list or a dict with a 'cases' list")

    return split_case_selector_tokens(text)


def resolve_selected_case_ids(
    selected_cases: list[str] | None,
    selected_cases_file: str,
) -> list[str] | None:
    raw_tokens: list[str] = []
    for item in selected_cases or []:
        raw_tokens.extend(split_case_selector_tokens(item))

    if selected_cases_file:
        raw_tokens.extend(load_case_selectors_from_file(Path(selected_cases_file)))

    if not raw_tokens:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        case_id = canonicalize_case_selector(token)
        if case_id not in seen:
            normalized.append(case_id)
            seen.add(case_id)
    return normalized


def discover_case_entries() -> list[dict[str, Any]]:
    case_entries: list[dict[str, Any]] = []
    for path in sorted(ORT_ROOT.glob("features_extensible_case_*")):
        if not path.is_dir() or path.name.endswith("_selected"):
            continue
        case_meta = infer_case_metadata(path)
        case_entries.append(
            {
                **case_meta,
                "case_dir": path,
            }
        )
    return case_entries


def op_shapes_path_for_combo(case_meta: dict[str, Any], combo: str) -> Path:
    batch_size, num_indices = split_combo(combo)
    if batch_size is None or num_indices is None:
        return case_meta["sweep_dir"] / "op_shapes" / f"{combo}.csv"
    return case_meta["sweep_dir"] / "op_shapes" / f"op_shapes_{batch_size}_{num_indices}.csv"


def make_row_uid(source_name: str, combo: str, op_idx: Any, node_name: str) -> str:
    if pd.notna(op_idx):
        return f"{source_name}::{combo}::opidx:{int(op_idx)}"
    normalized = normalize_node_name(node_name) or "unknown"
    return f"{source_name}::{combo}::node:{normalized}"


def load_raw_feature_csv(feature_csv: Path) -> pd.DataFrame:
    return pd.read_csv(feature_csv, usecols=lambda column: column in RAW_COLUMNS)


def profile_timeline_path_for_combo(case_meta: dict[str, Any], combo: str) -> Path:
    return case_meta["sweep_dir"] / "onnx_profiles" / combo / "branch_parallel_op_timeline.csv"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").abs().clip(lower=1e-9)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def load_profile_label_stats(
    case_meta: dict[str, Any],
    combo: str,
    *,
    drop_first_profile_batch: bool,
) -> pd.DataFrame:
    timeline_path = profile_timeline_path_for_combo(case_meta, combo)
    if not timeline_path.exists():
        raise FileNotFoundError(f"Profile timeline CSV does not exist: {timeline_path}")

    timeline_df = pd.read_csv(timeline_path, usecols=["batch_idx", "node_name", "dur_us"])
    timeline_df["batch_idx"] = pd.to_numeric(timeline_df["batch_idx"], errors="coerce")
    timeline_df["dur_us"] = pd.to_numeric(timeline_df["dur_us"], errors="coerce")
    timeline_df = timeline_df[
        timeline_df["batch_idx"].notna()
        & timeline_df["dur_us"].notna()
    ].copy()
    if timeline_df.empty:
        return pd.DataFrame()

    timeline_df["batch_idx"] = timeline_df["batch_idx"].astype(int)
    timeline_df["node_name_normalized"] = timeline_df["node_name"].map(normalize_node_name)
    timeline_df = timeline_df[timeline_df["node_name_normalized"] != ""].copy()
    if timeline_df.empty:
        return pd.DataFrame()

    by_batch = (
        timeline_df.groupby(["node_name_normalized", "batch_idx"], dropna=False)["dur_us"]
        .sum()
        .reset_index()
    )
    all_batches = sorted(by_batch["batch_idx"].dropna().unique().tolist())
    dropped_batches = all_batches[:1] if drop_first_profile_batch else []
    kept_batches = [batch for batch in all_batches if batch not in dropped_batches]

    all_stats = by_batch.groupby("node_name_normalized", dropna=False)["dur_us"].agg(
        profile_batch_count_total="count",
        profile_label_all_batch_mean_us="mean",
    )

    kept_stats = (
        by_batch[by_batch["batch_idx"].isin(kept_batches)]
        .groupby("node_name_normalized", dropna=False)["dur_us"]
        .agg(
            profile_batch_count_kept="count",
            profile_label_kept_batch_mean_us="mean",
        )
    )

    pivot = by_batch.pivot_table(
        index="node_name_normalized",
        columns="batch_idx",
        values="dur_us",
        aggfunc="first",
    ).sort_index(axis=1)

    stats = all_stats.join(kept_stats, how="left").reset_index()
    stats["profile_batch_count_kept"] = pd.to_numeric(
        stats["profile_batch_count_kept"],
        errors="coerce",
    ).fillna(0).astype(int)
    stats["profile_dropped_batch_indices"] = ",".join(str(batch) for batch in dropped_batches)
    stats["profile_kept_batch_indices"] = ",".join(str(batch) for batch in kept_batches)

    if len(kept_batches) >= 2:
        last2_batches = kept_batches[-2:]
        last2_left = pd.to_numeric(pivot[last2_batches[0]], errors="coerce")
        last2_right = pd.to_numeric(pivot[last2_batches[1]], errors="coerce")
        pair_mean = (last2_left + last2_right) / 2.0
        pair_std = pd.concat([last2_left, last2_right], axis=1).std(axis=1, ddof=0)
        last2_metrics = pd.DataFrame(index=pivot.index)
        last2_metrics.index.name = "node_name_normalized"
        last2_metrics["profile_last2_abs_diff_us"] = (last2_left - last2_right).abs()
        last2_metrics["profile_last2_range_ratio"] = _safe_ratio(
            (last2_left - last2_right).abs(),
            pair_mean,
        )
        last2_metrics["profile_last2_cv"] = _safe_ratio(pair_std, pair_mean)
        stats = stats.merge(
            last2_metrics.reset_index(),
            on="node_name_normalized",
            how="left",
        )
    else:
        stats["profile_last2_abs_diff_us"] = np.nan
        stats["profile_last2_range_ratio"] = np.nan
        stats["profile_last2_cv"] = np.nan

    return stats


def build_rows_for_feature_csv(
    feature_csv: Path,
    case_meta: dict[str, Any],
    *,
    drop_first_profile_batch: bool,
) -> pd.DataFrame:
    combo = feature_csv.stem
    df = load_raw_feature_csv(feature_csv)
    if df.empty:
        return pd.DataFrame()

    if "batch_size" not in df.columns or df["batch_size"].isna().all():
        batch_size, _ = split_combo(combo)
        df["batch_size"] = batch_size
    if "num_indices_per_lookup" not in df.columns or df["num_indices_per_lookup"].isna().all():
        _, num_indices = split_combo(combo)
        df["num_indices_per_lookup"] = num_indices

    node_name = df.get("node_name", pd.Series("", index=df.index)).fillna("").astype(str)
    trace_op_name = df.get("trace_op_name", pd.Series("", index=df.index)).fillna("").astype(str)
    node_idx = pd.to_numeric(df.get("node_idx"), errors="coerce").astype("Int64") if "node_idx" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int64")
    inferred_idx = trace_op_name.map(parse_trace_op_idx).astype("Int64")

    df["op_idx"] = node_idx.where(node_idx.notna(), inferred_idx)
    df["node_name"] = node_name
    df["trace_op_name"] = trace_op_name
    df["op_type"] = df.get("op_type", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    df["arch_embedding_size"] = df.get("arch_embedding_size", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    df["arch_mlp_bot"] = df.get("arch_mlp_bot", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    df["arch_mlp_top"] = df.get("arch_mlp_top", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    df["node_name_normalized"] = df["node_name"].map(normalize_node_name)
    df["node_scope"] = df["node_name_normalized"].map(extract_node_scope)
    df["combo"] = combo
    df["feature_csv"] = str(feature_csv)
    df["case_id"] = case_meta["case_id"]
    df["source_name"] = case_meta["source_name"]
    df["source_mode"] = case_meta["source_mode"]
    df["sample_group"] = combo
    df["has_cpu_profile"] = pd.to_numeric(df.get("has_cpu_profile", 0), errors="coerce").fillna(0).astype(int)

    op_shapes_df = load_op_shapes_for_combo(op_shapes_path_for_combo(case_meta, combo))
    df = merge_op_shapes_into_feature_rows(df, op_shapes_df)
    df = add_engineered_features(df)
    df = build_stage2_candidate_feature_frame(
        df,
        combo_profile_dir=case_meta["sweep_dir"] / "onnx_profiles" / combo,
    )
    profile_stats_df = load_profile_label_stats(
        case_meta,
        combo,
        drop_first_profile_batch=drop_first_profile_batch,
    )
    if profile_stats_df.empty:
        raise RuntimeError(f"No profile label rows were recovered for {case_meta['case_id']}::{combo}")
    df = df.merge(profile_stats_df, on="node_name_normalized", how="left")

    df[TARGET_COLUMN] = pd.to_numeric(df.get("profile_label_kept_batch_mean_us"), errors="coerce")
    df["row_uid"] = [
        make_row_uid(case_meta["source_name"], combo, op_idx, node)
        for op_idx, node in zip(df["op_idx"], df["node_name"])
    ]
    return df


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in DATASET_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in BASELINE_CATEGORICAL_FEATURES:
        if column not in out.columns:
            out[column] = "unknown"
        out[column] = out[column].fillna("unknown").astype(str)

    if "op_idx" in out.columns:
        out["op_idx"] = pd.to_numeric(out["op_idx"], errors="coerce").astype("Int64")
    return out


def assign_group_splits(groups: pd.Series, seed: int, ratios: dict[str, float]) -> dict[str, str]:
    unique_groups = sorted({str(value) for value in groups.dropna().astype(str)})
    if not unique_groups:
        return {}

    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(unique_groups))
    total = len(shuffled)
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])

    split_map: dict[str, str] = {}
    for group in shuffled[:train_end]:
        split_map[group] = "train"
    for group in shuffled[train_end:val_end]:
        split_map[group] = "val"
    for group in shuffled[val_end:]:
        split_map[group] = "test"
    return split_map


def ordered_output_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in METADATA_COLUMNS:
        if column in df.columns:
            columns.append(column)
    for column in FEATURE_COLUMNS:
        if column in df.columns and column not in columns:
            columns.append(column)
    for column in ANALYSIS_NUMERIC_FEATURES:
        if column in df.columns and column not in columns:
            columns.append(column)
    if TARGET_COLUMN in df.columns:
        columns.append(TARGET_COLUMN)
    return columns


def build_dataset(
    output_dir: Path,
    case_pattern: str,
    selected_case_ids: list[str] | None,
    max_files_per_case: int,
    group_column: str,
    seed: int,
    ratios: dict[str, float],
    drop_first_profile_batch: bool,
    profile_instability_metric: str,
    profile_instability_threshold: float,
    disable_profile_stability_filter: bool,
) -> dict[str, Any]:
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    all_case_entries = discover_case_entries()
    if not all_case_entries:
        raise FileNotFoundError(f"No case directories were found under {ORT_ROOT}")

    case_entries = [
        entry
        for entry in all_case_entries
        if fnmatch.fnmatch(entry["case_dir"].name, case_pattern)
        and (selected_case_ids is None or entry["case_id"] in selected_case_ids)
    ]
    if selected_case_ids is not None:
        matched_case_ids = {entry["case_id"] for entry in case_entries}
        missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in matched_case_ids]
        if missing_case_ids:
            available_case_ids = [entry["case_id"] for entry in all_case_entries]
            raise FileNotFoundError(
                "Selected cases were not found after applying filters: "
                f"{missing_case_ids}. case_pattern={case_pattern!r}, available_cases={available_case_ids}"
            )

    if not case_entries:
        raise FileNotFoundError(f"No case directories matched pattern {case_pattern!r}")

    frames: list[pd.DataFrame] = []
    case_file_counts: dict[str, int] = {}
    for case_meta in case_entries:
        case_dir = Path(case_meta["case_dir"])
        feature_csvs = sorted(case_dir.glob("bs*_nip*.csv"))
        if max_files_per_case > 0:
            feature_csvs = feature_csvs[:max_files_per_case]
        case_file_counts[case_meta["case_id"]] = len(feature_csvs)
        for feature_csv in feature_csvs:
            frame = build_rows_for_feature_csv(
                feature_csv,
                case_meta,
                drop_first_profile_batch=drop_first_profile_batch,
            )
            if not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No rows were built from the selected case directories")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = normalize_feature_columns(dataset)
    dataset[TARGET_COLUMN] = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    dataset = dataset[dataset[TARGET_COLUMN].notna() & (dataset[TARGET_COLUMN] > 0)].copy()
    instability_column = f"profile_{profile_instability_metric}"
    if instability_column not in dataset.columns:
        raise KeyError(
            f"Unsupported instability metric {profile_instability_metric!r}; "
            f"expected one of {PROFILE_INSTABILITY_METRICS}"
        )
    dataset["profile_instability_metric_name"] = profile_instability_metric
    dataset["profile_instability_value"] = pd.to_numeric(
        dataset[instability_column],
        errors="coerce",
    )
    dataset["profile_is_stable"] = (
        dataset["profile_instability_value"].isna()
        | (dataset["profile_instability_value"] <= profile_instability_threshold)
    )
    pre_filter_row_count = int(len(dataset))
    pre_filter_case_count = int(dataset["case_id"].nunique())
    if not disable_profile_stability_filter:
        dataset = dataset[dataset["profile_is_stable"]].copy()
    dataset[group_column] = dataset[group_column].fillna(dataset["combo"]).astype(str)

    split_map = assign_group_splits(dataset[group_column], seed=seed, ratios=ratios)
    dataset["split"] = dataset[group_column].map(split_map)
    dataset = dataset[dataset["split"].notna()].copy()
    dataset = dataset.sort_values(["split", "source_name", "combo", "op_idx", "trace_op_name"], kind="stable").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_columns = ordered_output_columns(dataset)
    dataset = dataset[ordered_columns].copy()
    dataset.to_csv(output_dir / "dataset_full.csv", index=False)

    split_paths: dict[str, str] = {}
    split_row_counts: dict[str, int] = {}
    split_group_counts: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        split_df = dataset[dataset["split"] == split_name].copy()
        split_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths[split_name] = str(split_path)
        split_row_counts[split_name] = int(len(split_df))
        split_group_counts[split_name] = int(split_df[group_column].nunique())

    feature_manifest = {
        "categorical_features": list(BASELINE_CATEGORICAL_FEATURES),
        "numeric_features": list(BASELINE_NUMERIC_FEATURES),
        "analysis_numeric_features": list(ANALYSIS_NUMERIC_FEATURES),
        "all_features": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "group_column": group_column,
    }
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_manifest, handle, indent=2, ensure_ascii=False)

    instability_values = pd.to_numeric(
        pd.concat(frames, ignore_index=True).get(f"profile_{profile_instability_metric}"),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    instability_summary = {
        "count": int(instability_values.notna().sum()),
        "p50": float(instability_values.quantile(0.50)) if instability_values.notna().any() else 0.0,
        "p90": float(instability_values.quantile(0.90)) if instability_values.notna().any() else 0.0,
        "p95": float(instability_values.quantile(0.95)) if instability_values.notna().any() else 0.0,
        "max": float(instability_values.max()) if instability_values.notna().any() else 0.0,
    }
    post_filter_row_count = int(len(dataset))
    summary = {
        "ort_root": str(ORT_ROOT),
        "case_pattern": case_pattern,
        "selected_cases": selected_case_ids,
        "case_count": len(case_entries),
        "case_file_counts": case_file_counts,
        "feature_count": len(FEATURE_COLUMNS),
        "total_rows": post_filter_row_count,
        "pre_filter_rows": pre_filter_row_count,
        "rows_dropped_by_profile_filter": pre_filter_row_count - post_filter_row_count,
        "rows_dropped_by_profile_filter_fraction": (
            float((pre_filter_row_count - post_filter_row_count) / pre_filter_row_count)
            if pre_filter_row_count > 0
            else 0.0
        ),
        "split_row_counts": split_row_counts,
        "split_group_counts": split_group_counts,
        "unique_cases": int(dataset["case_id"].nunique()),
        "unique_cases_pre_filter": pre_filter_case_count,
        "unique_sources": int(dataset["source_name"].nunique()),
        "unique_groups": int(dataset[group_column].nunique()),
        "source_mode_counts": dataset["source_mode"].value_counts(dropna=False).to_dict(),
        "profile_label_policy": {
            "drop_first_profile_batch": drop_first_profile_batch,
            "dropped_batch_indices_default": [0] if drop_first_profile_batch else [],
            "instability_filter_enabled": not disable_profile_stability_filter,
            "instability_metric": profile_instability_metric,
            "instability_threshold": profile_instability_threshold,
            "recommended_threshold": 0.20,
            "recommended_threshold_note": (
                "0.20 aligns with the E2E stability audit's common unstable-CV threshold and, "
                "on the current all-case profiles, removes about 17% of operator rows after the first batch is dropped."
            ),
            "instability_distribution_after_first_batch_drop": instability_summary,
        },
        "output_files": {
            "dataset_full_csv": str(output_dir / "dataset_full.csv"),
            **split_paths,
            "feature_columns_json": str(output_dir / "feature_columns.json"),
        },
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    selected_case_ids = resolve_selected_case_ids(args.selected_cases, args.selected_cases_file)
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    summary = build_dataset(
        output_dir=Path(args.output_dir),
        case_pattern=args.case_pattern,
        selected_case_ids=selected_case_ids,
        max_files_per_case=args.max_files_per_case,
        group_column=args.group_column,
        seed=args.seed,
        ratios=ratios,
        drop_first_profile_batch=args.drop_first_profile_batch,
        profile_instability_metric=args.profile_instability_metric,
        profile_instability_threshold=args.profile_instability_threshold,
        disable_profile_stability_filter=args.disable_profile_stability_filter,
    )
    print(f"dataset_full_csv={summary['output_files']['dataset_full_csv']}")
    print(f"train_csv={summary['output_files']['train']}")
    print(f"val_csv={summary['output_files']['val']}")
    print(f"test_csv={summary['output_files']['test']}")


if __name__ == "__main__":
    main()
