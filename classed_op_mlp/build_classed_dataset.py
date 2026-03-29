from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_contract import ANALYTICAL_RESIDUAL_TARGET_COLUMN, TARGET_COLUMN  # noqa: E402
from feature_engineering import add_engineered_features  # noqa: E402

try:  # noqa: E402
    from .contracts import (
        BASELINE_COMPARE_DIR,
        CLASSED_FEATURE_DESCRIPTIONS,
        DEFAULT_FEATURE_BRANCH,
        DEFAULT_INPUT_DATASET_DIR,
        FEATURE_BRANCH_NO_ANALYTICAL,
        OP_CLASS_ORDER,
        SHARED_CATEGORICAL_FEATURES,
        SUPPORTED_FEATURE_BRANCHES,
        resolve_branch_features,
        resolve_model_group,
        resolve_model_group_op_types,
        resolve_model_group_order,
        resolve_op_class,
        resolve_output_dir,
    )
except ImportError:  # noqa: E402
    from contracts import (
        BASELINE_COMPARE_DIR,
        CLASSED_FEATURE_DESCRIPTIONS,
        DEFAULT_FEATURE_BRANCH,
        DEFAULT_INPUT_DATASET_DIR,
        FEATURE_BRANCH_NO_ANALYTICAL,
        OP_CLASS_ORDER,
        SHARED_CATEGORICAL_FEATURES,
        SUPPORTED_FEATURE_BRANCHES,
        resolve_branch_features,
        resolve_model_group,
        resolve_model_group_op_types,
        resolve_model_group_order,
        resolve_op_class,
        resolve_output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build branch-specific grouped datasets for the classed single-op MLP pipeline.",
    )
    parser.add_argument(
        "--input-data-dir",
        default=str(DEFAULT_INPUT_DATASET_DIR),
        help="Dataset directory containing dataset_full.csv and split CSVs. Defaults to dataset_all_no_trace.",
    )
    parser.add_argument(
        "--analytical-dir",
        default="",
        help="Directory produced by analytical_calibrated/run_pipeline.py. Ignored for no_analytical branch.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output root for classed dataset artifacts. Defaults to a branch-specific directory under artifacts/latest/classed_op_mlp.",
    )
    parser.add_argument(
        "--feature-branch",
        choices=list(SUPPORTED_FEATURE_BRANCHES),
        default=DEFAULT_FEATURE_BRANCH,
        help="Feature branch to export. with_analytical keeps ana_calib_*; no_analytical excludes them to isolate pure classed MLP behavior.",
    )
    return parser.parse_args()


def load_dataset(data_dir: Path) -> pd.DataFrame:
    dataset_csv = data_dir / "dataset_full.csv"
    if not dataset_csv.exists():
        raise FileNotFoundError(dataset_csv)
    frame = pd.read_csv(dataset_csv, low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    frame["_source_order"] = range(len(frame))
    return frame


def load_analytical_features(analytical_dir: Path) -> pd.DataFrame:
    feature_csv = analytical_dir / "analytical_features_full.csv"
    if not feature_csv.exists():
        raise FileNotFoundError(feature_csv)
    frame = pd.read_csv(feature_csv, low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    keep_columns = ["row_uid", "op_class", "ana_calib_family", "ana_calib_total_us", "ana_calib_mem_us", "ana_calib_compute_us", "ana_calib_overhead_us"]
    return frame[[column for column in keep_columns if column in frame.columns]].copy()


def build_gemm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.drop(
        columns=[column for column in frame.columns if column.startswith(("feat_", "ana_", "hw_"))],
        errors="ignore",
    ).copy()
    engineered = add_engineered_features(base)
    columns = [
        "row_uid",
        "feat_gemm_m",
        "feat_gemm_n",
        "feat_gemm_k",
        "feat_gemm_mac_count",
        "feat_gemm_bytes_per_mac",
    ]
    return engineered[[column for column in columns if column in engineered.columns]].copy()


def dataset_summary_op_type_map(frame: pd.DataFrame) -> dict[str, str]:
    mapping = (
        frame[["op_type", "op_class"]]
        .dropna()
        .drop_duplicates(subset=["op_type"], keep="first")
        .sort_values("op_type")
    )
    return dict(zip(mapping["op_type"].astype(str), mapping["op_class"].astype(str)))


def feature_manifest_payload(model_group: str, feature_branch: str) -> dict[str, Any]:
    branch_features = resolve_branch_features(feature_branch)
    numeric_features = list(branch_features[model_group])
    categorical_features = list(SHARED_CATEGORICAL_FEATURES)
    return {
        "routing_policy": "static_op_type",
        "feature_branch": feature_branch,
        "analytical_enabled": feature_branch != FEATURE_BRANCH_NO_ANALYTICAL,
        "model_group": model_group,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "analysis_numeric_features": numeric_features,
        "shared_categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        "model_group_order": list(resolve_model_group_order(feature_branch)),
        "per_model_group_numeric_features": {key: list(value) for key, value in branch_features.items()},
        "op_type_class_map": {},  # filled by the top-level manifest
        "op_type_model_group_map": {},  # filled by the top-level manifest
        "target_column": TARGET_COLUMN,
        "target_columns": [TARGET_COLUMN] if feature_branch == FEATURE_BRANCH_NO_ANALYTICAL else [TARGET_COLUMN, ANALYTICAL_RESIDUAL_TARGET_COLUMN],
        "analytical_feature_columns": [] if feature_branch == FEATURE_BRANCH_NO_ANALYTICAL else ["ana_calib_total_us", "ana_calib_mem_us", "ana_calib_compute_us", "ana_calib_overhead_us", "ana_calib_family", "op_class"],
        "all_features": categorical_features + numeric_features,
        "baseline_compare_dir": str(BASELINE_COMPARE_DIR),
    }


def build_classed_dataset_artifacts(
    input_data_dir: Path,
    analytical_dir: Path | None,
    output_dir: Path,
    *,
    feature_branch: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    branch_features = resolve_branch_features(feature_branch)
    model_group_order = resolve_model_group_order(feature_branch)
    model_group_op_types = resolve_model_group_op_types(feature_branch)

    base_df = load_dataset(input_data_dir)
    gemm_df = build_gemm_columns(base_df)
    merged = base_df.merge(gemm_df, on="row_uid", how="left", validate="one_to_one")
    merged["op_class"] = merged["op_type"].map(resolve_op_class).fillna("mixed_balanced").astype(str)
    merged["model_group"] = merged["op_type"].map(lambda op_type: resolve_model_group(feature_branch, op_type)).astype(str)
    if feature_branch == FEATURE_BRANCH_NO_ANALYTICAL:
        merged["ana_calib_family"] = "not_used"
    else:
        if analytical_dir is None:
            raise RuntimeError("analytical_dir is required for with_analytical branch")
        analytical_df = load_analytical_features(analytical_dir)
        merged = merged.merge(
            analytical_df,
            on="row_uid",
            how="left",
            validate="one_to_one",
        )
        missing = int(merged["op_class"].isna().sum()) if "op_class" in merged.columns else len(merged)
        if missing > 0:
            raise RuntimeError(f"analytical_calibrated features are missing for {missing} rows")
        merged["op_class"] = merged["op_class"].fillna("mixed_balanced").astype(str)
        merged["model_group"] = merged["op_class"].fillna("mixed_balanced").astype(str)
        merged["ana_calib_family"] = merged["ana_calib_family"].fillna("not_used").astype(str)
    merged_path = output_dir / "classed_dataset_full.csv"
    merged.to_csv(merged_path, index=False)

    model_group_summary: dict[str, Any] = {}
    for model_group in model_group_order:
        group_dir = datasets_dir / model_group
        group_dir.mkdir(parents=True, exist_ok=True)
        group_df = merged[merged["model_group"] == model_group].copy()
        group_df = group_df.sort_values("_source_order", kind="stable").reset_index(drop=True)
        for split_name in ["train", "val", "test"]:
            split_df = group_df[group_df["split"] == split_name].copy()
            split_df.to_csv(group_dir / f"{split_name}.csv", index=False)
        group_df.to_csv(group_dir / "dataset_full.csv", index=False)

        manifest = feature_manifest_payload(model_group, feature_branch)
        manifest["op_type_class_map"] = dict(dataset_summary_op_type_map(merged))
        manifest["op_type_model_group_map"] = {
            op_type: resolved_group
            for resolved_group, op_types in model_group_op_types.items()
            for op_type in op_types
        }
        manifest["feature_descriptions"] = {
            **{key: CLASSED_FEATURE_DESCRIPTIONS[key] for key in SHARED_CATEGORICAL_FEATURES if key in CLASSED_FEATURE_DESCRIPTIONS},
            **{key: CLASSED_FEATURE_DESCRIPTIONS[key] for key in branch_features[model_group] if key in CLASSED_FEATURE_DESCRIPTIONS},
        }
        with (group_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        model_group_summary[model_group] = {
            "class_dir": str(group_dir),
            "op_class": str(group_df["op_class"].iloc[0]) if not group_df.empty else "",
            "op_types": list(model_group_op_types.get(model_group, ())),
            "row_count": int(len(group_df)),
            "split_row_counts": {
                split_name: int((group_df["split"] == split_name).sum())
                for split_name in ["train", "val", "test"]
            },
            "numeric_features": list(branch_features[model_group]),
            "categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        }

    analytical_feature_descriptions: dict[str, str] = {}
    if feature_branch != FEATURE_BRANCH_NO_ANALYTICAL:
        for key in ["ana_calib_total_us", "ana_calib_mem_us", "ana_calib_compute_us", "ana_calib_overhead_us", "ana_calib_family"]:
            if key in CLASSED_FEATURE_DESCRIPTIONS:
                analytical_feature_descriptions[key] = CLASSED_FEATURE_DESCRIPTIONS[key]

    summary_payload = {
        "input_data_dir": str(input_data_dir),
        "feature_branch": feature_branch,
        "analytical_enabled": feature_branch != FEATURE_BRANCH_NO_ANALYTICAL,
        "analytical_dir": "" if analytical_dir is None else str(analytical_dir),
        "output_dir": str(output_dir),
        "baseline_compare_dir": str(BASELINE_COMPARE_DIR),
        "routing_policy": "static_op_type",
        "op_type_class_map": dataset_summary_op_type_map(merged),
        "op_type_model_group_map": {
            op_type: resolved_group
            for resolved_group, op_types in model_group_op_types.items()
            for op_type in op_types
        },
        "shared_categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        "model_group_order": list(model_group_order),
        "per_model_group_numeric_features": {key: list(value) for key, value in branch_features.items()},
        "per_class_numeric_features": {key: list(value) for key, value in branch_features.items()},
        "feature_descriptions": CLASSED_FEATURE_DESCRIPTIONS,
        "analytical_feature_descriptions": analytical_feature_descriptions,
        "merged_dataset_csv": str(merged_path),
        "model_groups": model_group_summary,
        "classes": model_group_summary,
    }
    summary_json = output_dir / "dataset_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    return summary_payload


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.feature_branch)
    analytical_dir = None
    if args.feature_branch != FEATURE_BRANCH_NO_ANALYTICAL:
        analytical_dir = Path(args.analytical_dir) if args.analytical_dir else PROJECT_ROOT / "artifacts" / "latest" / "analytical_calibrated"
    payload = build_classed_dataset_artifacts(
        Path(args.input_data_dir),
        analytical_dir,
        output_dir,
        feature_branch=args.feature_branch,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
