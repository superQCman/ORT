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

from analytical_calibrated.contracts import (  # noqa: E402
    ANALYTICAL_FEATURE_COLUMNS,
    ANALYTICAL_FEATURE_DESCRIPTIONS,
    BASELINE_COMPARE_DIR,
    DEFAULT_INPUT_DATASET_DIR,
    FEATURE_DESCRIPTIONS,
    OP_CLASS_ORDER,
    PER_CLASS_NUMERIC_FEATURES,
    SHARED_CATEGORICAL_FEATURES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build three class-specific datasets for the classed single-op MLP pipeline.",
    )
    parser.add_argument(
        "--input-data-dir",
        default=str(DEFAULT_INPUT_DATASET_DIR),
        help="Dataset directory containing dataset_full.csv and split CSVs. Defaults to dataset_all_no_trace.",
    )
    parser.add_argument(
        "--analytical-dir",
        default=str(PROJECT_ROOT / "artifacts" / "latest" / "analytical_calibrated"),
        help="Directory produced by analytical_calibrated/run_pipeline.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "latest" / "classed_op_mlp"),
        help="Output root for classed dataset artifacts.",
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


def feature_manifest_payload(op_class: str) -> dict[str, Any]:
    numeric_features = list(PER_CLASS_NUMERIC_FEATURES[op_class])
    categorical_features = list(SHARED_CATEGORICAL_FEATURES)
    return {
        "routing_policy": "static_op_type",
        "op_class": op_class,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "analysis_numeric_features": numeric_features,
        "shared_categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        "per_class_numeric_features": {key: list(value) for key, value in PER_CLASS_NUMERIC_FEATURES.items()},
        "op_type_class_map": {},  # filled by the top-level manifest
        "target_column": TARGET_COLUMN,
        "target_columns": [TARGET_COLUMN, ANALYTICAL_RESIDUAL_TARGET_COLUMN],
        "analytical_feature_columns": list(ANALYTICAL_FEATURE_COLUMNS),
        "all_features": categorical_features + numeric_features,
        "baseline_compare_dir": str(BASELINE_COMPARE_DIR),
    }


def build_classed_dataset_artifacts(
    input_data_dir: Path,
    analytical_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_dataset(input_data_dir)
    analytical_df = load_analytical_features(analytical_dir)
    gemm_df = build_gemm_columns(base_df)

    merged = base_df.merge(
        analytical_df,
        on="row_uid",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        gemm_df,
        on="row_uid",
        how="left",
        validate="one_to_one",
    )

    missing = int(merged["op_class"].isna().sum()) if "op_class" in merged.columns else len(merged)
    if missing > 0:
        raise RuntimeError(f"analytical_calibrated features are missing for {missing} rows")

    merged["op_class"] = merged["op_class"].fillna("mixed_balanced").astype(str)
    merged_path = output_dir / "classed_dataset_full.csv"
    merged.to_csv(merged_path, index=False)

    class_summary: dict[str, Any] = {}
    for op_class in OP_CLASS_ORDER:
        class_dir = datasets_dir / op_class
        class_dir.mkdir(parents=True, exist_ok=True)
        class_df = merged[merged["op_class"] == op_class].copy()
        class_df = class_df.sort_values("_source_order", kind="stable").reset_index(drop=True)
        for split_name in ["train", "val", "test"]:
            split_df = class_df[class_df["split"] == split_name].copy()
            split_df.to_csv(class_dir / f"{split_name}.csv", index=False)
        class_df.to_csv(class_dir / "dataset_full.csv", index=False)

        manifest = feature_manifest_payload(op_class)
        manifest["op_type_class_map"] = dict(dataset_summary_op_type_map(merged))
        manifest["feature_descriptions"] = {
            **{key: FEATURE_DESCRIPTIONS[key] for key in SHARED_CATEGORICAL_FEATURES if key in FEATURE_DESCRIPTIONS},
            **{key: FEATURE_DESCRIPTIONS[key] for key in PER_CLASS_NUMERIC_FEATURES[op_class] if key in FEATURE_DESCRIPTIONS},
            **{key: ANALYTICAL_FEATURE_DESCRIPTIONS[key] for key in PER_CLASS_NUMERIC_FEATURES[op_class] if key in ANALYTICAL_FEATURE_DESCRIPTIONS},
        }
        with (class_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        class_summary[op_class] = {
            "class_dir": str(class_dir),
            "row_count": int(len(class_df)),
            "split_row_counts": {
                split_name: int((class_df["split"] == split_name).sum())
                for split_name in ["train", "val", "test"]
            },
            "numeric_features": list(PER_CLASS_NUMERIC_FEATURES[op_class]),
            "categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        }

    summary_payload = {
        "input_data_dir": str(input_data_dir),
        "analytical_dir": str(analytical_dir),
        "output_dir": str(output_dir),
        "baseline_compare_dir": str(BASELINE_COMPARE_DIR),
        "routing_policy": "static_op_type",
        "op_type_class_map": dataset_summary_op_type_map(merged),
        "shared_categorical_features": list(SHARED_CATEGORICAL_FEATURES),
        "per_class_numeric_features": {key: list(value) for key, value in PER_CLASS_NUMERIC_FEATURES.items()},
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "analytical_feature_descriptions": ANALYTICAL_FEATURE_DESCRIPTIONS,
        "merged_dataset_csv": str(merged_path),
        "classes": class_summary,
    }
    summary_json = output_dir / "dataset_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    return summary_payload


def main() -> None:
    args = parse_args()
    payload = build_classed_dataset_artifacts(
        Path(args.input_data_dir),
        Path(args.analytical_dir),
        Path(args.output_dir),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
