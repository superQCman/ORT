#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_utils import add_engineered_features, add_operator_hardware_context, add_real_targets, load_selected_feature_rows
from gem5_utils import collect_gem5_label_rows
from hardware_utils import flatten_hardware_features, load_hardware_profile
from model_utils import ensure_parent_dir, split_combo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training-ready operator dataset for ORT DLRM CPU modeling.")
    parser.add_argument(
        "--features-root",
        default="/data/qc/dlrm/ORT/features_selected",
        help="Directory containing bs*_nip*.csv compact feature files",
    )
    parser.add_argument(
        "--hw-profile",
        default="/data/qc/dlrm/ORT/model/hardware_profiles/kunpeng920_gem5.yaml",
        help="YAML hardware profile whose fields are appended as hw_* features",
    )
    parser.add_argument(
        "--hardware-name",
        default="kunpeng920_gem5",
        help="Logical hardware name written into the dataset",
    )
    parser.add_argument(
        "--gem5-root",
        action="append",
        default=[],
        help="Optional root containing gem5 stats.txt files; can be passed multiple times",
    )
    parser.add_argument(
        "--gem5-default-combo",
        default="",
        help="Fallback combo for gem5 roots that do not encode bs*_nip* in the path",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the merged training dataset CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features_root = Path(args.features_root)
    hw_profile = load_hardware_profile(Path(args.hw_profile))
    hw_features = flatten_hardware_features(hw_profile)

    df = load_selected_feature_rows(features_root)
    df = add_engineered_features(df)
    df["hardware_name"] = args.hardware_name
    df["sample_group"] = df["hardware_name"].astype(str) + "::" + df["combo"].astype(str)
    for key, value in hw_features.items():
        df[key] = value
    df = add_operator_hardware_context(df)

    hw_clock_ghz = hw_features.get("hw_core_cpu_clock")
    if isinstance(hw_clock_ghz, (int, float)) and hw_clock_ghz > 0:
        df = add_real_targets(df, float(hw_clock_ghz))
    else:
        df = add_real_targets(df, None)

    gem5_roots = [Path(root) for root in args.gem5_root]
    if gem5_roots:
        gem5_df = collect_gem5_label_rows(
            gem5_roots,
            default_combo=args.gem5_default_combo or None,
        )
        if not gem5_df.empty:
            df = df.merge(
                gem5_df,
                on=["combo", "op_idx"],
                how="left",
                suffixes=("", "_gem5"),
            )

    ensure_parent_dir(Path(args.output_csv))
    df.to_csv(args.output_csv, index=False)

    gem5_count = int(df["label_gem5_sim_seconds"].notna().sum()) if "label_gem5_sim_seconds" in df.columns else 0
    print(f"rows={len(df)}")
    print(f"real_target_rows={int(df['label_real_dur_us'].notna().sum())}")
    print(f"gem5_target_rows={gem5_count}")
    print(args.output_csv)


if __name__ == "__main__":
    main()
