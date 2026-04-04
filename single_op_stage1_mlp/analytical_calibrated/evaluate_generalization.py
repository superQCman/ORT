from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_contract import TARGET_COLUMN  # noqa: E402
import evaluate_analytical_generalization as analytical_eval  # noqa: E402

try:  # noqa: E402
    from .build_analytical_features import (
        add_calibrated_analytical_columns,
        prepare_heavy_prediction_frame,
        rebuild_local_features,
    )
    from .contracts import (
        DEFAULT_INPUT_CSV,
        DEFAULT_OUTPUT_DIR,
        GENERIC_MEMORY_OP_TYPES,
        GENERIC_MIXED_OP_TYPES,
        HEAVY_FAMILIES,
        OP_AWARE_LIGHT_OP_TYPES,
    )
except ImportError:  # noqa: E402
    from build_analytical_features import (
        add_calibrated_analytical_columns,
        prepare_heavy_prediction_frame,
        rebuild_local_features,
    )
    from contracts import (
        DEFAULT_INPUT_CSV,
        DEFAULT_OUTPUT_DIR,
        GENERIC_MEMORY_OP_TYPES,
        GENERIC_MIXED_OP_TYPES,
        HEAVY_FAMILIES,
        OP_AWARE_LIGHT_OP_TYPES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated analytical generalization on dataset_all_no_trace.",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
        help="Input dataset_full.csv. Defaults to dataset_all_no_trace/dataset_full.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory under analytical_calibrated artifacts.",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["leave_one_case_out", "leave_one_combo_out"],
        choices=["leave_one_case_out", "leave_one_combo_out"],
        help="Generalization schemes to run.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Coordinate-descent passes used per fold.",
    )
    return parser.parse_args()


def compute_error_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {
            "mape": 0.0,
            "dwre": 0.0,
            "mae_us": 0.0,
            "rmse_us": 0.0,
        }
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = np.clip(target, a_min=1e-9, a_max=None)
    abs_err = np.abs(pred - target)
    return {
        "mape": float(np.mean(abs_err / denominator)),
        "dwre": float(np.sum(abs_err) / np.sum(denominator)),
        "mae_us": float(np.mean(abs_err)),
        "rmse_us": float(np.sqrt(np.mean(np.square(pred - target)))),
    }


def group_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = compute_error_metrics(
            pd.to_numeric(group[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(group["ana_calib_total_us"], errors="coerce").to_numpy(dtype=float),
        )
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "row_count": int(len(group)),
                "actual_sum_us": float(pd.to_numeric(group[TARGET_COLUMN], errors="coerce").fillna(0.0).sum()),
                **metrics,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_light_proxy(light_metrics_df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    split_df = light_metrics_df[light_metrics_df["split"] == split_name].copy()
    if split_df.empty:
        return {}
    by_op_type = (
        split_df.groupby("op_type", as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            mean_mape=("mape", "mean"),
            max_mape=("mape", "max"),
            mean_dwre=("dwre", "mean"),
            total_actual_us=("actual_sum_us", "sum"),
        )
        .sort_values(["mean_mape", "row_count", "op_type"], ascending=[False, False, True])
    )
    by_class = (
        split_df.groupby("op_class", as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            mean_mape=("mape", "mean"),
            max_mape=("mape", "max"),
            mean_dwre=("dwre", "mean"),
            total_actual_us=("actual_sum_us", "sum"),
        )
        .sort_values(["mean_mape", "row_count", "op_class"], ascending=[False, False, True])
    )
    return {
        "split": split_name,
        "by_op_type": by_op_type.to_dict(orient="records"),
        "by_op_class": by_class.to_dict(orient="records"),
    }


def render_markdown(
    input_csv: Path,
    heavy_summaries: dict[str, dict[str, Any]],
    light_summaries: dict[str, dict[str, Any]],
    params_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Calibrated Analytical Generalization")
    lines.append("")
    lines.append(f"- Input dataset: `{input_csv}`")
    lines.append(f"- Heavy families: `{', '.join(HEAVY_FAMILIES)}`")
    lines.append(f"- Op-aware light families: `{', '.join(OP_AWARE_LIGHT_OP_TYPES)}`")
    lines.append(f"- Generic memory proxy: `{', '.join(GENERIC_MEMORY_OP_TYPES)}`")
    lines.append(f"- Generic mixed proxy: `{', '.join(GENERIC_MIXED_OP_TYPES)}`")
    lines.append("")

    for scheme, summary in heavy_summaries.items():
        lines.append(f"## {scheme}")
        lines.append("")
        test_summary = summary.get("test", {})
        if test_summary:
            lines.append(f"- Heavy test mean macro MAPE: `{test_summary['macro_mape_mean'] * 100.0:.2f}%`")
            lines.append(f"- Heavy test worst fold macro MAPE: `{test_summary['macro_mape_max'] * 100.0:.2f}%`")
            lines.append(f"- Heavy test weighted family MAPE: `{test_summary['weighted_family_mape'] * 100.0:.2f}%`")
            lines.append(f"- Heavy test duration-weighted RE: `{test_summary['duration_weighted_relative_error'] * 100.0:.2f}%`")
            lines.append("")
            lines.append("### Heavy Test Family Metrics")
            lines.append("")
            lines.append("| family | mean MAPE | mean DWRE | max MAPE | folds |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for row in test_summary["family_summary"]:
                lines.append(
                    f"| `{row['family']}` | {row['mean_mape'] * 100.0:.2f}% | "
                    f"{row['mean_dwre'] * 100.0:.2f}% | {row['max_mape'] * 100.0:.2f}% | {int(row['folds'])} |"
                )
            lines.append("")

        light_test = light_summaries.get(scheme, {}).get("test", {})
        if light_test:
            lines.append("### Light Proxy Test Metrics By Op Type")
            lines.append("")
            lines.append("| op_type | rows | mean MAPE | mean DWRE |")
            lines.append("| --- | ---: | ---: | ---: |")
            for row in light_test["by_op_type"]:
                lines.append(
                    f"| `{row['op_type']}` | {int(row['row_count'])} | "
                    f"{row['mean_mape'] * 100.0:.2f}% | {row['mean_dwre'] * 100.0:.2f}% |"
                )
            lines.append("")

        scheme_params = params_df[params_df["scheme"] == scheme].copy()
        if not scheme_params.empty:
            lines.append("### Fold Parameters")
            lines.append("")
            lines.append("| fold | rho_copy_inf | kappa_reduce | rho_gather_inf | rho_fma_inf | rho_tiny_inf | m_stride |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _, row in scheme_params.sort_values("fold").iterrows():
                lines.append(
                    f"| `{row['fold']}` | {row['rho_copy_inf']:.3f} | {row['kappa_reduce']:.3f} | "
                    f"{row['rho_gather_inf']:.3f} | {row['rho_fma_inf']:.3f} | {row['rho_tiny_inf']:.3f} | {row['m_stride']:.3f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def evaluate_generalization(
    input_csv: Path,
    output_dir: Path,
    *,
    schemes: list[str],
    passes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(input_csv, low_memory=False)
    rebuilt_df = rebuild_local_features(raw_df)
    heavy_df = prepare_heavy_prediction_frame(rebuilt_df)

    heavy_metric_rows: list[dict[str, Any]] = []
    light_metric_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    heavy_summaries: dict[str, dict[str, Any]] = {}
    light_summaries: dict[str, dict[str, Any]] = {}

    for scheme in schemes:
        for fold_name, heavy_train_df, heavy_test_df in analytical_eval.build_folds(heavy_df, scheme):
            params = analytical_eval.calibrate_params(
                heavy_train_df,
                passes=passes,
                variant="baseline",
                matmul_formulation="tiny_occ",
            )
            param_rows.append({"scheme": scheme, "fold": fold_name, **params})
            heavy_metric_rows.extend(
                analytical_eval.family_metric_rows(
                    heavy_train_df,
                    params,
                    "baseline",
                    "tiny_occ",
                    scheme,
                    fold_name,
                    "train",
                )
            )
            heavy_metric_rows.extend(
                analytical_eval.family_metric_rows(
                    heavy_test_df,
                    params,
                    "baseline",
                    "tiny_occ",
                    scheme,
                    fold_name,
                    "test",
                )
            )

            if scheme == "leave_one_case_out":
                test_mask = rebuilt_df["case_id"].astype(str) == str(fold_name)
            else:
                test_mask = rebuilt_df["combo"].astype(str) == str(fold_name)
            rebuilt_test = rebuilt_df[test_mask].copy()
            heavy_test_prepared = heavy_df[heavy_df["row_uid"].astype(str).isin(rebuilt_test["row_uid"].astype(str))].copy()
            fold_features = add_calibrated_analytical_columns(rebuilt_test, heavy_test_prepared, params)
            light_family_names = {"generic_memory", "generic_mixed", *OP_AWARE_LIGHT_OP_TYPES}
            light_test = fold_features[fold_features["ana_calib_family"].astype(str).isin(light_family_names)].copy()
            if not light_test.empty:
                group_df = group_metrics(light_test, ["op_class", "op_type"])
                group_df["scheme"] = scheme
                group_df["fold"] = fold_name
                group_df["split"] = "test"
                light_metric_rows.extend(group_df.to_dict(orient="records"))

        scheme_heavy_df = pd.DataFrame([row for row in heavy_metric_rows if row["scheme"] == scheme])
        heavy_summaries[scheme] = {
            "train": analytical_eval.summarize_scheme(scheme_heavy_df, "train"),
            "test": analytical_eval.summarize_scheme(scheme_heavy_df, "test"),
        }
        scheme_light_df = pd.DataFrame([row for row in light_metric_rows if row["scheme"] == scheme])
        light_summaries[scheme] = {
            "test": summarize_light_proxy(scheme_light_df, "test"),
        }

    heavy_metrics_df = pd.DataFrame(heavy_metric_rows)
    light_metrics_df = pd.DataFrame(light_metric_rows)
    params_df = pd.DataFrame(param_rows)

    heavy_metrics_csv = output_dir / "heavy_fold_metrics.csv"
    light_metrics_csv = output_dir / "light_proxy_fold_metrics.csv"
    params_csv = output_dir / "fold_parameters.csv"
    heavy_metrics_df.to_csv(heavy_metrics_csv, index=False)
    light_metrics_df.to_csv(light_metrics_csv, index=False)
    params_df.to_csv(params_csv, index=False)

    summary_payload = {
        "input_csv": str(input_csv),
        "heavy": heavy_summaries,
        "light": light_summaries,
        "output_files": {
            "heavy_metrics_csv": str(heavy_metrics_csv),
            "light_metrics_csv": str(light_metrics_csv),
            "params_csv": str(params_csv),
        },
    }
    summary_json = output_dir / "generalization_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    markdown = render_markdown(input_csv, heavy_summaries, light_summaries, params_df)
    summary_md = output_dir / "generalization_summary.md"
    summary_md.write_text(markdown, encoding="utf-8")
    return {
        "summary": summary_payload,
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "heavy_metrics_csv": str(heavy_metrics_csv),
        "light_metrics_csv": str(light_metrics_csv),
        "params_csv": str(params_csv),
    }


def main() -> None:
    args = parse_args()
    result = evaluate_generalization(
        Path(args.input_csv),
        Path(args.output_dir),
        schemes=list(args.schemes),
        passes=args.passes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
