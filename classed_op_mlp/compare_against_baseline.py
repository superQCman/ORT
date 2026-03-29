from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_op_type_metrics import (  # noqa: E402
    combo_op_type_total_duration_weighted_mape,
    enrich_with_metadata,
    per_op_type_metrics,
    resolve_data_dir,
    summarize_frame,
)

from analytical_calibrated.contracts import BASELINE_COMPARE_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the classed-op MLP results against model_all_no_trace.",
    )
    parser.add_argument(
        "--classed-model-root",
        default=str(PROJECT_ROOT / "artifacts" / "latest" / "classed_op_mlp" / "models"),
        help="Model root produced by train_class_models.py.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(BASELINE_COMPARE_DIR),
        help="Baseline model directory. Defaults to model_all_no_trace.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit comparison output dir. Defaults to <classed-model-root>/comparison.",
    )
    return parser.parse_args()


def _load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    return frame


def build_overall_comparison(
    classed_root: Path,
    baseline_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    markdown_lines = [
        "# Classed MLP vs model_all_no_trace",
        "",
        f"- Classed model root: `{classed_root}`",
        f"- Baseline model root: `{baseline_dir}`",
        "",
        "## Overall Metrics",
        "",
        "| split | model | MAE (us) | RMSE (us) | R2 | MAPE | median APE | combo-op duration-weighted MAPE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split_name in ["train", "val", "test"]:
        classed_path = classed_root / "combined" / f"combined_predictions_{split_name}.csv"
        baseline_path = baseline_dir / f"predictions_{split_name}.csv"
        classed = _load_prediction(classed_path)
        baseline = _load_prediction(baseline_path)
        if "op_type" not in baseline.columns or "combo" not in baseline.columns:
            baseline = enrich_with_metadata(
                baseline,
                resolve_data_dir(baseline_dir, ""),
                split_name,
            )

        classed_summary = summarize_frame(classed)
        classed_summary["combo_op_type_total_duration_weighted_mape"] = combo_op_type_total_duration_weighted_mape(classed)
        baseline_summary = summarize_frame(baseline)
        baseline_summary["combo_op_type_total_duration_weighted_mape"] = combo_op_type_total_duration_weighted_mape(baseline)

        for model_name, summary in [("classed_op_mlp", classed_summary), ("model_all_no_trace", baseline_summary)]:
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    **summary,
                }
            )
            markdown_lines.append(
                f"| `{split_name}` | `{model_name}` | {summary['mae_us']:.3f} | {summary['rmse_us']:.3f} | "
                f"{summary['r2']:.6f} | {summary['mape'] * 100.0:.2f}% | {summary['median_ape'] * 100.0:.2f}% | "
                f"{summary['combo_op_type_total_duration_weighted_mape'] * 100.0:.2f}% |"
            )
        markdown_lines.append("")

        classed_op = per_op_type_metrics(classed).rename(
            columns={
                "row_count": "classed_row_count",
                "mae_us": "classed_mae_us",
                "rmse_us": "classed_rmse_us",
                "r2": "classed_r2",
                "mape": "classed_mape",
                "median_ape": "classed_median_ape",
                "p90_ape": "classed_p90_ape",
            }
        )
        baseline_op = per_op_type_metrics(baseline).rename(
            columns={
                "row_count": "baseline_row_count",
                "mae_us": "baseline_mae_us",
                "rmse_us": "baseline_rmse_us",
                "r2": "baseline_r2",
                "mape": "baseline_mape",
                "median_ape": "baseline_median_ape",
                "p90_ape": "baseline_p90_ape",
            }
        )
        per_op = classed_op.merge(
            baseline_op,
            on="op_type",
            how="outer",
        )
        for metric in ["mae_us", "rmse_us", "mape", "median_ape", "p90_ape"]:
            classed_column = f"classed_{metric}"
            baseline_column = f"baseline_{metric}"
            per_op[f"delta_{metric}"] = per_op[classed_column] - per_op[baseline_column]
        per_op_csv = output_dir / f"per_op_type_comparison_{split_name}.csv"
        per_op.to_csv(per_op_csv, index=False)

    overall_df = pd.DataFrame(rows)
    overall_csv = output_dir / "overall_comparison.csv"
    overall_df.to_csv(overall_csv, index=False)
    markdown = "\n".join(markdown_lines) + "\n"
    markdown_path = output_dir / "comparison_summary.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    payload = {
        "overall_csv": str(overall_csv),
        "markdown_path": str(markdown_path),
        "comparison_dir": str(output_dir),
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    args = parse_args()
    classed_root = Path(args.classed_model_root)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else classed_root / "comparison"
    payload = build_overall_comparison(classed_root, baseline_dir, output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
