from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "latest"
    / "classed_op_mlp"
)
DEFAULT_TARGET_COLUMN = "label_operator_actual_dur_us"
DEFAULT_SPLIT = "test"
DEFAULT_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the analytical feature columns that are currently active inputs "
            "for classed_op_mlp and export one CSV summary."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Root directory produced by classed_op_mlp/build_classed_dataset.py or run_pipeline.py.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the single CSV output file.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split used for validation. Defaults to test.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COLUMN,
        help="Latency target column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="MAPE pass threshold. Defaults to 0.30.",
    )
    return parser.parse_args()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mape(actual: pd.Series, pred: pd.Series) -> float:
    y = _safe_numeric(actual).to_numpy(dtype=float)
    x = _safe_numeric(pred).to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    denom = np.clip(y[mask], a_min=1e-9, a_max=None)
    return float(np.mean(np.abs(x[mask] - y[mask]) / denom))


def _pearson(actual: pd.Series, pred: pd.Series) -> float:
    y = _safe_numeric(actual).to_numpy(dtype=float)
    x = _safe_numeric(pred).to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if int(np.sum(mask)) <= 1:
        return float("nan")
    if np.allclose(y[mask], y[mask][0]) or np.allclose(x[mask], x[mask][0]):
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def load_group_manifest(group_dir: Path) -> dict[str, Any]:
    manifest_path = group_dir / "feature_columns.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_group(
    datasets_dir: Path,
    model_group: str,
    *,
    split: str,
    target_col: str,
    threshold: float,
) -> list[dict[str, Any]]:
    group_dir = datasets_dir / model_group
    manifest = load_group_manifest(group_dir)
    numeric_features = list(manifest.get("numeric_features", []))
    active_analytical = [column for column in numeric_features if column.startswith("ana_calib_")]
    split_csv = group_dir / f"{split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(split_csv)
    frame = pd.read_csv(split_csv, low_memory=False)

    if not active_analytical:
        return [
            {
                "model_group": model_group,
                "split": split,
                "feature": "",
                "rows": int(len(frame)),
                "mape_vs_actual": float("nan"),
                "pearson_r": float("nan"),
                "threshold": float(threshold),
                "passes_threshold": True,
                "status": "no_active_analytical_features",
            }
        ]

    rows: list[dict[str, Any]] = []
    for feature in active_analytical:
        feature_mape = _mape(frame[target_col], frame[feature])
        rows.append(
            {
                "model_group": model_group,
                "split": split,
                "feature": feature,
                "rows": int(len(frame)),
                "mape_vs_actual": feature_mape,
                "pearson_r": _pearson(frame[target_col], frame[feature]),
                "threshold": float(threshold),
                "passes_threshold": bool(np.isfinite(feature_mape) and feature_mape < threshold),
                "status": "active_analytical_input",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    datasets_dir = data_root / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(datasets_dir)

    group_dirs = sorted(path for path in datasets_dir.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for group_dir in group_dirs:
        rows.extend(
            validate_group(
                datasets_dir,
                group_dir.name,
                split=args.split,
                target_col=args.target_col,
                threshold=args.threshold,
            )
        )

    result = pd.DataFrame(rows)
    status_order = {"active_analytical_input": 0, "no_active_analytical_features": 1}
    result["_status_order"] = result["status"].map(status_order).fillna(99)
    result = result.sort_values(["_status_order", "model_group", "feature"], kind="stable").drop(columns=["_status_order"])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(output_csv)


if __name__ == "__main__":
    main()
