from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_mlp  # noqa: E402

try:  # noqa: E402
    from .contracts import DEFAULT_OUTPUT_ROOT, DEFAULT_VARIANT_MODE, SUPPORTED_VARIANT_MODES
except ImportError:  # noqa: E402
    from contracts import DEFAULT_OUTPUT_ROOT, DEFAULT_VARIANT_MODE, SUPPORTED_VARIANT_MODES


DEFAULT_SPLITS = ("val", "test")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    dropped_features: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable feature ablation experiments on a prepared dataset directory or on a "
            "classed_op_mlp experiment root."
        ),
    )
    parser.add_argument(
        "--source-experiment-root",
        default="",
        help=(
            "Optional existing experiment root containing datasets/<model-group> and "
            "models/<model-group>, for example classed_op_mlp_test_2_analytical_5_200_iter."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional explicit dataset directory containing train.csv, val.csv, test.csv, and feature_columns.json.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        default="",
        help="Optional explicit baseline model directory containing metrics.json and prediction CSVs.",
    )
    parser.add_argument(
        "--model-group",
        default="",
        help="Logical model group name, for example gather. Required when --source-experiment-root is used.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for the ablation run. Defaults to artifacts/latest/feature_ablation/<source>/<group>.",
    )
    parser.add_argument(
        "--ablation-feature",
        action="append",
        default=[],
        help="Feature to ablate. Repeat this flag to study multiple features.",
    )
    parser.add_argument(
        "--variant-mode",
        choices=list(SUPPORTED_VARIANT_MODES),
        default=DEFAULT_VARIANT_MODE,
        help="How to build variants from --ablation-feature when --variant is not enough.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Custom variant in the form name=feat_a,feat_b . Use name= for a custom baseline-like variant.",
    )
    parser.add_argument(
        "--summary-splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Prediction splits to summarize and compare against baseline.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain variant models even if metrics.json already exists under the output directory.",
    )
    parser.add_argument(
        "--disable-reuse-source-baseline",
        action="store_true",
        help="Force retraining the baseline instead of reusing the source model directory when available.",
    )
    parser.add_argument(
        "--hidden-layers",
        default="",
        help="Optional comma-separated hidden sizes. Defaults to baseline metrics.json when available.",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Optional override for training batch size.")
    parser.add_argument("--max-iter", type=int, default=0, help="Optional override for max epochs.")
    parser.add_argument("--alpha", type=float, default=-1.0, help="Optional override for AdamW weight decay.")
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=-1.0,
        help="Optional override for optimizer learning rate.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="Optional override for training seed.")
    parser.add_argument(
        "--train-device",
        default="auto",
        help="Training device passed to train_mlp.py. Defaults to auto.",
    )
    parser.add_argument("--npu-device-id", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument(
        "--disable-log-target",
        action="store_true",
        help="Train directly on microseconds instead of inheriting log-target behavior from the baseline.",
    )
    parser.add_argument(
        "--enable-onnx-export",
        action="store_true",
        help="Export ONNX for each trained ablation variant. Disabled by default to keep the experiment light.",
    )
    parser.add_argument("--onnx-opset", type=int, default=17)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    source_experiment_root = Path(args.source_experiment_root) if args.source_experiment_root else None
    data_dir = Path(args.data_dir) if args.data_dir else None
    baseline_model_dir = Path(args.baseline_model_dir) if args.baseline_model_dir else None
    model_group = str(args.model_group).strip()

    if source_experiment_root is not None:
        if not model_group:
            raise ValueError("--model-group is required when --source-experiment-root is used")
        if data_dir is None:
            data_dir = source_experiment_root / "datasets" / model_group
        if baseline_model_dir is None:
            candidate = source_experiment_root / "models" / model_group
            if candidate.exists():
                baseline_model_dir = candidate

    if data_dir is None:
        raise ValueError("Either --data-dir or --source-experiment-root must be provided")
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    if not model_group:
        model_group = data_dir.name
    return data_dir, baseline_model_dir, model_group


def resolve_output_dir(args: argparse.Namespace, data_dir: Path, model_group: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    source_root = Path(args.source_experiment_root) if args.source_experiment_root else data_dir.parent
    source_name = source_root.name
    return DEFAULT_OUTPUT_ROOT / source_name / model_group


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_feature_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "feature_columns.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return load_json(manifest_path)


def load_baseline_metrics(baseline_model_dir: Path | None) -> dict[str, Any] | None:
    if baseline_model_dir is None:
        return None
    metrics_path = baseline_model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return load_json(metrics_path)


def parse_variant_definition(text: str) -> VariantSpec:
    if "=" not in text:
        raise ValueError(f"Variant must be in name=feature_a,feature_b form: {text!r}")
    name, raw_features = text.split("=", 1)
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError(f"Variant name cannot be empty: {text!r}")
    features = tuple(sorted({item.strip() for item in raw_features.split(",") if item.strip()}))
    return VariantSpec(name=cleaned_name, dropped_features=features)


def build_variants(
    base_numeric_features: list[str],
    ablation_features: list[str],
    custom_variants: list[str],
    *,
    variant_mode: str,
) -> list[VariantSpec]:
    available = set(base_numeric_features)
    cleaned_ablation_features = []
    for feature in ablation_features:
        candidate = str(feature).strip()
        if not candidate:
            continue
        if candidate not in available:
            raise KeyError(f"Ablation feature {candidate!r} is not in the baseline numeric feature list")
        if candidate not in cleaned_ablation_features:
            cleaned_ablation_features.append(candidate)

    variants: list[VariantSpec] = [VariantSpec(name="baseline", dropped_features=tuple())]
    if cleaned_ablation_features and variant_mode == DEFAULT_VARIANT_MODE:
        for feature in cleaned_ablation_features:
            variants.append(VariantSpec(name=f"drop_{feature}", dropped_features=(feature,)))
        variants.append(
            VariantSpec(
                name="drop_all_selected",
                dropped_features=tuple(sorted(cleaned_ablation_features)),
            )
        )
    elif cleaned_ablation_features and variant_mode not in SUPPORTED_VARIANT_MODES:
        raise ValueError(f"Unsupported variant_mode={variant_mode!r}")

    for item in custom_variants:
        variants.append(parse_variant_definition(item))

    deduped: list[VariantSpec] = []
    seen_names: set[str] = set()
    for variant in variants:
        if variant.name in seen_names:
            raise ValueError(f"Duplicate variant name: {variant.name}")
        seen_names.add(variant.name)
        deduped.append(variant)
    return deduped


def prepare_variant_manifest(
    base_manifest: dict[str, Any],
    *,
    model_group: str,
    variant: VariantSpec,
) -> dict[str, Any]:
    manifest = copy.deepcopy(base_manifest)
    categorical_features = list(manifest.get("categorical_features", []))
    base_numeric_features = list(manifest.get("numeric_features", []))
    dropped = set(variant.dropped_features)
    kept_numeric_features = [feature for feature in base_numeric_features if feature not in dropped]
    manifest["numeric_features"] = kept_numeric_features
    manifest["analysis_numeric_features"] = kept_numeric_features
    manifest["all_features"] = categorical_features + kept_numeric_features
    if "per_model_group_numeric_features" in manifest and isinstance(manifest["per_model_group_numeric_features"], dict):
        manifest["per_model_group_numeric_features"][model_group] = kept_numeric_features
    if "per_class_numeric_features" in manifest and isinstance(manifest["per_class_numeric_features"], dict):
        manifest["per_class_numeric_features"][model_group] = kept_numeric_features
    manifest["ablation"] = {
        "variant_name": variant.name,
        "model_group": model_group,
        "dropped_features": list(variant.dropped_features),
        "kept_numeric_features": kept_numeric_features,
    }
    return manifest


def copy_variant_dataset(
    source_data_dir: Path,
    target_data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    target_data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["train.csv", "val.csv", "test.csv", "dataset_full.csv"]:
        source_path = source_data_dir / filename
        if source_path.exists():
            frame = pd.read_csv(source_path, low_memory=False)
            frame.to_csv(target_data_dir / filename, index=False)
    with (target_data_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def resolve_training_config(args: argparse.Namespace, baseline_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = baseline_metrics or {}
    hidden_layers = (
        train_mlp.parse_hidden_layers(args.hidden_layers)
        if args.hidden_layers
        else tuple(metrics.get("hidden_layers", [128, 128, 128, 128, 128]))
    )
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(metrics.get("batch_size", 1024))
    max_iter = int(args.max_iter) if args.max_iter > 0 else int(metrics.get("max_iter", 120))
    alpha = float(args.alpha) if args.alpha >= 0.0 else float(metrics.get("alpha", 1e-4))
    learning_rate_init = (
        float(args.learning_rate_init)
        if args.learning_rate_init >= 0.0
        else float(metrics.get("learning_rate_init", 1e-3))
    )
    seed = int(args.seed) if args.seed >= 0 else int(metrics.get("seed", 42))
    inherited_log_target = bool(metrics.get("log_target_requested", True))
    return {
        "hidden_layers": hidden_layers,
        "batch_size": batch_size,
        "max_iter": max_iter,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "seed": seed,
        "log_target": False if args.disable_log_target else inherited_log_target,
        "train_device": str(args.train_device).strip() or "auto",
        "npu_device_id": int(args.npu_device_id),
        "early_stopping_patience": int(args.early_stopping_patience),
        "export_onnx": bool(args.enable_onnx_export),
        "onnx_opset": int(args.onnx_opset),
    }


def load_predictions(model_dir: Path, split_name: str) -> pd.DataFrame:
    path = model_dir / f"predictions_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame["row_uid"] = frame["row_uid"].astype(str)
    return frame


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compute_delta(current: float, baseline: float, *, invert: bool = False) -> float:
    if np.isnan(current) or np.isnan(baseline):
        return float("nan")
    return (baseline - current) if invert else (current - baseline)


def compute_relative_change(current: float, baseline: float, *, invert: bool = False) -> float:
    if np.isnan(current) or np.isnan(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    raw = (baseline - current) if invert else (current - baseline)
    return raw / baseline


def paired_error_delta_summary(baseline_model_dir: Path, model_dir: Path, split_name: str) -> dict[str, float]:
    baseline = load_predictions(baseline_model_dir, split_name)
    current = load_predictions(model_dir, split_name)
    joined = baseline.merge(
        current,
        on="row_uid",
        how="inner",
        suffixes=("_baseline", "_variant"),
        validate="one_to_one",
    )
    if joined.empty:
        return {
            "paired_row_count": 0,
            "mean_abs_error_us_delta": float("nan"),
            "median_abs_error_us_delta": float("nan"),
            "mean_ape_delta": float("nan"),
            "median_ape_delta": float("nan"),
            "improved_row_fraction": float("nan"),
            "worsened_row_fraction": float("nan"),
        }
    abs_error_delta = (
        pd.to_numeric(joined["abs_error_us_variant"], errors="coerce")
        - pd.to_numeric(joined["abs_error_us_baseline"], errors="coerce")
    )
    ape_delta = pd.to_numeric(joined["ape_variant"], errors="coerce") - pd.to_numeric(joined["ape_baseline"], errors="coerce")
    return {
        "paired_row_count": int(len(joined)),
        "mean_abs_error_us_delta": float(abs_error_delta.mean()),
        "median_abs_error_us_delta": float(abs_error_delta.median()),
        "mean_ape_delta": float(ape_delta.mean()),
        "median_ape_delta": float(ape_delta.median()),
        "improved_row_fraction": float((abs_error_delta < 0.0).mean()),
        "worsened_row_fraction": float((abs_error_delta > 0.0).mean()),
    }


def summarize_variant(
    variant: VariantSpec,
    metrics_payload: dict[str, Any],
    baseline_metrics_payload: dict[str, Any],
    baseline_model_dir: Path,
    model_dir: Path,
    summary_splits: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant.name,
        "dropped_features": ",".join(variant.dropped_features),
        "dropped_feature_count": len(variant.dropped_features),
        "numeric_feature_count": len(metrics_payload.get("numeric_features", [])),
    }
    for split_name in summary_splits:
        metrics = metrics_payload.get("metrics", {}).get(split_name, {})
        baseline_metrics = baseline_metrics_payload.get("metrics", {}).get(split_name, {})
        for metric_name in ["mae_us", "rmse_us", "r2", "mape", "median_ape"]:
            current_value = safe_float(metrics.get(metric_name))
            baseline_value = safe_float(baseline_metrics.get(metric_name))
            row[f"{split_name}_{metric_name}"] = current_value
            row[f"{split_name}_{metric_name}_delta_vs_baseline"] = compute_delta(
                current=current_value,
                baseline=baseline_value,
                invert=(metric_name == "r2"),
            )
            row[f"{split_name}_{metric_name}_relative_change_vs_baseline"] = compute_relative_change(
                current=current_value,
                baseline=baseline_value,
                invert=(metric_name == "r2"),
            )
        paired = paired_error_delta_summary(baseline_model_dir, model_dir, split_name)
        for key, value in paired.items():
            row[f"{split_name}_{key}"] = value
    return row


def build_summary_markdown(summary_df: pd.DataFrame, summary_splits: list[str]) -> str:
    if summary_df.empty:
        return "# Feature Ablation Summary\n\nNo variants were executed.\n"
    lines = ["# Feature Ablation Summary", ""]
    for split_name in summary_splits:
        metric_column = f"{split_name}_mape_delta_vs_baseline"
        if metric_column not in summary_df.columns:
            continue
        ranked = summary_df.copy()
        ranked = ranked.sort_values(metric_column, ascending=True, na_position="last").reset_index(drop=True)
        lines.append(f"## {split_name}")
        for _, row in ranked.iterrows():
            lines.append(
                (
                    f"- `{row['variant']}`: dropped=`{row['dropped_features'] or 'none'}`, "
                    f"{split_name}_mape={safe_float(row.get(f'{split_name}_mape')):.6f}, "
                    f"delta_vs_baseline={safe_float(row.get(metric_column)):.6f}, "
                    f"improved_row_fraction={safe_float(row.get(f'{split_name}_improved_row_fraction')):.6f}"
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def train_variant(
    variant: VariantSpec,
    source_data_dir: Path,
    output_dir: Path,
    base_manifest: dict[str, Any],
    *,
    model_group: str,
    training_config: dict[str, Any],
    force_retrain: bool,
) -> dict[str, Any]:
    variant_root = output_dir / "variants" / variant.name
    data_dir = variant_root / "data"
    model_dir = variant_root / "model"
    manifest = prepare_variant_manifest(base_manifest, model_group=model_group, variant=variant)
    copy_variant_dataset(source_data_dir, data_dir, manifest)

    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists() and not force_retrain:
        return load_json(metrics_path)

    return train_mlp.train_model(
        data_dir=data_dir,
        output_dir=model_dir,
        hidden_layers=training_config["hidden_layers"],
        batch_size=training_config["batch_size"],
        max_iter=training_config["max_iter"],
        alpha=training_config["alpha"],
        learning_rate_init=training_config["learning_rate_init"],
        seed=training_config["seed"],
        log_target=training_config["log_target"],
        target_mode="direct_us",
        train_device=training_config["train_device"],
        npu_device_id=training_config["npu_device_id"],
        early_stopping_patience=training_config["early_stopping_patience"],
        export_onnx=training_config["export_onnx"],
        onnx_opset=training_config["onnx_opset"],
    )


def main() -> None:
    args = parse_args()
    data_dir, baseline_model_dir, model_group = resolve_inputs(args)
    output_dir = resolve_output_dir(args, data_dir, model_group)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_feature_manifest(data_dir)
    base_numeric_features = list(base_manifest.get("numeric_features", []))
    variants = build_variants(
        base_numeric_features=base_numeric_features,
        ablation_features=list(args.ablation_feature),
        custom_variants=list(args.variant),
        variant_mode=args.variant_mode,
    )
    baseline_metrics = load_baseline_metrics(baseline_model_dir)
    training_config = resolve_training_config(args, baseline_metrics)
    reuse_source_baseline = baseline_model_dir is not None and not args.disable_reuse_source_baseline

    variant_metrics: dict[str, dict[str, Any]] = {}
    variant_model_dirs: dict[str, Path] = {}
    for variant in variants:
        if variant.name == "baseline" and reuse_source_baseline and baseline_metrics is not None:
            variant_metrics[variant.name] = baseline_metrics
            variant_model_dirs[variant.name] = baseline_model_dir
            continue
        trained_metrics = train_variant(
            variant=variant,
            source_data_dir=data_dir,
            output_dir=output_dir,
            base_manifest=base_manifest,
            model_group=model_group,
            training_config=training_config,
            force_retrain=bool(args.force_retrain),
        )
        variant_metrics[variant.name] = trained_metrics
        variant_model_dirs[variant.name] = output_dir / "variants" / variant.name / "model"

    if "baseline" not in variant_metrics:
        raise RuntimeError("Missing baseline variant result")

    baseline_metrics_payload = variant_metrics["baseline"]
    baseline_model_dir_resolved = variant_model_dirs["baseline"]
    summary_rows = [
        summarize_variant(
            variant=variant,
            metrics_payload=variant_metrics[variant.name],
            baseline_metrics_payload=baseline_metrics_payload,
            baseline_model_dir=baseline_model_dir_resolved,
            model_dir=variant_model_dirs[variant.name],
            summary_splits=list(args.summary_splits),
        )
        for variant in variants
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["test_mape_delta_vs_baseline", "val_mape_delta_vs_baseline", "variant"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    summary_csv = output_dir / "ablation_summary.csv"
    summary_json = output_dir / "ablation_summary.json"
    summary_md = output_dir / "ablation_summary.md"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "source_experiment_root": str(args.source_experiment_root),
        "data_dir": str(data_dir),
        "baseline_model_dir": "" if baseline_model_dir is None else str(baseline_model_dir),
        "reuse_source_baseline": reuse_source_baseline,
        "model_group": model_group,
        "output_dir": str(output_dir),
        "training_config": {
            **training_config,
            "hidden_layers": list(training_config["hidden_layers"]),
        },
        "ablation_features": list(args.ablation_feature),
        "variant_mode": args.variant_mode,
        "variants": [
            {
                "name": variant.name,
                "dropped_features": list(variant.dropped_features),
                "model_dir": str(variant_model_dirs[variant.name]),
                "metrics_json": str(variant_model_dirs[variant.name] / "metrics.json"),
            }
            for variant in variants
        ],
        "summary_splits": list(args.summary_splits),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "summary_rows": summary_rows,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    summary_md.write_text(build_summary_markdown(summary_df, list(args.summary_splits)), encoding="utf-8")

    print(f"ablation_summary_csv={summary_csv}")
    print(f"ablation_summary_json={summary_json}")
    print(f"ablation_summary_md={summary_md}")


if __name__ == "__main__":
    main()
