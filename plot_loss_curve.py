from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-epoch training/validation loss curves for single-op MLP.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model output directory containing metrics.json and/or training_history.csv.",
    )
    parser.add_argument(
        "--output-png",
        default="",
        help="Optional explicit output PNG path. Defaults to <model-dir>/loss_curve.png.",
    )
    return parser.parse_args()


def import_matplotlib_pyplot() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_history(model_dir: Path) -> tuple[pd.DataFrame, str]:
    history_csv = model_dir / "training_history.csv"
    if history_csv.exists():
        return pd.read_csv(history_csv), "training_history.csv"

    metrics_json = model_dir / "metrics.json"
    if not metrics_json.exists():
        raise FileNotFoundError(f"Neither {history_csv} nor {metrics_json} exists")

    payload = json.loads(metrics_json.read_text())
    if "training_history" in payload:
        return pd.DataFrame(payload["training_history"]), "metrics.json:training_history"
    if "training_history_tail" in payload:
        return pd.DataFrame(payload["training_history_tail"]), "metrics.json:training_history_tail"
    raise RuntimeError("No training history found in metrics.json")


def plot_history(history_df: pd.DataFrame, output_png: Path) -> None:
    if history_df.empty:
        raise RuntimeError("Training history is empty")

    plt = import_matplotlib_pyplot()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_df["epoch"], history_df["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        axis.plot(history_df["epoch"], history_df["val_loss"], label="val_loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("MLP Training Loss Curve")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_png = Path(args.output_png) if args.output_png else model_dir / "loss_curve.png"
    history_df, history_source = load_history(model_dir)
    plot_history(history_df, output_png)
    print(f"output_png={output_png}")
    print(f"history_source={history_source}")
    print(f"epochs_plotted={len(history_df)}")


if __name__ == "__main__":
    main()
