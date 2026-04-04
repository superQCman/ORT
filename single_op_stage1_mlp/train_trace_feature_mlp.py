from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from trace_feature_contract import (
    ALL_TRACE_PROXY_TARGET_COLUMNS,
    DEFAULT_TRACE_PROXY_TARGET_COLUMNS,
    TRACE_PROXY_CATEGORICAL_FEATURES,
    TRACE_PROXY_INPUT_COLUMNS,
    TRACE_PROXY_LOG_SCALE_TARGET_COLUMNS,
    TRACE_PROXY_METADATA_COLUMNS,
    TRACE_PROXY_NUMERIC_FEATURES,
)


MISSING_CATEGORICAL_TOKEN = "__missing__"
UNKNOWN_CATEGORICAL_TOKEN = "__unknown__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a separate PyTorch MLP to predict trace-derived features from non-trace inputs.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv from the prepared dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the saved trace-feature model artifacts.",
    )
    parser.add_argument(
        "--hidden-layers",
        default="256,128",
        help="Comma-separated hidden layer sizes.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=120, help="Maximum training epochs.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Weight decay used by AdamW.")
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-device",
        default="auto",
        help="PyTorch training device: auto, cpu, cuda, or npu.",
    )
    parser.add_argument(
        "--npu-device-id",
        type=int,
        default=0,
        help="Ascend NPU device id used when train-device resolves to npu.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop after this many epochs without validation improvement.",
    )
    parser.add_argument(
        "--disable-log-targets",
        action="store_true",
        help="Disable the default log1p transform on wide-range positive trace targets.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="*",
        default=None,
        help=(
            "Optional explicit trace target list. Supports space-separated or comma-separated "
            f"columns from: {', '.join(ALL_TRACE_PROXY_TARGET_COLUMNS)}"
        ),
    )
    return parser.parse_args()


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    cleaned = [part.strip() for part in str(text).split(",") if part.strip()]
    if not cleaned:
        raise ValueError("At least one hidden layer size is required")
    return tuple(int(part) for part in cleaned)


def split_target_tokens(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        pieces = [piece.strip() for piece in str(value).replace(",", " ").split()]
        tokens.extend([piece for piece in pieces if piece])
    return tokens


def _import_torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for train_trace_feature_mlp.py. Install it first, for example: pip install torch"
        ) from exc
    try:
        import torch_npu
    except ImportError:
        torch_npu = None
    return torch, nn, DataLoader, TensorDataset, torch_npu


def sanitize_categorical_series(series: pd.Series) -> pd.Series:
    text = series.fillna(MISSING_CATEGORICAL_TOKEN).astype(str).str.strip()
    return text.replace({"": MISSING_CATEGORICAL_TOKEN, "nan": MISSING_CATEGORICAL_TOKEN, "None": MISSING_CATEGORICAL_TOKEN})


def configure_torch_npu(torch: Any, torch_npu: Any, npu_device_id: int) -> None:
    if torch_npu is None:
        raise RuntimeError(
            "Requested NPU training, but torch_npu is not installed. "
            "Please install the Ascend PyTorch stack first."
        )
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device_name = f"npu:{int(npu_device_id)}"
    if hasattr(torch_npu.npu, "set_device"):
        torch_npu.npu.set_device(device_name)
    elif hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        torch.npu.set_device(device_name)


def select_training_device(torch: Any, torch_npu: Any, requested_device: str, npu_device_id: int) -> Any:
    choice = str(requested_device).strip().lower()
    if choice in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch_npu is not None and hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
            configure_torch_npu(torch, torch_npu, npu_device_id)
            return torch.device(f"npu:{int(npu_device_id)}")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested train_device=cuda, but CUDA is not available")
        return torch.device("cuda")
    if choice == "npu":
        configure_torch_npu(torch, torch_npu, npu_device_id)
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available()):
            raise RuntimeError("Requested train_device=npu, but PyTorch NPU support is not available after importing torch_npu")
        return torch.device(f"npu:{int(npu_device_id)}")
    return torch.device(choice)


def load_split_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for split_name in ["train", "val", "test"]:
        path = data_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        tables[split_name] = pd.read_csv(path)
    return tables


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_target_columns(tables: dict[str, pd.DataFrame], requested_targets: list[str] | None) -> list[str]:
    available_columns = set(tables["train"].columns)
    requested = split_target_tokens(requested_targets)
    if requested:
        unknown = [column for column in requested if column not in ALL_TRACE_PROXY_TARGET_COLUMNS]
        if unknown:
            raise ValueError(f"Unsupported trace target columns requested: {unknown}")
        missing = [column for column in requested if column not in available_columns]
        if missing:
            raise ValueError(f"Requested trace target columns are not present in the dataset: {missing}")
        return requested

    defaults = [column for column in DEFAULT_TRACE_PROXY_TARGET_COLUMNS if column in available_columns]
    if not defaults:
        raise RuntimeError(
            "None of the default trace target columns were found in the dataset. "
            "Pass --target-columns explicitly if you prepared a custom dataset."
        )
    return defaults


def filter_complete_target_rows(
    tables: dict[str, pd.DataFrame],
    target_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    filtered: dict[str, pd.DataFrame] = {}
    stats: dict[str, dict[str, int]] = {}
    for split_name, frame in tables.items():
        work = frame.copy()
        target_frame = work[target_columns].apply(pd.to_numeric, errors="coerce")
        keep_mask = target_frame.notna().all(axis=1)
        filtered[split_name] = work.loc[keep_mask].reset_index(drop=True)
        stats[split_name] = {
            "rows_before": int(len(work)),
            "rows_after": int(len(filtered[split_name])),
            "rows_dropped_missing_targets": int((~keep_mask).sum()),
        }
    return filtered, stats


def fit_input_preprocessor_state(
    train_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    numeric_stats: list[dict[str, Any]] = []
    categorical_stats: list[dict[str, Any]] = []
    transformed_feature_names: list[str] = []

    for column in numeric_features:
        if column in train_df.columns:
            values = pd.to_numeric(train_df[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=train_df.index, dtype=float)
        median = _safe_float(values.median(), 0.0)
        filled = values.fillna(median).astype(float)
        mean = _safe_float(filled.mean(), 0.0)
        std = _safe_float(filled.std(ddof=0), 1.0)
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        numeric_stats.append({"name": column, "median": median, "mean": mean, "std": std})
        transformed_feature_names.append(column)

    offset = len(numeric_stats)
    for column in categorical_features:
        if column in train_df.columns:
            values = sanitize_categorical_series(train_df[column])
        else:
            values = pd.Series(MISSING_CATEGORICAL_TOKEN, index=train_df.index, dtype=str)
        unique_values = sorted({value for value in values.tolist() if value != UNKNOWN_CATEGORICAL_TOKEN})
        vocabulary = [UNKNOWN_CATEGORICAL_TOKEN, *unique_values]
        categorical_stats.append(
            {
                "name": column,
                "vocabulary": vocabulary,
                "offset": offset,
                "size": len(vocabulary),
            }
        )
        transformed_feature_names.extend([f"{column}={token}" for token in vocabulary])
        offset += len(vocabulary)

    return {
        "version": 1,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "input_dim": offset,
        "transformed_feature_names": transformed_feature_names,
        "missing_categorical_token": MISSING_CATEGORICAL_TOKEN,
        "unknown_categorical_token": UNKNOWN_CATEGORICAL_TOKEN,
    }


def transform_input_features(df: pd.DataFrame, preprocessor_state: dict[str, Any]) -> np.ndarray:
    rows = len(df)
    input_dim = int(preprocessor_state["input_dim"])
    matrix = np.zeros((rows, input_dim), dtype=np.float32)

    for index, entry in enumerate(preprocessor_state["numeric_stats"]):
        name = str(entry["name"])
        if name in df.columns:
            values = pd.to_numeric(df[name], errors="coerce")
        else:
            values = pd.Series(np.nan, index=df.index, dtype=float)
        median = float(entry["median"])
        mean = float(entry["mean"])
        std = float(entry["std"]) if float(entry["std"]) > 0.0 else 1.0
        matrix[:, index] = ((values.fillna(median).astype(float) - mean) / std).to_numpy(dtype=np.float32)

    for entry in preprocessor_state["categorical_stats"]:
        name = str(entry["name"])
        if name in df.columns:
            values = sanitize_categorical_series(df[name])
        else:
            values = pd.Series(MISSING_CATEGORICAL_TOKEN, index=df.index, dtype=str)
        vocabulary = list(entry["vocabulary"])
        mapping = {token: idx for idx, token in enumerate(vocabulary)}
        offset = int(entry["offset"])
        encoded_indices = [mapping.get(value, 0) for value in values.tolist()]
        row_indices = np.arange(rows, dtype=np.int64)
        matrix[row_indices, offset + np.asarray(encoded_indices, dtype=np.int64)] = 1.0
    return matrix


def fit_target_state(
    train_df: pd.DataFrame,
    target_columns: list[str],
    enable_log_targets: bool,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for column in target_columns:
        values = pd.to_numeric(train_df[column], errors="coerce").astype(float)
        if values.isna().any():
            raise RuntimeError(f"Target column {column!r} still contains NaN after filtering")
        use_log = (
            enable_log_targets
            and column in TRACE_PROXY_LOG_SCALE_TARGET_COLUMNS
            and float(values.min()) >= 0.0
        )
        base_values = np.log1p(values.to_numpy(dtype=np.float64)) if use_log else values.to_numpy(dtype=np.float64)
        mean = float(np.mean(base_values))
        std = float(np.std(base_values))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        entries.append(
            {
                "name": column,
                "transform": "log1p" if use_log else "identity",
                "mean": mean,
                "std": std,
                "min_train": float(np.min(values)),
                "max_train": float(np.max(values)),
            }
        )
    return {
        "version": 1,
        "target_columns": target_columns,
        "entries": entries,
    }


def transform_targets(df: pd.DataFrame, target_state: dict[str, Any]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for entry in target_state["entries"]:
        name = str(entry["name"])
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        if entry["transform"] == "log1p":
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        values = (values - float(entry["mean"])) / max(float(entry["std"]), 1e-12)
        outputs.append(values.astype(np.float32))
    return np.stack(outputs, axis=1) if outputs else np.empty((len(df), 0), dtype=np.float32)


def inverse_transform_targets(matrix: np.ndarray, target_state: dict[str, Any]) -> np.ndarray:
    if matrix.size == 0:
        return np.empty_like(matrix)
    restored = np.asarray(matrix, dtype=np.float64).copy()
    for index, entry in enumerate(target_state["entries"]):
        restored[:, index] = restored[:, index] * float(entry["std"]) + float(entry["mean"])
        if entry["transform"] == "log1p":
            restored[:, index] = np.expm1(restored[:, index])
    return restored.astype(np.float32)


class TorchTraceFeatureMLP:
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int, nn_module: Any) -> None:
        layers: list[Any] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn_module.Linear(prev_dim, hidden_dim))
            layers.append(nn_module.ReLU())
            prev_dim = hidden_dim
        layers.append(nn_module.Linear(prev_dim, output_dim))
        self.network = nn_module.Sequential(*layers)

    def __call__(self, inputs: Any) -> Any:
        return self.network(inputs)

    def parameters(self) -> Any:
        return self.network.parameters()

    def to(self, device: Any) -> "TorchTraceFeatureMLP":
        self.network = self.network.to(device)
        return self

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def state_dict(self) -> dict[str, Any]:
        return self.network.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.network.load_state_dict(state_dict)


def _make_tensor_dataset(torch: Any, features: np.ndarray, targets: np.ndarray) -> Any:
    _, _, _, TensorDataset, _ = _import_torch()
    return TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )


def _evaluate_loss(model: TorchTraceFeatureMLP, features: np.ndarray, targets: np.ndarray, device: Any, torch: Any, criterion: Any) -> float | None:
    if len(features) == 0:
        return None
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features.astype(np.float32)).to(device)
        labels = torch.from_numpy(targets.astype(np.float32)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    return float(loss.detach().cpu().item())


def predict_with_torch(model: TorchTraceFeatureMLP, features: np.ndarray, batch_size: int, device: Any, torch: Any) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, 0), dtype=np.float32)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), max(1, int(batch_size))):
            batch = torch.from_numpy(features[start:start + max(1, int(batch_size))].astype(np.float32)).to(device)
            pred = model(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(pred)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "median_ape": 0.0,
        }
    clipped_pred = np.asarray(y_pred, dtype=float)
    target = np.asarray(y_true, dtype=float)
    denominator = np.clip(np.abs(target), a_min=1e-9, a_max=None)
    ape = np.abs(clipped_pred - target) / denominator
    return {
        "mae": float(mean_absolute_error(target, clipped_pred)),
        "rmse": float(np.sqrt(mean_squared_error(target, clipped_pred))),
        "r2": float(r2_score(target, clipped_pred)) if len(target) >= 2 else float("nan"),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
    }


def aggregate_target_metrics(per_target: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_target:
        return {}
    keys = ["mae", "rmse", "r2", "mape", "median_ape"]
    summary: dict[str, float] = {}
    for key in keys:
        values = [metrics.get(key, np.nan) for metrics in per_target.values()]
        summary[f"mean_{key}"] = float(np.nanmean(values))
        summary[f"median_{key}"] = float(np.nanmedian(values))
    return summary


def save_split_outputs(
    split_name: str,
    frame: pd.DataFrame,
    target_columns: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    prediction_df = frame[[column for column in TRACE_PROXY_METADATA_COLUMNS if column in frame.columns]].copy()
    per_target_metrics: dict[str, dict[str, float]] = {}
    metrics_rows: list[dict[str, Any]] = []

    for index, column in enumerate(target_columns):
        true_values = y_true[:, index] if y_true.size else np.empty((0,), dtype=np.float32)
        pred_values = y_pred[:, index] if y_pred.size else np.empty((0,), dtype=np.float32)
        abs_error = np.abs(pred_values - true_values)
        ape = abs_error / np.clip(np.abs(true_values), a_min=1e-9, a_max=None)
        prediction_df[f"target__{column}"] = true_values
        prediction_df[f"pred__{column}"] = pred_values
        prediction_df[f"abs_error__{column}"] = abs_error
        prediction_df[f"ape__{column}"] = ape

        metrics = target_metrics(true_values, pred_values)
        metrics["target_column"] = column
        metrics["row_count"] = int(len(true_values))
        metrics["target_mean"] = float(np.mean(true_values)) if len(true_values) else 0.0
        metrics["pred_mean"] = float(np.mean(pred_values)) if len(pred_values) else 0.0
        per_target_metrics[column] = {key: float(value) for key, value in metrics.items() if key != "target_column"}
        metrics_rows.append(metrics)

    prediction_path = output_dir / f"trace_feature_predictions_{split_name}.csv"
    prediction_df.to_csv(prediction_path, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = output_dir / f"trace_feature_metrics_{split_name}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    return {
        "aggregate": aggregate_target_metrics(per_target_metrics),
        "per_target": per_target_metrics,
        "prediction_csv": str(prediction_path),
        "metrics_csv": str(metrics_csv),
    }


def save_training_history(history: list[dict[str, Any]], output_dir: Path) -> Path:
    history_df = pd.DataFrame(history)
    history_path = output_dir / "trace_feature_training_history.csv"
    history_df.to_csv(history_path, index=False)
    return history_path


def train_trace_feature_model(
    data_dir: Path,
    output_dir: Path,
    hidden_layers: tuple[int, ...],
    batch_size: int,
    max_iter: int,
    alpha: float,
    learning_rate_init: float,
    seed: int,
    train_device: str = "auto",
    npu_device_id: int = 0,
    early_stopping_patience: int = 12,
    enable_log_targets: bool = True,
    requested_target_columns: list[str] | None = None,
) -> dict[str, Any]:
    torch, nn, DataLoader, _, torch_npu = _import_torch()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_tables = load_split_tables(data_dir)
    target_columns = resolve_target_columns(raw_tables, requested_target_columns)
    tables, row_filter_stats = filter_complete_target_rows(raw_tables, target_columns)

    numeric_features = [column for column in TRACE_PROXY_NUMERIC_FEATURES if column in tables["train"].columns]
    categorical_features = [column for column in TRACE_PROXY_CATEGORICAL_FEATURES if column in tables["train"].columns]
    if not numeric_features and not categorical_features:
        raise RuntimeError("No non-trace input feature columns were found in the training table")

    input_state = fit_input_preprocessor_state(
        train_df=tables["train"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    target_state = fit_target_state(
        train_df=tables["train"],
        target_columns=target_columns,
        enable_log_targets=enable_log_targets,
    )

    feature_matrices = {split_name: transform_input_features(frame, input_state) for split_name, frame in tables.items()}
    original_targets = {
        split_name: frame[target_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        for split_name, frame in tables.items()
    }
    transformed_targets = {split_name: transform_targets(frame, target_state) for split_name, frame in tables.items()}

    device = select_training_device(torch, torch_npu, train_device, npu_device_id)
    model = TorchTraceFeatureMLP(
        input_dim=int(input_state["input_dim"]),
        hidden_layers=hidden_layers,
        output_dim=len(target_columns),
        nn_module=nn,
    ).to(device)

    train_dataset = _make_tensor_dataset(torch, feature_matrices["train"], transformed_targets["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(int(batch_size), max(1, len(train_dataset)))),
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_init, weight_decay=alpha)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = None
    epochs_without_improvement = 0
    train_history: list[dict[str, Any]] = []

    for epoch in range(1, int(max_iter) + 1):
        model.train()
        epoch_losses: list[float] = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_loss = _evaluate_loss(
            model=model,
            features=feature_matrices["val"],
            targets=transformed_targets["val"],
            device=device,
            torch=torch,
            criterion=criterion,
        )
        score_loss = val_loss if val_loss is not None else train_loss
        improved = best_val_loss is None or score_loss < best_val_loss
        if improved:
            best_val_loss = score_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epochs_without_improvement >= max(1, int(early_stopping_patience)):
            break

    model.load_state_dict(best_state)

    split_metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for split_name, frame in tables.items():
        raw_predictions = predict_with_torch(
            model=model,
            features=feature_matrices[split_name],
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
        predictions = inverse_transform_targets(raw_predictions, target_state)
        split_metrics[split_name] = save_split_outputs(
            split_name=split_name,
            frame=frame,
            target_columns=target_columns,
            y_true=original_targets[split_name],
            y_pred=predictions,
            output_dir=output_dir,
        )
        artifacts[f"predictions_{split_name}_csv"] = split_metrics[split_name]["prediction_csv"]
        artifacts[f"metrics_{split_name}_csv"] = split_metrics[split_name]["metrics_csv"]

    model_path = output_dir / "trace_feature_mlp_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_layers": list(hidden_layers),
            "input_dim": int(input_state["input_dim"]),
            "output_dim": len(target_columns),
            "target_columns": target_columns,
        },
        model_path,
    )

    state_path = output_dir / "trace_feature_preprocessor_state.json"
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_state": input_state,
                "target_state": target_state,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    training_history_path = save_training_history(train_history, output_dir)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "model_type": "pytorch_trace_feature_mlp",
        "input_feature_count": len(numeric_features) + len(categorical_features),
        "input_dim_after_encoding": int(input_state["input_dim"]),
        "input_numeric_features": numeric_features,
        "input_categorical_features": categorical_features,
        "target_columns": target_columns,
        "target_log_columns_used": [
            entry["name"] for entry in target_state["entries"] if entry["transform"] == "log1p"
        ],
        "hidden_layers": list(hidden_layers),
        "batch_size": batch_size,
        "max_iter": max_iter,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "seed": seed,
        "train_device": str(device),
        "torch_npu_enabled": bool(torch_npu is not None),
        "npu_device_id": int(npu_device_id),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_trained": len(train_history),
        "training_history": train_history,
        "training_history_tail": train_history[-10:],
        "row_filter_stats": row_filter_stats,
        "metrics": split_metrics,
        "artifacts": {
            "model_pt": str(model_path),
            "preprocessor_state_json": str(state_path),
            "training_history_csv": str(training_history_path),
            "metrics_json": str(output_dir / "trace_feature_metrics.json"),
            **artifacts,
        },
    }

    with (output_dir / "trace_feature_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    summary = train_trace_feature_model(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        hidden_layers=parse_hidden_layers(args.hidden_layers),
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        seed=args.seed,
        train_device=args.train_device,
        npu_device_id=args.npu_device_id,
        early_stopping_patience=args.early_stopping_patience,
        enable_log_targets=not args.disable_log_targets,
        requested_target_columns=args.target_columns,
    )
    print(f"model_pt={summary['artifacts']['model_pt']}")
    print(f"preprocessor_state_json={summary['artifacts']['preprocessor_state_json']}")
    print(f"metrics_json={summary['artifacts']['metrics_json']}")
    print(f"val_mean_r2={summary['metrics']['val']['aggregate']['mean_r2']:.6f}")
    print(f"test_mean_r2={summary['metrics']['test']['aggregate']['mean_r2']:.6f}")


if __name__ == "__main__":
    main()
