#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_utils import feature_columns_for_training
from model_utils import ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare multiple operator-level regressors.")
    parser.add_argument(
        "--dataset-csv",
        action="append",
        required=True,
        help="Training dataset CSV; can be passed multiple times and will be concatenated",
    )
    parser.add_argument(
        "--target",
        default="label_real_dur_us",
        help="Target column to predict",
    )
    parser.add_argument(
        "--group-column",
        default="sample_group",
        help="Grouping column for validation splits",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits when enough groups are available",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for stochastic models",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for leaderboard, fold metrics, and the best model artifact",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge_log", "random_forest_log", "hist_gbdt_log"],
        help="Subset of models to train",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of trees for the random forest",
    )
    parser.add_argument(
        "--hgbt-iterations",
        type=int,
        default=250,
        help="Number of boosting iterations for HistGradientBoosting",
    )
    return parser.parse_args()


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ],
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    rf_estimators: int,
    hgbt_iterations: int,
) -> dict[str, Pipeline]:
    return {
        "ridge_log": Pipeline(
            [
                ("pre", make_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=RidgeCV(alphas=np.logspace(-3, 3, 13)),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "random_forest_log": Pipeline(
            [
                ("pre", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=RandomForestRegressor(
                            n_estimators=rf_estimators,
                            min_samples_leaf=2,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
        "hist_gbdt_log": Pipeline(
            [
                ("pre", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=HistGradientBoostingRegressor(
                            max_depth=8,
                            max_iter=hgbt_iterations,
                            learning_rate=0.05,
                            random_state=random_state,
                        ),
                        func=np.log1p,
                        inverse_func=np.expm1,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0),
    }


def build_splitter(groups: pd.Series, n_splits: int, random_state: int):
    unique_groups = groups.dropna().astype(str).nunique()
    if unique_groups >= 2:
        return GroupKFold(n_splits=min(n_splits, unique_groups)), True
    return KFold(n_splits=min(n_splits, 5), shuffle=True, random_state=random_state), False


def main() -> None:
    args = parse_args()

    frames = [pd.read_csv(path) for path in args.dataset_csv]
    df = pd.concat(frames, ignore_index=True)
    if args.target not in df.columns:
        raise KeyError(f"Target column not found: {args.target}")

    df = df[df[args.target].notna()].copy()
    if df.empty:
        raise ValueError(f"No rows with target={args.target}")
    if (df[args.target] <= 0).any():
        raise ValueError(f"Target must be positive for log-space training: {args.target}")

    numeric_features, categorical_features = feature_columns_for_training(df, args.target)
    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("No usable feature columns found")

    X = df[feature_columns]
    y = df[args.target].astype(float).to_numpy()
    groups = df[args.group_column].astype(str) if args.group_column in df.columns else pd.Series(["all"] * len(df))

    splitter, uses_groups = build_splitter(groups, args.n_splits, args.random_state)
    all_models = build_models(
        numeric_features,
        categorical_features,
        args.random_state,
        args.rf_estimators,
        args.hgbt_iterations,
    )
    unknown_models = sorted(set(args.models) - set(all_models))
    if unknown_models:
        raise ValueError(f"Unknown model names: {unknown_models}")
    models = {name: all_models[name] for name in args.models}

    fold_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        predictions = np.full(shape=len(df), fill_value=np.nan, dtype=float)

        if uses_groups:
            split_iter = splitter.split(X, y, groups)
        else:
            split_iter = splitter.split(X, y)

        for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)
            predictions[test_idx] = y_pred

            metrics = regression_metrics(y_test, y_pred)
            metrics.update(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                }
            )
            fold_rows.append(metrics)

        valid_mask = ~np.isnan(predictions)
        summary = regression_metrics(y[valid_mask], predictions[valid_mask])
        summary.update(
            {
                "model": model_name,
                "rows": int(valid_mask.sum()),
                "feature_count": len(feature_columns),
            }
        )
        leaderboard_rows.append(summary)

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("mae")
    best_model_name = str(leaderboard_df.iloc[0]["model"])
    best_model = models[best_model_name]
    best_model.fit(X, y)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = output_dir / "leaderboard.csv"
    folds_path = output_dir / "fold_metrics.csv"
    model_path = output_dir / "best_model.joblib"
    summary_path = output_dir / "training_summary.json"

    leaderboard_df.to_csv(leaderboard_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    joblib.dump(best_model, model_path)

    summary = {
        "target": args.target,
        "best_model": best_model_name,
        "trained_models": list(models.keys()),
        "dataset_rows": int(len(df)),
        "feature_count": len(feature_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "group_column": args.group_column,
        "uses_group_kfold": uses_groups,
        "rf_estimators": args.rf_estimators,
        "hgbt_iterations": args.hgbt_iterations,
        "dataset_csvs": args.dataset_csv,
        "leaderboard_csv": str(leaderboard_path),
        "fold_metrics_csv": str(folds_path),
        "best_model_artifact": str(model_path),
    }
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"best_model={best_model_name}")
    print(leaderboard_path)
    print(model_path)


if __name__ == "__main__":
    main()
