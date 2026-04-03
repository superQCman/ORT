from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

from static_pipeline_eval.artifact_loader import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ORT_ROOT,
    build_combo_specs,
    load_op_shapes_frame,
    load_prediction_frame,
    load_timeline_frame,
)
from static_pipeline_eval.schedule_engine import build_op_graph


@pytest.fixture(scope="session")
def prediction_df() -> pd.DataFrame:
    return load_prediction_frame(DEFAULT_ARTIFACT_ROOT, split="test")


@pytest.fixture(scope="session")
def combo_specs(prediction_df):
    return build_combo_specs(prediction_df, DEFAULT_ORT_ROOT)


@pytest.fixture(scope="session")
def combo_spec_map(combo_specs):
    return {(spec.case_id, spec.combo): spec for spec in combo_specs}


@pytest.fixture(scope="session")
def combo_catalog(prediction_df) -> pd.DataFrame:
    catalog = (
        prediction_df.groupby(
            ["case_id", "combo", "batch_size", "num_indices_per_lookup", "inter_threads"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "observed_count"})
        .sort_values(
            ["inter_threads", "observed_count", "case_id", "batch_size", "num_indices_per_lookup"],
            ascending=[True, False, True, True, True],
        )
        .reset_index(drop=True)
    )
    catalog["is_full_graph"] = catalog["observed_count"] == 60
    return catalog


@pytest.fixture(scope="session")
def select_combo_spec(combo_catalog, combo_spec_map) -> Callable[[int, bool | None], object]:
    def _select(inter_threads: int, require_full: bool | None = None):
        subset = combo_catalog[combo_catalog["inter_threads"] == inter_threads]
        if require_full is True:
            subset = subset[subset["is_full_graph"]]
        elif require_full is False:
            subset = subset[~subset["is_full_graph"]]
        if subset.empty:
            raise AssertionError(
                f"Could not find combo for inter_threads={inter_threads}, require_full={require_full}"
            )
        row = subset.iloc[0]
        return combo_spec_map[(row["case_id"], row["combo"])]

    return _select


@pytest.fixture(scope="session")
def load_combo_context(prediction_df):
    def _load(combo_spec):
        combo_rows = prediction_df[
            (prediction_df["case_id"] == combo_spec.case_id)
            & (prediction_df["combo"] == combo_spec.combo)
        ].copy()
        graph = build_op_graph(load_op_shapes_frame(combo_spec.artifact_paths.shape_csv))
        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        return combo_rows, graph, timeline_df

    return _load
