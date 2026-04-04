from __future__ import annotations

from run_static_pipeline_eval import _build_sorted_frame


def test_build_sorted_frame_preserves_columns_for_empty_rows():
    frame = _build_sorted_frame(
        rows=[],
        columns=["case_id", "combo", "ape"],
        sort_keys=["case_id", "combo"],
    )

    assert frame.empty
    assert frame.columns.tolist() == ["case_id", "combo", "ape"]


def test_build_sorted_frame_sorts_non_empty_rows():
    frame = _build_sorted_frame(
        rows=[
            {"case_id": "case_b", "combo": "bs2_nip2", "ape": 0.2},
            {"case_id": "case_a", "combo": "bs1_nip1", "ape": 0.1},
        ],
        columns=["case_id", "combo", "ape"],
        sort_keys=["case_id", "combo"],
    )

    assert frame["case_id"].tolist() == ["case_a", "case_b"]
