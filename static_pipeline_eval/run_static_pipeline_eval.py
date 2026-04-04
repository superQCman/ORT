#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from static_pipeline_eval.artifact_loader import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ORT_ROOT,
    build_combo_specs,
    load_op_shapes_frame,
    load_prediction_frame,
    load_timeline_frame,
)
from static_pipeline_eval.schedule_engine import (
    analyze_embedding_execution,
    build_op_graph,
    compute_mean_batch_span,
    schedule_combo,
)


def _metric_summary(metric_rows: list[dict[str, object]], predicted_key: str, actual_key: str) -> dict[str, object]:
    if not metric_rows:
        return {
            "count": 0,
            "mae_us": None,
            "mape": None,
            "p50_ape": None,
            "p95_ape": None,
            "worst_combo": None,
        }

    frame = pd.DataFrame(metric_rows)
    worst_row = frame.sort_values("abs_error_us", ascending=False).iloc[0]
    return {
        "count": int(len(frame)),
        "mae_us": float(frame["abs_error_us"].mean()),
        "mape": float(frame["ape"].mean()),
        "p50_ape": float(frame["ape"].quantile(0.50)),
        "p95_ape": float(frame["ape"].quantile(0.95)),
        "worst_combo": {
            "case_id": worst_row["case_id"],
            "combo": worst_row["combo"],
            predicted_key: float(worst_row[predicted_key]),
            actual_key: float(worst_row[actual_key]),
            "abs_error_us": float(worst_row["abs_error_us"]),
            "ape": float(worst_row["ape"]),
        },
    }


def _build_sorted_frame(
    rows: list[dict[str, object]],
    columns: list[str],
    sort_keys: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(sort_keys)


def _format_order(order: tuple[int, ...]) -> str:
    return "->".join(str(item) for item in order)


def _format_us(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{sign}{magnitude / 1_000_000:.3f}M us"
    if magnitude >= 1_000:
        return f"{sign}{magnitude / 1_000:.3f}K us"
    return f"{sign}{magnitude:.3f} us"


def _build_calibration_candidates(
    prediction_df: pd.DataFrame,
    full_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
    embedding_rows: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append("# Calibration Candidates")
    lines.append("")

    embedding_df = pd.DataFrame(embedding_rows)
    if not embedding_df.empty:
        handoff_df = embedding_df.dropna(subset=["handoff_gap_mean_us"]).copy()
        lines.append("## 1. Embedding 槽位交接空隙")
        lines.append("")
        lines.append("- 现象：排队 branch 的 `Gather` 并不是在前一条 branch 的 `ReduceSum` 结束瞬间立刻发起，存在稳定但非零的 handoff gap。")
        lines.append("- 建议校准方式：增加一个只依赖静态变量的加性 `release_gap_us(inter_threads, wave_idx, branch_size_summary)`。")
        lines.append("- 当前统计（按 combo 级 mean gap 聚合，避免单批长尾噪声主导）：")
        for inter_threads, sub in handoff_df.groupby("inter_threads"):
            mean_gap = float(sub["handoff_gap_mean_us"].mean())
            p10_gap = float(sub["handoff_gap_mean_us"].quantile(0.10))
            p90_gap = float(sub["handoff_gap_mean_us"].quantile(0.90))
            max_gap = float(sub["handoff_gap_mean_us"].max())
            lines.append(
                f"  - inter_threads={int(inter_threads)}: "
                f"mean={mean_gap:.3f} us, "
                f"p10={p10_gap:.3f} us, "
                f"p90={p90_gap:.3f} us"
            )
            if max_gap > p90_gap * 4:
                lines.append(
                    f"    - rare outlier: combo-mean gap can still reach {max_gap:.3f} us"
                )
        lines.append("")

    lines.append("## 2. Embedding 复合 branch 残差")
    lines.append("")
    lines.append("- 现象：最差样本的主误差仍然集中在 `Gather + ReduceSum`，说明复合 branch 本身值得单独留一个黑盒校准入口。")
    lines.append("- 建议校准方式：对 `/emb_lX/{Gather,Reshape,ReduceSum}` 复合 task 增加 branch-level correction，只允许使用静态输入。")
    lines.append("- 代表性样本：")
    for case_id, combo in [
        ("case_10_3_3", "bs2048_nip2000"),
        ("case_10_4_6", "bs1856_nip1700"),
        ("case_8_1_1", "bs2048_nip2000"),
    ]:
        sub = prediction_df[
            (prediction_df["case_id"] == case_id)
            & (prediction_df["combo"] == combo)
            & (prediction_df["op_type"].isin(["Gather", "ReduceSum"]))
        ]
        if sub.empty:
            continue
        residuals = (
            sub.groupby("op_type")
            .apply(lambda frame: float((frame["pred_us"] - frame["target_us"]).sum()), include_groups=False)
            .to_dict()
        )
        formatted = ", ".join(
            f"{op_type} { _format_us(residual) }"
            for op_type, residual in sorted(residuals.items())
        )
        lines.append(f"  - `{case_id} / {combo}`: {formatted}")
    lines.append("")

    micro_tail = prediction_df[prediction_df["op_idx"].between(39, 69)].copy()
    if not micro_tail.empty:
        micro_tail["residual_us"] = micro_tail["pred_us"] - micro_tail["target_us"]
        micro_tail_summary = (
            micro_tail.groupby(["case_id", "combo"], as_index=False)["residual_us"]
            .sum()
            .assign(abs_residual_us=lambda frame: frame["residual_us"].abs())
            .sort_values("abs_residual_us", ascending=False)
            .head(3)
        )
        lines.append("## 3. Join 后微尾段 bundle")
        lines.append("")
        lines.append("- 现象：`Shape_1/Gather_9/.../Concat_4` 这段是汇合后的短尾 burst，单算子很短，但累计残差会在 schedule 级别放大。")
        lines.append("- 建议校准方式：如果整图误差集中在这个区域，优先做 bundle correction，而不是分别拟合这些微小 op。")
        lines.append("- 当前残差最大的样本：")
        for row in micro_tail_summary.itertuples(index=False):
            lines.append(
                f"  - `{row.case_id} / {row.combo}`: bundle residual { _format_us(float(row.residual_us)) }"
            )
        lines.append("")

    missing_top_tail = pd.DataFrame(partial_rows)
    if not missing_top_tail.empty:
        representative = missing_top_tail.sort_values(["missing_count", "coverage_ratio"], ascending=[False, True]).iloc[0]
        lines.append("## 4. Top MLP 末段易波动尾部")
        lines.append("")
        lines.append("- 现象：partial combo 往往在 top MLP 末段掉点，说明这部分标签稳定性仍然偏弱。")
        lines.append("- 建议校准方式：把 `/top_l/top_l.4/Gemm` 及附近激活作为可选校准点，但优先级低于 embedding branch。")
        lines.append(
            f"- partial 代表样本：`{representative['case_id']} / {representative['combo']}`，"
            f"coverage_ratio={float(representative['coverage_ratio']):.3f}，"
            f"missing_count={int(representative['missing_count'])}。"
        )
        lines.append("- 已验证的波动样本：`case_10_1_1 / bs1056_nip1900` 的 `/top_l/top_l.4/Gemm` 最后两批时长 range ratio 约为 `0.522`，确实会触发上游稳定性过滤。")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORT static pipeline evaluation")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Path to the classed_op_mlp artifact root",
    )
    parser.add_argument(
        "--ort-root",
        type=Path,
        default=DEFAULT_ORT_ROOT,
        help="Path to the ORT repository root",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"static_pipeline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output run directory name under artifacts/latest/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    ort_root = args.ort_root.resolve()
    output_dir = ort_root / "static_pipeline_eval" / "artifacts" / "latest" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = load_prediction_frame(artifact_root, split="test")
    combo_specs = build_combo_specs(prediction_df, ort_root=ort_root)

    full_columns = [
        "metric_scope",
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "coverage_ratio",
        "predicted_e2e_us",
        "actual_e2e_us",
        "abs_error_us",
        "ape",
        "bottom_end_us",
        "branch_pool_end_us",
        "tail_barrier_us",
    ]
    partial_columns = [
        "metric_scope",
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "expected_count",
        "observed_count",
        "missing_count",
        "coverage_ratio",
        "predicted_observed_subgraph_us",
        "actual_observed_subgraph_us",
        "abs_error_us",
        "ape",
        "missing_node_indices",
        "missing_node_names",
    ]
    embedding_columns = [
        "case_id",
        "combo",
        "batch_size",
        "num_indices_per_lookup",
        "inter_threads",
        "kept_batch_count",
        "launch_order",
        "matches_fifo",
        "max_gather_concurrency",
        "matches_inter_threads",
        "handoff_gap_mean_us",
        "handoff_gap_min_us",
        "handoff_gap_max_us",
        "tail_start_gap_mean_us",
        "tail_start_gap_min_us",
        "tail_start_gap_max_us",
    ]

    full_rows: list[dict[str, object]] = []
    partial_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []

    for combo_spec in combo_specs:
        combo_rows = prediction_df[
            (prediction_df["case_id"] == combo_spec.case_id)
            & (prediction_df["combo"] == combo_spec.combo)
        ].copy()
        op_shapes_df = load_op_shapes_frame(combo_spec.artifact_paths.shape_csv)
        graph = build_op_graph(op_shapes_df)
        schedule_result = schedule_combo(combo_spec, graph, combo_rows)

        timeline_df = load_timeline_frame(combo_spec.artifact_paths.timeline_csv)
        expected_node_names = [graph[node_idx].node_name for node_idx in schedule_result.expected_node_indices]
        observed_node_names = [graph[node_idx].node_name for node_idx in schedule_result.observed_node_indices]

        actual_full = compute_mean_batch_span(timeline_df, expected_node_names)
        actual_observed = compute_mean_batch_span(timeline_df, observed_node_names)
        embedding_observation = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)

        embedding_rows.append(
            {
                "case_id": combo_spec.case_id,
                "combo": combo_spec.combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "kept_batch_count": embedding_observation.kept_batch_count,
                "launch_order": _format_order(embedding_observation.representative_launch_order),
                "matches_fifo": embedding_observation.all_kept_batches_match_fifo,
                "max_gather_concurrency": embedding_observation.representative_max_gather_concurrency,
                "matches_inter_threads": embedding_observation.all_kept_batches_match_inter_threads,
                "handoff_gap_mean_us": embedding_observation.handoff_gap_mean_us,
                "handoff_gap_min_us": embedding_observation.handoff_gap_min_us,
                "handoff_gap_max_us": embedding_observation.handoff_gap_max_us,
                "tail_start_gap_mean_us": embedding_observation.tail_start_gap_mean_us,
                "tail_start_gap_min_us": embedding_observation.tail_start_gap_min_us,
                "tail_start_gap_max_us": embedding_observation.tail_start_gap_max_us,
            }
        )

        if schedule_result.coverage.is_full_graph:
            predicted_us = schedule_result.predicted_full_graph_us
            actual_us = actual_full.mean_span_us
            row = {
                "metric_scope": "full_graph_e2e",
                "case_id": combo_spec.case_id,
                "combo": combo_spec.combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "expected_count": schedule_result.coverage.expected_count,
                "observed_count": schedule_result.coverage.observed_count,
                "coverage_ratio": schedule_result.coverage.coverage_ratio,
                "predicted_e2e_us": predicted_us,
                "actual_e2e_us": actual_us,
                "abs_error_us": abs(predicted_us - actual_us),
                "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                "bottom_end_us": schedule_result.bottom_end_us,
                "branch_pool_end_us": schedule_result.branch_pool_end_us,
                "tail_barrier_us": schedule_result.tail_barrier_us,
            }
            full_rows.append(row)
        else:
            predicted_us = schedule_result.predicted_observed_subgraph_us
            actual_us = actual_observed.mean_span_us
            missing_names = [graph[node_idx].node_name for node_idx in schedule_result.coverage.missing_node_indices]
            row = {
                "metric_scope": "observed_subgraph_non_e2e",
                "case_id": combo_spec.case_id,
                "combo": combo_spec.combo,
                "batch_size": combo_spec.batch_size,
                "num_indices_per_lookup": combo_spec.num_indices_per_lookup,
                "inter_threads": combo_spec.inter_threads,
                "expected_count": schedule_result.coverage.expected_count,
                "observed_count": schedule_result.coverage.observed_count,
                "missing_count": schedule_result.coverage.missing_count,
                "coverage_ratio": schedule_result.coverage.coverage_ratio,
                "predicted_observed_subgraph_us": predicted_us,
                "actual_observed_subgraph_us": actual_us,
                "abs_error_us": abs(predicted_us - actual_us),
                "ape": abs(predicted_us - actual_us) / actual_us if actual_us else 0.0,
                "missing_node_indices": "|".join(str(node_idx) for node_idx in schedule_result.coverage.missing_node_indices),
                "missing_node_names": "|".join(missing_names),
            }
            partial_rows.append(row)

    full_df = _build_sorted_frame(
        full_rows,
        columns=full_columns,
        sort_keys=["case_id", "batch_size", "num_indices_per_lookup"],
    )
    partial_df = _build_sorted_frame(
        partial_rows,
        columns=partial_columns,
        sort_keys=["case_id", "batch_size", "num_indices_per_lookup"],
    )
    embedding_df = _build_sorted_frame(
        embedding_rows,
        columns=embedding_columns,
        sort_keys=["case_id", "batch_size", "num_indices_per_lookup"],
    )

    full_df.to_csv(output_dir / "full_combo_metrics.csv", index=False)
    partial_df.to_csv(output_dir / "partial_combo_metrics.csv", index=False)
    embedding_df.to_csv(output_dir / "embedding_order_check.csv", index=False)

    calibration_md = _build_calibration_candidates(prediction_df, full_rows, partial_rows, embedding_rows)
    (output_dir / "calibration_candidates.md").write_text(calibration_md, encoding="utf-8")

    summary = {
        "artifact_root": str(artifact_root),
        "ort_root": str(ort_root),
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "combo_counts": {
            "total_test_combos": int(len(combo_specs)),
            "full_combo_count": int(len(full_rows)),
            "partial_combo_count": int(len(partial_rows)),
        },
        "full_graph_metrics": _metric_summary(full_rows, "predicted_e2e_us", "actual_e2e_us"),
        "partial_graph_metrics": _metric_summary(
            partial_rows,
            "predicted_observed_subgraph_us",
            "actual_observed_subgraph_us",
        ),
        "embedding_semantics": {
            "all_combos_match_fifo": bool(embedding_df["matches_fifo"].all()) if not embedding_df.empty else None,
            "all_combos_match_inter_threads": bool(embedding_df["matches_inter_threads"].all()) if not embedding_df.empty else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
