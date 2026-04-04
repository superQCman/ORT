from __future__ import annotations

import heapq

from static_pipeline_eval.schedule_engine import (
    BRANCH_FIFO_ORDER,
    analyze_embedding_execution,
    compute_mean_batch_span,
    schedule_combo,
)


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start_us, end_us in intervals:
        events.append((start_us, 1))
        events.append((end_us, -1))
    events.sort(key=lambda item: (item[0], 0 if item[1] < 0 else 1))

    active = 0
    max_active = 0
    for _, delta in events:
        active += delta
        max_active = max(max_active, active)
    return max_active


def test_build_op_graph_recovers_expected_modeled_nodes(select_combo_spec, load_combo_context):
    combo_spec = select_combo_spec(inter_threads=4, require_full=True)
    _, graph, _ = load_combo_context(combo_spec)

    modeled_nodes = [node for node in graph.values() if node.op_type != "Constant"]
    assert len(modeled_nodes) == 60
    assert graph[42].node_name == "/Concat"
    assert set(graph[42].predecessors) == {3, 7, 11, 15, 19, 23, 27, 31, 35}
    assert graph[70].predecessors == (69,)


def test_schedule_combo_enforces_fifo_slots(select_combo_spec, load_combo_context):
    combo_spec = select_combo_spec(inter_threads=4, require_full=True)
    combo_rows, graph, _ = load_combo_context(combo_spec)
    schedule_result = schedule_combo(combo_spec, graph, combo_rows)
    task_span_map = schedule_result.task_span_map()

    branch_spans = [task_span_map[f"branch:{branch_idx}"] for branch_idx in BRANCH_FIFO_ORDER]
    branch_starts = [span.start_us for span in branch_spans]
    assert branch_starts == sorted(branch_starts)
    assert _max_overlap([(span.start_us, span.end_us) for span in branch_spans]) == combo_spec.inter_threads

    active_heap: list[tuple[float, int]] = []
    for branch_idx in BRANCH_FIFO_ORDER[: combo_spec.inter_threads]:
        heapq.heappush(active_heap, (branch_spans[branch_idx].end_us, branch_idx))
    for branch_idx in BRANCH_FIFO_ORDER[combo_spec.inter_threads :]:
        freed_end_us, _ = heapq.heappop(active_heap)
        assert branch_spans[branch_idx].start_us >= freed_end_us
        heapq.heappush(active_heap, (branch_spans[branch_idx].end_us, branch_idx))

    assert schedule_result.tail_barrier_us == max(
        schedule_result.bottom_end_us,
        schedule_result.branch_pool_end_us,
    )


def test_compute_mean_batch_span_drops_first_batch(select_combo_spec, load_combo_context):
    combo_spec = select_combo_spec(inter_threads=1, require_full=True)
    _, graph, timeline_df = load_combo_context(combo_spec)
    node_names = [graph[node_idx].node_name for node_idx in sorted(node_idx for node_idx, node in graph.items() if node.op_type != "Constant")]

    timing = compute_mean_batch_span(timeline_df, node_names)

    subset = timeline_df[timeline_df["normalized_node_name"].isin(node_names)].copy()
    grouped = (
        subset.groupby(["batch_idx", "normalized_node_name"], as_index=False)
        .agg(start_us=("start_us", "min"), end_us=("end_us", "max"))
    )
    manual_spans = []
    for batch_idx, batch_df in grouped.groupby("batch_idx"):
        if set(batch_df["normalized_node_name"]) == set(node_names):
            manual_spans.append((int(batch_idx), float(batch_df["end_us"].max() - batch_df["start_us"].min())))
    manual_spans.sort(key=lambda item: item[0])
    manual_mean = sum(span for _, span in manual_spans[1:]) / len(manual_spans[1:])

    assert timing.dropped_batch_indices == (manual_spans[0][0],)
    assert timing.mean_span_us == manual_mean


def test_partial_combo_is_classified_as_filtered_subgraph(select_combo_spec, load_combo_context):
    combo_spec = select_combo_spec(inter_threads=5, require_full=False)
    combo_rows, graph, _ = load_combo_context(combo_spec)
    schedule_result = schedule_combo(combo_spec, graph, combo_rows)

    assert not schedule_result.coverage.is_full_graph
    assert schedule_result.coverage.expected_count == 60
    assert schedule_result.coverage.observed_count < 60
    assert schedule_result.coverage.missing_count > 0
    assert schedule_result.predicted_observed_subgraph_us > 0


def test_embedding_execution_matches_runner_semantics(select_combo_spec, load_combo_context):
    for inter_threads in [1, 3, 4, 5, 6]:
        combo_spec = select_combo_spec(inter_threads=inter_threads, require_full=None)
        _, _, timeline_df = load_combo_context(combo_spec)
        observation = analyze_embedding_execution(timeline_df, combo_spec.inter_threads)

        assert observation.representative_launch_order == BRANCH_FIFO_ORDER
        assert observation.all_kept_batches_match_fifo
        assert observation.representative_max_gather_concurrency == combo_spec.inter_threads
        assert observation.all_kept_batches_match_inter_threads
        assert observation.handoff_gap_mean_us is None or observation.handoff_gap_mean_us >= 0.0
