#!/usr/bin/env python3
"""
Analyze per-op DynamoRIO trace outputs and generate a compact ranking table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    rows = []
    with Path(args.summary_csv).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["trace_bytes"] = int(row.get("trace_bytes") or 0)
            try:
                row["latency_avg_ms"] = float(row.get("latency_avg_ms") or 0.0)
            except ValueError:
                row["latency_avg_ms"] = 0.0
            row["exit_code"] = int(row.get("exit_code") or 0)
            rows.append(row)

    ok = [x for x in rows if x["exit_code"] == 0]
    fail = [x for x in rows if x["exit_code"] != 0]

    top_trace = sorted(ok, key=lambda x: x["trace_bytes"], reverse=True)[: args.topk]
    top_lat = sorted(ok, key=lambda x: x["latency_avg_ms"], reverse=True)[: args.topk]

    print(f"total={len(rows)} success={len(ok)} failed={len(fail)}")
    print("\nTop by trace_bytes:")
    for i, x in enumerate(top_trace, 1):
        print(
            f"{i:02d}. idx={x['op_idx']} type={x['op_type']} "
            f"trace={x['trace_bytes']}B lat={x['latency_avg_ms']:.3f}ms"
        )

    print("\nTop by latency_avg_ms:")
    for i, x in enumerate(top_lat, 1):
        print(
            f"{i:02d}. idx={x['op_idx']} type={x['op_type']} "
            f"lat={x['latency_avg_ms']:.3f}ms trace={x['trace_bytes']}B"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
