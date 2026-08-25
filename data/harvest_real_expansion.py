#!/usr/bin/env python3
"""Re-audit real-source expansion traces with source-native answer metrics."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.real_expansion_agent import validate_real_task, verify_real_trace


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def harvest_real_traces(
    tasks: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_by_id = {}
    for task in tasks:
        validate_real_task(task)
        task_id = task["task_id"]
        if task_id in task_by_id:
            raise ValueError(f"duplicate source task: {task_id}")
        task_by_id[task_id] = task

    accepted = []
    rejected = []
    reasons: Counter[str] = Counter()
    accepted_by_source: Counter[str] = Counter()
    seen = set()
    for trace in traces:
        task_id = trace.get("task_id")
        if task_id in seen:
            raise ValueError(f"duplicate generated trace: {task_id}")
        seen.add(task_id)
        if task_id not in task_by_id:
            raise ValueError(f"trace has no source task: {task_id}")
        task = task_by_id[task_id]
        canonical = copy.deepcopy(trace)
        verification = verify_real_trace(task, canonical.get("messages", []))
        canonical["verification"] = verification
        canonical["harvest_version"] = "real-expansion-harvest-v1"
        canonical["source_dataset"] = task["source_dataset"]
        canonical["source_row_id"] = task["source_row_id"]
        canonical["gold_answer"] = task["gold_answer"]
        canonical["support_segment_ids"] = task["support_segment_ids"]
        canonical["segment_count"] = len(task["segments"])
        reasons[verification["reason"]] += 1
        if verification["accepted"]:
            accepted.append(canonical)
            accepted_by_source[task["source_dataset"]] += 1
        else:
            rejected.append(canonical)

    report = {
        "status": "complete",
        "source_tasks": len(tasks),
        "generated_traces": len(traces),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(traces) if traces else None,
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "reasons": dict(sorted(reasons.items())),
    }
    return accepted, rejected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted, rejected, report = harvest_real_traces(
        _read_jsonl(args.tasks), _read_jsonl(args.traces)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "accepted.jsonl", accepted)
    _write_jsonl(args.output / "rejected.jsonl", rejected)
    with (args.output / "report.json").open("w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
