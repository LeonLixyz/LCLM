#!/usr/bin/env python3
"""Audit, deduplicate, and harvest verified synthetic expansion-agent traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.synthetic_expansion_agent import (
    EXPAND_TOOL,
    MEMORY_END,
    MEMORY_START,
    MIN_SEGMENT_WORDS,
    verify_trace,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _audit_trace(task: Mapping[str, Any], trace: Mapping[str, Any]) -> None:
    task_id = task.get("task_id")
    if trace.get("task_id") != task_id:
        raise ValueError(f"{task_id}: task/trace ID mismatch")
    if trace.get("tools") != [EXPAND_TOOL]:
        raise ValueError(f"{task_id}: trace does not contain the canonical expand tool")
    if trace.get("compression_scope") != "input_segments":
        raise ValueError(f"{task_id}: wrong compression scope")
    if not trace.get("verification", {}).get("accepted"):
        raise ValueError(f"{task_id}: trace is not marked accepted")

    messages = trace.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError(f"{task_id}: incomplete message trajectory")
    training_system_prompt = task.get("training_system_prompt")
    user_index = 1 if training_system_prompt else 0
    if training_system_prompt:
        if messages[0] != {"role": "system", "content": training_system_prompt}:
            raise ValueError(f"{task_id}: task system prompt was not preserved exactly")
    elif messages[0].get("role") == "system":
        raise ValueError(f"{task_id}: generation-only system prompt leaked into training")
    if messages[user_index].get("role") != "user":
        raise ValueError(f"{task_id}: expected initial task user message")

    marker_messages: list[int] = []
    call_ids: list[str] = []
    result_ids: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"{task_id}: message {index} is not a mapping")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"{task_id}: message {index} content is not a string")
        if "<think>" in content or "</think>" in content:
            raise ValueError(f"{task_id}: thinking tags are not allowed")
        if MEMORY_START in content or MEMORY_END in content:
            marker_messages.append(index)

        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            if content:
                raise ValueError(f"{task_id}: tool-call turn contains assistant prose")
            for call in message["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if function.get("name") != "expand" or not isinstance(arguments, Mapping):
                    raise ValueError(f"{task_id}: non-native expand call")
                if set(arguments) != {"segment_id"}:
                    raise ValueError(f"{task_id}: invalid expand arguments")
                call_ids.append(call.get("id"))
        elif role == "tool":
            if message.get("name") != "expand":
                raise ValueError(f"{task_id}: non-expand tool result")
            result_ids.append(message.get("tool_call_id"))

    if marker_messages != [user_index]:
        raise ValueError(f"{task_id}: memory markers must occur only in the initial user context")
    if messages[user_index]["content"].count(MEMORY_START) != len(task["segments"]):
        raise ValueError(f"{task_id}: initial context does not contain every compressed segment")
    if messages[user_index]["content"].count(MEMORY_END) != len(task["segments"]):
        raise ValueError(f"{task_id}: unbalanced initial memory segments")
    for segment in task["segments"]:
        segment_id = segment.get("segment_id")
        text = segment.get("text")
        if not isinstance(segment_id, str) or not isinstance(text, str):
            raise ValueError(f"{task_id}: invalid task segment")
        if len(text.split()) < MIN_SEGMENT_WORDS:
            raise ValueError(f"{task_id}: {segment_id} is shorter than the word floor")
        expected_block = f"{segment_id}\n{MEMORY_START}{text}{MEMORY_END}"
        if expected_block not in messages[user_index]["content"]:
            raise ValueError(
                f"{task_id}: training context does not contain full text for {segment_id}"
            )
    if call_ids != result_ids:
        raise ValueError(f"{task_id}: tool calls and results do not match in order")

    verification = verify_trace(task, messages)
    if not verification.accepted:
        raise ValueError(
            f"{task_id}: verifier rejected trace during harvest: {verification.reason}"
        )


def harvest_run_directories(
    run_directories: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    run_counts: dict[str, int] = {}

    for run_directory in run_directories:
        tasks_path = run_directory / "tasks.jsonl"
        accepted_path = run_directory / "accepted.jsonl"
        if not tasks_path.is_file() or not accepted_path.is_file():
            raise ValueError(f"{run_directory}: missing tasks.jsonl or accepted.jsonl")
        tasks = {task["task_id"]: task for task in _read_jsonl(tasks_path)}
        accepted = _read_jsonl(accepted_path)
        run_counts[run_directory.name] = len(accepted)
        for trace in accepted:
            task_id = trace.get("task_id")
            if task_id not in tasks:
                raise ValueError(f"{run_directory}: no task for trace {task_id}")
            _audit_trace(tasks[task_id], trace)
            canonical = dict(trace)
            canonical["harvest_version"] = "synthetic-expansion-harvest-v1"
            if task_id in unique:
                if canonical != unique[task_id]:
                    raise ValueError(f"conflicting duplicate task_id: {task_id}")
                duplicates += 1
                continue
            unique[task_id] = canonical

    harvested = [unique[task_id] for task_id in sorted(unique)]
    report = {
        "schema_version": 1,
        "status": "complete",
        "runs": run_counts,
        "duplicates_removed": duplicates,
        "harvested": len(harvested),
        "families": dict(
            sorted(Counter(row["sub_dataset"] for row in harvested).items())
        ),
        "tool_calls": sum(row["tool_call_count"] for row in harvested),
        "compression_scope": "input_segments",
    }
    return harvested, report


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="generation run directories")
    parser.add_argument("--output", type=Path, required=True, help="harvest output directory")
    args = parser.parse_args()

    harvested, report = harvest_run_directories(args.runs)
    _write_jsonl(args.output / "accepted.jsonl", harvested)
    with (args.output / "report.json").open("w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
