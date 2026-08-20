import copy
import json

from data.harvest_synthetic_expansion import harvest_run_directories
from data.synthetic_expansion_agent import (
    EXPAND_TOOL,
    MEMORY_END,
    MEMORY_START,
    TASK_FAMILIES,
    expand,
    generate_task,
    generate_tasks,
    messages_for_openai_api,
    run_agent_rollout,
)


def _expand_call(call_id, segment_id):
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "expand",
                    "arguments": {"segment_id": segment_id},
                },
            }
        ],
    }


def test_generation_is_deterministic_and_uses_only_positional_segment_ids():
    first = generate_tasks(len(TASK_FAMILIES), seed=7, distractors=8)
    second = generate_tasks(len(TASK_FAMILIES), seed=7, distractors=8)

    assert first == second
    assert [task["family"] for task in first] == list(TASK_FAMILIES)
    for task in first:
        assert [segment["segment_id"] for segment in task["segments"]] == [
            f"seg_{index}" for index in range(1, len(task["segments"]) + 1)
        ]
        assert all(segment_id.startswith("seg_") for segment_id in task["support_segment_ids"])


def test_training_context_contains_full_long_segments_and_rollout_sees_only_summaries():
    task = generate_task(0, seed=8, distractors=5)

    assert task["user_prompt"].count(MEMORY_START) == len(task["segments"])
    assert task["user_prompt"].count(MEMORY_END) == len(task["segments"])
    assert MEMORY_START not in task["rollout_user_prompt"]
    assert MEMORY_END not in task["rollout_user_prompt"]
    for segment in task["segments"]:
        assert f"{segment['segment_id']}\n{MEMORY_START}{segment['text']}{MEMORY_END}" in task[
            "user_prompt"
        ]
        assert len(segment["text"].split()) >= 512
        assert segment["summary"] in task["rollout_user_prompt"]
        assert segment["text"] not in task["rollout_user_prompt"]


def test_memory_markers_exist_only_in_the_initial_user_context():
    task = generate_task(0, seed=81, distractors=3)
    responses = iter(
        [
            _expand_call("call-1", task["support_segment_ids"][0]),
            {"content": task["expected_final"], "tool_calls": []},
        ]
    )

    trace = run_agent_rollout(task, lambda _messages, _tools: next(responses))

    marker_messages = [
        message
        for message in trace["messages"]
        if MEMORY_START in message.get("content", "") or MEMORY_END in message.get("content", "")
    ]
    assert marker_messages == [trace["messages"][1]]


def test_expand_returns_original_text_for_exact_seg_i():
    task = generate_task(0, seed=9, distractors=2)
    segment = task["segments"][1]

    assert expand(task, segment["segment_id"]) == segment["text"]


def test_native_rollout_expands_all_support_then_answers():
    task = generate_task(2, seed=10, distractors=4, family="multi_key_lookup")
    seen_prompts = []
    responses = []
    for index, segment_id in enumerate(task["support_segment_ids"]):
        responses.append(_expand_call(f"call-{index}", segment_id))
    responses.append({"content": task["expected_final"], "tool_calls": []})
    scripted = iter(responses)

    def complete(messages, _tools):
        seen_prompts.append(messages[1]["content"])
        return next(scripted)

    trace = run_agent_rollout(task, complete)

    assert trace["verification"]["accepted"]
    assert trace["compression_scope"] == "input_segments"
    assert trace["tool_call_count"] == len(task["support_segment_ids"])
    assert trace["tools"] == [EXPAND_TOOL]
    assert all(prompt == task["rollout_user_prompt"] for prompt in seen_prompts)
    assert trace["messages"][1]["content"] == task["training_user_prompt"]
    tool_results = [message for message in trace["messages"] if message["role"] == "tool"]
    assert [message["content"] for message in tool_results] == [
        expand(task, segment_id) for segment_id in task["support_segment_ids"]
    ]
    assert all(MEMORY_START not in message["content"] for message in tool_results)


def test_parallel_native_expand_calls_are_all_executed_and_preserved():
    task = generate_task(3, seed=11, distractors=3, family="numeric_comparison")
    calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {
                "name": "expand",
                "arguments": {"segment_id": segment_id},
            },
        }
        for index, segment_id in enumerate(task["support_segment_ids"])
    ]
    scripted = iter(
        [
            {"content": "", "tool_calls": calls},
            {"content": task["expected_final"], "tool_calls": []},
        ]
    )

    trace = run_agent_rollout(task, lambda _messages, _tools: next(scripted))

    assert trace["verification"]["accepted"]
    assistant_calls = [
        message for message in trace["messages"] if message["role"] == "assistant" and message.get("tool_calls")
    ]
    assert assistant_calls[0]["tool_calls"] == calls
    assert len([message for message in trace["messages"] if message["role"] == "tool"]) == len(calls)


def test_answer_without_expansion_is_rejected():
    task = generate_task(0, seed=12, distractors=2)
    trace = run_agent_rollout(
        task,
        lambda _messages, _tools: {"content": task["expected_final"], "tool_calls": []},
    )

    assert not trace["verification"]["accepted"]
    assert trace["verification"]["reason"] == "no_tool_call"


def test_missing_required_expansion_is_rejected_even_with_correct_answer():
    task = generate_task(1, seed=13, distractors=2, family="two_hop_join")
    scripted = iter(
        [
            _expand_call("call-1", task["support_segment_ids"][0]),
            {"content": task["expected_final"], "tool_calls": []},
        ]
    )

    trace = run_agent_rollout(task, lambda _messages, _tools: next(scripted))

    assert not trace["verification"]["accepted"]
    assert trace["verification"]["reason"] == "missing_support"


def test_unknown_segment_call_is_rejected_without_inserting_a_tool_result():
    task = generate_task(0, seed=14, distractors=2)
    trace = run_agent_rollout(
        task,
        lambda _messages, _tools: _expand_call("call-1", "seg_9999"),
    )

    assert not trace["verification"]["accepted"]
    assert trace["verification"]["reason"] == "unknown_segment"
    assert not any(message["role"] == "tool" for message in trace["messages"])


def test_openai_wire_format_stringifies_expand_arguments_without_mutation():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "expand",
                        "arguments": {"segment_id": "seg_3"},
                    },
                }
            ],
        }
    ]
    original = copy.deepcopy(messages)

    wire = messages_for_openai_api(messages)

    assert messages == original
    assert json.loads(wire[0]["tool_calls"][0]["function"]["arguments"]) == {
        "segment_id": "seg_3"
    }


def test_harvest_replays_verification_and_deduplicates_runs(tmp_path):
    task = generate_task(0, seed=15, distractors=2)

    def make_trace():
        scripted = iter([
            *[
                _expand_call(f"call-{index}", segment_id)
                for index, segment_id in enumerate(task["support_segment_ids"])
            ],
            {"content": task["expected_final"], "tool_calls": []},
        ])
        return run_agent_rollout(task, lambda _messages, _tools: next(scripted))

    for name in ("run-a", "run-b"):
        run = tmp_path / name
        run.mkdir()
        (run / "tasks.jsonl").write_text(json.dumps(task) + "\n")
        (run / "accepted.jsonl").write_text(json.dumps(make_trace()) + "\n")

    harvested, report = harvest_run_directories([tmp_path / "run-a", tmp_path / "run-b"])

    assert len(harvested) == 1
    assert report["duplicates_removed"] == 1
    assert report["harvested"] == 1
    assert report["tool_calls"] == len(task["support_segment_ids"])
