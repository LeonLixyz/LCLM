import json
import sys
import types

import pytest

from data.build_stage3_agent_mixture import (
    COLDSTART_AGENT_DATASET,
    DEFAULT_AGENT_DATASET,
    DEFAULT_REASONING_SUBDATASETS,
    DEFAULT_STAGE3_DATASET,
    MEMORY_END,
    MEMORY_START,
    SCHEMA_KEYS,
    agent_row_is_successful,
    agent_row_is_primary_trace,
    build_parser,
    build_source_specs,
    split_reasoning_target,
    reasoning_row_is_compressed,
    transform_agent_row,
    transform_stage3_row,
    weighted_interleave,
    write_jsonl,
    SourceSpec,
    _iter_source,
)


def stage3_row(sub_dataset="reasoning_data", target="work\nFinal Answer: 42"):
    return {
        "prompt": [{"role": "user", "content": "question"}],
        "compression_prompt": [
            {
                "role": "user",
                "content": f"{MEMORY_START}question{MEMORY_END}",
            }
        ],
        "target": target,
        "sub_dataset": sub_dataset,
        "quality": 0.9,
    }


class WhitespaceOffsetTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_attention_mask,
        return_offsets_mapping,
    ):
        del add_special_tokens, return_attention_mask, return_offsets_mapping
        matches = list(__import__("re").finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


def test_reasoning_stage3_moves_compression_to_tagged_analysis():
    row = stage3_row(target="preamble <think>first\nsecond</think>\nThe answer is 7.")

    transformed = transform_stage3_row(row)

    assert transformed["compression_prompt"] == row["prompt"]
    assert transformed["target"] == (
        f"preamble <think>{MEMORY_START}first\nsecond{MEMORY_END}"
        "</think>\nThe answer is 7."
    )
    assert transformed["compression_scope"] == "target_analysis"
    assert transformed["reasoning_split"].startswith("tag:")
    assert json.loads(transformed["metadata_json"]) == {"quality": 0.9}
    assert tuple(transformed) == SCHEMA_KEYS


def test_reasoning_stage3_uses_final_answer_boundary():
    row = stage3_row(target="step one\nstep two\nFinal Answer: 42")

    transformed = transform_stage3_row(row)

    assert transformed["target"] == (
        f"{MEMORY_START}step one\nstep two\n{MEMORY_END}Final Answer: 42"
    )
    assert transformed["reasoning_split"] == "marker:Final Answer:"


def test_reasoning_mixture_uncompressed_arm_keeps_original_target():
    row = stage3_row(target="step one\nstep two\nFinal Answer: 42")

    transformed = transform_stage3_row(row, compress_reasoning=False)

    assert transformed["compression_prompt"] == row["prompt"]
    assert transformed["target"] == row["target"]
    assert transformed["compression_scope"] == "none"
    assert transformed["reasoning_split"] == "mixture:uncompressed"


def test_reasoning_mixture_assignment_is_stable_and_honors_endpoints():
    key = "train-00001-of-02033.parquet:17"
    assert reasoning_row_is_compressed(key, 0.5) == reasoning_row_is_compressed(key, 0.5)
    assert not reasoning_row_is_compressed(key, 0.0)
    assert reasoning_row_is_compressed(key, 1.0)


def test_splitter_fails_closed_without_a_safe_boundary():
    row = stage3_row(target="Reasoning which eventually says the result is 42.")

    transformed = transform_stage3_row(row)

    assert transformed["compression_prompt"] == row["prompt"]
    assert transformed["target"] == row["target"]
    assert transformed["compression_scope"] == "none"
    assert transformed["reasoning_split"] == "none"


def test_splitter_falls_back_to_last_n_tokens_without_rewriting_text():
    target = "one two three four five six"
    split = split_reasoning_target(
        target,
        fallback_tokenizer=WhitespaceOffsetTokenizer(),
        fallback_answer_tokens=2,
    )

    assert split is not None
    assert split.analysis == "one two three four "
    assert split.suffix == "five six"
    assert split.method == "fallback:last-2-tokens"
    assert split.analysis + split.suffix == target


def test_fallback_leaves_short_target_fully_trainable():
    assert (
        split_reasoning_target(
            "one two",
            fallback_tokenizer=WhitespaceOffsetTokenizer(),
            fallback_answer_tokens=2,
        )
        is None
    )


def test_splitter_keeps_boxed_conclusion_trainable():
    split = split_reasoning_target("derive\ncarefully\n\\boxed{C}")
    assert split is not None
    assert split.analysis == "derive\ncarefully\n"
    assert split.suffix == r"\boxed{C}"
    assert split.method == "pattern:boxed"


def test_splitter_accepts_gsm8k_hash_answer_only_on_last_line():
    split = split_reasoning_target("derive carefully\n#### 42")
    assert split is not None
    assert split.analysis == "derive carefully\n"
    assert split.suffix == "#### 42"
    assert split.method == "pattern:gsm8k-final-line"


def test_splitter_does_not_treat_markdown_h4_as_an_answer():
    assert (
        split_reasoning_target(
            "derive\n#### Partial derivative\ncontinue deriving",
        )
        is None
    )


def test_splitter_keeps_answer_line_trainable():
    split = split_reasoning_target("step one\nstep two\nThe answer is (I).")
    assert split is not None
    assert split.analysis == "step one\nstep two\n"
    assert split.suffix == "The answer is (I)."
    assert split.method == "pattern:answer-line"


def test_splitter_uses_balanced_last_code_fence_for_programming_solution():
    split = split_reasoning_target(
        "scratch reasoning\n```python\nprint(42)\n```\nExplanation"
    )
    assert split is not None
    assert split.analysis == "scratch reasoning\n"
    assert split.suffix.startswith("```python")
    assert split.method == "pattern:last-code-fence"


def test_splitter_rejects_unbalanced_code_fence():
    assert split_reasoning_target("reasoning\n```python\nprint(42)") is None


@pytest.mark.parametrize(
    "target",
    [
        "<think>reasoning only</think>",
        "<think>one</think> suffix <think>two</think> suffix",
        "Final Answer: 42",
        "reasoning mentions Final Answer: inline but keeps reasoning",
    ],
)
def test_splitter_rejects_missing_or_ambiguous_suffixes(target):
    assert split_reasoning_target(target) is None


def test_splitter_allows_explicit_custom_markers():
    split = split_reasoning_target(
        "derive carefully<RESULT =>9",
        analysis_tag_pairs=(),
        final_answer_markers=("<RESULT =>",),
    )

    assert split is not None
    assert split.analysis == "derive carefully"
    assert split.suffix == "<RESULT =>9"


def test_splitter_fails_closed_on_malformed_explicit_tags():
    assert (
        split_reasoning_target("<think>unfinished\nFinal Answer: 3") is None
    )


def test_generic_answer_marker_requires_a_paragraph_boundary():
    assert split_reasoning_target("step\nAnswer: intermediate") is None
    split = split_reasoning_target("step\n\nAnswer: final")
    assert split is not None
    assert split.suffix == "Answer: final"


def test_normal_stage3_row_keeps_training_fields_unchanged():
    row = stage3_row(sub_dataset="repo_summarize", target="summary")

    transformed = transform_stage3_row(row)

    assert transformed["prompt"] == row["prompt"]
    assert transformed["compression_prompt"] == row["compression_prompt"]
    assert transformed["target"] == row["target"]
    assert transformed["sub_dataset"] == row["sub_dataset"]
    assert transformed["compression_scope"] == "prompt"
    assert transformed["reasoning_split"] == "not_selected"


def test_default_reasoning_selectors_are_explicit_and_conservative():
    assert DEFAULT_REASONING_SUBDATASETS == (
        "reasoning_data",
        "dolci_think",
    )


def test_agent_row_preserves_native_multiturn_trajectory():
    conversations = [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "I will inspect."},
        {"role": "tool", "content": "test output"},
        {"role": "assistant", "content": "fixed"},
    ]
    row = {
        "conversations": conversations,
        "task": "repository task",
        "trace_source": "swesmith",
        "agent": "terminus-2",
        "custom": {"score": 1},
    }

    transformed = transform_agent_row(row)

    assert transformed["messages"] == conversations
    assert transformed["messages"] is not conversations
    assert transformed["conversations"] is None
    assert transformed["tools"] is None
    assert transformed["prompt"] is None
    assert transformed["target"] is None
    assert transformed["compression_scope"] == "none"
    assert json.loads(transformed["metadata_json"]) == {"custom": {"score": 1}}
    assert tuple(transformed) == SCHEMA_KEYS


@pytest.mark.parametrize(
    "row",
    [
        {"conversations": [{"role": "assistant", "content": f"bad {MEMORY_START}"}]},
        {"messages": [{"role": "user", "content": "ok"}], "task": f"bad {MEMORY_END}"},
        {"messages": [{"role": "assistant", "content": {"nested": "<|memory|>"}}]},
    ],
)
def test_agent_rows_reject_memory_markers_recursively(row):
    with pytest.raises(ValueError, match="memory marker"):
        transform_agent_row(row)


def test_agent_row_requires_a_native_trajectory():
    with pytest.raises(ValueError, match="messages or conversations"):
        transform_agent_row({"task": "not enough"})


def test_agent_row_preserves_native_tool_schema():
    tools = [
        {
            "type": "function",
            "function": {"name": "terminal", "parameters": {"type": "object"}},
        }
    ]
    transformed = transform_agent_row(
        {"messages": [{"role": "user", "content": "task"}], "tools": tools}
    )
    assert transformed["tools"] == tools
    assert transformed["tools"] is not tools


def test_agent_error_rows_are_not_successful_by_default():
    assert agent_row_is_successful({"result": None})
    assert agent_row_is_successful({"result": ""})
    assert not agent_row_is_successful({"result": "AgentTimeoutError"})


def test_only_primary_agent_trace_variants_are_kept_by_default():
    assert agent_row_is_primary_trace({})
    assert agent_row_is_primary_trace({"trace_source": "main"})
    assert agent_row_is_primary_trace({"trace_source": "swesmith"})
    assert not agent_row_is_primary_trace({"trace_source": "summarization-1-summary"})


def test_agent_max_rows_counts_emitted_successes_not_filtered_failures(monkeypatch):
    rows = [
        {
            "messages": [{"role": "user", "content": "failed"}],
            "result": "AgentTimeoutError",
        },
        {"messages": [{"role": "user", "content": "one"}], "result": None},
        {"messages": [{"role": "user", "content": "two"}], "result": None},
    ]
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: rows
    monkeypatch.setitem(sys.modules, "datasets", module)

    output = list(
        _iter_source(
            SourceSpec("agent/test", "agent", max_rows=2),
            split="train",
            reasoning_subdatasets=set(),
            analysis_tag_pairs=(),
            final_answer_markers=(),
        )
    )
    assert [row["messages"][0]["content"] for row in output] == ["one", "two"]


def test_weighted_interleave_is_seeded_and_does_not_flatten_rows():
    streams = [
        [{"source": "a", "value": 1}, {"source": "a", "value": 2}],
        [{"source": "b", "value": 3}, {"source": "b", "value": 4}],
    ]

    first = list(weighted_interleave(streams, [1, 2], seed=7))
    second = list(weighted_interleave(streams, [1, 2], seed=7))

    assert first == second
    assert sorted(row["value"] for row in first) == [1, 2, 3, 4]


def test_write_jsonl_streams_canonical_rows(tmp_path):
    output = tmp_path / "mixture.jsonl"
    rows = [
        transform_stage3_row(stage3_row()),
        transform_agent_row({"messages": [{"role": "user", "content": "task"}]}),
    ]

    count = write_jsonl(iter(rows), output, overwrite=False)

    assert count == 2
    loaded = [json.loads(line) for line in output.read_text().splitlines()]
    assert loaded[0]["data_type"] == "stage3"
    assert loaded[1]["data_type"] == "agent_trajectory"


def parse_args(*arguments):
    return build_parser().parse_args([*arguments, "--output", "out.jsonl"])


def test_cli_defaults_include_only_main_agent_dataset():
    specs = build_source_specs(parse_args())

    assert [(spec.dataset_id, spec.kind) for spec in specs] == [
        (DEFAULT_STAGE3_DATASET, "stage3"),
        (DEFAULT_AGENT_DATASET, "agent"),
    ]


def test_coldstart_is_exposed_separately_without_being_a_duplicate_default():
    specs = build_source_specs(parse_args("--include-coldstart"))

    assert [spec.dataset_id for spec in specs].count(DEFAULT_AGENT_DATASET) == 1
    assert [spec.dataset_id for spec in specs].count(COLDSTART_AGENT_DATASET) == 1


def test_explicit_agent_datasets_replace_the_implicit_default():
    specs = build_source_specs(
        parse_args(
            "--agent-dataset",
            "example/one",
            "--agent-dataset",
            "example/two",
            "--agent-weight",
            "2",
            "--agent-weight",
            "3",
        )
    )

    assert [spec.dataset_id for spec in specs[1:]] == ["example/one", "example/two"]
    assert [spec.weight for spec in specs[1:]] == [2, 3]


def test_agent_repeat_is_explicit_upsampling():
    specs = build_source_specs(parse_args("--agent-repeat", "12"))
    assert specs[0].repeat == 1
    assert specs[1].repeat == 12


def test_duplicate_dataset_ids_are_rejected():
    args = parse_args("--agent-dataset", COLDSTART_AGENT_DATASET, "--include-coldstart")
    with pytest.raises(ValueError, match="unique"):
        build_source_specs(args)

