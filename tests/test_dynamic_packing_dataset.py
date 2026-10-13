import pytest

from data.dynamic_packing_dataset import DynamicPackedDataset


MEMORY_START_ID = 100
MEMORY_END_ID = 101
MEMORY_ID = 102


class StubEmbedTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.encodings[text])


def make_dataset():
    dataset = object.__new__(DynamicPackedDataset)
    dataset.compression_ratio = 2
    dataset.memory_start_id = MEMORY_START_ID
    dataset.memory_end_id = MEMORY_END_ID
    dataset.memory_id = MEMORY_ID
    dataset.embed_tokenizer = StubEmbedTokenizer(
        {
            "first": [1, 2, 3, 4, 5],
            "second": [6, 7],
        }
    )
    return dataset


def test_expand_example_uses_source_positions_and_emits_consistent_spans():
    dataset = make_dataset()
    example = {
        "base_input_ids": [
            9,
            MEMORY_START_ID,
            MEMORY_ID,
            MEMORY_END_ID,
            8,
            MEMORY_START_ID,
            MEMORY_ID,
            MEMORY_END_ID,
            7,
        ],
        "base_labels": [9, 100, 102, 101, 8, 100, 102, 101, 7],
        "memory_strings": ["first", "second"],
        "memory_positions": [1, 5],
    }

    result = dataset._expand_example(example)

    assert result is not None
    assert result["seq_len"] == 11
    assert result["processed"] == {
        "input_ids": [[
            9,
            MEMORY_START_ID,
            MEMORY_ID,
            MEMORY_ID,
            MEMORY_ID,
            MEMORY_END_ID,
            8,
            MEMORY_START_ID,
            MEMORY_ID,
            MEMORY_END_ID,
            7,
        ]],
        "labels": [[
            9,
            -100,
            -100,
            -100,
            -100,
            -100,
            8,
            -100,
            -100,
            -100,
            7,
        ]],
        "memory_token_ids": [[[1, 2, 3, 4, 5], [6, 7]]],
        "memory_positions": [[(2, 5), (8, 9)]],
        "latent_counts": [[3, 1]],
    }
    # Expansion must not overwrite or mutate the compact source positions.
    assert example["memory_positions"] == [1, 5]


def test_expand_example_accepts_intentionally_uncompressed_example():
    dataset = make_dataset()
    example = {
        "base_input_ids": [11, 12, 13],
        "base_labels": [-100, 12, 13],
        "memory_strings": [],
        "memory_positions": [],
    }

    result = dataset._expand_example(example)

    assert result == {
        "processed": {
            "input_ids": [[11, 12, 13]],
            "labels": [[-100, 12, 13]],
            "memory_token_ids": [[]],
            "memory_positions": [[]],
            "latent_counts": [[]],
        },
        "seq_len": 3,
    }


@pytest.mark.parametrize(
    ("memory_strings", "memory_positions"),
    [
        (["first"], []),
        ([], [1]),
        (["first", "second"], [1]),
    ],
)
def test_expand_example_rejects_inconsistent_memory_metadata(
    memory_strings, memory_positions
):
    dataset = make_dataset()
    example = {
        "base_input_ids": [9, MEMORY_START_ID, MEMORY_ID, MEMORY_END_ID, 8],
        "base_labels": [9, -100, -100, -100, 8],
        "memory_strings": memory_strings,
        "memory_positions": memory_positions,
    }

    assert dataset._expand_example(example) is None


@pytest.mark.parametrize(
    "example",
    [
        {
            "base_input_ids": [9, MEMORY_START_ID, 999, MEMORY_END_ID, 8],
            "base_labels": [9, -100, -100, -100, 8],
            "memory_strings": ["first"],
            "memory_positions": [1],
        },
        {
            "base_input_ids": [9, MEMORY_START_ID, MEMORY_ID, MEMORY_END_ID, 8],
            "base_labels": [9, -100, -100, -100, 8],
            "memory_strings": ["first"],
            "memory_positions": [4],
        },
        {
            "base_input_ids": [9, MEMORY_START_ID, MEMORY_ID, MEMORY_END_ID, 8],
            "base_labels": [9, -100],
            "memory_strings": ["first"],
            "memory_positions": [1],
        },
    ],
)
def test_expand_example_rejects_malformed_compact_representation(example):
    dataset = make_dataset()

    assert dataset._expand_example(example) is None

