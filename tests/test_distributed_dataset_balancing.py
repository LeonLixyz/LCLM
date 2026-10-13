import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.dynamic_packing_dataset import DynamicPackedDataset
from data.stateful_dataset import StatefulDataset


MEMORY_START_ID = 100
MEMORY_END_ID = 101
MEMORY_ID = 102


class StubDecoderTokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {
            "<|memory_start|>": MEMORY_START_ID,
            "<|memory_end|>": MEMORY_END_ID,
            "<|memory|>": MEMORY_ID,
        }[token]


class StubEmbedTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _write_parquet_dir(path, row_counts, dataset_kind):
    path.mkdir()
    global_row = 0
    for file_idx, row_count in enumerate(row_counts):
        row_ids = list(range(global_row, global_row + row_count))
        global_row += row_count
        if dataset_kind == "stateful":
            table = pa.table(
                {
                    "prompt": pa.array(
                        [f"prompt-{row_id}" for row_id in row_ids],
                        type=pa.string(),
                    ),
                    "target": pa.array(
                        [f"target-{row_id}" for row_id in row_ids],
                        type=pa.string(),
                    ),
                }
            )
        else:
            packed_rows = []
            for row_id in row_ids:
                example = {
                    "base_input_ids": [row_id],
                    "base_labels": [row_id],
                    "memory_strings": [],
                    "memory_positions": [],
                }
                packed_rows.append(pickle.dumps([example]))
            table = pa.table(
                {
                    "packed_batch_bytes": pa.array(
                        packed_rows,
                        type=pa.binary(),
                    )
                }
            )
        pq.write_table(table, path / f"part-{file_idx:02d}.parquet")
    return path


def _make_dataset(dataset_kind, parquet_dir, **kwargs):
    common = {
        "parquet_path": str(parquet_dir),
        "num_processes": kwargs.pop("num_processes", 1),
        "process_rank": kwargs.pop("process_rank", 0),
        "seed": kwargs.pop("seed", 17),
        "shuffle": kwargs.pop("shuffle", False),
        "shuffle_files": kwargs.pop("shuffle_files", False),
        "drop_last_files": kwargs.pop("drop_last_files", True),
    }
    assert not kwargs
    if dataset_kind == "stateful":
        return StatefulDataset(**common)
    return DynamicPackedDataset(
        **common,
        decoder_tokenizer=StubDecoderTokenizer(),
        embed_tokenizer=StubEmbedTokenizer(),
        compression_ratio=2,
    )


def _collect_row_ids(dataset_kind, dataset):
    if dataset_kind == "stateful":
        return [int(item["prompt"].removeprefix("prompt-")) for item in dataset]
    return [int(item["input_ids"][0, 0]) for item in dataset]


@pytest.mark.parametrize("dataset_kind", ["stateful", "dynamic"])
def test_drop_mode_balances_rows_and_drops_only_global_remainder(
    tmp_path, dataset_kind
):
    # The leading empty file exercises duplicate parquet-prefix offsets in the
    # metadata plan. Unequal file sizes would be badly imbalanced by file sharding.
    parquet_dir = _write_parquet_dir(
        tmp_path / dataset_kind, [0, 5, 2, 4], dataset_kind
    )
    ranks = [
        _make_dataset(
            dataset_kind,
            parquet_dir,
            num_processes=3,
            process_rank=rank,
            drop_last_files=True,
        )
        for rank in range(3)
    ]

    rows = [_collect_row_ids(dataset_kind, dataset) for dataset in ranks]

    assert [len(dataset) for dataset in ranks] == [3, 3, 3]
    assert rows == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert sorted(row for rank_rows in rows for row in rank_rows) == list(range(9))
    assert all(len(dataset._source_manifest) == 4 for dataset in ranks)


@pytest.mark.parametrize("dataset_kind", ["stateful", "dynamic"])
def test_non_drop_mode_balances_by_padding_global_prefix(tmp_path, dataset_kind):
    parquet_dir = _write_parquet_dir(
        tmp_path / dataset_kind, [5, 2, 4], dataset_kind
    )
    ranks = [
        _make_dataset(
            dataset_kind,
            parquet_dir,
            num_processes=3,
            process_rank=rank,
            drop_last_files=False,
        )
        for rank in range(3)
    ]

    rows = [_collect_row_ids(dataset_kind, dataset) for dataset in ranks]

    assert [len(dataset) for dataset in ranks] == [4, 4, 4]
    assert rows == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 0]]
    flattened = [row for rank_rows in rows for row in rank_rows]
    assert sorted(set(flattened)) == list(range(11))
    assert flattened.count(0) == 2


@pytest.mark.parametrize("dataset_kind", ["stateful", "dynamic"])
def test_total_rows_smaller_than_world_size_fails_clearly(tmp_path, dataset_kind):
    parquet_dir = _write_parquet_dir(
        tmp_path / dataset_kind, [0, 1, 0], dataset_kind
    )

    with pytest.raises(ValueError, match="found 1 total rows for 2 ranks"):
        _make_dataset(
            dataset_kind,
            parquet_dir,
            num_processes=2,
            process_rank=0,
        )


@pytest.mark.parametrize("dataset_kind", ["stateful", "dynamic"])
def test_resume_at_assignment_boundary_is_exact_and_fingerprinted(
    tmp_path, dataset_kind
):
    parquet_dir = _write_parquet_dir(
        tmp_path / dataset_kind, [2, 3, 4], dataset_kind
    )
    original = _make_dataset(
        dataset_kind,
        parquet_dir,
        num_processes=2,
        process_rank=0,
        seed=29,
        shuffle=True,
    )
    iterator = iter(original)
    next(iterator)
    next(iterator)  # Last row of the first assigned parquet range.
    checkpoint = original.state_dict()
    expected_remaining = _collect_row_ids(dataset_kind, iterator)

    resumed = _make_dataset(
        dataset_kind,
        parquet_dir,
        num_processes=2,
        process_rank=0,
        seed=29,
        shuffle=True,
    )
    resumed.load_state_dict(checkpoint)

    assert checkpoint["row_assignments"] == resumed._row_assignments
    assert checkpoint["current_assignment"] == resumed._row_assignments[0]
    assert checkpoint["row_idx"] == 2
    assert _collect_row_ids(dataset_kind, resumed) == expected_remaining

    changed_seed = _make_dataset(
        dataset_kind,
        parquet_dir,
        num_processes=2,
        process_rank=0,
        seed=30,
        shuffle=True,
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        changed_seed.load_state_dict(checkpoint)

    changed_shuffle = _make_dataset(
        dataset_kind,
        parquet_dir,
        num_processes=2,
        process_rank=0,
        seed=29,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        changed_shuffle.load_state_dict(checkpoint)


def test_stateful_dataset_preserves_native_agent_columns_and_adds_aliases(tmp_path):
    parquet_dir = tmp_path / "agent"
    parquet_dir.mkdir()
    native_row = {
        "source_prompt": "fallback prompt",
        "source_target": "fallback target",
        "messages": [
            {"role": "user", "content": "Search for the paper."},
            {"role": "assistant", "content": "I'll search now."},
        ],
        "tools": [
            {"name": "search", "description": "Search the web."},
        ],
        "trace_id": "trace-123",
    }
    pq.write_table(pa.Table.from_pylist([native_row]), parquet_dir / "part.parquet")
    dataset = StatefulDataset(
        parquet_path=str(parquet_dir),
        shuffle=False,
        shuffle_files=False,
        prompt_column="source_prompt",
        target_column="source_target",
    )

    row = next(iter(dataset))

    assert row["prompt"] == native_row["source_prompt"]
    assert row["target"] == native_row["source_target"]
    assert row["messages"] == native_row["messages"]
    assert row["tools"] == native_row["tools"]
    assert row["trace_id"] == native_row["trace_id"]

