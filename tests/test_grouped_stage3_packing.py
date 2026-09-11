import pickle

import pytest

from data.grouped_stage3_packing import (
    CATEGORIES, LENGTHS, REASONING_SOURCES, ShuffledSequenceWriter,
    apply_reasoning_policy, category_for, make_packers, validate_pack,
    exclusion_reason, proportional_interleave,
)


def example(key, category, length):
    return {'_source_row_id': key, '_packing_category': category,
            '_sub_dataset': category, '_trainable_tokens': 1,
            'estimated_seq_len': length, 'base_input_ids': [1] * length,
            'base_labels': [-100] * (length - 1) + [1],
            'memory_positions': [], 'memory_strings': []}


def test_category_taxonomy_is_explicit():
    for source in REASONING_SOURCES:
        assert category_for('base', source) == 'reasoning'
    assert category_for('native', 'reasoning_data') == 'agent'
    assert category_for('base', 'dolci_instruct') == 'other'
    assert category_for('base', 'nemotron_math_4plus') == 'other'
    with pytest.raises(ValueError):
        category_for('base', 'new_unknown_source')
    with pytest.raises(ValueError):
        category_for('expansion', 'unreviewed')


@pytest.mark.parametrize('length', LENGTHS)
def test_packers_keep_categories_and_length_boundaries(tmp_path, length):
    packers = make_packers(length, 73)
    writer = ShuffledSequenceWriter(tmp_path, length, 73, buffer_size=9)
    expected = set()
    for category in CATEGORIES:
        for index, size in enumerate([18, 300, length - 18, length, 101]):
            key = f'{category}:{index}'
            expected.add(key)
            for batch in packers[category].add_example(example(key, category, size)):
                writer.write(batch)
    for packer in packers.values():
        for batch in packer.finalize():
            writer.write(batch)
    writer.flush()
    import pyarrow.parquet as pq
    observed = []
    categories = set()
    for path in sorted(tmp_path.glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            batch = pickle.loads(row['packed_batch_bytes'])
            category, actual = validate_pack(batch, length)
            assert category == row['packing_category']
            assert actual == row['expanded_seq_len_cs16']
            categories.add(category)
            observed.extend(item['_source_row_id'] for item in batch)
    assert len(observed) == len(set(observed))
    assert set(observed) == expected
    assert categories == set(CATEGORIES)


def test_mixed_category_pack_rejected():
    with pytest.raises(AssertionError):
        validate_pack([example('a', 'agent', 18), example('r', 'reasoning', 18)], 16384)


def test_overlength_pack_rejected():
    with pytest.raises(AssertionError):
        validate_pack([example('a', 'agent', 16385)], 16384)


def test_existing_cot_rewrite_is_preserved():
    prompt = [{'role': 'user', 'content': 'Solve 2+2.'}]
    row = {'sub_dataset': 'reasoning_data', 'prompt': prompt, 'compression_prompt': prompt,
           'target': '<|memory_start|>2 and 2<|memory_end|>Final Answer: 4'}
    corrected, policy = apply_reasoning_policy(row, 'base:x:0', None)
    assert corrected == row
    assert policy == 'preserved_cot_memory'


def test_math_reconstruction_preserves_original_compression_policy():
    prompt = [{'role': 'user', 'content': 'Solve 2+2.'}]
    target = 'Add two pairs together.\nFinal Answer: 4'
    row = {'sub_dataset': 'nemotron_math_4plus', 'prompt': prompt,
           'compression_prompt': [{'role': 'user', 'content': '<|memory_start|>Solve 2+2.<|memory_end|>'}],
           'target': target}
    corrected, policy = apply_reasoning_policy(row, 'base:file:0', None)
    assert corrected == row
    assert corrected['target'] == target
    assert policy == 'not_reasoning'
    assert row['compression_prompt'] != prompt


def test_existing_reasoning_prompt_compression_fails_closed():
    with pytest.raises(ValueError, match='differs'):
        apply_reasoning_policy({'sub_dataset': 'dolci_think', 'prompt': [],
                                'compression_prompt': ['bad'], 'target': 'answer'}, 'base:x:0', None)


def test_exact_length_exclusions_including_worker_sentinel():
    sentinel = {'_skipped_max_seq_len': True, 'estimated_seq_len': 32769}
    for length in LENGTHS:
        assert exclusion_reason(sentinel, length) == 'overlength'
    assert exclusion_reason({'estimated_seq_len': 16385}, 16384) == 'overlength'
    assert exclusion_reason({'estimated_seq_len': 16385}, 32768) is None
    assert exclusion_reason({'estimated_seq_len': 16384}, 16384) is None


def test_weighted_interleave_preserves_rows_and_mixes_native():
    output = list(proportional_interleave({'base': [f'b{i}' for i in range(160)],
                                          'native': [f'n{i}' for i in range(10)]},
                                         {'base': 160, 'native': 10}))
    assert len(set(output)) == 170
    assert [v for v in output if v.startswith('b')] == [f'b{i}' for i in range(160)]
    assert [v for v in output if v.startswith('n')] == [f'n{i}' for i in range(10)]
    assert output[1] == 'n0'
    assert output.index('n9') < 160


def test_no_cot_is_invented_for_math_reconstruction():
    prompt = [{'role': 'user', 'content': 'Solve 2+2.'}]
    row = {'sub_dataset': 'nemotron_math_4plus', 'prompt': prompt,
           'compression_prompt': [{'role': 'user', 'content': '<|memory_start|>question<|memory_end|>'}],
           'target': 'Work\nFinal Answer: 4'}
    corrected, policy = apply_reasoning_policy(row, 'base:file:1', None)
    assert corrected == row
    assert policy == 'not_reasoning'
