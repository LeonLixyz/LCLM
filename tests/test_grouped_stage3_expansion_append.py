import copy
import json
from pathlib import Path

import pytest

from data import grouped_stage3_expansion_append as append
from data.grouped_stage3_packing import LENGTHS, ShuffledSequenceWriter, exclusion_reason


def row(task='task-1'):
    text = 'The river gauge reading is forty two. ' * 90
    return {'task_id': task, 'source_dataset': 'fixture', 'sub_dataset': 'fixture_expansion',
            'compression_scope': 'input_segments', 'data_type': 'agent_trajectory',
            'messages': [
                {'role': 'system', 'content': 'Use expand to inspect a segment, then answer with FINAL.'},
                {'role': 'user', 'content': 'seg_1\n<|memory_start|>' + text + '<|memory_end|>\nWhat is the reading?'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call-1', 'type': 'function',
                    'function': {'name': 'expand', 'arguments': {'segment_id': 'seg_1'}}}]},
                {'role': 'tool', 'tool_call_id': 'call-1', 'name': 'expand', 'content': text},
                {'role': 'assistant', 'content': 'FINAL: forty two'}],
            'tools': [{'type': 'function', 'function': {'name': 'expand', 'description': 'Read a segment',
                'parameters': {'type': 'object', 'properties': {'segment_id': {'type': 'string'}},
                               'required': ['segment_id']}}}]}


def manifest():
    return {'schema': append.SCHEMA, 'status': 'reviewed_complete', 'selection_sha256': 'a' * 64,
            'approval': {'status': 'approved', 'selection_sha256': 'a' * 64,
                         'artifacts': [{'path': '/review.json', 'sha256': 'b' * 64}]},
            'transport_root': '/transport', 'rows': 2, 'task_ids_sha256': 'c' * 64,
            'files': [{'name': 'part.parquet', 'rows': 2, 'sha256': 'd' * 64}],
            'base_snapshot': {'root': '/packs', 'manifest_sha256': {str(v): 'e' * 64 for v in LENGTHS}}}


def test_manifest_requires_explicit_matching_approval_and_safe_shard_names():
    append.validate_manifest(manifest())
    for change in ('unreviewed', 'mismatch', 'traversal'):
        value = manifest()
        if change == 'unreviewed': value['status'] = 'generated'
        elif change == 'mismatch': value['approval']['selection_sha256'] = 'f' * 64
        else: value['files'][0]['name'] = '../part.parquet'
        with pytest.raises(AssertionError): append.validate_manifest(value)


def test_transport_preserves_tool_memory_and_rejects_unstripped_thought():
    original = row()
    transport = {**original, 'messages': json.dumps(original['messages']), 'tools': json.dumps(original['tools'])}
    assert append.normalize_transport_row(transport) == original
    dirty = row(); dirty['messages'][2]['reasoning_content'] = 'private thought'
    with pytest.raises(AssertionError): append.normalize_transport_row(dirty)
    dirty = row(); dirty.pop('task_id')
    with pytest.raises(AssertionError): append.normalize_transport_row(dirty)


@pytest.mark.parametrize('length,expected', [(16385, ('overlength', None)), (32769, ('overlength', 'overlength'))])
def test_once_processed_expansion_feeds_exact_both_length_decisions(monkeypatch, length, expected):
    from data import preprocess_for_dynamic_packing as pre
    calls = []
    compact = {'base_input_ids': [1, 2, 3, 4], 'base_labels': [-100, -100, -100, 4],
               'memory_positions': [0], 'memory_strings': ['retained input segment'],
               'estimated_seq_len': length, '_trainable_tokens': 1}
    if length > 32768: compact['_skipped_max_seq_len'] = True
    def process(payload):
        calls.append(payload)
        return copy.deepcopy(compact)
    monkeypatch.setattr(pre, 'worker_process_example', process)
    info, result = append.process_item((row(), 'part.parquet:0', 'a' * 64))
    assert len(calls) == 1 and calls[0][3] == 32768
    assert info['category'] == result['_packing_category'] == 'agent'
    assert result['memory_strings'] == compact['memory_strings']
    assert tuple(exclusion_reason(result, v) for v in LENGTHS) == expected


def test_reviewed_append_actual_worker_loader_and_immutable_mixed_outputs(tmp_path):
    """Tiny real-tokenizer append, no generation/selection or production data."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.grouped_stage3_packing_modal import tokenizers
    from data.packed_file_discovery import discover_packed_parquet_files
    tokenizers()
    _, example = append.process_item((row('fixture-base'), 'fixture:0', 'a' * 64))
    base_root = tmp_path / 'grouped'
    base_hashes = {}
    mixed_hashes = {}
    for length in LENGTHS:
        output = base_root / f'packed-cs16-{length}'
        mixed = output / 'data/mixed/part-000'
        base_example = {**example, '_packing_category': 'other'}
        writer = ShuffledSequenceWriter(mixed, length, 7)
        writer.write([base_example]); writer.flush()
        mixed_hashes[length] = append.sha256_file(mixed / writer.files[0])
        base = {'status': 'complete', 'all_output_byte_digests_verified': True,
                'max_packed_length': length, 'reference_chunk_size': 16,
                'decoder_tokenizer_revision': append.DECODER_REVISION,
                'encoder_tokenizer_revision': append.ENCODER_REVISION,
                'expansion_included': False, 'counts': {'agent': {}, 'other': dict(writer.counts['other']), 'reasoning': {}},
                'sub_datasets': {}, 'raw_input_rows': 1, 'files': 1, 'packed_sequences': 1}
        path = output / 'manifest.json'; path.write_text(json.dumps(base))
        base_hashes[str(length)] = append.sha256_file(path)
    transport = tmp_path / 'transport'; transport.mkdir()
    rows = [row('task-1'), row('task-2')]
    rows = [{**value, 'messages': json.dumps(value['messages']), 'tools': json.dumps(value['tools'])} for value in rows]
    shard = transport / 'accepted.parquet'
    pq.write_table(pa.Table.from_pylist(rows), shard)
    review = tmp_path / 'review.json'; review.write_text('{"status":"approved","fixture":true}')
    value = manifest()
    value.update(transport_root=str(transport), task_ids_sha256=append.task_ids_digest(['task-1', 'task-2']),
                 files=[{'name': shard.name, 'rows': 2, 'sha256': append.sha256_file(shard)}],
                 base_snapshot={'root': str(base_root), 'manifest_sha256': base_hashes})
    value['approval']['artifacts'] = [{'path': str(review), 'sha256': append.sha256_file(review)}]
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(value))
    digest = append.sha256_file(path)
    result = append.pack_partition(path, digest, 0, 1, 'fixture-owner')
    assert result['task_ids'] == ['task-1', 'task-2']
    for length in LENGTHS:
        output = base_root / f'packed-cs16-{length}'
        assert all('/data/mixed/' in p for p in discover_packed_parquet_files(output))
    final = append.finalize(path, digest, 1)
    for length in LENGTHS:
        output = base_root / f'packed-cs16-{length}'
        files = discover_packed_parquet_files(output)
        assert len(files) == 2 and any('/data/expansion/' in p for p in files)
        assert append.sha256_file(output / 'data/mixed/part-000/packed-000000.parquet') == mixed_hashes[length]
        assert append.sha256_file(output / 'base-native-manifest.json') == base_hashes[str(length)]
        combined = json.loads((output / 'manifest.json').read_text())
        assert combined['expansion_included'] and combined['raw_input_rows'] == 3
        assert combined['counts']['agent']['packed_rows'] == 2
        assert combined['raw_input_content_sha256_manifest'][-1]['content_sha256'] == append.sha256_file(shard)
        assert combined['raw_input_content_sha256_manifest_digest'] == append.canonical_digest(combined['raw_input_content_sha256_manifest'])
        assert combined['reasoning_policies']['reviewed_expansion_input_segments'] == 2
        assert final[str(length)]['runtime_loader_samples'][0]['examples'] == 2
    assert append.pack_partition(path, digest, 0, 1, 'fixture-owner') == result
    assert append.finalize(path, digest, 1) == final
