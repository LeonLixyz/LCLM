"""Append explicitly reviewed expansion to completed grouped packs on Modal.

No generation, source selection, export, or publication occurs here. Call
pack_partition(manifest_path, manifest_sha256, partition, partitions, owner_id)
on CPU workers, then finalize(manifest_path, manifest_sha256, partitions).
Pass volume.commit as checkpoint when hosted on Modal. Nothing is launched by
importing this module. All expansion shards stay outside loader discovery until
all partitions and both lengths pass final verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import uuid
from collections import Counter
from pathlib import Path

from data.grouped_stage3_packing import LENGTHS, ShuffledSequenceWriter, exclusion_reason, canonical_digest
from data.stage3_tokenizers import DECODER, DECODER_REVISION, ENCODER, ENCODER_REVISION

SCHEMA = 'reviewed-expansion-transport-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def task_ids_digest(ids):
    return hashlib.sha256(('\n'.join(sorted(ids)) + '\n').encode()).hexdigest()


def validate_manifest(value):
    assert value['schema'] == SCHEMA and value['status'] == 'reviewed_complete'
    assert value['rows'] > 0 and value['files']
    approval = value['approval']
    assert approval['status'] == 'approved'
    assert approval['selection_sha256'] == value['selection_sha256']
    assert approval['artifacts'], 'Frozen source-specific review evidence is required'
    for digest in [value['selection_sha256'], value['task_ids_sha256'],
                   *value['base_snapshot']['manifest_sha256'].values()]:
        assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    assert set(value['base_snapshot']['manifest_sha256']) == {str(v) for v in LENGTHS}
    assert Path(value['transport_root']).is_absolute()
    assert Path(value['base_snapshot']['root']).is_absolute()
    names = [entry['name'] for entry in value['files']]
    assert len(set(names)) == len(names)
    assert all(name == Path(name).name and name.endswith('.parquet') for name in names)
    assert all(entry['rows'] > 0 and len(entry['sha256']) == 64 for entry in value['files'])
    assert sum(entry['rows'] for entry in value['files']) == value['rows']
    return value


def load_manifest(path, expected_sha256):
    assert sha256_file(path) == expected_sha256, 'Reviewed manifest changed'
    value = validate_manifest(json.loads(Path(path).read_text()))
    for artifact in value['approval']['artifacts']:
        assert Path(artifact['path']).is_absolute()
        assert sha256_file(artifact['path']) == artifact['sha256'], 'Review evidence changed'
    return value


def verify_base(manifest):
    root = Path(manifest['base_snapshot']['root'])
    for length in LENGTHS:
        output = root / f'packed-cs16-{length}'
        saved = output / 'base-native-manifest.json'
        path = saved if saved.exists() else output / 'manifest.json'
        assert sha256_file(path) == manifest['base_snapshot']['manifest_sha256'][str(length)]
        base = json.loads(path.read_text())
        assert base['status'] == 'complete' and base['all_output_byte_digests_verified']
        assert base['max_packed_length'] == length and base['reference_chunk_size'] == 16
        assert base['decoder_tokenizer_revision'] == DECODER_REVISION
        assert base['encoder_tokenizer_revision'] == ENCODER_REVISION
        assert base['expansion_included'] is False
    return root


def normalize_transport_row(row):
    from data.harvest_expansion_trace import harvest_training_messages
    row = dict(row)
    assert isinstance(row.get('task_id'), str) and row['task_id'] and '\n' not in row['task_id'], 'Transport must preserve task_id explicitly'
    assert row['compression_scope'] == 'input_segments'
    assert row['data_type'] == 'agent_trajectory'
    for key in ('messages', 'tools'):
        if isinstance(row[key], str):
            row[key] = json.loads(row[key])
        assert isinstance(row[key], list) and row[key]
    assert harvest_training_messages(row['messages'])[0] == row['messages'], 'Unstripped assistant thought/preamble'
    assert row['source_dataset'] and row['sub_dataset']
    return row


def process_item(item):
    """Exactly one tokenization per trajectory feeds both length variants."""
    from data import preprocess_for_dynamic_packing as pre
    row, identity, selection_sha = item
    row = normalize_transport_row(row)
    info = {'task_id': row['task_id'], 'row_id': identity, 'sub_dataset': row['sub_dataset'],
            'source_dataset': row['source_dataset'], 'category': 'agent'}
    example = pre.worker_process_example((row, 16, 'compression_prompt', 32768, None))
    if example is not None:
        example.update(_source_row_id=f'expansion:{selection_sha}:{row["task_id"]}',
                       _packing_category='agent', _reasoning_policy='reviewed_expansion_input_segments',
                       _expansion_task_id=row['task_id'], _expansion_selection_sha256=selection_sha,
                       _expansion_transport_row_id=identity)
        if not example.get('_skipped_max_seq_len'):
            assert len(example['base_input_ids']) == len(example['base_labels'])
            assert len(example['memory_strings']) == len(example['memory_positions'])
            assert example['memory_strings'], 'Expansion must retain its input-segment memory'
            assert any(v != -100 for v in example['base_labels'])
            for position in example['memory_positions']:
                assert example['base_labels'][position:position + 3] == [-100, -100, -100]
    return info, example


def pack_partition(manifest_path, manifest_sha256, partition, partitions, owner_id, checkpoint=None):
    import multiprocessing as mp
    import pyarrow.parquet as pq
    from data.grouped_stage3_repair_modal import bounded_imap
    from data.preprocess_for_dynamic_packing import worker_init, StreamingPacker
    checkpoint = checkpoint or (lambda: None)
    manifest = load_manifest(manifest_path, manifest_sha256)
    assert 0 <= partition < partitions and owner_id
    root = verify_base(manifest)
    audit = root / 'expansion-append' / manifest_sha256
    part = audit / f'part-{partition:03d}'
    report_path = part / 'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['manifest_sha256'] == manifest_sha256 and report['partitions'] == partitions
        return report
    part.mkdir(parents=True, exist_ok=True)
    owner_path = part / 'owner.json'
    if owner_path.exists():
        assert json.loads(owner_path.read_text())['owner_id'] == owner_id, 'Another call owns this partition'
    else:
        with owner_path.open('x') as handle:
            json.dump({'owner_id': owner_id}, handle)
    attempt = part / ('attempt-' + uuid.uuid4().hex)
    attempt.mkdir()
    versions = {}
    for length in LENGTHS:
        output = root / f'packed-cs16-{length}'
        assert not (output / 'data/expansion').exists(), 'Expansion component already published'
        stage = output / 'expansion-staging' / manifest_sha256 / f'part-{partition:03d}'
        if stage.exists():
            os.replace(stage, attempt / f'partial-{length}')
        versions[length] = {'path': stage, 'writer': ShuffledSequenceWriter(stage, length, 20260909 + partition),
                            'packer': StreamingPacker(length, 128, 20260909 + partition, 32),
                            'counts': Counter(input_rows=0, eligible_rows=0, packed_rows=0,
                                              packed_sequences=0, compressed_rows=0,
                                              overlength=0, under_18=0, processing_rejected=0),
                            'sources': {}}
    (attempt / 'started.json').write_text(json.dumps({'owner_id': owner_id, 'partition': partition}))
    checkpoint()
    ids = []
    files = []
    exclusions = []

    def inputs():
        for entry in manifest['files'][partition::partitions]:
            path = Path(manifest['transport_root']) / entry['name']
            assert sha256_file(path) == entry['sha256'], 'Transport shard changed'
            parquet = pq.ParquetFile(path)
            assert parquet.metadata.num_rows == entry['rows']
            files.append(entry)
            row_number = 0
            for batch in parquet.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    yield row, f'{entry["name"]}:{row_number}', manifest['selection_sha256']
                    row_number += 1

    with mp.get_context('spawn').Pool(8, initializer=worker_init,
            initargs=(DECODER, ENCODER, DECODER_REVISION, ENCODER_REVISION)) as pool:
        for info, example in bounded_imap(pool, process_item, inputs(), 16):
            ids.append(info['task_id'])
            for length, version in versions.items():
                count = version['counts']
                source = version['sources'].setdefault(info['sub_dataset'], Counter())
                count['input_rows'] += 1
                source['input_rows'] += 1
                reason = exclusion_reason(example, length)
                if reason:
                    count[reason] += 1
                    source[reason] += 1
                    exclusions.append({**info, 'max_length': length, 'reason': reason,
                                       'expanded_length': example['estimated_seq_len'] if example else None})
                    continue
                count['eligible_rows'] += 1
                count['compressed_rows'] += bool(example['memory_strings'])
                source['eligible_rows'] += 1
                for packed in version['packer'].add_example(example):
                    version['writer'].write(packed)
    assert len(ids) == len(set(ids)), 'Duplicate expansion task within partition'
    outputs = {}
    for length, version in versions.items():
        for packed in version['packer'].finalize():
            version['writer'].write(packed)
        version['writer'].flush()
        count = version['counts']
        count.update(version['writer'].counts['agent'])
        assert count['eligible_rows'] == count['packed_rows']
        assert count['input_rows'] == sum(count[key] for key in ('eligible_rows', 'overlength', 'under_18', 'processing_rejected'))
        outputs[str(length)] = {'path': str(version['path']), 'files': version['writer'].files,
            'output_file_bytes_sha256': version['writer'].digest.hexdigest(), 'counts': dict(count),
            'sub_datasets': {key: dict(value) for key, value in version['sources'].items()}}
    (part / 'exclusions.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in exclusions))
    result = {'status': 'complete', 'manifest_sha256': manifest_sha256, 'partition': partition,
              'partitions': partitions, 'input_files': files, 'task_ids': ids, 'outputs': outputs}
    report_path.write_text(json.dumps(result, indent=2))
    checkpoint()
    return result


def finalize(manifest_path, manifest_sha256, partitions, checkpoint=None):
    import pyarrow.parquet as pq
    from data.dynamic_packing_dataset import DynamicPackedDataset
    from data.grouped_stage3_packing_modal import tokenizers
    checkpoint = checkpoint or (lambda: None)
    manifest = load_manifest(manifest_path, manifest_sha256)
    root = verify_base(manifest)
    audit = root / 'expansion-append' / manifest_sha256
    complete = audit / 'summary.json'
    if complete.exists():
        return json.loads(complete.read_text())
    reports = [json.loads((audit / f'part-{i:03d}/report.json').read_text()) for i in range(partitions)]
    assert all(r['manifest_sha256'] == manifest_sha256 and r['partitions'] == partitions for r in reports)
    ids = [task for report in reports for task in report['task_ids']]
    assert len(ids) == len(set(ids)) == manifest['rows'], 'Duplicate or missing expansion task IDs'
    assert task_ids_digest(ids) == manifest['task_ids_sha256']
    assert sorted((entry for r in reports for entry in r['input_files']), key=lambda x: x['name']) == sorted(manifest['files'], key=lambda x: x['name'])
    decoder, encoder = tokenizers()
    results = {}
    for length in LENGTHS:
        output = root / f'packed-cs16-{length}'
        stage = output / 'expansion-staging' / manifest_sha256
        published = output / 'data/expansion'
        component = published if published.exists() else stage
        counts = Counter()
        sources = {}
        samples = []
        file_count = 0
        rows = 0
        for report in reports:
            part = report['outputs'][str(length)]
            values = dict(part['counts'])
            maximum = values.pop('max_sequence_length', 0)
            counts.update(values)
            counts['max_sequence_length'] = max(counts['max_sequence_length'], maximum)
            for key, value in part['sub_datasets'].items():
                sources.setdefault(key, Counter()).update(value)
            digest = hashlib.sha256()
            directory = component / f'part-{report["partition"]:03d}'
            files = sorted(directory.glob('*.parquet'))
            assert [p.name for p in files] == part['files']
            for path in files:
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                meta = pq.read_table(path, columns=['packing_category', 'expanded_seq_len_cs16', 'example_count'])
                assert all(v == 'agent' for v in meta['packing_category'].to_pylist())
                assert all(0 < v <= length for v in meta['expanded_seq_len_cs16'].to_pylist())
                rows += meta.num_rows
                file_count += 1
            assert digest.hexdigest() == part['output_file_bytes_sha256']
            if files and len(samples) < 3:
                table = next(pq.ParquetFile(files[0]).iter_batches(batch_size=1)).to_pydict()
                examples = pickle.loads(table['packed_batch_bytes'][0])
                dataset = DynamicPackedDataset(str(directory), decoder, encoder, 16,
                    shuffle=False, shuffle_files=False, target_length=length)
                expanded = [dataset._expand_example(ex) for ex in examples]
                assert all(ex is not None for ex in expanded)
                assert sum(ex['seq_len'] for ex in expanded) == table['expanded_seq_len_cs16'][0]
                for before, after in zip(examples, expanded):
                    assert before['_packing_category'] == 'agent'
                    labels = after['processed']['labels'][0]
                    assert sum(v != -100 for v in labels) == before['_trainable_tokens']
                    for start, end in after['processed']['memory_positions'][0]:
                        assert all(v == -100 for v in labels[start - 1:end + 1])
                batch = next(iter(dataset))
                assert batch['input_ids'].shape == batch['labels'].shape
                assert batch['input_ids'].shape[1] == length
                samples.append({'partition': report['partition'], 'examples': len(examples),
                                'actual_length': sum(ex['seq_len'] for ex in expanded)})
        assert rows == counts['packed_sequences'] and counts['input_rows'] == manifest['rows']
        results[str(length)] = {'status': 'complete', 'manifest_sha256': manifest_sha256,
            'selection_sha256': manifest['selection_sha256'], 'counts': dict(counts),
            'sub_datasets': {key: dict(value) for key, value in sources.items()}, 'files': file_count,
            'runtime_loader_samples': samples, 'all_output_byte_digests_verified': True}
    # All data for BOTH lengths is validated before publishing either component.
    for length in LENGTHS:
        output = root / f'packed-cs16-{length}'
        part = results[str(length)]
        component = output / 'data/expansion'
        if not component.exists():
            os.replace(output / 'expansion-staging' / manifest_sha256, component)
        saved_base = output / 'base-native-manifest.json'
        if not saved_base.exists():
            saved_base.write_bytes((output / 'manifest.json').read_bytes())
        assert sha256_file(saved_base) == manifest['base_snapshot']['manifest_sha256'][str(length)]
        combined = copy.deepcopy(json.loads(saved_base.read_text()))
        count = Counter(combined['counts']['agent'])
        maximum = max(count.pop('max_sequence_length', 0), part['counts'].get('max_sequence_length', 0))
        additions = dict(part['counts']); additions.pop('max_sequence_length', None)
        count.update(additions); count['max_sequence_length'] = maximum
        combined['counts']['agent'] = dict(count)
        for key, value in part['sub_datasets'].items():
            merged = Counter(combined['sub_datasets'].get(key, {})); merged.update(value)
            combined['sub_datasets'][key] = dict(merged)
        combined['raw_input_rows'] += manifest['rows']
        combined['files'] += part['files']
        combined['packed_sequences'] += part['counts']['packed_sequences']
        combined['expansion_included'] = True
        combined['base_native_manifest_sha256'] = sha256_file(saved_base)
        combined['expansion_append'] = part
        combined['expansion_reviewed_transport_manifest'] = manifest
        source_hashes = list(combined.get('raw_input_content_sha256_manifest', []))
        source_hashes.extend({'kind': 'expansion', 'root': manifest['transport_root'],
            'name': entry['name'], 'rows': entry['rows'], 'content_sha256': entry['sha256']}
            for entry in manifest['files'])
        combined['raw_input_content_sha256_manifest'] = source_hashes
        combined['raw_input_content_sha256_manifest_digest'] = canonical_digest(source_hashes)
        policies = Counter(combined.get('reasoning_policies', {}))
        policies['reviewed_expansion_input_segments'] += manifest['rows']
        combined['reasoning_policies'] = dict(policies)
        combined['runtime_loader_samples'] = list(combined.get('runtime_loader_samples', [])) + [
            {'component': 'expansion', **sample} for sample in part['runtime_loader_samples']]
        (output / 'manifest.json').write_text(json.dumps(combined, indent=2))
    complete.write_text(json.dumps(results, indent=2))
    checkpoint()
    return results
