"""Bounded content certificates and the unchanged grouped-pack loader audit.

Only checksum orchestration changes: every packed byte is still read and hashed,
and each certificate binds full relative paths, sizes, file hashes, report bytes,
the exact source configuration, and the original ordered concatenation digest.
"""
from pathlib import Path
import hashlib
import json
import time

from data import grouped_stage3_packing_modal as core
from data.grouped_stage3_packing import CATEGORIES, LENGTHS, canonical_digest
from data.stage3_tokenizers import DECODER, DECODER_REVISION, ENCODER, ENCODER_REVISION

tokenizers = core.tokenizers
file_sha256 = core.file_sha256


def bindings(root, partition, expected_manifest_sha):
    root = Path(root)
    report_path = root / 'partitions' / f'part-{partition:03d}' / 'report.json'
    config_path = root / 'source-manifest.json'
    config = json.loads(config_path.read_text())
    report = json.loads(report_path.read_text())
    assert canonical_digest(config) == expected_manifest_sha
    assert report['source_manifest_sha256'] == expected_manifest_sha
    assert report['partition'] == partition and report['status'] == 'complete'
    return report, {'partition': partition, 'root': str(root),
                    'source_manifest_sha256': expected_manifest_sha,
                    'source_manifest_file_sha256': file_sha256(config_path),
                    'partition_report_relative_path': str(report_path.relative_to(root)),
                    'partition_report_sha256': file_sha256(report_path)}


def certificate_path(audit_root, partition):
    return Path(audit_root) / 'certificates' / f'part-{partition:03d}.json'


def validate_certificate(root, audit_root, partition, expected_manifest_sha):
    """Verify immutable bindings and metadata without redundantly rereading bytes."""
    root = Path(root)
    report, expected = bindings(root, partition, expected_manifest_sha)
    certificate = json.loads(certificate_path(audit_root, partition).read_text())
    assert certificate['schema'] == 'grouped-content-certificate-v1'
    assert certificate['status'] == 'complete' and certificate['bindings'] == expected
    assert set(certificate['outputs']) == {str(v) for v in LENGTHS}
    for length in LENGTHS:
        result = report['outputs'][str(length)]
        output = root / f'packed-cs16-{length}' / 'data/mixed' / f'part-{partition:03d}'
        assert Path(result['path']) == output
        files = sorted(output.glob('*.parquet'))
        assert [p.name for p in files] == result['files']
        audited = certificate['outputs'][str(length)]
        assert audited['output_file_bytes_sha256'] == result['output_file_bytes_sha256']
        assert len(audited['files']) == len(files)
        for path, entry in zip(files, audited['files']):
            assert not path.is_symlink()
            assert entry['relative_path'] == str(path.relative_to(root))
            assert entry['bytes'] == path.stat().st_size
            assert len(entry['sha256']) == 64 and all(c in '0123456789abcdef' for c in entry['sha256'])
        assert audited['files_binding_sha256'] == canonical_digest(audited['files'])
    return certificate


def audit_partition(root, audit_root, partition, expected_manifest_sha, checkpoint=lambda: None):
    import pyarrow.parquet as pq
    root, audit_root = Path(root), Path(audit_root)
    report, expected = bindings(root, partition, expected_manifest_sha)
    target = certificate_path(audit_root, partition)
    if target.exists():
        return validate_certificate(root, audit_root, partition, expected_manifest_sha)
    certificate = {'schema': 'grouped-content-certificate-v1', 'status': 'complete',
                   'bindings': expected, 'outputs': {}, 'started_at_unix': time.time()}
    for length in LENGTHS:
        result = report['outputs'][str(length)]
        output = root / f'packed-cs16-{length}' / 'data/mixed' / f'part-{partition:03d}'
        assert Path(result['path']) == output
        files = sorted(output.glob('*.parquet'))
        assert [p.name for p in files] == result['files']
        aggregate = hashlib.sha256()
        entries = []
        for path in files:
            assert not path.is_symlink()
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    aggregate.update(chunk)
                    digest.update(chunk)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            entries.append({'relative_path': str(path.relative_to(root)), 'bytes': after.st_size,
                            'sha256': digest.hexdigest(), 'parquet_rows': pq.ParquetFile(path).metadata.num_rows})
        assert aggregate.hexdigest() == result['output_file_bytes_sha256'], str(output)
        certificate['outputs'][str(length)] = {'files': entries,
            'files_binding_sha256': canonical_digest(entries),
            'output_file_bytes_sha256': aggregate.hexdigest()}
    assert bindings(root, partition, expected_manifest_sha)[1] == expected
    certificate['completed_at_unix'] = time.time()
    target.parent.mkdir(parents=True, exist_ok=True)
    # A partition has one dispatched owner; platform retries reuse its certificate.
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(certificate, indent=2))
    temporary.replace(target)
    checkpoint()
    return validate_certificate(root, audit_root, partition, expected_manifest_sha)


def original_finalizer_terminal(graph, call_id):
    found = []
    def visit(nodes):
        for node in nodes:
            if node.function_call_id == call_id:
                found.append(node.status.name)
            visit(node.children)
    visit(graph)
    assert found, 'Original finalizer is absent from the known coordinator graph'
    assert all(status in {'SUCCESS', 'FAILURE', 'TERMINATED', 'TIMEOUT'} for status in found), found
    return {'function_call_id': call_id, 'input_statuses': found}


def finalize_verified(root, audit_root, expected_manifest_sha, checkpoint=lambda: None, *, partitions=64, expected_raw=None, before_publish=lambda: None):
    ROOT = Path(root)
    pilot = False
    PARTITIONS = partitions
    EXPECTED_RAW = expected_raw or core.EXPECTED_RAW
    import pickle
    import hashlib
    import tempfile
    import pyarrow.parquet as pq
    from collections import Counter
    from data.dynamic_packing_dataset import DynamicPackedDataset
    reports_root = ROOT / ('pilot' if pilot else 'partitions')
    reports = [json.loads(p.read_text()) for p in sorted(reports_root.glob('part-*/report.json'))]
    assert len(reports) == (1 if pilot else PARTITIONS), len(reports)
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    assert canonical_digest(manifest) == expected_manifest_sha
    certificates = {r['partition']: validate_certificate(ROOT, audit_root, r['partition'], expected_manifest_sha) for r in reports}
    assert all(r['source_manifest_sha256'] == canonical_digest(manifest) for r in reports)
    raw = Counter()
    policies = Counter()
    for report in reports:
        raw.update(report['raw_category_counts'])
        policies.update(report['reasoning_policies'])
    if not pilot:
        assert sum(raw.values()) == sum(EXPECTED_RAW.values()), raw
    decoder, encoder = tokenizers()
    versions = {}
    for length in LENGTHS:
        output_root = ROOT / ('pilot' if pilot else '') / f'packed-cs16-{length}'
        counts = {category: Counter() for category in CATEGORIES}
        source_counts = {}
        total_files = 0
        total_metadata_sequences = 0
        validation = []
        for report in reports:
            result = report['outputs'][str(length)]
            for category in CATEGORIES:
                part_counts = dict(result['counts'][category])
                maximum = part_counts.pop('max_sequence_length', 0)
                counts[category].update(part_counts)
                counts[category]['max_sequence_length'] = max(counts[category]['max_sequence_length'], maximum)
            for sub, count in result['sub_datasets'].items():
                source_counts.setdefault(sub, Counter()).update(count)
            files = sorted(Path(result['path']).glob('*.parquet'))
            assert [f.name for f in files] == result['files']
            total_files += len(files)
            certificate = certificates[report['partition']]['outputs'][str(length)]
            assert certificate['output_file_bytes_sha256'] == result['output_file_bytes_sha256']
            for path, audited in zip(files, certificate['files']):
                rows = pq.ParquetFile(path).metadata.num_rows
                assert rows == audited['parquet_rows']
                total_metadata_sequences += rows
        assert total_metadata_sequences == sum(c['packed_sequences'] for c in counts.values())
        chosen = sorted({0, len(reports) // 2, len(reports) - 1})
        with tempfile.TemporaryDirectory(prefix='grouped-loader-') as temp:
            for report_index in chosen:
                part = reports[report_index]['outputs'][str(length)]
                sampled = set()
                for name in part['files']:
                    path = Path(part['path']) / name
                    table = pq.read_table(path)
                    for category in CATEGORIES:
                        if category in sampled:
                            continue
                        indexes = [i for i, value in enumerate(table['packing_category'].to_pylist()) if value == category]
                        if not indexes:
                            continue
                        index = indexes[0]
                        # Real loader receives the exact saved sequence bytes.
                        import pyarrow as pa
                        sample_path = Path(temp) / f'{report_index}-{category}.parquet'
                        pq.write_table(table.slice(index, 1), sample_path)
                        sample_dataset = DynamicPackedDataset(str(temp), decoder, encoder, 16,
                            shuffle=False, shuffle_files=False, target_length=length)
                        examples = pickle.loads(table['packed_batch_bytes'][index].as_py())
                        expanded = [sample_dataset._expand_example(ex) for ex in examples]
                        assert all(ex is not None for ex in expanded)
                        actual = sum(ex['seq_len'] for ex in expanded)
                        assert actual == table['expanded_seq_len_cs16'][index].as_py()
                        assert actual <= length
                        assert all(ex['_packing_category'] == category for ex in examples)
                        for before, after in zip(examples, expanded):
                            labels = after['processed']['labels'][0]
                            assert sum(x != -100 for x in labels) == before['_trainable_tokens']
                            for start, end in after['processed']['memory_positions'][0]:
                                assert all(x == -100 for x in labels[start - 1:end + 1])
                        validation.append({'partition': reports[report_index]['partition'],
                            'category': category, 'examples': len(examples),
                            'actual_length': actual, 'file': str(path), 'parquet_row': index})
                        sampled.add(category)
                    if sampled == set(CATEGORIES):
                        break
                assert sampled == {c for c in CATEGORIES if part['counts'][c].get('packed_rows')}, sampled
            rank_lengths = []
            for rank in range(2):
                dataset = DynamicPackedDataset(temp, decoder, encoder, 16, num_processes=2,
                    process_rank=rank, seed=20260909, shuffle=True, shuffle_files=True,
                    drop_last_files=False, target_length=length)
                rank_lengths.append(len(dataset))
                for batch in dataset:
                    assert batch['input_ids'].shape == batch['labels'].shape
                    assert batch['input_ids'].shape[1] == length
                    assert batch['labels'].ne(-100).any()
                    assert sum(batch['sample_lens'][0]) <= length
            assert len(set(rank_lengths)) == 1
        result = {'status': 'complete', 'max_packed_length': length, 'reference_chunk_size': 16,
            'source_manifest_sha256': canonical_digest(manifest),
            'counts': {k: dict(v) for k, v in counts.items()},
            'sub_datasets': {k: dict(v) for k, v in source_counts.items()},
            'files': total_files, 'packed_sequences': total_metadata_sequences,
            'runtime_loader_samples': validation, 'two_rank_sample_loader_lengths': rank_lengths,
            'raw_input_rows': sum(raw.values()), 'reasoning_policies': dict(policies),
            'decoder_tokenizer': DECODER, 'decoder_tokenizer_revision': DECODER_REVISION,
            'encoder_tokenizer': ENCODER, 'encoder_tokenizer_revision': ENCODER_REVISION,
            'category_policy': 'each sequence has one category; completed sequences are shuffled into shards',
            'shuffle_buffer_sequences': 512, 'expansion_included': False,
            'validation_scope': 'all rows checked while writing; all file metadata recounted; representative actual runtime loader checks; no model training',
            'format': 'dynamic packed; decoder IDs/labels pretokenized; memory strings encoder-tokenized at runtime',
            'training_loader': 'DynamicPackedDataset with compression_ratio=16 and target_length=max_packed_length; shuffle=True and shuffle_files=True',
            'source_terms': 'Original source provenance/terms review remains separate from technical trainability'}
        result['raw_input_content_sha256_manifest'] = [entry for report in reports for entry in report['input_files']]
        result['raw_input_content_sha256_manifest_digest'] = canonical_digest(result['raw_input_content_sha256_manifest'])
        result['all_output_byte_digests_verified'] = True
        result['parallel_content_audit'] = {
            'schema': 'grouped-content-certificate-v1',
            'certificates': [{'relative_path': str(certificate_path(audit_root, r['partition']).relative_to(ROOT)),
                              'sha256': file_sha256(certificate_path(audit_root, r['partition']))}
                             for r in reports],
            'verified_output_bytes': sum(entry['bytes'] for certificate in certificates.values()
                                         for entry in certificate['outputs'][str(length)]['files'])}
        versions[str(length)] = result
    # All byte certificates and both runtime loaders passed before publication.
    before_publish()
    for length, result in versions.items():
        (ROOT / f'packed-cs16-{length}' / 'manifest.json').write_text(json.dumps(result, indent=2))
    from data.grouped_stage3_expansion_append import verify_base
    verify_base({'base_snapshot': {'root': str(ROOT), 'manifest_sha256': {length: file_sha256(ROOT / f'packed-cs16-{length}' / 'manifest.json') for length in versions}}})
    (reports_root / 'summary.json').write_text(json.dumps(versions, indent=2))
    checkpoint()
    return versions
