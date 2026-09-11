import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from data import grouped_stage3_parallel_audit as audit
from data.grouped_stage3_packing import CATEGORIES, LENGTHS, ShuffledSequenceWriter, canonical_digest


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / 'grouped'
    root.mkdir()
    config = {'fixture': True, 'lengths': list(LENGTHS), 'categories': list(CATEGORIES),
              'decoder_revision': audit.DECODER_REVISION, 'encoder_revision': audit.ENCODER_REVISION}
    (root / 'source-manifest.json').write_text(json.dumps(config))
    sha = canonical_digest(config)
    report = {'status': 'complete', 'partition': 0, 'source_manifest_sha256': sha,
              'raw_category_counts': {c: 1 for c in CATEGORIES}, 'reasoning_policies': {},
              'input_files': [], 'outputs': {}}
    for length in LENGTHS:
        output = root / f'packed-cs16-{length}' / 'data/mixed/part-000'
        writer = ShuffledSequenceWriter(output, length, 17)
        for category in CATEGORIES:
            item = {'_source_row_id': category, '_packing_category': category,
                    '_sub_dataset': category, '_trainable_tokens': 1,
                    'estimated_seq_len': 18, 'base_input_ids': [1] * 18,
                    'base_labels': [-100] * 17 + [1], 'memory_positions': [], 'memory_strings': []}
            writer.write([item])
        writer.flush()
        report['outputs'][str(length)] = {'path': str(output), 'files': writer.files,
            'output_file_bytes_sha256': writer.digest.hexdigest(),
            'counts': {k: dict(v) for k, v in writer.counts.items()}, 'sub_datasets': {}}
    report_path = root / 'partitions/part-000/report.json'
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report))
    return root, root / 'parallel-audit-v1', sha, report


def test_certificates_bind_all_files_and_resume(fixture):
    root, store, sha, report = fixture
    value = audit.audit_partition(root, store, 0, sha)
    assert audit.audit_partition(root, store, 0, sha) == value
    for length in LENGTHS:
        files = value['outputs'][str(length)]['files']
        assert len(files) == 1 and files[0]['relative_path'].startswith(f'packed-cs16-{length}/data/mixed/part-000/')
        assert files[0]['sha256'] == audit.file_sha256(root / files[0]['relative_path'])
        assert files[0]['parquet_rows'] == 3


@pytest.mark.parametrize('change', ['report', 'config', 'relative_path', 'size', 'digest'])
def test_certificate_rejects_changed_bindings(fixture, change):
    root, store, sha, report = fixture
    audit.audit_partition(root, store, 0, sha)
    if change in ('report', 'config'):
        path = root / ('partitions/part-000/report.json' if change == 'report' else 'source-manifest.json')
        value = json.loads(path.read_text()); value['changed'] = True
    else:
        path = audit.certificate_path(store, 0)
        value = json.loads(path.read_text()); entry = value['outputs']['16384']['files'][0]
        if change == 'relative_path': entry['relative_path'] = '../other.parquet'
        elif change == 'size': entry['bytes'] += 1
        else: entry['sha256'] = 'a' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError): audit.validate_certificate(root, store, 0, sha)


def test_same_size_content_corruption_cannot_get_certificate(fixture):
    root, store, sha, report = fixture
    path = Path(report['outputs']['16384']['path']) / report['outputs']['16384']['files'][0]
    data = bytearray(path.read_bytes()); data[80] ^= 1; path.write_bytes(data)
    with pytest.raises(Exception): audit.audit_partition(root, store, 0, sha)
    assert not audit.certificate_path(store, 0).exists()


def test_terminal_gate_checks_matching_child_not_root():
    node = lambda call, status, children=[]: SimpleNamespace(function_call_id=call, status=SimpleNamespace(name=status), children=children)
    graph = [node('coordinator', 'PENDING', [node('finalizer', 'TERMINATED')])]
    assert audit.original_finalizer_terminal(graph, 'finalizer')['input_statuses'] == ['TERMINATED']
    graph[0].children[0].status.name = 'PENDING'
    with pytest.raises(AssertionError): audit.original_finalizer_terminal(graph, 'finalizer')
    with pytest.raises(AssertionError): audit.original_finalizer_terminal(graph, 'missing')


def test_actual_loaders_and_append_base_consumer(fixture):
    root, store, sha, report = fixture
    audit.audit_partition(root, store, 0, sha)
    gates = []
    result = audit.finalize_verified(root, store, sha, partitions=1, expected_raw={'fixture': 3},
                                    before_publish=lambda: gates.append('single-writer gate'))
    assert gates == ['single-writer gate']
    for length in LENGTHS:
        value = result[str(length)]
        assert value['all_output_byte_digests_verified'] and value['packed_sequences'] == 3
        assert len(value['runtime_loader_samples']) == 3
    # Same consumer used by the frozen running expansion completion chain.
    from data.expansion_completion_chain import base_snapshot
    selected = base_snapshot({'base': {'root': str(root)}})
    assert selected['root'] == str(root) and set(selected['manifest_sha256']) == {'16384', '32768'}
    assert selected['summary']['sha256'] == audit.file_sha256(root / 'partitions/summary.json')
