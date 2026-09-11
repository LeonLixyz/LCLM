"""Immutable 16k/32k category-isolated CPU rebuild, deployed once on Modal.

Deploy: modal deploy -m data.grouped_stage3_packing_modal
Invoke the deployed preflight, then launch functions with Function.from_name.
No generation app, original raw source, or original packed source is modified.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from data.grouped_stage3_packing import CATEGORIES, LENGTHS, canonical_digest
from data.stage3_tokenizers import DECODER, DECODER_REVISION, ENCODER, ENCODER_REVISION

APP_NAME = 'lclm-stage3-grouped-packing-20260909-v3'
ROOT = Path('/data/stage3-build-20260906/grouped-packing-20260909-v3')
SOURCE_ROOTS = {
    'base': Path('/data/stage3-final-mixture-cot50-v1'),
    'native': Path('/data/stage3-build-20260906/agents-transport'),
}
EXPECTED_RAW = {'base': 20326114, 'native': 1244170}
PARTITIONS = 64
app = modal.App(APP_NAME)
volume = modal.Volume.from_name('lclm-stage3-data')
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.8.0', index_url='https://download.pytorch.org/whl/cpu')
    .pip_install('transformers==4.57.1', 'pyarrow==21.0.0', 'datasets==3.6.0', 'pytest==8.4.2')
    .env({'PYTHONPATH': '/opt/lclm', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'})
    .add_local_dir('data', '/opt/lclm/data', ignore=['__pycache__'])
    .add_local_dir('tests', '/opt/lclm/tests', ignore=['__pycache__'])
)
options = dict(image=image, volumes={'/data': volume}, secrets=[modal.Secret.from_name('huggingface')])


def tokenizers():
    from data.preprocess_for_dynamic_packing import worker_init
    from data import preprocess_for_dynamic_packing as pre
    worker_init(DECODER, ENCODER, DECODER_REVISION, ENCODER_REVISION)
    return pre._worker_tokenizer, pre._worker_embed_tokenizer


def file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    # Source exports are already complete and have audited total counts. Keep
    # launch inventory finite: row metadata and content digests are collected by
    # each partition while streaming, not by reopening thousands of shards here.
    manifest = {'sources': {}, 'expected_raw_rows': EXPECTED_RAW,
                'decoder': DECODER, 'decoder_revision': DECODER_REVISION,
                'encoder': ENCODER, 'encoder_revision': ENCODER_REVISION,
                'reference_chunk_size': 16, 'lengths': list(LENGTHS),
                'reasoning_sources': ['reasoning_data', 'dolci_think'],
                'math_policy': 'other: passage-reconstruction task; preserve existing raw policy and content',
                'math_semantics_evidence': '/data/stage3-build-20260906/nemotron-math-task-semantics-20260909.json',
                'base_policy': 'preserve existing cot50-v1 unchanged; audit reasoning prompts remain uncompressed',
                'expansion_included': False,
                'input_selection': 'raw once; no legacy packed/recovery inputs',
                'source_provenance_report': '/data/stage3-build-20260906/base-source-provenance-inspection.json'}
    for kind, root in SOURCE_ROOTS.items():
        files = sorted(root.glob('*.parquet'))
        if not files:
            raise ValueError(f'No raw {kind} files')
        if kind == 'native' and not (root / 'report.json').exists():
            raise ValueError('Native transport export is incomplete')
        entries = [{'name': path.name, 'bytes': path.stat().st_size} for path in files]
        rows = EXPECTED_RAW[kind]
        manifest['sources'][kind] = {'root': str(root), 'files': entries, 'rows': rows}
    return manifest


@app.function(**options, cpu=4, memory=16384, timeout=3600)
def preflight():
    import pickle
    import subprocess
    import pyarrow.parquet as pq
    from collections import Counter
    from data.grouped_stage3_packing import (REASONING_SOURCES, OTHER_SOURCES,
        process_raw_item, has_memory)
    from data.dynamic_packing_dataset import DynamicPackedDataset
    ROOT.mkdir(parents=True, exist_ok=True)
    report_file = ROOT / 'preflight.json'
    if report_file.exists():
        return json.loads(report_file.read_text())
    started = time.time()
    completed = subprocess.run(['python', '-m', 'pytest', '-q',
        '/opt/lclm/tests/test_grouped_stage3_packing.py',
        '/opt/lclm/tests/test_preprocess_for_dynamic_packing.py',
        '/opt/lclm/tests/test_packed_file_discovery.py',
        '/opt/lclm/tests/test_dynamic_packing_dataset.py'],
        text=True, capture_output=True)
    print(completed.stdout, completed.stderr, flush=True)
    assert completed.returncode == 0
    manifest = inventory()
    print('Raw source metadata inventory complete', flush=True)
    (ROOT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2))
    decoder, encoder = tokenizers()
    runtime = object.__new__(DynamicPackedDataset)
    runtime.decoder_tokenizer = decoder
    runtime.embed_tokenizer = encoder
    runtime.compression_ratio = 16
    runtime.memory_start_id = decoder.convert_tokens_to_ids('<|memory_start|>')
    runtime.memory_end_id = decoder.convert_tokens_to_ids('<|memory_end|>')
    runtime.memory_id = decoder.convert_tokens_to_ids('<|memory|>')
    runtime.pooling = 'mean'
    selected = Counter()
    raw_samples = {}
    audits = []
    desired = REASONING_SOURCES | {'nemotron_math_4plus'}
    for kind, info in manifest['sources'].items():
        indexes = sorted({round(i * (len(info['files']) - 1) / 7) for i in range(8)})
        for file_index in indexes:
            entry = info['files'][file_index]
            path = Path(info['root']) / entry['name']
            row_number = 0
            # Files are source-mixed. Scan only a small first batch from each
            # until every source is sampled, without loading the full raw corpus.
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for raw in batch.to_pylist():
                    sub = raw['sub_dataset']
                    key = f'{kind}:{sub}'
                    index = row_number
                    row_number += 1
                    if selected[key] >= 3:
                        continue
                    row_id = f'{kind}:{entry["name"]}:{index}'
                    metadata, packed = process_raw_item((kind, row_id, raw, True))
                    if packed is None or packed.get('_skipped_max_seq_len'):
                        audits.append({**metadata, 'excluded': 'processing_rejected' if packed is None else 'over_32768'})
                        continue
                    expanded = runtime._expand_example(packed)
                    assert expanded is not None
                    assert expanded['seq_len'] == packed['estimated_seq_len']
                    assert sum(v != -100 for v in expanded['processed']['labels'][0]) == packed['_trainable_tokens']
                    selected[key] += 1
                    audits.append({**metadata, 'exact_expanded_length': expanded['seq_len'],
                                   'memory_blocks': len(packed['memory_strings']),
                                   'fits_16k': expanded['seq_len'] <= 16384,
                                   'fits_32k': expanded['seq_len'] <= 32768})
                    if sub == 'nemotron_math_4plus' or selected[key] == 1:
                        raw_samples.setdefault(key, []).append({
                            'row_id': row_id, 'prompt': raw.get('prompt'),
                            'compression_prompt': raw.get('compression_prompt'),
                            'target': str(raw.get('target', ''))[:12000],
                            'target_chars': len(str(raw.get('target', ''))),
                            'original_prompt_compressed': has_memory(raw.get('compression_prompt')),
                        })
                break
            if kind == 'base' and all(selected[f'base:{source}'] >= 3 for source in desired):
                break
    assert sum(v for key, v in selected.items() if key.startswith('native:')) >= 3
    assert all(selected[f'base:{source}'] >= 1 for source in desired), selected
    report = {'status': 'passed', 'seconds': time.time() - started,
              'tests': completed.stdout, 'source_manifest_sha256': canonical_digest(manifest),
              'sample_counts': dict(selected), 'raw_samples': raw_samples,
              'actual_tokenizer_runtime_checks': audits,
              'scope': 'CPU tests and representative raw-row/runtime expansion checks; no GPU eval'}
    report_file.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(**options, cpu=8, memory=32768, timeout=86400, max_containers=32)
def pack_partition(partition: int, manifest_sha: str, pilot_limit: int = 0):
    import multiprocessing as mp
    import pyarrow.parquet as pq
    from collections import Counter
    from data.grouped_stage3_packing import (process_raw_item, make_packers,
        ShuffledSequenceWriter, exclusion_reason, proportional_interleave)
    from data.preprocess_for_dynamic_packing import worker_init
    volume.reload()
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    assert canonical_digest(manifest) == manifest_sha
    report_root = ROOT / ('pilot' if pilot_limit else 'partitions') / f'part-{partition:03d}'
    report_file = report_root / 'report.json'
    if report_file.exists():
        result = json.loads(report_file.read_text())
        assert result['source_manifest_sha256'] == manifest_sha
        return result
    report_root.mkdir(parents=True, exist_ok=True)
    lock = report_root / 'STARTED.json'
    with lock.open('x') as handle:
        json.dump({'started': time.time(), 'function_call_id': modal.current_function_call_id()}, handle)
    started = time.time()
    versions = {}
    for length in LENGTHS:
        output_root = ROOT / ('pilot' if pilot_limit else '') / f'packed-cs16-{length}'
        staging = output_root / 'in-progress' / f'part-{partition:03d}'
        final = output_root / 'data/mixed' / f'part-{partition:03d}'
        if final.exists() or staging.exists():
            raise ValueError(f'Existing output requires inspection: {staging}')
        versions[length] = {
            'root': output_root, 'staging': staging, 'final': final,
            'packers': make_packers(length, 20260909 + partition),
            'writer': ShuffledSequenceWriter(staging, length, 20260909 + partition),
            'counts': {category: Counter() for category in CATEGORIES},
            'sources': {},
        }
    raw_counts = Counter()
    source_counts = Counter()
    policies = Counter()
    exclusions = []
    input_files = []

    def source_inputs(kind, info):
        selected = info['files'][partition::PARTITIONS]
        seen = 0
        for entry in selected:
            path = Path(info['root']) / entry['name']
            assert path.stat().st_size == entry['bytes']
            parquet = pq.ParquetFile(path)
            source_identity = {'kind': kind, **entry, 'rows': parquet.metadata.num_rows,
                               'content_sha256': file_sha256(path)}
            input_files.append(source_identity)
            index = 0
            for batch in parquet.iter_batches(batch_size=32):
                for raw in batch.to_pylist():
                    row_id = f'{kind}:{entry["name"]}:{index}'
                    index += 1
                    seen += 1
                    yield kind, row_id, raw, True
                    if pilot_limit and seen >= pilot_limit:
                        break
                if pilot_limit and seen >= pilot_limit:
                    break
            if pilot_limit and seen >= pilot_limit:
                break

    def inputs():
        streams = {kind: source_inputs(kind, info) for kind, info in manifest['sources'].items()}
        sizes = {kind: min(pilot_limit, info['rows']) if pilot_limit else info['rows']
                 for kind, info in manifest['sources'].items()}
        yield from proportional_interleave(streams, sizes)

    with mp.get_context('spawn').Pool(8, initializer=worker_init,
            initargs=(DECODER, ENCODER, DECODER_REVISION, ENCODER_REVISION)) as pool:
        for info, example in pool.imap_unordered(process_raw_item, inputs(), chunksize=16):
            category, sub = info['category'], info['sub_dataset']
            raw_counts[category] += 1
            source_counts[sub] += 1
            policies[info['policy']] += 1
            for length, version in versions.items():
                count = version['counts'][category]
                by_source = version['sources'].setdefault(sub, Counter())
                count['input_rows'] += 1
                by_source['input_rows'] += 1
                reason = exclusion_reason(example, length)
                if reason:
                    count[reason] += 1
                    by_source[reason] += 1
                    exclusions.append({**info, 'max_length': length, 'reason': reason,
                        'expanded_length': example['estimated_seq_len'] if example else None})
                    continue
                count['eligible_rows'] += 1
                by_source['eligible_rows'] += 1
                count['uncompressed_rows' if not example['memory_strings'] else 'compressed_rows'] += 1
                for packed in version['packers'][category].add_example(example):
                    version['writer'].write(packed)
            if sum(raw_counts.values()) % 10000 == 0:
                progress = {'partition': partition, 'input_rows': sum(raw_counts.values()),
                            'seconds': time.time() - started}
                (report_root / 'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
    outputs = {}
    for length, version in versions.items():
        for packer in version['packers'].values():
            for packed in packer.finalize():
                version['writer'].write(packed)
        version['writer'].flush()
        for category in CATEGORIES:
            count = version['counts'][category]
            assert count['input_rows'] == sum(count[key] for key in
                ('eligible_rows', 'processing_rejected', 'under_18', 'overlength'))
            assert count['eligible_rows'] == version['writer'].counts[category]['packed_rows']
            count.update(version['writer'].counts[category])
        version['final'].parent.mkdir(parents=True, exist_ok=True)
        os.replace(version['staging'], version['final'])
        outputs[str(length)] = {'counts': {k: dict(v) for k, v in version['counts'].items()},
            'sub_datasets': {k: dict(v) for k, v in version['sources'].items()},
            'path': str(version['final']), 'files': version['writer'].files,
            'output_file_bytes_sha256': version['writer'].digest.hexdigest()}
    with (report_root / 'exclusions.jsonl').open('w') as handle:
        for exclusion in exclusions:
            handle.write(json.dumps(exclusion) + '\n')
    result = {'status': 'complete', 'partition': partition, 'partitions': PARTITIONS,
              'seconds': time.time() - started, 'source_manifest_sha256': manifest_sha,
              'input_files': input_files, 'raw_category_counts': dict(raw_counts),
              'raw_source_counts': dict(source_counts), 'reasoning_policies': dict(policies),
              'outputs': outputs, 'pilot_limit_per_component': pilot_limit,
              'function_call_id': modal.current_function_call_id()}
    report_file.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(**options, cpu=4, memory=16384, timeout=3600)
def finalize(pilot: bool = False):
    import pickle
    import hashlib
    import tempfile
    import pyarrow.parquet as pq
    from collections import Counter
    from data.dynamic_packing_dataset import DynamicPackedDataset
    volume.reload()
    reports_root = ROOT / ('pilot' if pilot else 'partitions')
    reports = [json.loads(p.read_text()) for p in sorted(reports_root.glob('part-*/report.json'))]
    assert len(reports) == (1 if pilot else PARTITIONS), len(reports)
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
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
            output_digest = hashlib.sha256()
            for path in files:
                total_metadata_sequences += pq.ParquetFile(path).metadata.num_rows
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                        output_digest.update(chunk)
            assert output_digest.hexdigest() == result['output_file_bytes_sha256'], result['path']
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
        (output_root / 'manifest.json').write_text(json.dumps(result, indent=2))
        versions[str(length)] = result
    (reports_root / 'summary.json').write_text(json.dumps(versions, indent=2))
    volume.commit()
    return versions


@app.function(**options, cpu=1, memory=2048, timeout=86400, max_containers=1)
def launch(pilot: bool = False):
    volume.reload()
    pre = json.loads((ROOT / 'preflight.json').read_text())
    assert pre['status'] == 'passed'
    if not pilot:
        assert (ROOT / 'pilot/summary.json').exists(), 'Small raw packing/loader pilot must pass first'
    begin = time.time()
    reports = list(pack_partition.starmap(
        ((i, pre['source_manifest_sha256'], 256 if pilot else 0)
         for i in (range(1) if pilot else range(PARTITIONS))), order_outputs=False))
    print(json.dumps({'completed_partitions': len(reports), 'seconds': time.time() - begin}), flush=True)
    return finalize.remote(pilot)


@app.function(**options, cpu=1, memory=2048, timeout=120)
def status():
    volume.reload()
    reports = []
    progress = []
    for path in sorted((ROOT / 'partitions').glob('part-*')):
        marker = path / 'report.json'
        if marker.exists():
            report = json.loads(marker.read_text())
            reports.append({'partition': report['partition'], 'seconds': report['seconds'],
                'raw_category_counts': report['raw_category_counts'],
                'outputs': {length: value['counts'] for length, value in report['outputs'].items()}})
        elif (path / 'progress.json').exists():
            progress.append(json.loads((path / 'progress.json').read_text()))
    return {'complete': reports, 'progress': progress,
            'final_summary_exists': (ROOT / 'partitions/summary.json').exists()}
