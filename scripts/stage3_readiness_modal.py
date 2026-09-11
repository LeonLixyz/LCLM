"""Bounded verification of the final packs and distributed training path."""
import json
import subprocess
from pathlib import Path

import modal

app = modal.App('lclm-stage3-readiness-20260910-v1')
volume = modal.Volume.from_name('lclm-stage3-data')
ROOT = Path('/data/stage3-build-20260906/grouped-packing-20260909-v3')
OUT = Path('/data/stage3-build-20260906/readiness-20260910-v1')


def code(image):
    image = image.env({'PYTHONPATH': '/opt/lclm', 'TOKENIZERS_PARALLELISM': 'false',
                      'OMP_NUM_THREADS': '1', 'WANDB_MODE': 'disabled', 'NCCL_DEBUG': 'WARN'})
    for name in ('data', 'latent_context', 'train', 'utils', 'tests', 'scripts'):
        image = image.add_local_dir(name, f'/opt/lclm/{name}', ignore=['__pycache__'])
    return image


cpu_image = code(modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy', 'pyarrow==21.0.0'))
gpu_image = code(modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel')
    .pip_install('transformers==4.57.1', 'datasets==3.6.0', 'pyarrow==21.0.0',
                 'accelerate==1.10.1', 'peft==0.17.1', 'torchdata==0.11.0',
                 'pytest==8.4.2', 'einops', 'omegaconf', 'liger-kernel==0.6.2',
                 'wandb==0.21.1', 'python-dotenv', 'ninja', 'packaging')
    .run_commands('MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation'))


@app.function(image=cpu_image, cpu=8, memory=16384, timeout=3600, volumes={'/data': volume})
def audit():
    import hashlib
    import random
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import pyarrow.parquet as pq
    from data.packed_file_discovery import discover_packed_parquet_files
    volume.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for length in (16384, 32768):
        root = ROOT / f'packed-cs16-{length}'
        manifest_bytes = (root / 'manifest.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        files = discover_packed_parquet_files(root)
        random.Random(4231).shuffle(files)

        def read(path):
            table = pq.read_table(path, columns=['packing_category', 'expanded_seq_len_cs16', 'example_count'])
            categories = table['packing_category'].to_pylist()
            lengths = table['expanded_seq_len_cs16'].to_pylist()
            assert all(0 < n <= length for n in lengths)
            return {'path': path, 'rows': table.num_rows, 'bytes': Path(path).stat().st_size,
                    'categories': categories, 'examples': sum(table['example_count'].to_pylist())}

        with ThreadPoolExecutor(max_workers=32) as pool:
            rows = list(pool.map(read, files))
        total = sum(r['rows'] for r in rows)
        assert total == manifest['packed_sequences']
        categories = [c for r in rows for c in r['categories']]
        counts = Counter(categories)
        assert counts == {k: v['packed_sequences'] for k, v in manifest['counts'].items()}
        expansion = np.concatenate([np.full(r['rows'], '/data/expansion/' in r['path']) for r in rows])
        ranks = []
        for rank in range(8):
            start, stop = rank * (total // 8), (rank + 1) * (total // 8)
            cats = categories[start:stop]
            exp = expansion[start:stop]
            ranks.append({'rank': rank, 'rows': len(cats), 'categories': dict(Counter(cats)),
                          'expansion_sequences': int(exp.sum()),
                          'first_4096_categories': dict(Counter(cats[:4096])),
                          'first_4096_expansion': int(exp[:4096].sum())})
        results[str(length)] = {'status': 'complete', 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'files': len(files), 'bytes': sum(r['bytes'] for r in rows), 'rows': total,
            'examples': sum(r['examples'] for r in rows), 'categories': dict(counts),
            'expansion_sequences': int(expansion.sum()), 'ranks': ranks,
            'shuffle_scope': 'file order and rows within each file; not a global row permutation'}
        (OUT / f'{length}-files.json').write_text(json.dumps([{k:v for k,v in r.items() if k != 'categories'} for r in rows]))
        (OUT / 'data-audit.json').write_text(json.dumps(results, indent=2))
        volume.commit()
    return results


@app.function(image=gpu_image, gpu='H200:8', cpu=16, memory=65536,
              timeout=7200, volumes={'/data': volume}, secrets=[modal.Secret.from_name('huggingface')])
def gpu_checks():
    volume.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    commands = [
        ('regression', ['python', '-m', 'pytest', '-q',
            'tests/test_distributed_dataset_balancing.py', 'tests/test_dynamic_packing_dataset.py',
            'tests/test_packed_file_discovery.py', 'tests/test_trainer_no_memory_optimizer.py',
            'tests/test_processor_target_memory.py', 'tests/test_unpacked_agent_collate.py',
            'tests/test_nan_checks.py', 'tests/test_model_mixed_compression.py',
            'tests/test_distributed_mixed_compression.py', 'tests/test_packed_flash_parity.py']),
        ('nccl8', ['torchrun', '--standalone', '--nproc_per_node=8', 'scripts/stage3_nccl_smoke.py']),
        ('fsdp8', ['torchrun', '--standalone', '--nproc_per_node=8', 'scripts/stage3_nccl_smoke.py', '--fsdp']),
    ]
    for name, cmd in commands:
        with (OUT / f'{name}.log').open('w') as stream:
            run = subprocess.run(cmd, cwd='/opt/lclm', stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        result = {'check': name, 'exit_code': run.returncode,
                  'tail': (OUT / f'{name}.log').read_text()[-6000:]}
        results.append(result)
        (OUT / 'gpu-report.json').write_text(json.dumps(results, indent=2))
        volume.commit()
        print(json.dumps(result), flush=True)
    return results


@app.function(image=gpu_image, cpu=8, memory=16384, timeout=3600,
              volumes={'/data': volume}, secrets=[modal.Secret.from_name('huggingface')])
def prepare_real_training():
    import random
    import pickle
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from data.stage3_tokenizers import DECODER, DECODER_REVISION, ENCODER, ENCODER_REVISION
    from data.packed_file_discovery import discover_packed_parquet_files
    volume.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, repo, revision in [('decoder', DECODER, DECODER_REVISION), ('encoder', ENCODER, ENCODER_REVISION)]:
        snapshot_download(repo, revision=revision, local_dir=str(OUT / 'models' / name),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    for length in (16384, 32768):
        files = discover_packed_parquet_files(ROOT / f'packed-cs16-{length}')
        random.Random(4231).shuffle(files)
        samples = {key: [] for key in ('native', 'expansion', 'reasoning', 'other')}
        # Search expansion separately because it is a small part of the union.
        files = [f for f in files if '/data/expansion/' in f][:1] + files
        for path in files:
            meta = pq.read_table(path, columns=['packing_category']).to_pydict()['packing_category']
            needed = []
            for index, category in enumerate(meta):
                kind = ('expansion' if '/data/expansion/' in path else 'native') if category == 'agent' else category
                if len(samples[kind]) + sum(k == kind for _, k in needed) < 8:
                    needed.append((index, kind))
            if needed:
                table = pq.read_table(path)
                for index, kind in needed:
                    row = table.slice(index, 1).to_pylist()[0]
                    examples = pickle.loads(row['packed_batch_bytes'])
                    assert all(ex['_packing_category'] == row['packing_category'] for ex in examples)
                    if kind == 'native':
                        assert not any(ex['memory_strings'] for ex in examples)
                    row['sample_kind'] = kind
                    row['source_path'] = path
                    row['source_row'] = index
                    samples[kind].append(row)
            if all(len(rows) == 8 for rows in samples.values()):
                break
        assert all(len(rows) == 8 for rows in samples.values())
        target = OUT / f'fixture-{length}'; target.mkdir(parents=True, exist_ok=True)
        # Eight rows per kind suffice for rank-specific full-length checks.
        pq.write_table(pa.Table.from_pylist([row for rows in samples.values() for row in rows]), target / 'samples.parquet')
    volume.commit()
    return {'status': 'complete', 'fixture_rows_per_length': 32}


@app.function(image=gpu_image, gpu='H200:8', cpu=32, memory=262144,
              timeout=7200, volumes={'/data': volume})
def real_training():
    import os
    volume.reload()
    env = dict(os.environ, ACCELERATE_USE_FSDP='true', FSDP_AUTO_WRAP_POLICY='TRANSFORMER_BASED_WRAP',
        FSDP_SHARDING_STRATEGY='FULL_SHARD', FSDP_USE_ORIG_PARAMS='true',
        FSDP_SYNC_MODULE_STATES='true', ACCELERATE_MIXED_PRECISION='bf16')
    cmd = ['torchrun', '--standalone', '--nproc_per_node=8', 'scripts/stage3_real_training_smoke.py']
    with (OUT / 'real-training.log').open('w') as stream:
        run = subprocess.run(cmd, cwd='/opt/lclm', env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=6600)
    result = {'exit_code': run.returncode, 'tail': (OUT / 'real-training.log').read_text()[-10000:]}
    (OUT / 'real-training-report.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=gpu_image, cpu=8, memory=16384, timeout=1800, volumes={'/data': volume})
def final_loader():
    """Load actual globally shuffled LargeBinary shards with pinned tokenizers."""
    import pickle
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from data.dynamic_packing_dataset import DynamicPackedDataset
    from data.packed_file_discovery import discover_packed_parquet_files
    volume.reload()
    root = Path('/data/stage3-build-20260906/global-shuffle-20260910-v2')
    assert (root / 'completion.json').exists()
    decoder = AutoTokenizer.from_pretrained(OUT / 'models/decoder')
    encoder = AutoTokenizer.from_pretrained(OUT / 'models/encoder')
    decoder.add_special_tokens({'additional_special_tokens': ['<|memory_start|>', '<|memory_end|>', '<|memory|>']})
    decoder.pad_token = decoder.pad_token or decoder.eos_token
    results = {}
    for length in (16384, 32768):
        source = root / f'packed-cs16-{length}'
        manifest = json.loads((source / 'manifest.json').read_text())
        files = discover_packed_parquet_files(source)
        assert len(files) == manifest['files']
        chosen = {}
        for path in files[:8]:
            table = pq.read_table(path)
            for index in range(table.num_rows):
                examples = pickle.loads(table['packed_batch_bytes'][index].as_py())
                kind = table['packing_category'][index].as_py()
                if kind == 'agent':
                    kind = 'expansion' if any(example['memory_strings'] for example in examples) else 'native'
                if kind not in chosen:
                    chosen[kind] = table.slice(index, 1)
            if set(chosen) == {'native', 'expansion', 'reasoning', 'other'}:
                break
        assert len(chosen) == 4
        rows = pa.concat_tables(list(chosen.values()))
        assert pa.types.is_large_binary(rows.schema.field('packed_batch_bytes').type)
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(rows, Path(directory) / 'samples.parquet')
            dataset = DynamicPackedDataset(directory, decoder, encoder, 16,
                shuffle=False, shuffle_files=False, target_length=length)
            checks = []
            for index, batch in enumerate(dataset):
                examples = pickle.loads(rows['packed_batch_bytes'][index].as_py())
                assert batch['input_ids'].shape == batch['labels'].shape == (1, length)
                assert sum(batch['sample_lens'][0]) == rows['expanded_seq_len_cs16'][index].as_py()
                labels = int((batch['labels'] != -100).sum())
                assert labels == sum(example['_trainable_tokens'] for example in examples)
                for start, stop in batch['memory_positions'][0]:
                    assert (batch['labels'][0, start - 1:stop + 1] == -100).all()
                checks.append({'kind': list(chosen)[index], 'labeled_tokens': labels,
                               'actual_length': sum(batch['sample_lens'][0])})
        results[str(length)] = {'status': 'complete', 'discovered_files': len(files), 'samples': checks}
    (OUT / 'final-loader-report.json').write_text(json.dumps(results, indent=2))
    volume.commit()
    return results


@app.local_entrypoint()
def main(check: str = 'audit'):
    function = {'audit': audit, 'gpu': gpu_checks, 'prepare': prepare_real_training,
                'train': real_training, 'loader': final_loader}[check]
    print(json.dumps(function.remote(), indent=2))
