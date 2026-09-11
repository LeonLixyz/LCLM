"""Create an immutable, globally shuffled copy of both completed Stage-3 packs."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import modal
from data.global_sequence_shuffle import BUCKET_ROWS, MODULUS, PARTITIONS, SEED

app = modal.App('lclm-stage3-global-shuffle-20260910-v2')
volume = modal.Volume.from_name('lclm-stage3-data')
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('numpy==2.2.6', 'pyarrow==21.0.0', 'pytest==8.4.2')
    .env({'PYTHONPATH': '/opt/lclm', 'OMP_NUM_THREADS': '1'})
    .add_local_dir('data', '/opt/lclm/data', ignore=['__pycache__'])
    .add_local_file('tests/test_global_sequence_shuffle.py', '/opt/lclm/tests/test_global_sequence_shuffle.py'))
SOURCE = Path('/data/stage3-build-20260906/grouped-packing-20260909-v3')
ROOT = Path('/data/stage3-build-20260906/global-shuffle-20260910-v2')
EXPECTED = {16384: '8cf25be8aa2412a7f18447c8d5d3076ac8ff8b22b34f96182d7fa5fa164bf6c4',
            32768: 'aacb4d29c6dec1096fab24a388fbb0df52119cabea987c4a10d3c341cb6d2dac'}
opts = dict(image=image, volumes={'/data': volume})


@app.function(**opts, cpu=4, memory=32768, timeout=7200, max_containers=16)
def worker(length, phase, partition):
    from data.global_sequence_shuffle import scatter, gather
    volume.reload()
    work = ROOT / f'work-{length}'
    plan = json.loads((work / 'plan.json').read_text())
    if phase == 'scatter':
        result = scatter(plan, partition, work)
    else:
        result = gather(plan, partition, work, ROOT / f'packed-cs16-{length}')
    volume.commit()
    return {'length': length, 'phase': phase, 'part': partition, 'rows': result['rows']}


@app.function(**opts, cpu=4, memory=8192, timeout=14400, max_containers=1)
def run():
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    from data.packed_file_discovery import discover_packed_parquet_files
    subprocess.run(['python', '-m', 'pytest', '-q', 'tests/test_global_sequence_shuffle.py'], cwd='/opt/lclm', check=True)
    volume.reload()
    plans = {}
    for length in (16384, 32768):
        source = SOURCE / f'packed-cs16-{length}'
        raw = (source / 'manifest.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == EXPECTED[length]
        manifest = json.loads(raw)
        files = discover_packed_parquet_files(source)
        def info(path):
            return {'path': path, 'rows': pq.read_metadata(path).num_rows, 'bytes': Path(path).stat().st_size}
        with ThreadPoolExecutor(max_workers=32) as pool:
            entries = list(pool.map(info, files))
        total = 0
        for entry in entries:
            entry['offset'] = total
            total += entry['rows']
        assert total == manifest['packed_sequences']
        plan = {'files': entries, 'rows': total, 'seed': SEED, 'partitions': PARTITIONS,
                'bucket_rows': BUCKET_ROWS, 'buckets': math.ceil(total / BUCKET_ROWS),
                'source_manifest_sha256': EXPECTED[length]}
        work = ROOT / f'work-{length}'; work.mkdir(parents=True, exist_ok=True)
        old = work / 'plan.json'
        if old.exists():
            assert json.loads(old.read_text()) == plan
        else:
            old.write_text(json.dumps(plan))
        plans[length] = plan
    volume.commit()
    for phase in ('scatter', 'gather'):
        args = [(length, phase, p) for length, plan in plans.items()
                for p in range(PARTITIONS if phase == 'scatter' else plan['buckets'])]
        for result in worker.starmap(args, order_outputs=False):
            print(json.dumps(result), flush=True)
    volume.reload()
    summary = {}
    for length, plan in plans.items():
        work = ROOT / f'work-{length}'
        inputs = [json.loads((work / f'scatter-{p:03d}.json').read_text()) for p in range(PARTITIONS)]
        outputs = [json.loads((work / f'gather-{p:03d}.json').read_text()) for p in range(plan['buckets'])]
        assert sum(r['rows'] for r in inputs) == sum(r['rows'] for r in outputs) == plan['rows']
        assert sum(int(r['payload_digest']) for r in inputs) % MODULUS == sum(int(r['payload_digest']) for r in outputs) % MODULUS
        source = SOURCE / f'packed-cs16-{length}'
        manifest = json.loads((source / 'manifest.json').read_text())
        counts = Counter()
        for report in outputs:
            counts.update(report['counts'])
        assert counts == {k: v['packed_sequences'] for k, v in manifest['counts'].items()}
        assert sum(r['examples'] for r in outputs) == sum(c['packed_rows'] for c in manifest['counts'].values())
        manifest['global_shuffle'] = {'algorithm': 'numpy.PCG64.permutation of all packed sequences',
            'seed': SEED, 'source_manifest_sha256': EXPECTED[length],
            'all_positions_verified': True, 'payload_multiset_digest': str(sum(int(r['payload_digest']) for r in outputs) % MODULUS),
            'serialized_rows_preserved': True, 'all_outputs_reopened_and_verified': True}
        manifest['output_files'] = [f for r in outputs for f in r['files']]
        manifest['files'] = len(manifest['output_files'])
        manifest['training_loader'] = 'DynamicPackedDataset at this version root; compression_ratio=16; target_length=max_packed_length; shuffle=True; shuffle_files=True'
        target = ROOT / f'packed-cs16-{length}'
        (target / 'source-manifest.json').write_bytes((source / 'manifest.json').read_bytes())
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        summary[str(length)] = {'status': 'complete', 'path': str(target), 'rows': plan['rows'],
            'counts': dict(counts), 'manifest_sha256': hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest()}
    (ROOT / 'completion.json').write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), indent=2))
