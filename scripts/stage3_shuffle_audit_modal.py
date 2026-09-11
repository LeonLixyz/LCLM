"""Independent source binding and eight-rank mixture audit of global shuffle."""
import hashlib
import json
import random
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
from data.global_sequence_shuffle_modal import SOURCE, ROOT, EXPECTED, image, volume

app = modal.App('lclm-stage3-shuffle-audit-20260910-v1')


@app.function(image=image, cpu=1, memory=2048, timeout=10800, volumes={'/data': volume})
def after_shuffle():
    import time
    for _ in range(180):
        volume.reload()
        if (ROOT / 'completion.json').exists():
            return verify.remote()
        time.sleep(20)
    raise TimeoutError('Global shuffle did not finish within one hour')


@app.function(image=image, cpu=8, memory=16384, timeout=7200, volumes={'/data': volume})
def verify():
    import numpy as np
    import pyarrow.parquet as pq
    volume.reload()
    assert (ROOT / 'completion.json').exists(), 'Global shuffle is incomplete'
    result = {}
    for length in (16384, 32768):
        source = SOURCE / f'packed-cs16-{length}'
        target = ROOT / f'packed-cs16-{length}'
        manifest = json.loads((target / 'manifest.json').read_text())
        source_raw = (source / 'manifest.json').read_bytes()
        assert hashlib.sha256(source_raw).hexdigest() == EXPECTED[length]
        original = json.loads(source_raw)
        work = ROOT / f'work-{length}'
        plan = json.loads((work / 'plan.json').read_text())
        inputs = [entry for p in range(plan['partitions'])
                  for entry in json.loads((work / f'scatter-{p:03d}.json').read_text())['inputs']]
        digests = {entry['path']: entry['sha256'] for entry in inputs}
        assert len(digests) == len(inputs) == len(plan['files'])
        assert set(digests) == {entry['path'] for entry in plan['files']}
        certificates = original['parallel_content_audit']['certificates']
        def validate_certificate(entry):
            raw = (SOURCE / entry['relative_path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == entry['sha256']
            content = json.loads(raw)
            for record in content['outputs'][str(length)]['files']:
                assert digests[str(SOURCE / record['relative_path'])] == record['sha256']
            return len(content['outputs'][str(length)]['files'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            base_files = sum(pool.map(validate_certificate, certificates))
        expansion_manifest = original['expansion_append']['manifest_sha256']
        def validate_expansion(partition):
            report = json.loads((SOURCE / 'expansion-append' / expansion_manifest / f'part-{partition:03d}/report.json').read_text())
            expected = report['outputs'][str(length)]
            folder = source / 'data/expansion' / f'part-{partition:03d}'
            digest = hashlib.sha256()
            for name in expected['files']:
                path = folder / name
                raw = path.read_bytes()
                assert digests[str(path)] == hashlib.sha256(raw).hexdigest()
                digest.update(raw)
            assert digest.hexdigest() == expected['output_file_bytes_sha256']
            return len(expected['files'])
        with ThreadPoolExecutor(max_workers=8) as pool:
            expansion_files = sum(pool.map(validate_expansion, range(16)))
        assert base_files + expansion_files == len(digests)

        # Reproduce the loader's file order, rank intervals and row RNG. Only
        # metadata columns are needed because payload preservation is exhaustive.
        files = sorted(manifest['output_files'], key=lambda entry: entry['relative_path'])
        random.Random(4231).shuffle(files)
        ends = np.cumsum([entry['rows'] for entry in files]).tolist()
        total = ends[-1]
        assert total == manifest['packed_sequences']
        expansion_origin = np.zeros(total, dtype=bool)
        for entry in plan['files']:
            if '/data/expansion/' in entry['path']:
                expansion_origin[entry['offset']:entry['offset'] + entry['rows']] = True
        def rank_audit(rank):
            quota = total // 8
            cursor, stop = rank * quota, (rank + 1) * quota
            rng = random.Random(4231 + rank)
            counts = Counter()
            expansion = 0
            sampled = 0
            while cursor < stop and sampled < 4096:
                index = bisect_right(ends, cursor)
                entry = files[index]
                offset = ends[index - 1] if index else 0
                start = cursor - offset
                end = min(entry['rows'], stop - offset)
                indices = list(range(start, end)); rng.shuffle(indices)
                chosen = indices[:4096 - sampled]
                path = target / entry['relative_path']
                table = pq.read_table(path, columns=['packing_category', '_origin_row', 'expanded_seq_len_cs16'])
                assert table.num_rows == entry['rows']
                categories = table['packing_category'].to_pylist()
                origins = table['_origin_row'].to_pylist()
                lengths = table['expanded_seq_len_cs16'].to_pylist()
                for row in chosen:
                    assert 0 < lengths[row] <= length
                    counts[categories[row]] += 1
                    expansion += int(expansion_origin[origins[row]])
                sampled += len(chosen)
                cursor = offset + end
            assert sampled == 4096 and set(counts) == {'agent', 'reasoning', 'other'} and expansion > 0
            return {'rank': rank, 'assigned_rows': quota, 'sampled_first_sequences': sampled,
                    'category_counts': dict(counts), 'expansion_sequences': expansion}
        with ThreadPoolExecutor(max_workers=8) as pool:
            ranks = list(pool.map(rank_audit, range(8)))
        result[str(length)] = {'status': 'complete', 'source_files_bound': len(digests),
            'source_base_certificate_files': base_files, 'source_expansion_audit_files': expansion_files,
            'all_categories_and_expansion_present_on_every_rank': True,
            'rows': total, 'per_rank_rows': total // 8, 'training_drop_remainder': total % 8,
            'ranks': ranks, 'manifest_sha256': hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest()}
        (ROOT / 'independent-audit.json').write_text(json.dumps(result, indent=2))
        volume.commit()
    return result


@app.function(image=image, cpu=2, memory=2048, timeout=3600, volumes={'/data': volume})
def cleanup_scratch():
    import shutil
    volume.reload()
    audit = json.loads((ROOT / 'independent-audit.json').read_text())
    assert all(audit[str(length)]['status'] == 'complete' for length in (16384, 32768))
    folders = []
    for length in (16384, 32768):
        work = ROOT / f'work-{length}'
        plan = json.loads((work / 'plan.json').read_text())
        folders.extend(work / f'bucket-{bucket:03d}' for bucket in range(plan['buckets']))
    def remove(folder):
        assert not folder.is_symlink()
        if not folder.exists():
            return {'files': 0, 'bytes': 0}
        files = list(folder.iterdir())
        assert all(path.is_file() and path.name.startswith('scatter-') and path.suffix == '.parquet' for path in files)
        result = {'files': len(files), 'bytes': sum(path.stat().st_size for path in files)}
        shutil.rmtree(folder)
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        removed = list(pool.map(remove, folders))
    result = {'status': 'complete', 'removed_files': sum(r['files'] for r in removed),
              'removed_bytes': sum(r['bytes'] for r in removed),
              'preserved': 'original packs, final shuffled packs, plans, source digests and all audit reports'}
    (ROOT / 'scratch-cleanup.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(cleanup: bool = False):
    print(json.dumps((cleanup_scratch if cleanup else after_shuffle).remote(), indent=2))
