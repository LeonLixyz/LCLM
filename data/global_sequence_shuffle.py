"""Exact global permutation of intact packed rows, using bounded scatter/gather."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20260910
BUCKET_ROWS = 32768
PARTITIONS = 64
MODULUS = 1 << 256


def order(total, seed=SEED):
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(total)
    inverse = np.empty(total, dtype=np.int64)
    inverse[permutation] = np.arange(total)
    return permutation, inverse


def payload_digest(table):
    """Commutative digest bound to each original row identity and payload."""
    value = 0
    for identity, payload in zip(table['_origin_row'].to_pylist(), table['packed_batch_bytes'].to_pylist()):
        value = (value + int.from_bytes(hashlib.sha256(identity.to_bytes(8, 'big') + payload).digest(), 'big')) % MODULUS
    return value


def scatter(plan, partition, work):
    import pyarrow as pa
    import pyarrow.parquet as pq
    work = Path(work)
    done = work / f'scatter-{partition:03d}.json'
    if done.exists():
        return json.loads(done.read_text())
    _, inverse = order(plan['rows'], plan['seed'])
    tables = []
    inputs = []
    for index in range(partition, len(plan['files']), plan['partitions']):
        entry = plan['files'][index]
        raw = Path(entry['path']).read_bytes()
        assert len(raw) == entry['bytes']
        table = pq.read_table(pa.BufferReader(raw))
        # A partition exceeds Arrow binary's 2 GiB offset limit. Packed payloads
        # keep identical bytes; only the Arrow offset width changes for sorting.
        field_index = table.schema.get_field_index('packed_batch_bytes')
        table = table.set_column(field_index, 'packed_batch_bytes', table['packed_batch_bytes'].cast(pa.large_binary()))
        assert table.num_rows == entry['rows']
        ids = np.arange(entry['offset'], entry['offset'] + entry['rows'], dtype=np.int64)
        table = table.append_column('_origin_row', pa.array(ids))
        table = table.append_column('_shuffle_position', pa.array(inverse[ids]))
        tables.append(table)
        inputs.append({'path': entry['path'], 'sha256': hashlib.sha256(raw).hexdigest()})
    table = pa.concat_tables(tables) if tables else None
    outputs = []
    total_digest = 0
    if table is not None:
        total_digest = payload_digest(table)
        table = table.sort_by([('_shuffle_position', 'ascending')])
        positions = table['_shuffle_position'].to_numpy()
        for bucket in range(plan['buckets']):
            start, stop = np.searchsorted(positions, [bucket * plan['bucket_rows'], (bucket + 1) * plan['bucket_rows']])
            if start == stop:
                continue
            directory = work / f'bucket-{bucket:03d}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'scatter-{partition:03d}.parquet'
            subset = table.slice(int(start), int(stop - start))
            pq.write_table(subset, path, compression='snappy', row_group_size=256)
            outputs.append({'path': str(path), 'rows': subset.num_rows})
    report = {'partition': partition, 'rows': 0 if table is None else table.num_rows,
              'payload_digest': str(total_digest), 'inputs': inputs, 'outputs': outputs}
    done.write_text(json.dumps(report))
    return report


def gather(plan, bucket, work, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq
    work, destination = Path(work), Path(destination)
    done = work / f'gather-{bucket:03d}.json'
    if done.exists():
        return json.loads(done.read_text())
    paths = sorted((work / f'bucket-{bucket:03d}').glob('scatter-*.parquet'))
    table = pa.concat_tables([pq.read_table(path) for path in paths]).sort_by([('_shuffle_position', 'ascending')])
    start = bucket * plan['bucket_rows']
    stop = min(start + plan['bucket_rows'], plan['rows'])
    positions = table['_shuffle_position'].to_numpy()
    assert np.array_equal(positions, np.arange(start, stop)), 'Missing or repeated shuffle positions'
    permutation, _ = order(plan['rows'], plan['seed'])
    assert np.array_equal(table['_origin_row'].to_numpy(), permutation[start:stop]), 'Wrong permutation'
    digest = payload_digest(table)
    counts = Counter(table['packing_category'].to_pylist())
    directory = destination / 'data/mixed' / f'part-{bucket:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    verified_digest = 0
    for offset in range(0, table.num_rows, 512):
        path = directory / f'packed-{offset // 512:06d}.parquet'
        subset = table.slice(offset, min(512, table.num_rows - offset))
        pq.write_table(subset, path, compression='snappy', row_group_size=64)
        # Reopen output bytes and verify both the serialized payload and metadata.
        raw = path.read_bytes()
        restored = pq.read_table(pa.BufferReader(raw))
        assert restored.equals(subset)
        verified_digest = (verified_digest + payload_digest(restored)) % MODULUS
        outputs.append({'relative_path': str(path.relative_to(destination)), 'rows': subset.num_rows,
                        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    assert verified_digest == digest
    report = {'bucket': bucket, 'rows': table.num_rows, 'payload_digest': str(digest),
              'counts': dict(counts), 'examples': sum(table['example_count'].to_pylist()),
              'files': outputs}
    done.write_text(json.dumps(report))
    return report
