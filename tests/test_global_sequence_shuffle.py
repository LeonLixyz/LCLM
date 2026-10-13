import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from data.global_sequence_shuffle import MODULUS, gather, order, scatter


def test_global_shuffle_preserves_all_payloads_and_permutation(tmp_path):
    files = []
    expected = []
    offset = 0
    for index, size in enumerate((5, 2, 9, 3)):
        payloads = [f'original-{offset + n}'.encode() for n in range(size)]
        expected.extend(payloads)
        path = tmp_path / f'input-{index}.parquet'
        pq.write_table(pa.table({'packed_batch_bytes': payloads,
            'packing_category': [['agent', 'reasoning', 'other'][index % 3]] * size,
            'example_count': [2] * size, 'expanded_seq_len_cs16': [100] * size}), path)
        files.append({'path': str(path), 'bytes': path.stat().st_size, 'rows': size, 'offset': offset})
        offset += size
    plan = {'files': files, 'rows': offset, 'seed': 123, 'partitions': 2, 'buckets': 4, 'bucket_rows': 5}
    work = tmp_path / 'work'; work.mkdir()
    target = tmp_path / 'result'
    scatter_reports = [scatter(plan, p, work) for p in range(2)]
    gather_reports = [gather(plan, b, work, target) for b in range(4)]
    input_digest = sum(int(r['payload_digest']) for r in scatter_reports) % MODULUS
    assert input_digest == sum(int(r['payload_digest']) for r in gather_reports) % MODULUS
    outputs = sorted(target.glob('data/mixed/part-*/*.parquet'))
    table = pa.concat_tables([pq.read_table(path) for path in outputs])
    permutation, _ = order(offset, 123)
    assert table['packed_batch_bytes'].to_pylist() == [expected[i] for i in permutation]
    assert table['_shuffle_position'].to_pylist() == list(range(offset))
    assert sum(r['rows'] for r in gather_reports) == offset
    assert scatter(plan, 0, work) == scatter_reports[0]
    assert gather(plan, 0, work, target) == gather_reports[0]
