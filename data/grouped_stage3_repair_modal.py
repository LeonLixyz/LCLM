"""Repair terminal-failed v3 packing partitions without touching healthy work.

Uses the exact v3 processor, category policy, tokenizers and length limits.
Differences are bounded in-flight multiprocessing chunks, 64GiB memory, and
durable exception reporting. Old partial output is quarantined, never deleted.
"""
from __future__ import annotations

import json
import time
from collections import deque
from itertools import islice

import modal

from data import grouped_stage3_packing_modal as core

APP_NAME = 'lclm-stage3-grouped-repair-20260909-v2'
app = modal.App(APP_NAME)


def process_chunk(function, items):
    return [function(item) for item in items]


def bounded_imap(pool, function, iterable, chunksize=1):
    """At most 2*workers chunks may hold input or processed output in RAM."""
    source = iter(iterable)
    pending = deque()
    capacity = max(1, 2 * pool._processes)
    exhausted = False
    while pending or not exhausted:
        while len(pending) < capacity and not exhausted:
            chunk = list(islice(source, chunksize))
            if not chunk:
                exhausted = True
                break
            pending.append(pool.apply_async(process_chunk, (function, chunk)))
        if pending:
            yield from pending.popleft().get()


def validate_owner(owner, function_call_id):
    # Modal guarantees that a preempted input is restarted with the same call.
    # Permit that retry; a second independently submitted call must fail closed.
    if owner['function_call_id'] != function_call_id:
        raise ValueError('Different repair call already owns this partition')


@app.function(**core.options, cpu=8, memory=65536, timeout=86400, max_containers=3)
def repair(partition: int, manifest_sha: str):
    import os
    import traceback
    import multiprocessing.pool
    import uuid
    from pathlib import Path
    if partition not in (9, 21, 30):
        raise ValueError('Only the three audited terminal-failed inputs are eligible')
    core.volume.reload()
    root = core.ROOT
    report_dir = root / 'partitions' / f'part-{partition:03d}'
    if (report_dir / 'report.json').exists():
        return json.loads((report_dir / 'report.json').read_text())
    repair_base = root / 'repair-audit-v2' / f'part-{partition:03d}'
    repair_base.mkdir(parents=True, exist_ok=True)
    function_call_id = modal.current_function_call_id()
    owner_marker = repair_base / 'owner.json'
    if owner_marker.exists():
        validate_owner(json.loads(owner_marker.read_text()), function_call_id)
    else:
        with owner_marker.open('x') as handle:
            json.dump({'function_call_id': function_call_id}, handle)
    repair_root = repair_base / ('attempt-' + uuid.uuid4().hex)
    repair_root.mkdir()
    launch_marker = repair_root / 'launch.json'
    launch = {'partition': partition, 'started_at_unix': time.time(),
              'function_call_id': function_call_id,
              'original_map_call': 'fc-01M22CD6E8JS9J1K5JGJJ3DHDR',
              'original_status': 'Preempted; retry hit the original exclusive launch marker',
              'changes': ['max in-flight chunks=16 at chunksize16', 'memory=65536MiB',
                          'durable traceback capture', 'same-call preemption retry with per-attempt quarantine'],
              'unchanged': ['source selection', 'row IDs', 'reasoning policy', 'categories',
                            'tokenizer pins', 'length limits', 'loss masks'], 'quarantined': []}
    for label, path in [('partition-report', report_dir)] + [
            (f'partial-{length}', root / f'packed-cs16-{length}/in-progress' / f'part-{partition:03d}')
            for length in core.LENGTHS]:
        if path.exists():
            destination = repair_root / 'quarantine' / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            files = list(path.rglob('*'))
            launch['quarantined'].append({'old_path': str(path), 'new_path': str(destination),
                'files': [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size}
                          for p in files if p.is_file()]})
            os.replace(path, destination)
    # A completed output without its report is also kept outside loader discovery.
    for length in core.LENGTHS:
        path = root / f'packed-cs16-{length}/data/mixed' / f'part-{partition:03d}'
        if path.exists():
            destination = repair_root / 'quarantine' / f'unreported-final-{length}'
            os.replace(path, destination)
            launch['quarantined'].append({'old_path': str(path), 'new_path': str(destination)})
    launch_marker.write_text(json.dumps(launch, indent=2))
    core.volume.commit()
    original_imap = multiprocessing.pool.Pool.imap_unordered
    multiprocessing.pool.Pool.imap_unordered = bounded_imap
    try:
        result = core.pack_partition.get_raw_f()(partition, manifest_sha, 0)
        (repair_root / 'result.json').write_text(json.dumps({'status': 'complete',
            'partition': partition, 'seconds': time.time() - launch['started_at_unix'],
            'source_manifest_sha256': manifest_sha,
            'counts': {k: v['counts'] for k, v in result['outputs'].items()}}, indent=2))
        core.volume.commit()
        return result
    except BaseException as error:
        failure = {'status': 'failed', 'partition': partition,
                   'exception_type': type(error).__name__, 'exception': str(error),
                   'traceback': traceback.format_exc()}
        print(json.dumps(failure), flush=True)
        (repair_root / 'error.json').write_text(json.dumps(failure, indent=2))
        core.volume.commit()
        raise
    finally:
        multiprocessing.pool.Pool.imap_unordered = original_imap


@app.function(**core.options, cpu=1, memory=2048, timeout=120)
def bounded_smoke():
    """Check bounded scheduling and exact item coverage without a data scan."""
    class Result:
        def __init__(self, pool, func, args): self.pool, self.func, self.args = pool, func, args
        def get(self):
            self.pool.pending -= 1
            return self.func(*self.args)
    class Pool:
        _processes = 2
        pending = 0
        maximum = 0
        def apply_async(self, func, args):
            self.pending += 1
            self.maximum = max(self.maximum, self.pending)
            return Result(self, func, args)
    pool = Pool()
    results = list(bounded_imap(pool, lambda value: value * 2, range(103), 16))
    assert results == [i * 2 for i in range(103)]
    assert pool.maximum == 4 and pool.pending == 0
    validate_owner({'function_call_id': 'same'}, 'same')
    try:
        validate_owner({'function_call_id': 'old'}, 'different')
    except ValueError:
        pass
    else:
        raise AssertionError('Separate call should not share partition ownership')
    return {'status': 'passed', 'rows': len(results), 'max_pending_chunks': pool.maximum,
            'preemption_retry_owner_guard': 'passed'}
