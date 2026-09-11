"""Bounded completion of the immutable v3 pack, including preempted inputs.

No changes to data semantics: this calls the exact frozen v3 raw processor.
Only missing partitions after the original map and previous repairs terminate
are eligible. Partial attempts are retained outside loader discovery.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from data import grouped_stage3_packing_modal as core
from data.grouped_stage3_repair_modal import bounded_imap, validate_owner

APP_NAME = 'lclm-stage3-grouped-completion-20260909-v2'
app = modal.App(APP_NAME)
ORIGINAL_CALL = 'fc-01M22CD5A03YKF3RB02Q6WRJ3K'
ORIGINAL_MAP = 'fc-01M22CD6E8JS9J1K5JGJJ3DHDR'
KNOWN_REPAIRS = {
    9: 'fc-01M22DHK1W9EKTNMQF758WTJH9',
    21: 'fc-01M22D8P9BTSRX2A59A5N58HFV',
    30: 'fc-01M22D8PBP9WYTQJ82MM5TAXSA',
}
AUDIT = core.ROOT / 'repair-completion-v2'
MAX_REPAIRS = 16


def call_status(call_id):
    graph = modal.FunctionCall.from_id(call_id).get_call_graph()
    assert len(graph) == 1, graph
    return graph[0].status.name


def original_snapshot():
    graph = modal.FunctionCall.from_id(ORIGINAL_CALL).get_call_graph()
    assert len(graph) == 1
    children = [child for child in graph[0].children if child.function_call_id == ORIGINAL_MAP]
    assert len(children) == core.PARTITIONS, len(children)
    return {'root_status': graph[0].status.name,
            'inputs': [{'input_id': child.input_id, 'status': child.status.name} for child in children]}


def assert_original_terminal(snapshot):
    assert len(snapshot['inputs']) == core.PARTITIONS
    assert all(item['status'] != 'PENDING' for item in snapshot['inputs']), 'Original map still active'


def completed_reports():
    return {int(path.parent.name.split('-')[-1]): path
            for path in (core.ROOT / 'partitions').glob('part-*/report.json')}


@app.function(**core.options, cpu=8, memory=65536, timeout=86400, max_containers=MAX_REPAIRS)
def repair(partition: int, manifest_sha: str):
    import multiprocessing.pool
    import os
    import traceback
    import uuid

    assert 0 <= partition < core.PARTITIONS
    core.volume.reload()
    report_dir = core.ROOT / 'partitions' / f'part-{partition:03d}'
    if (report_dir / 'report.json').exists():
        result = json.loads((report_dir / 'report.json').read_text())
        assert result['source_manifest_sha256'] == manifest_sha
        return {'partition': partition, 'status': 'already_complete'}
    snapshot = original_snapshot()
    assert_original_terminal(snapshot)
    # Public call graphs do not expose map arguments. Waiting for every original
    # input avoids assuming positional correspondence or racing a healthy writer.
    assert any(item['status'] != 'SUCCESS' for item in snapshot['inputs'])
    if partition in KNOWN_REPAIRS:
        assert call_status(KNOWN_REPAIRS[partition]) != 'PENDING', 'Earlier repair still active'
    base = AUDIT / f'part-{partition:03d}'
    base.mkdir(parents=True, exist_ok=True)
    owner = base / 'owner.json'
    call_id = modal.current_function_call_id()
    if owner.exists():
        validate_owner(json.loads(owner.read_text()), call_id)
    else:
        with owner.open('x') as handle:
            json.dump({'function_call_id': call_id, 'partition': partition}, handle)
    attempt = base / ('attempt-' + uuid.uuid4().hex)
    attempt.mkdir()
    launch = {'partition': partition, 'function_call_id': call_id,
              'started_at_unix': time.time(), 'original_snapshot': snapshot,
              'source_manifest_sha256': manifest_sha, 'quarantined': [],
              'semantics': 'exact v3 processor; bounded chunks and retry-safe ownership only'}
    candidates = [('partition-report', report_dir)]
    for length in core.LENGTHS:
        output = core.ROOT / f'packed-cs16-{length}'
        candidates.extend([(f'partial-{length}', output / 'in-progress' / f'part-{partition:03d}'),
                           (f'unreported-final-{length}', output / 'data/mixed' / f'part-{partition:03d}')])
    for label, path in candidates:
        if path.exists():
            target = attempt / 'quarantine' / label
            target.parent.mkdir(parents=True, exist_ok=True)
            listing = [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size}
                       for p in path.rglob('*') if p.is_file()]
            os.replace(path, target)
            launch['quarantined'].append({'old_path': str(path), 'new_path': str(target), 'files': listing})
    (attempt / 'launch.json').write_text(json.dumps(launch, indent=2))
    core.volume.commit()
    previous = multiprocessing.pool.Pool.imap_unordered
    multiprocessing.pool.Pool.imap_unordered = bounded_imap
    try:
        result = core.pack_partition.get_raw_f()(partition, manifest_sha, 0)
        summary = {'partition': partition, 'status': 'complete',
                   'seconds': time.time() - launch['started_at_unix'],
                   'counts': {length: value['counts'] for length, value in result['outputs'].items()}}
        (attempt / 'result.json').write_text(json.dumps(summary, indent=2))
        core.volume.commit()
        return summary
    except BaseException as error:
        failure = {'partition': partition, 'status': 'failed', 'exception_type': type(error).__name__,
                   'exception': str(error), 'traceback': traceback.format_exc()}
        print(json.dumps(failure), flush=True)
        (attempt / 'error.json').write_text(json.dumps(failure, indent=2))
        core.volume.commit()
        raise
    finally:
        multiprocessing.pool.Pool.imap_unordered = previous


@app.function(**core.options, cpu=1, memory=2048, timeout=14400, max_containers=1)
def complete():
    """Watch every 45 seconds, repair at most 16 concurrently, then verify packs."""
    core.volume.reload()
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest_sha = json.loads((core.ROOT / 'preflight.json').read_text())['source_manifest_sha256']
    started = time.time()
    while time.time() - started < 3 * 3600:
        core.volume.reload()
        summary_path = core.ROOT / 'partitions/summary.json'
        if summary_path.exists():
            return {'status': 'complete', 'summary': str(summary_path)}
        reports = completed_reports()
        snapshot = original_snapshot()
        pending_original = sum(item['status'] == 'PENDING' for item in snapshot['inputs'])
        tracked = dict(KNOWN_REPAIRS)
        for owner_path in AUDIT.glob('part-*/owner.json'):
            owner = json.loads(owner_path.read_text())
            tracked[owner['partition']] = owner['function_call_id']
        state_path = AUDIT / 'coordinator.json'
        if state_path.exists():
            old = json.loads(state_path.read_text())
            tracked.update({int(part): call for part, call in old.get('repair_calls', {}).items()})
        active = {}
        errors = {}
        for partition, call in tracked.items():
            if partition in reports:
                continue
            state = call_status(call)
            if state == 'PENDING':
                active[partition] = call
            elif partition not in KNOWN_REPAIRS or call != KNOWN_REPAIRS[partition]:
                errors[partition] = {'call': call, 'status': state}
        state = {'completed': len(reports), 'pending_original': pending_original,
                 'active_repairs': active, 'repair_calls': tracked, 'terminal_repair_errors': errors,
                 'updated_at_unix': time.time(), 'function_call_id': modal.current_function_call_id()}
        state_path.write_text(json.dumps(state, indent=2))
        core.volume.commit()
        print(json.dumps(state), flush=True)
        if errors:
            raise RuntimeError(f'Terminal generalized repair failure requires investigation: {errors}')
        if len(reports) == core.PARTITIONS:
            # Let the original coordinator finish if every original input passed.
            if snapshot['root_status'] == 'PENDING' and all(item['status'] == 'SUCCESS' for item in snapshot['inputs']):
                time.sleep(45)
                continue
            final = modal.Function.from_name(core.APP_NAME, 'finalize').remote(False)
            return {'status': 'complete', 'summary': str(summary_path),
                    'versions': {length: {'counts': value['counts'], 'packed_sequences': value['packed_sequences']}
                                 for length, value in final.items()}}
        if not pending_original:
            assert_original_terminal(snapshot)
            for partition in range(core.PARTITIONS):
                if len(active) >= MAX_REPAIRS:
                    break
                if partition in reports or partition in active:
                    continue
                # Existing failed v1/v2 attempt is now safely replaceable.
                call = repair.spawn(partition, manifest_sha)
                tracked[partition] = call.object_id
                active[partition] = call.object_id
                state['repair_calls'] = tracked
                state['active_repairs'] = active
                state_path.write_text(json.dumps(state, indent=2))
                core.volume.commit()
        time.sleep(45)
    raise TimeoutError('Completion coordinator exceeded bounded three-hour watch')


@app.function(**core.options, cpu=1, memory=2048, timeout=120)
def smoke():
    from data.grouped_stage3_repair_modal import bounded_smoke
    result = bounded_smoke.get_raw_f()()
    snapshot = {'inputs': [{'status': 'SUCCESS'}] * 63 + [{'status': 'FAILURE'}]}
    assert_original_terminal(snapshot)
    try:
        assert_original_terminal({'inputs': [{'status': 'PENDING'}] * 64})
    except AssertionError:
        pass
    else:
        raise AssertionError('Active original inputs must block repair')
    return {**result, 'terminal_guard': 'passed', 'original_snapshot': original_snapshot()}
