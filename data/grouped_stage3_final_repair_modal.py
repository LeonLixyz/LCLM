"""Isolated partition-51 repair with explicit lost-worker detection.

Reuses the frozen v3 processor and existing quarantine/report implementation.
It never drops a row or changes category, tokenization, masks, or length policy.
"""
import json
import time
from collections import deque
from itertools import islice
import multiprocessing as mp

import modal
from data import grouped_stage3_packing_modal as core
from data import grouped_stage3_repair_coordinator_modal as previous

app = modal.App('lclm-grouped-final-repair-20260909-v1')
OLD_CALL = 'fc-01M22EW25AR4Z860XBBBQW6RPF'


def safe_chunk(function, items):
    result = []
    for item in items:
        try:
            result.append(function(item))
        except BaseException as error:
            identity = item[1] if isinstance(item, tuple) and len(item) > 1 else repr(item)
            raise RuntimeError(f'Fatal worker exception on {identity}: {type(error).__name__}: {error}') from error
    return result


def guarded_imap(pool, function, iterable, chunksize=1):
    source = iter(iterable)
    pending = deque()
    workers = tuple(pool._pool)
    capacity = max(1, 2 * pool._processes)
    exhausted = False
    while pending or not exhausted:
        while len(pending) < capacity and not exhausted:
            chunk = list(islice(source, chunksize))
            if not chunk:
                exhausted = True
                break
            identities = [item[1] if isinstance(item, tuple) and len(item) > 1 else repr(item) for item in chunk]
            pending.append((pool.apply_async(safe_chunk, (function, chunk)), identities))
        if pending:
            future, identities = pending.popleft()
            started = time.monotonic()
            while True:
                try:
                    values = future.get(timeout=10)
                    break
                except mp.TimeoutError:
                    lost = [{'pid': worker.pid, 'exitcode': worker.exitcode}
                            for worker in workers if worker.exitcode is not None]
                    if lost:
                        raise RuntimeError('Lost multiprocessing worker: ' + json.dumps(
                            {'workers': lost, 'pending_rows': identities}))
                    if time.monotonic() - started > 600:
                        raise TimeoutError('Worker chunk exceeded 600 seconds: ' + json.dumps(identities))
            yield from values


@app.function(**core.options, cpu=8, memory=65536, timeout=86400, max_containers=1)
def repair(manifest_sha):
    core.volume.reload()
    report = core.ROOT / 'partitions/part-051/report.json'
    if report.exists():
        value = json.loads(report.read_text())
        assert value['source_manifest_sha256'] == manifest_sha
        return {'status': 'already_complete', 'partition': 51}
    # Result polling is authoritative; a child graph includes its enclosing root.
    try:
        modal.FunctionCall.from_id(OLD_CALL).get(timeout=0)
    except TimeoutError:
        raise RuntimeError('Old partition-51 call is still active; do not race it')
    except BaseException:
        pass
    else:
        raise RuntimeError('Old call succeeded without its expected report; inspect before retry')
    owner = json.loads((core.ROOT / 'repair-completion-v2/part-051/owner.json').read_text())
    assert owner['function_call_id'] == OLD_CALL
    previous.AUDIT = core.ROOT / 'final-repair-v1'
    previous.bounded_imap = guarded_imap
    for attempt in range(2):
        try:
            return previous.repair.get_raw_f()(51, manifest_sha)
        except RuntimeError as error:
            if not str(error).startswith('Lost multiprocessing worker:') or attempt:
                raise
            print(json.dumps({'retry': 'one bounded lost-worker retry', 'error': str(error)}), flush=True)
    raise AssertionError('Unreachable')


@app.function(**core.options, cpu=1, memory=2048, timeout=120)
def smoke():
    class Worker:
        pid = 7
        exitcode = None
    class Result:
        def __init__(self, function, args, lost=False):
            self.function, self.args, self.lost = function, args, lost
        def get(self, timeout):
            if self.lost:
                raise mp.TimeoutError()
            return self.function(*self.args)
    class Pool:
        _processes = 2
        def __init__(self, lost=False): self._pool, self.lost = [Worker()], lost
        def apply_async(self, function, args):
            if self.lost: self._pool[0].exitcode = -9
            return Result(function, args, self.lost)
    assert list(guarded_imap(Pool(), lambda value: value * 2, range(103), 16)) == [v * 2 for v in range(103)]
    try:
        list(guarded_imap(Pool(lost=True), lambda value: value, range(3), 2))
    except RuntimeError as error:
        assert str(error).startswith('Lost multiprocessing worker:')
    else:
        raise AssertionError('Worker loss must not leave an unresolved future')
    def fatal(_): raise SystemExit('fixture')
    try:
        safe_chunk(fatal, [1])
    except RuntimeError as error:
        assert 'SystemExit' in str(error)
    else:
        raise AssertionError('BaseException must be transported to parent explicitly')
    return {'status': 'passed', 'coverage_rows': 103, 'lost_worker_detection': True,
            'fatal_worker_exception_transport': True}
