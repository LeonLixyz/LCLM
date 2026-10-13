"""Interleaved trajectories must match serial generation exactly."""
from collections import deque
from copy import deepcopy

from compaction.controller import run
from compaction.scheduler import serve

LENGTHS = {1: 100, 2: 3400, 3: 1900, 4: 700, 5: 2100}


class FakeBackend:
    """Problem ``i`` emits ``LENGTHS[i]`` copies of one letter; completes requests out of order."""

    def __init__(self):
        self.pending = {}
        self.steps = 0
        self.submission_log = []

    def prefix(self, messages):
        return [int(messages[0]['content'])]

    def decode(self, ids, final=False):
        return ''.join(chr(i) for i in ids)

    def clear_cache(self, namespace=None):
        assert all(v['namespace'] != namespace for v in self.pending.values())

    def generate(self, request, allowance):
        ident = request['prefix_ids'][0]
        generated = sum(len(m['token_ids']) for m in request['memories']) + len(request['recent_ids'])
        n = min(allowance, LENGTHS[ident] - generated)
        letter = 65 + ident
        # Responses must never mix trajectories.
        assert all(t == letter for t in request['recent_ids'])
        assert all(t == letter for m in request['memories'] for t in m['token_ids'])
        positions = 1 + len(request['recent_ids']) + sum(len(m['token_ids']) // 16 + 2 for m in request['memories'])
        return dict(token_ids=[letter] * n, finish_reason='stop' if generated + n == LENGTHS[ident] else 'length',
                    prompt_positions=positions)

    def submit_many(self, items):
        for item in items:
            self.submission_log.append((self.steps, item['request_id'], set(self.pending)))
            self.pending[item['request_id']] = deepcopy(item)

    def poll(self):
        self.steps += 1
        # Short problems finish before earlier long requests: deliberately out of order.
        _, rid = min((v['request']['prefix_ids'][0], k) for k, v in self.pending.items())
        item = self.pending.pop(rid)
        return [(rid, self.generate(item['request'], item['allowance']))]


def job(problem, compact, name=None):
    return (name or f'{problem}-{int(compact)}', [{'role': 'user', 'content': str(problem)}], compact)


def test_interleaving_matches_serial_controller():
    jobs = deque(job(i, c) for i in (2, 1, 5, 3, 4) for c in (False, True))
    rows = {}
    stats = serve(FakeBackend(), lambda: jobs.popleft() if jobs else None,
                  lambda jid, row: rows.update({jid: row}), max_active=4)
    assert len(rows) == 10 and stats['max_active'] == 4
    assert stats['refills_while_other_requests_pending'] > 0
    for jid, row in rows.items():
        _, messages, compact = job(int(jid.split('-')[0]), jid.endswith('-1'))
        reference = run(FakeBackend(), messages, compact=compact)
        for key, value in reference.items():
            assert row[key] == value, (jid, key)


def test_refills_slot_before_unrelated_long_request_finishes():
    jobs = deque([job(2, False, 'long'), job(1, False, 'fast'), job(3, False, 'next')])
    backend = FakeBackend()
    finished = []
    serve(backend, lambda: jobs.popleft() if jobs else None, lambda jid, row: finished.append(jid), max_active=2)
    next_submit = next(x for x in backend.submission_log if x[1].startswith('next:'))
    assert any(k.startswith('long:') for k in next_submit[2])
    assert finished[0] == 'fast'
