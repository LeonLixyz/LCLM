"""Continuous batching of independent trajectories on one backend.

Each trajectory has at most one in-flight request. Whenever a request
finishes, its trajectory's next request (or a new trajectory, if it ended)
is submitted without waiting for unrelated requests.
"""
import time

from .controller import MAX_CONTEXT, trajectory


def serve(backend, next_job, on_complete, max_active=16, on_progress=None, **trajectory_kwargs):
    """Run jobs until ``next_job()`` returns ``None`` and all trajectories finish.

    ``next_job()`` returns ``(job_id, messages, compact)``. ``on_complete(job_id, result)``
    is called with each finished trajectory's result. ``trajectory_kwargs`` are
    passed to ``compaction.controller.trajectory``.
    """
    trajectory_kwargs.setdefault('max_context', getattr(backend, 'max_context', MAX_CONTEXT))
    active, pending, ready = {}, {}, []
    ended = False
    counter = 0
    started = last_progress = time.monotonic()
    stats = dict(completed=0, requests_submitted=0, max_active=0, refills_while_other_requests_pending=0)

    def prepare(state, request, allowance):
        nonlocal counter
        counter += 1
        rid = f"{state['id']}:{counter}"
        assert all(p is not state for p in pending.values())
        pending[rid] = state
        return dict(namespace=state['id'], request_id=rid, request=request, allowance=allowance)

    while active or not ended:
        while not ended and len(active) < max_active:
            job = next_job()
            if job is None:
                ended = True
                break
            job_id, messages, compact = job
            assert job_id not in active
            stream = trajectory(backend.prefix(messages), backend.decode, compact=compact, **trajectory_kwargs)
            state = dict(id=job_id, stream=stream, started=time.monotonic())
            active[job_id] = state
            ready.append(prepare(state, *next(stream)))
        if ready:
            if len(pending) > len(ready):
                stats['refills_while_other_requests_pending'] += 1
            backend.submit_many(ready)
            stats['requests_submitted'] += len(ready)
            stats['max_active'] = max(stats['max_active'], len(active))
            ready = []
        if not pending:
            break
        for rid, output in backend.poll():
            state = pending.pop(rid)
            try:
                ready.append(prepare(state, *state['stream'].send(output)))
            except StopIteration as done:
                result = done.value
                result['wall_seconds'] = time.monotonic() - state['started']
                backend.clear_cache(state['id'])
                active.pop(state['id'])
                on_complete(state['id'], result)
                stats['completed'] += 1
        if on_progress is not None and time.monotonic() - last_progress > 5:
            on_progress(dict(stats, active=len(active), elapsed_seconds=time.monotonic() - started))
            last_progress = time.monotonic()
    assert not pending and not active and not ready
    return dict(stats, elapsed_seconds=time.monotonic() - started)
