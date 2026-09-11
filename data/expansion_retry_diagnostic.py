"""Deterministic failed-only diagnostic sampling and stage-specific accounting."""
import hashlib
import heapq
from collections import Counter, defaultdict


class RawCaptureError(RuntimeError):
    pass


class TruncatedResponseError(RuntimeError):
    pass


class IncompleteResponseError(RuntimeError):
    pass


def response_token_ids(choice):
    ids = choice.get('response_token_ids')
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise RawCaptureError('Server returned missing/invalid pre-parser token IDs')
    return ids


def ensure_complete_response(finish_reason, phase):
    if finish_reason == 'length':
        raise TruncatedResponseError('Truncated ' + phase + ' response')
    if finish_reason not in ('stop', 'tool_calls'):
        raise IncompleteResponseError(phase + ' response finished with ' + str(finish_reason))


def fair_quotas(capacities, total):
    if total < 0 or total > sum(capacities.values()):
        raise ValueError('Requested sample exceeds available failures')
    quotas = dict.fromkeys(sorted(capacities), 0)
    for _ in range(total):
        key = min((k for k in quotas if quotas[k] < capacities[k]),
                  key=lambda k: (quotas[k], k))
        quotas[key] += 1
    return {k: n for k, n in quotas.items() if n}


def select_failures(rows, count):
    """Balance failure reasons; choose stable hash samples within each reason."""
    heaps = defaultdict(list)
    capacities = Counter()
    for row in rows:
        provenance = row.get('_attempt_provenance', {})
        if provenance.get('kind') != 'retry_rejected':
            continue
        reason = provenance['previous_reason']
        capacities[reason] += 1
        score = int(hashlib.sha256(('failed-1k-20260909:' + row['task_id']).encode()).hexdigest(), 16)
        item = (-score, row['task_id'], row)
        heap = heaps[reason]
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    quotas = fair_quotas(capacities, count)
    selected = [item[2] for reason, n in quotas.items()
                for item in sorted(heaps[reason], key=lambda x: (-x[0], x[1]))[:n]]
    if len({r['task_id'] for r in selected}) != count:
        raise ValueError('Duplicate selected task ID')
    return sorted(selected, key=lambda r: r['task_id']), dict(capacities), quotas


def summarize(results):
    by_source = defaultdict(Counter)
    transitions = Counter()
    for row in results:
        counts = by_source[row['source']]
        counts['completed'] += 1
        counts['tasks_with_valid_expansion'] += bool(row.get('tool_call_count'))
        counts['rule_accepted'] += row.get('rule_verification', {}).get('accepted') is True
        counts['automatic_accepted'] += row.get('verification', {}).get('accepted') is True
        reason = row.get('verification', {}).get('reason', 'unscored')
        counts['outcome:' + reason] += 1
        if row.get('error'):
            counts['error_phase:' + row['error']['phase']] += 1
        transitions[(row['previous_reason'], reason)] += 1
    total = Counter()
    for counts in by_source.values():
        total.update(counts)
    return {'completed': len(results), 'counts': dict(total),
            'sources': {k: dict(v) for k, v in sorted(by_source.items())},
            'transitions': [{'previous': a, 'new': b, 'count': n}
                            for (a, b), n in sorted(transitions.items())]}
