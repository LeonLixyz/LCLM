"""Lossless candidate partitioning, not semantic or publication approval.

No I/O and no release-selection integration. Call only with a completed review;
all original accepted rows must receive exactly one disposition. A future writer
must use separate output paths, preserve originals, and audit/review the subset.
"""
import hashlib
import json
from collections import Counter


def digest(content):
    return hashlib.sha256(content).hexdigest()


def jsonl(content):
    if not isinstance(content, bytes): raise ValueError('Expected raw JSONL bytes')
    for line in content.splitlines(keepends=True):
        if not line.endswith(b'\n') or not line.strip(): raise ValueError('Truncated/blank JSONL row')
        row = json.loads(line)
        if not isinstance(row, dict): raise ValueError('Expected JSONL object')
        yield line, row


def completed_decisions(content, report):
    """Validate full multi-source coverage/counts before partitioning any source."""
    manifest = report['manifest']; sources = manifest['sources']
    if (report.get('status') != 'complete' or report.get('remaining') != 0
            or type(report.get('remaining')) is not int or type(report.get('reviewed')) is not int
            or type(manifest.get('candidate_rows')) is not int
            or report.get('training_rows_modified') is not False
            or manifest.get('training_rows_modified') is not False
            or report.get('decisions_sha256') != digest(content)):
        raise ValueError('Incomplete, mutated or unbound review')
    expected = {r['source']: r['rows'] for r in sources}
    if (not expected or len(expected) != len(sources)
            or any(type(n) is not int or n <= 0 for n in expected.values())):
        raise ValueError('Invalid source manifest')
    rows = {}; counts = {s: Counter() for s in expected}
    for _, row in jsonl(content):
        task_id = row.get('task_id'); source = row.get('source')
        if (not isinstance(task_id, str) or not task_id or task_id in rows
                or not isinstance(source, str) or source not in expected or type(row.get('keep')) is not bool
                or ('error' in row and (row['keep'] or not isinstance(row['error'], str) or not row['error']))):
            raise ValueError('Invalid/duplicate review decision')
        rows[task_id] = row
        counts[source]['error' if 'error' in row else 'kept' if row['keep'] else 'rejected'] += 1
    saved_counts = report.get('counts')
    if (not isinstance(saved_counts, dict) or set(saved_counts) != set(expected)
            or any(not isinstance(c, dict) or any(type(n) is not int or n < 0 for n in c.values())
                   for c in saved_counts.values())):
        raise ValueError('Malformed saved review counts')
    if (len(rows) != sum(expected.values()) or report.get('reviewed') != len(rows)
            or manifest.get('candidate_rows') != len(rows)
            or any(sum(counts[s].values()) != n for s, n in expected.items())
            or report.get('counts') != {s: dict(c) for s, c in counts.items()}):
        raise ValueError('Review counts/coverage do not reconcile')
    return rows


def partition_candidate(source, original_bytes, decision_bytes, review_report, generation_report):
    decisions = completed_decisions(decision_bytes, review_report)
    entries = {r['source']: r for r in review_report['manifest']['sources']}
    if source not in entries or digest(original_bytes) != entries[source]['accepted_file_sha256']:
        raise ValueError('Original accepted source hash mismatch')
    reasons = generation_report.get('reasons', {})
    if (generation_report.get('status') != 'complete' or generation_report.get('source') != source
            or not reasons or any(not isinstance(k, str) or type(n) is not int or n < 0 for k, n in reasons.items())
            or sum(n for k, n in reasons.items() if k.startswith('accepted:')) != entries[source]['rows']):
        raise ValueError('Incomplete/inconsistent original generation report')
    source_decisions = {k: r for k, r in decisions.items() if r['source'] == source}
    kept = []; excluded = []; ledger = []; seen = set(); counts = Counter()
    for line, row in jsonl(original_bytes):
        task_id = row.get('task_id')
        if (not isinstance(task_id, str) or task_id not in source_decisions or task_id in seen
                or row.get('verification', {}).get('accepted') is not True):
            raise ValueError('Unverified, duplicate or unmatched original row')
        seen.add(task_id); decision = source_decisions[task_id]
        outcome = 'review_error' if 'error' in decision else 'retained_candidate' if decision['keep'] else 'review_rejected'
        counts[outcome] += 1
        (kept if decision['keep'] else excluded).append(line)
        ledger.append({'task_id': task_id, 'source': source, 'disposition': outcome,
            'original_row_sha256': digest(line), 'decision': decision})
    if seen != set(source_decisions) or len(seen) != entries[source]['rows']:
        raise ValueError('Not every original row has exactly one disposition')
    retained_bytes = b''.join(kept); excluded_bytes = b''.join(excluded)
    ledger_bytes = b''.join((json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n').encode() for r in ledger)
    original_rejections = {k: n for k, n in reasons.items() if not k.startswith('accepted:')}
    accounted = sum(original_rejections.values()) + sum(counts.values())
    if accounted != sum(reasons.values()): raise ValueError('Original generation accounting mismatch')
    manifest = {'status': 'candidate_partitioned_unreleased', 'source': source,
        'original_accepted_file_sha256': digest(original_bytes), 'review_decisions_sha256': digest(decision_bytes),
        'review_report_sha256': digest(json.dumps(review_report, sort_keys=True).encode()),
        'original_generation_report': generation_report, 'counts': dict(counts),
        'original_generation_rejection_reasons': original_rejections,
        'original_accepted_rows': len(seen), 'total_original_attempts_accounted': accounted,
        'retained_file_sha256': digest(retained_bytes), 'excluded_file_sha256': digest(excluded_bytes),
        'ledger_sha256': digest(ledger_bytes), 'approved_for_release': False,
        'original_rows_rewritten': False,
        'limits': 'Only bookkeeping validated. Retained means judge-kept candidate, not verified correct. Original generation rejection ledger, source provenance, subset format audit and fresh manual review remain required.'}
    return {'retained': retained_bytes, 'excluded': excluded_bytes, 'ledger': ledger_bytes, 'manifest': manifest}
