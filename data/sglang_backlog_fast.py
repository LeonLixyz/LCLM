"""Reviewed source probes, immutable input references, and bounded continuous work."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from data.sglang_backlog import (
    EXPECTED_PENDING, MODEL, REVISION, IMAGE, atomic_json, file_sha, sha,
    generation_config as base_generation_config, validate_scale_decision,
)

REPLICAS = 8
REQUESTS_PER_REPLICA = 16
CONCURRENCY = REPLICAS * REQUESTS_PER_REPLICA
RESERVATION_BLOCK = 64
PROBE_SIZE = 64
TEACHER_GUIDANCE_VERSION = 'pending-source-prompt-canary-v1'
TEACHER_GUIDANCE = (
    'Give a concise direct answer supported by the expanded evidence. Preserve material '
    'conditions, exceptions, thresholds, deadlines, and distinct effective dates. Do not '
    'infer additional eligibility restrictions, procedural exclusions, or unstated requirements. '
    'For numerical questions, give the answer directly in the units stated by the question; '
    'do not include calculation prose. Use FINAL: for the answer.'
)
BILLSUM_GUIDANCE = (
    'For this bill summary, write exactly one concise paragraph after FINAL:. Do not use '
    'bullets, headings, or newlines in the final answer. Retain the material conditions '
    'and exceptions and preserve distinct effective dates and earlier-of or later-of '
    'deadline conditions when the evidence specifies them.'
)


def teacher_messages(messages, source, phase):
    """Change only a request copy; stored task/training messages remain untouched."""
    import copy
    from data.synthetic_expansion_agent import messages_for_openai_api
    result = copy.deepcopy(messages_for_openai_api(messages))
    if phase == 'rollout':
        if not result or result[0]['role'] != 'system' or not isinstance(result[0]['content'], str):
            raise ValueError('Expected a rollout teacher system message')
        guidance = TEACHER_GUIDANCE + ('\n' + BILLSUM_GUIDANCE if source == 'billsum' else '')
        result[0]['content'] += '\n\n' + guidance
    return result


def generation_config():
    config = base_generation_config()
    config.update(concurrency=CONCURRENCY, reservation_block=RESERVATION_BLOCK,
        server_layout='8 independent TP1 replicas; one visible H200 each',
        server_dtype='bfloat16', requests_per_replica=REQUESTS_PER_REPLICA,
        cuda_graphs=True, cuda_graph_max_bs_decode=16,
        max_mamba_cache_size=80, chunked_prefill_size=32768,
        source_probe_policy='balanced-config-family-or-attempt-kind-v1',
        scheduler='bounded-continuous-queue-v1')
    config['infrastructure_recovery'] = {'max_attempts_per_task': 3,
        'automatic_retry_scope': 'identified same Modal function-call/input restart, identical frozen chunk/config',
        'one_final_result_per_task': True, 'committed_failures_are_not_retried': True}
    config['teacher_guidance'] = {'version': TEACHER_GUIDANCE_VERSION,
        'general_text': TEACHER_GUIDANCE, 'billsum_text': BILLSUM_GUIDANCE,
        'general_sha256': sha(TEACHER_GUIDANCE.encode()),
        'billsum_sha256': sha(BILLSUM_GUIDANCE.encode()),
        'scope': 'teacher rollout request copies only; task/training prompts and validators unchanged',
        'comparison': 'source prompt canary; not an isolated model comparison'}
    for name in ('sglang_backlog_fast.py', 'sglang_backlog_fast_modal.py',
                 'sglang_27b_serving.py', 'sglang_throughput_smoke.py',
                 'expansion_source_probes.py'):
        config['code_sha256'][name] = file_sha(Path(__file__).parent / name)
    return config


def checked_rows(spec):
    """Hash the exact bytes consumed, and validate effective rows/IDs on exhaustion."""
    excluded = set(spec.get('exclude_task_ids', []))
    if len(excluded) != len(spec.get('exclude_task_ids', [])):
        raise ValueError('Duplicate excluded task ID')
    digest, count, ids, removed = hashlib.sha256(), 0, [], set()
    with Path(spec['path']).open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.endswith(b'\n'):
                raise ValueError('Partial referenced task line')
            task = json.loads(raw)
            if task['task_id'] in excluded:
                if task['task_id'] in removed:
                    raise ValueError('Duplicate excluded task in referenced chunk')
                removed.add(task['task_id'])
                continue
            count += 1
            ids.append(task['task_id'])
            yield task
    if digest.hexdigest() != spec['sha256'] or count != spec['rows'] or removed != excluded:
        raise ValueError('Referenced chunk bytes/count/exclusions changed')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate referenced task ID')
    if 'task_ids_sha256' in spec and sha(json.dumps(sorted(ids)).encode()) != spec['task_ids_sha256']:
        raise ValueError('Effective referenced task IDs changed')


def original_specs(source, parent_root):
    return [{**chunk, 'path': str(Path(parent_root) / 'inputs' / source['source'] / chunk['name'])}
            for chunk in source['chunks']]


def prepare_source(source, parent_root, output_root, forbidden_ids):
    """Write only a representative probe; reference existing continuation bytes."""
    from data.expansion_source_probes import select_source_probe, select_balanced_probe, DEFAULT_PROBE_SEED
    name = source['source']
    output = Path(output_root) / name
    specs = original_specs(source, parent_root)
    binding = {'source': name, 'parent_source': source, 'generation_config': generation_config(),
               'forbidden_ids_sha256': sha(json.dumps(sorted(forbidden_ids)).encode())}
    report_path = output / 'prepared.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['binding'] != binding:
            raise ValueError('Changed source preparation binding')
        for spec in report['chunks']:
            list(checked_rows(spec))
        return report
    marker = output / 'input-binding.json'
    if output.exists() and any(output.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()) != binding:
            raise ValueError('Partial fast preparation has changed/unclassified inputs')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, binding)
    def tasks():
        for spec in specs:
            for task in checked_rows(spec):
                if task['task_id'] in forbidden_ids:
                    raise ValueError('Pending input overlaps pilot or original accepted task')
                kind = task.get('_attempt_provenance', {}).get('kind')
                if kind not in EXPECTED_PENDING:
                    raise ValueError('Pending task lacks failed/unattempted provenance')
                yield task
    probe_count = min(PROBE_SIZE, source['rows'])
    if name in ('lex_glue', 'synthetic'):
        selected, capacities, quotas = select_source_probe(tasks(), probe_count, source=name)
    else:
        selected, capacities, quotas = select_balanced_probe(tasks(), probe_count,
            strata=tuple(EXPECTED_PENDING), classify=lambda task: task['_attempt_provenance']['kind'])
    selected_ids = {row['task_id'] for row in selected}
    if len(selected_ids) != probe_count or sum(capacities.values()) != source['rows']:
        raise ValueError('Probe/parent selection count disagrees')
    chunks, counts, seen_probe, coverage = [], Counter(), set(), []
    if selected:
        probe_path = output / 'probe.tasks.jsonl'
        raw = b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode() for row in selected)
        probe_path.write_bytes(raw)
        chunks.append({'index': 0, 'kind': 'probe', 'path': str(probe_path), 'rows': len(selected),
            'sha256': sha(raw), 'exclude_task_ids': [],
            'task_ids_sha256': sha(json.dumps(sorted(selected_ids)).encode())})
        coverage.extend({'task_id': task['task_id'], 'kind': task['_attempt_provenance']['kind'], 'chunk': 0}
                        for task in selected)
    for base in specs:
        remaining, excluded, remaining_kinds = [], [], {}
        for task in checked_rows(base):
            counts[task['_attempt_provenance']['kind']] += 1
            if task['task_id'] in selected_ids:
                excluded.append(task['task_id'])
                seen_probe.add(task['task_id'])
            else:
                remaining.append(task['task_id'])
                remaining_kinds[task['task_id']] = task['_attempt_provenance']['kind']
        if remaining:
            coverage.extend({'task_id': key, 'kind': remaining_kinds[key], 'chunk': len(chunks)} for key in remaining)
            chunks.append({'index': len(chunks), 'kind': 'continuation', 'path': base['path'],
                'sha256': base['sha256'], 'rows': len(remaining), 'exclude_task_ids': sorted(excluded),
                'task_ids_sha256': sha(json.dumps(sorted(remaining)).encode())})
    if (seen_probe != selected_ids or sum(c['rows'] for c in chunks) != source['rows']
            or dict(counts) != source['counts']):
        raise ValueError('Probe + continuation accounting failed')
    coverage_path = output / 'coverage.ids.jsonl'
    coverage_raw = b''.join((json.dumps(row) + '\n').encode() for row in coverage)
    coverage_path.write_bytes(coverage_raw)
    report = {'binding': binding, 'source': name, 'rows': source['rows'], 'counts': dict(counts),
        'probe_rows': probe_count, 'probe_ids': sorted(selected_ids), 'probe_capacities': capacities,
        'probe_quotas': quotas, 'probe_seed': DEFAULT_PROBE_SEED, 'chunks': chunks,
        'coverage': {'path': str(coverage_path), 'rows': len(coverage), 'sha256': sha(coverage_raw)},
        'retained_prior_outputs': source['retained_prior_outputs'], 'approved_for_release': False}
    atomic_json(report_path, report)
    return report


def validate_serving_evidence(report, manifest, *, report_sha, manifest_sha, requested_report_sha):
    if report_sha != requested_report_sha or report.get('manifest_sha256') != manifest_sha:
        raise ValueError('Serving evidence hash mismatch')
    if (report.get('status') != 'complete_serving_smoke' or report.get('total_replayed_requests') != 128
            or manifest.get('model') != MODEL or manifest.get('model_revision') != REVISION
            or manifest.get('image') != IMAGE):
        raise ValueError('Incomplete or different serving smoke')
    counts = report['all_requests']['counts']
    if counts.get('successful_http_and_capture') != 128 or any(k.startswith('error:') and v for k, v in counts.items()):
        raise ValueError('Serving smoke has unresolved request/parser failures')
    if counts.get('nonempty_reasoning_fields', 0) or counts.get('generated_reasoning_markers', 0):
        raise ValueError('Serving smoke contains generated reasoning')
    if counts.get('valid_native_call', 0) < 1:
        raise ValueError('Serving smoke did not demonstrate native tool parsing')
    if not report['eight_replica_wave']['requests_per_second'] > 0:
        raise ValueError('Serving smoke has no measured throughput')


def validate_source_decision(decision, manifest, *, stage, base_decision_sha=None, probe_reports=None):
    """Additional source-scope gate; call after the shared completed-pilot gate."""
    available = {source['source'] for source in manifest['sources'] if source['rows']}
    if stage == 'probe':
        allowed = decision.get('probe_sources')
        if decision.get('scope') != 'source_probes_only' or not isinstance(allowed, list):
            raise ValueError('Expected explicit probe-only source review')
        if not allowed or len(set(allowed)) != len(allowed) or not set(allowed) <= available:
            raise ValueError('Invalid/duplicate probe source allowlist')
        return set(allowed)
    if stage != 'continuation' or decision.get('scope') != 'reviewed_sources':
        raise ValueError('Expected reviewed-source continuation decision')
    if not isinstance(base_decision_sha, str) or not base_decision_sha or decision.get('probe_decision_sha256') != base_decision_sha:
        raise ValueError('Continuation is bound to a different probe decision')
    reviews = decision.get('source_reviews')
    if not isinstance(reviews, dict) or not reviews or not set(reviews) <= available:
        raise ValueError('Continuation source allowlist is absent or invalid')
    for source, review in reviews.items():
        if review.get('approved_for_generation') is not True or not str(review.get('evidence_review', '')).strip():
            raise ValueError('Missing source-specific quality review')
        probe = (probe_reports or {}).get(source)
        expected, = [item for item in manifest['sources'] if item['source'] == source]
        if (not probe or review.get('probe_report_sha256') != probe['sha256']
                or probe['report'].get('status') != 'complete'
                or probe['report'].get('source') != source
                or probe['report'].get('chunk') != 0
                or type(probe['report'].get('rows')) is not int
                or type(probe['report'].get('completed')) is not int
                or not probe['report']['rows'] > 0
                or probe['report']['completed'] != expected['probe_rows']
                or probe['report']['rows'] != expected['probe_rows']
                or probe['report'].get('input_sha256') != expected['chunks'][0]['sha256']
                or probe['report'].get('backlog_manifest_sha256') != decision['backlog_manifest_sha256']
                or probe['report'].get('decision_sha256') != base_decision_sha):
            raise ValueError('Reviewed source probe is incomplete or changed')
    return set(reviews)


def continuous_queue(tasks, *, run, reserve, save, stop, concurrency=CONCURRENCY,
                     replicas=REPLICAS, reservation_block=RESERVATION_BLOCK):
    """Keep free replica slots filled; commit every reservation before submission.

    A process interruption may leave reserved IDs unresolved. This function never
    retries an ID. Stop drains submitted futures and reports reserved-but-unsent
    IDs explicitly so callers can require reconciliation rather than claim coverage.
    """
    if concurrency <= 0 or replicas <= 0 or concurrency % replicas or reservation_block <= 0:
        raise ValueError('Invalid scheduler capacities')
    tasks = list(tasks)
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate scheduler task IDs')
    cursor, waiting, active, completed = 0, [], {}, []
    free_slots = [i % replicas for i in range(concurrency)]
    stopped, failures = False, []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while True:
            stopped = stopped or bool(stop(completed))
            if not stopped and free_slots and not waiting and cursor < len(tasks):
                batch = tasks[cursor:cursor + reservation_block]
                try:
                    reserve(batch)  # Durable before any task in this block executes.
                    cursor += len(batch)
                    waiting.extend(batch)
                except Exception as exc:
                    failures.append({'phase': 'reservation', 'task_ids': [t['task_id'] for t in batch],
                                     'type': type(exc).__name__, 'message': str(exc)})
                    stopped = True
            while free_slots and waiting and not stopped:
                task, replica = waiting.pop(0), free_slots.pop(0)
                active[pool.submit(run, task, replica)] = (task['task_id'], replica)
            if not stopped and free_slots and cursor < len(tasks):
                continue
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task_id, replica = active.pop(future)
                try:
                    result = future.result()
                    if result['task_id'] != task_id:
                        raise ValueError('Worker returned a different task')
                    save(result)
                    completed.append(result)
                except Exception as exc:
                    failures.append({'task_id': task_id, 'type': type(exc).__name__, 'message': str(exc)})
                    stopped = True
                free_slots.append(replica)
    return {'completed': len(completed), 'stopped': stopped,
            'worker_or_persistence_errors': failures,
            'never_reserved_ids': [t['task_id'] for t in tasks[cursor:]],
            'reserved_not_submitted_ids': [t['task_id'] for t in waiting]}


def append_reservations(path, tasks, call_id):
    with Path(path).open('a') as handle:
        for task in tasks:
            handle.write(json.dumps({'task_id': task['task_id'],
                'task_sha256': sha(json.dumps(task, sort_keys=True).encode()),
                'reserved_at': time.time(), 'function_call_id': call_id}) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def recover_attempts(tasks, journal, results, *, previous_owner, identity, binding, max_attempts=3):
    """Recover missing results only on the same unfinished Modal input restart."""
    if not identity.get('function_call_id') or not identity.get('input_id'):
        raise ValueError('A durable Modal call/input identity is required')
    expected = {task['task_id']: sha(json.dumps(task, sort_keys=True).encode()) for task in tasks}
    if len(expected) != len(tasks):
        raise ValueError('Duplicate input task')
    history = {}
    for row in journal:
        key, number = row['task_id'], row['attempt']
        prior = history.setdefault(key, [])
        if (key not in expected or row['task_sha256'] != expected[key]
                or type(number) is not int or number != len(prior) + 1 or number > max_attempts
                or row.get('binding') != binding):
            raise ValueError('Attempt journal has unknown/changed/duplicate attempts')
        prior.append(row)
    committed = {}
    for row in results:
        key = row['task_id']
        if (key in committed or key not in history or row.get('task_sha256') != expected[key]
                or row.get('attempt') != history[key][-1]['attempt']):
            raise ValueError('Committed result has inconsistent attempt/task provenance')
        committed[key] = row
    missing = sorted(set(history) - set(committed))
    same_restart = bool(previous_owner and previous_owner.get('binding') == binding
        and previous_owner.get('identity') == identity and previous_owner.get('finished') is False)
    if same_restart and any(history[key][-1].get('identity') != identity for key in missing):
        same_restart = False
    if missing and not same_restart:
        return {'status': 'needs_reconciliation', 'pending': [], 'committed': committed,
                'unresolved_ids': missing, 'interrupted_attempts': [], 'exhausted_ids': []}
    if previous_owner and previous_owner.get('binding') != binding:
        raise ValueError('Owner marker has changed input/configuration')
    interrupted = [{'task_id': key, 'attempt': history[key][-1]['attempt'],
                    'status': 'interrupted_without_committed_result',
                    'recovery': 'identified_same_modal_input_restart'} for key in missing]
    exhausted = [key for key in missing if len(history[key]) >= max_attempts]
    pending = []
    for task in tasks:
        key = task['task_id']
        if key in committed or key in exhausted:
            continue
        pending.append({'task': task, 'attempt': len(history.get(key, [])) + 1,
                        'interrupted_prior_attempts': len(history.get(key, []))})
    return {'status': 'exhausted' if exhausted else 'ready', 'pending': pending,
            'committed': committed, 'unresolved_ids': missing, 'interrupted_attempts': interrupted,
            'exhausted_ids': exhausted}
