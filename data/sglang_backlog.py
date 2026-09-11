"""Frozen backlog selection and one-attempt, checkpointed expansion generation."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from data.prepare_expansion_retry import SOURCES, validate_prepared

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
IMAGE = 'lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376'
CHUNK_SIZE = 512
INITIAL_CHUNK_SIZE = 64
CONCURRENCY = 16
MAX_REQUESTS = 64
EXPECTED_ORIGINAL = {'retry_rejected': 67088, 'previously_unattempted': 207095}
EXPECTED_PENDING = {'retry_rejected': 66088, 'previously_unattempted': 207095}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete JSONL line: {path}')
            yield json.loads(line)


def generation_config():
    from data.qwen38_pilot_sampling import sampling_kwargs
    from data.synthetic_expansion_agent import TEACHER_SYSTEM_PROMPT, EXPAND_TOOL
    from data.expansion_semantic_review import REVIEW_VERSION, CLAIM_REVIEW_VERSION
    dependencies = ('sglang_backlog.py', 'sglang_backlog_modal.py',
        'synthetic_expansion_agent.py', 'real_expansion_agent.py',
        'expansion_task_normalization.py', 'harvest_expansion_trace.py',
        'clean_agent_trajectories.py', 'full_expansion_rollouts.py',
        'expansion_semantic_review.py', 'expansion_retry_diagnostic.py',
        'qwen38_pilot_sampling.py', 'pubmedqa_split.py')
    return {'model': MODEL, 'model_revision': REVISION, 'image': IMAGE,
        'sampling': sampling_kwargs('recommended'), 'context_length': 32768,
        'max_tool_calls': 16, 'max_requests_per_task': MAX_REQUESTS,
        'concurrency': CONCURRENCY, 'chunk_size': CHUNK_SIZE,
        'initial_chunk_size': INITIAL_CHUNK_SIZE,
        'teacher_prompt_sha256': sha(TEACHER_SYSTEM_PROMPT.encode()),
        'tool_schema_sha256': sha(json.dumps(EXPAND_TOOL, sort_keys=True).encode()),
        'semantic_review': REVIEW_VERSION, 'summary_claim_review': CLAIM_REVIEW_VERSION,
        'code_sha256': {name: file_sha(Path(__file__).parent / name) for name in dependencies},
        'training_harvest': 'native-calls-and-explicit-final-v1',
        'approved_for_release': False}


def pilot_selection(pilot_root):
    root = Path(pilot_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    ids = {}
    for source in manifest['sources']:
        path = root / (source['source'] + '.inputs.json')
        if file_sha(path) != source['inputs_sha256']:
            raise ValueError('Pilot frozen input hash changed')
        rows = json.loads(path.read_text())
        if len(rows) != source['count']:
            raise ValueError('Pilot selection count mismatch')
        for row in rows:
            task_id = row['task']['task_id']
            if task_id in ids or row['baseline']['verification']['accepted'] is not False:
                raise ValueError('Pilot contains duplicate/non-failed task')
            ids[task_id] = row['source']
    if len(ids) != 1000 or manifest['rows'] != 1000:
        raise ValueError('Expected exactly 1000 reserved pilot attempts')
    return manifest, ids, file_sha(root / 'manifest.json')


def select_pending(rows, excluded_ids):
    """Yield untouched original tasks, excluding all pilot IDs regardless of outcome."""
    seen = set()
    for row in rows:
        task_id = row['task_id']
        if not isinstance(task_id, str) or task_id in seen:
            raise ValueError('Duplicate/invalid prepared task ID')
        seen.add(task_id)
        kind = row.get('_attempt_provenance', {}).get('kind')
        if kind not in EXPECTED_ORIGINAL:
            raise ValueError('Prepared task is neither failed nor unattempted')
        if task_id in excluded_ids:
            if kind != 'retry_rejected':
                raise ValueError('Pilot task was not a prior failure')
            continue
        yield row
    if not set(excluded_ids) <= seen:
        raise ValueError('Pilot IDs absent from prepared source')


def prepare_source(source, prepared_root, pilot_root, output_root, split_path):
    if source not in SOURCES:
        raise ValueError('Unknown expansion source')
    prepared, destination = Path(prepared_root), Path(output_root) / source
    _, pilot_ids, pilot_sha = pilot_selection(pilot_root)
    excluded = {key for key, value in pilot_ids.items() if value == source}
    parent_path = prepared / (source + '.build.json')
    parent = json.loads(parent_path.read_text())
    retry_manifest = json.loads((prepared / 'retry-manifest.json').read_text())
    if [r for r in retry_manifest['sources'] if r['source'] == source] != [parent]:
        raise ValueError('Prepared source report differs from retry manifest')
    validate_prepared(parent, prepared)
    binding = {'source': source, 'parent_report_sha256': file_sha(parent_path),
        'parent_retry_manifest_sha256': file_sha(prepared / 'retry-manifest.json'),
        'pilot_manifest_sha256': pilot_sha, 'generation_config': generation_config()}
    report_path = destination / 'prepared.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['binding'] != binding:
            raise ValueError('Existing backlog preparation has changed inputs/settings')
        for chunk in report['chunks']:
            if file_sha(destination / chunk['name']) != chunk['sha256']:
                raise ValueError('Existing backlog chunk changed')
        return report
    if destination.exists():
        raise ValueError('Partial backlog preparation exists; inspect before recovery')
    destination.mkdir(parents=True)
    allowed = None
    if source == 'pubmedqa_labeled':
        from data.pubmedqa_split import training_ids
        allowed = training_ids(json.loads(Path(split_path).read_text()))
    counts = Counter()
    chunks, batch = [], []
    def save_chunk():
        if not batch:
            return
        name = f'chunk-{len(chunks):05d}.tasks.jsonl'
        raw = b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode() for row in batch)
        (destination / name).write_bytes(raw)
        chunks.append({'name': name, 'index': len(chunks), 'rows': len(batch), 'sha256': sha(raw)})
        batch.clear()
    task_path = prepared / (source + '.tasks.jsonl')
    for row in select_pending(read_jsonl(task_path), excluded):
        if allowed is not None and row['source_row_id'] not in allowed:
            raise ValueError('PubMedQA non-training task')
        if source == 'multidoc2dial' and not row['_attempt_provenance'].get('corrected_multidoc2dial'):
            raise ValueError('Uncorrected MultiDoc2Dial source')
        counts[row['_attempt_provenance']['kind']] += 1
        batch.append(row)
        if len(batch) == (INITIAL_CHUNK_SIZE if not chunks else CHUNK_SIZE):
            save_chunk()
    save_chunk()
    expected = {key: parent['counts'].get(key, 0) - (len(excluded) if key == 'retry_rejected' else 0)
                for key in EXPECTED_ORIGINAL}
    if any(counts[key] != value for key, value in expected.items()):
        raise ValueError('Backlog selection counts disagree with frozen source')
    report = {'binding': binding, 'counts': dict(counts), 'rows': sum(counts.values()),
        'excluded_pilot_ids': sorted(excluded), 'chunks': chunks,
        'original_source_report': parent,
        'retained_prior_outputs': parent['parent_audit']['files'], 'approved_for_release': False}
    atomic_json(report_path, report)
    return report


def validate_scale_decision(decision, backlog_manifest, pilot_report, pilot_ids, results,
                            *, backlog_sha, pilot_manifest_sha, pilot_results_sha, pilot_report_sha):
    """Require an explicit evidence review; an acceptance count alone cannot unlock GPUs."""
    if (decision.get('decision') != 'generate_failed_and_unattempted_with_27b'
            or decision.get('reviewed_by') != 'root'
            or decision.get('approved_for_generation') is not True
            or decision.get('approved_for_release') is not False):
        raise ValueError('Missing root-reviewed generation-only scale decision')
    bindings = {'backlog_manifest_sha256': backlog_sha,
        'pilot_manifest_sha256': pilot_manifest_sha, 'pilot_results_sha256': pilot_results_sha,
        'pilot_report_sha256': pilot_report_sha}
    if any(decision.get(k) != v for k, v in bindings.items()):
        raise ValueError('Scale decision does not match frozen pilot/backlog artifacts')
    if (pilot_report.get('completed') != 1000
            or pilot_report.get('status') != 'complete_diagnostic_pending_review'
            or pilot_report.get('manifest_sha256') != pilot_manifest_sha
            or pilot_report.get('results_sha256') != pilot_results_sha):
        raise ValueError('Pilot is incomplete or report hashes disagree')
    if len(results) != 1000 or {r['task_id'] for r in results} != set(pilot_ids):
        raise ValueError('Pilot results do not cover exactly the 1000 reserved attempts')
    if backlog_manifest['pending_counts'] != EXPECTED_PENDING:
        raise ValueError('Unexpected pending task coverage')
    review = decision.get('evidence_review', {})
    for key in ('recovery', 'speed', 'quality', 'parser_failures', 'limitations'):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError('Scale decision lacks substantive ' + key + ' review')
    if not review.get('reviewed_task_ids') or not set(review['reviewed_task_ids']) <= set(pilot_ids):
        raise ValueError('Scale decision lacks reviewed pilot examples')
    if decision.get('generation_config') != backlog_manifest['generation_config']:
        raise ValueError('Scale decision has changed generation settings')


def attempt_state(tasks, started_rows, results):
    expected = {r['task_id']: sha(json.dumps(r, sort_keys=True).encode()) for r in tasks}
    if len(expected) != len(tasks):
        raise ValueError('Duplicate chunk task IDs')
    started = {}
    for row in started_rows:
        key = row['task_id']
        if key in started or expected.get(key) != row['task_sha256']:
            raise ValueError('Started journal contains duplicate or changed task')
        started[key] = row
    completed = {}
    for row in results:
        key = row['task_id']
        if key in completed or key not in started:
            raise ValueError('Result is duplicate or has no started journal record')
        completed[key] = row
    return {'pending': [r for r in tasks if r['task_id'] not in started],
            'unresolved': sorted(set(started) - set(completed)), 'completed': completed}


def summarize_results(results):
    counts = Counter()
    for row in results:
        counts['completed'] += 1
        counts['automatic_accepted'] += row['verification']['accepted'] is True
        counts['outcome:' + row['verification']['reason']] += 1
        counts['tasks_with_expansion'] += bool(row.get('tool_call_count'))
        if row.get('error'):
            counts['error_phase:' + row['error']['phase']] += 1
            counts['error_kind:' + row['error']['kind']] += 1
    return dict(counts)


def circuit_breaker(results):
    counts = summarize_results(results)
    n = counts.get('completed', 0)
    if n >= INITIAL_CHUNK_SIZE and not counts.get('automatic_accepted'):
        return 'zero_automatic_acceptance_after_64'
    infra = counts.get('error_kind:infrastructure', 0) + counts.get('error_kind:raw_capture', 0)
    if n >= 16 and infra / n >= 0.25:
        return 'infrastructure_or_capture_error_rate_at_least_25_percent'
    return None


def rollout_one(task, source, complete_request, provenance):
    """Same production checks, with stage/error provenance and no accepted export."""
    from data.synthetic_expansion_agent import run_agent_rollout, verify_trace, GENERATOR_VERSION
    from data.real_expansion_agent import verify_real_trace
    from data.clean_agent_trajectories import strip_cot
    from data.harvest_expansion_trace import harvest_training_messages
    from data.expansion_semantic_review import apply_semantic_review
    from data.expansion_retry_diagnostic import RawCaptureError, TruncatedResponseError, IncompleteResponseError
    phase, trace, requests = 'rollout', None, 0
    started = time.monotonic()
    result = {'task_id': task['task_id'], 'source': source,
        'source_dataset': task.get('source_dataset') or GENERATOR_VERSION,
        'source_row_id': task.get('source_row_id', str(task.get('index'))),
        'attempt_provenance': task['_attempt_provenance'], 'teacher': provenance,
        'approved_for_release': False}
    def request(messages, tools=None):
        nonlocal requests
        if requests >= MAX_REQUESTS:
            raise RequestBudgetError('Per-task request budget exhausted')
        requests += 1
        return complete_request(messages, tools, phase)
    def complete(messages, tools):
        message = request(messages, tools)
        return {'content': strip_cot(message.get('content')),
                'tool_calls': message.get('tool_calls') or []}
    try:
        trace = run_agent_rollout(task, complete, max_tool_calls=16)
        result['tool_call_count'] = trace['tool_call_count']
        if not trace['rollout_failure_reason']:
            phase = 'harvest'
            trace['messages'], trace['harvesting'] = harvest_training_messages(trace['messages'])
            phase = 'rule_verification'
            trace['verification'] = (verify_trace(task, trace['messages']).as_dict() if source == 'synthetic'
                                     else verify_real_trace(task, trace['messages']))
            result['rule_verification'] = trace['verification']
            phase = 'judge'
            if source != 'synthetic':
                trace['verification'] = apply_semantic_review(source, trace, lambda m: request(m).get('content'))
        result['verification'] = trace['verification']
    except Exception as exc:
        kind = ('truncation' if isinstance(exc, TruncatedResponseError) else
                'raw_capture' if isinstance(exc, RawCaptureError) else
                'incomplete_response' if isinstance(exc, IncompleteResponseError) else
                'request_budget' if isinstance(exc, RequestBudgetError) else
                'judge_output' if phase == 'judge' and isinstance(exc, (ValueError, TypeError, AttributeError)) else
                'format' if phase == 'harvest' else 'infrastructure' if phase in ('rollout', 'judge') else 'verification')
        result['error'] = {'phase': phase, 'kind': kind, 'type': type(exc).__name__, 'message': str(exc)}
        result['verification'] = {'accepted': False, 'reason': f'generation_error:{phase}:{kind}:{type(exc).__name__}'}
    if trace is not None:
        trace.update(model=MODEL, model_revision=REVISION, source_dataset=result['source_dataset'],
            source_row_id=result['source_row_id'], verification=result['verification'],
            approved_for_release=False, attempt_provenance=result['attempt_provenance'], generation=provenance)
        result['trace'] = trace
    result.update(requests=requests, seconds=time.monotonic() - started)
    return result


class RequestBudgetError(RuntimeError):
    pass
