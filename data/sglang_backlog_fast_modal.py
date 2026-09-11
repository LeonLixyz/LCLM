"""Two-stage reviewed probes/continuation. Import/deploy/prepare never starts GPUs."""
from __future__ import annotations

import json
from pathlib import Path
import modal

from data.sglang_backlog import MODEL, REVISION, IMAGE, SOURCES, EXPECTED_PENDING, atomic_json, file_sha, sha, read_jsonl
from data.sglang_backlog_fast import generation_config, checked_rows

APP_NAME = 'lclm-sglang27-backlog-fast-20260909-v2'
ROOT = Path('/data/stage3-build-20260906/sglang27-backlog-fast-20260909-v2')
PARENT = Path('/data/stage3-build-20260906/sglang27-backlog-20260909-v1')
PILOT = Path('/data/stage3-build-20260906/sglang-27b-failed-1k-20260909-v1')
SMOKE = Path('/data/stage3-build-20260906/sglang27-tp1x8-smoke-20260909-v1')
app = modal.App(APP_NAME)
volume = modal.Volume.from_name('lclm-stage3-data')
cache = modal.Volume.from_name('lclm-hf-cache')
ignore = ['.git', '.venv', '__pycache__', '**/__pycache__/**', '**/*.pyc', '_modal_run']
cpu_image = (modal.Image.debian_slim(python_version='3.11').pip_install('pytest==8.4.2')
    .env({'PYTHONPATH': '/opt/lclm'}).add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))
gpu_image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/opt/lclm', 'HF_XET_HIGH_PERFORMANCE': '1',
          'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'})
    .add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))
cpu_options = dict(image=cpu_image, volumes={'/data': volume})


def parent_manifest():
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    if manifest['pending_rows'] != 273183 or manifest['pending_counts'] != EXPECTED_PENDING:
        raise ValueError('Incomplete/unexpected v1 preparation')
    if {s['source'] for s in manifest['sources']} != set(SOURCES):
        raise ValueError('Parent must preserve all15 sources, including zero-pending Watsonx')
    return manifest


def accepted_ids(source):
    ids = set()
    for spec in source['retained_prior_outputs']:
        if not spec['path'].endswith('.accepted.jsonl'):
            continue
        for row in checked_rows(spec):
            if row['verification']['accepted'] is not True or row['task_id'] in ids:
                raise ValueError('Original accepted checkpoint is inconsistent')
            ids.add(row['task_id'])
    return ids


@app.function(**cpu_options, cpu=2, memory=8192, timeout=1800)
def tests():
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '-q',
        'tests/test_sglang_backlog_fast.py', 'tests/test_sglang_backlog.py',
        'tests/test_sglang_throughput_smoke.py', 'tests/test_expansion_source_probes.py',
        'tests/test_expansion_retry_diagnostic.py', 'tests/test_harvest_expansion_trace.py'],
        cwd='/opt/lclm', capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('Fast backlog CPU tests failed')
    report = {'status': 'passed', 'output': result.stdout, 'generation_config': generation_config()}
    atomic_json(ROOT / 'tests.json', report)
    volume.commit()
    return report


@app.function(**cpu_options, cpu=4, memory=16384, timeout=14400, max_containers=4, retries=0)
def prepare_one(source_name):
    from data.sglang_backlog import pilot_selection
    from data.sglang_backlog_fast import prepare_source
    volume.reload()
    parent = parent_manifest()
    source, = [s for s in parent['sources'] if s['source'] == source_name]
    _, pilot_ids, pilot_sha = pilot_selection(PILOT)
    if pilot_sha != parent['pilot_manifest_sha256']:
        raise ValueError('Pilot selection changed since parent preparation')
    forbidden = set(pilot_ids) | accepted_ids(source)
    try:
        report = prepare_source(source, PARENT, ROOT / 'inputs', forbidden)
        volume.commit()
        return {key: report[key] for key in ('source', 'rows', 'probe_rows', 'probe_quotas')}
    except Exception:
        volume.commit()
        raise


@app.function(**cpu_options, cpu=4, memory=16384, timeout=14400, retries=0)
def finalize_preparation():
    from collections import Counter
    from data.sglang_backlog import pilot_selection
    volume.reload()
    parent = parent_manifest()
    test_report = json.loads((ROOT / 'tests.json').read_text())
    if test_report['status'] != 'passed' or test_report['generation_config'] != generation_config():
        raise ValueError('Tests do not cover this code/configuration')
    _, pilot_ids, pilot_sha = pilot_selection(PILOT)
    accepted = set()
    for source in parent['sources']:
        current = accepted_ids(source)
        if accepted & current:
            raise ValueError('Duplicate original accepted ID across sources')
        accepted.update(current)
    seen, counts, sources = set(), Counter(), []
    for source in parent['sources']:
        path = ROOT / 'inputs' / source['source'] / 'prepared.json'
        report = json.loads(path.read_text())
        if report['binding']['parent_source'] != source or report['binding']['generation_config'] != generation_config():
            raise ValueError('Prepared source binding changed')
        chunk_ids = {spec['index']: [] for spec in report['chunks']}
        for task in checked_rows(report['coverage']):
            key = task['task_id']
            if key in seen or key in accepted or key in pilot_ids or task['chunk'] not in chunk_ids:
                raise ValueError('Probe/continuation/accepted/pilot overlap or unknown chunk')
            seen.add(key)
            counts[task['kind']] += 1
            chunk_ids[task['chunk']].append(key)
        for spec in report['chunks']:
            ids = chunk_ids[spec['index']]
            if len(ids) != spec['rows'] or sha(json.dumps(sorted(ids)).encode()) != spec['task_ids_sha256']:
                raise ValueError('Coverage ledger disagrees with effective chunk IDs')
            if spec['kind'] == 'probe':
                list(checked_rows(spec))
        sources.append({key: report[key] for key in ('source', 'rows', 'counts', 'probe_rows',
            'probe_ids', 'probe_capacities', 'probe_quotas', 'probe_seed', 'chunks', 'coverage', 'retained_prior_outputs')})
        sources[-1]['report_sha256'] = file_sha(path)
        print(json.dumps({'verified_source': source['source'], 'total_distinct_pending': len(seen)}), flush=True)
    if len(seen) != 273183 or dict(counts) != EXPECTED_PENDING:
        raise ValueError('Exact273183-task accounting failed')
    for name, expected in [('maud-choices.json', parent['maud_ontology_sha256'])]:
        original = PARENT / 'inputs' / name
        if file_sha(original) != expected:
            raise ValueError('Frozen ontology changed')
    manifest = {'status': 'prepared_pending_source_probe_review', 'pending_rows': len(seen),
        'pending_counts': dict(counts), 'sources': sources, 'generation_config': generation_config(),
        'parent_manifest_sha256': file_sha(PARENT / 'manifest.json'), 'pilot_manifest_sha256': pilot_sha,
        'excluded_pilot_attempts': len(pilot_ids), 'original_accepted_ids': len(accepted),
        'original_accepted_ids_sha256': sha(json.dumps(sorted(accepted)).encode()),
        'maud_ontology_path': str(PARENT / 'inputs' / 'maud-choices.json'),
        'maud_ontology_sha256': parent['maud_ontology_sha256'], 'pubmed_split_sha256': parent['pubmed_split_sha256'],
        'probe_rows': sum(s['probe_rows'] for s in sources),
        'continuation_rows': sum(s['rows'] - s['probe_rows'] for s in sources),
        'approved_for_generation': False, 'approved_for_release': False,
        'selection': 'Same exact v1 pending IDs. Balanced first probes + disjoint references to remaining v1 bytes.',
        'prior_accepts_and_pilot_candidates': 'Preserved separately with original provenance; no release/export selected.'}
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Fast manifest changed')
    atomic_json(path, manifest)
    volume.commit()
    return {'manifest_sha256': file_sha(path), 'pending_rows': len(seen),
            'probe_rows': manifest['probe_rows'], 'continuation_rows': manifest['continuation_rows'],
            'pending_counts': dict(counts), 'output': str(ROOT)}


@app.function(**cpu_options, cpu=1, memory=2048, timeout=86400, max_containers=1, retries=0)
def prepare():
    tests.remote()
    for report in prepare_one.map(SOURCES, order_outputs=False):
        print(json.dumps(report), flush=True)
    return finalize_preparation.remote()


def load_gate(decision_sha256, stage):
    from data.sglang_backlog import pilot_selection, validate_scale_decision
    from data.sglang_backlog_fast import validate_serving_evidence, validate_source_decision
    path = ROOT / ('probe-decision.json' if stage == 'probe' else 'continuation-decision.json')
    if not path.exists() or file_sha(path) != decision_sha256:
        raise ValueError('Missing/changed root-reviewed decision')
    decision = json.loads(path.read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['generation_config'] != generation_config() or file_sha(PARENT / 'manifest.json') != manifest['parent_manifest_sha256']:
        raise ValueError('Frozen backlog code/configuration/parent changed')
    _, pilot_ids, pilot_sha = pilot_selection(PILOT)
    results = list(read_jsonl(PILOT / 'results.jsonl'))
    report = json.loads((PILOT / 'report.json').read_text())
    validate_scale_decision(decision, manifest, report, pilot_ids, results,
        backlog_sha=file_sha(ROOT / 'manifest.json'), pilot_manifest_sha=pilot_sha,
        pilot_results_sha=file_sha(PILOT / 'results.jsonl'), pilot_report_sha=file_sha(PILOT / 'report.json'))
    smoke = json.loads((SMOKE / 'report.json').read_text())
    smoke_manifest = json.loads((SMOKE / 'manifest.json').read_text())
    validate_serving_evidence(smoke, smoke_manifest, report_sha=file_sha(SMOKE / 'report.json'),
        manifest_sha=file_sha(SMOKE / 'manifest.json'), requested_report_sha=decision.get('serving_smoke_report_sha256'))
    base_sha, probes = None, {}
    if stage == 'continuation':
        base_sha = file_sha(ROOT / 'probe-decision.json')
        _, _, initially_allowed = load_gate(base_sha, 'probe')
        if not set(decision.get('source_reviews', {})) <= initially_allowed:
            raise ValueError('Continuation source was never authorized for a probe')
        for source in decision.get('source_reviews', {}):
            report_path = ROOT / 'outputs' / source / 'chunk-00000' / 'report.json'
            probes[source] = {'sha256': file_sha(report_path), 'report': json.loads(report_path.read_text())}
    allowed = validate_source_decision(decision, manifest, stage=stage, base_decision_sha=base_sha, probe_reports=probes)
    return manifest, decision, allowed


@app.function(**cpu_options, cpu=4, memory=16384, timeout=1800)
def check_gate(decision_sha256: str, stage: str):
    if stage not in ('probe', 'continuation'):
        raise ValueError('Unknown review stage')
    volume.reload()
    manifest, _, allowed = load_gate(decision_sha256, stage)
    return {'ready_for_generation': True, 'stage': stage, 'allowed_sources': sorted(allowed),
            'pending_rows': manifest['pending_rows'], 'approved_for_release': False}


def read_chunk_state(output, tasks, binding, identity):
    from data.sglang_backlog_fast import recover_attempts
    owner_path = output / 'owner.json'
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
    journal = list(read_jsonl(output / 'attempts.jsonl')) if (output / 'attempts.jsonl').exists() else []
    results = [json.loads(path.read_text()) for path in sorted((output / 'results').glob('*.json'))]
    for row in results:
        raw_path = output / 'attempts' / row['task_id'] / f"attempt-{row['attempt']:02d}" / 'raw.json'
        if file_sha(raw_path) != row['raw_responses_sha256']:
            raise ValueError('Committed raw response bytes changed')
    return recover_attempts(tasks, journal, results, previous_owner=owner, identity=identity, binding=binding)


def subgroup_counts(results, source):
    from data.sglang_backlog import summarize_results
    return {name: summarize_results([row for row in results if row.get('source_probe_stratum') == name])
            for name in source['probe_capacities']}


@app.cls(image=gpu_image, gpu='H200:8', cpu=32, memory=262144, timeout=21600,
         max_containers=1, scaledown_window=90, retries=0,
         volumes={'/data': volume, '/cache': cache}, secrets=[modal.Secret.from_name('huggingface')])
class Generator:
    @modal.enter()
    def enter(self):
        self.servers = None

    def start_servers(self):
        from data.sglang_27b_serving import Replicas
        if self.servers is not None:
            if not self.servers.healthy():
                raise RuntimeError('A serving replica exited; restart requires infrastructure review')
            return
        self.server_root = ROOT / 'server-runs' / modal.current_function_call_id() / modal.current_input_id()
        # A repeated input has a fresh container; preserve its earlier startup log.
        import uuid
        self.server_root = self.server_root / str(uuid.uuid4())
        self.servers = Replicas(self.server_root, volume.commit)
        self.servers.start()

    @modal.method()
    def chunk(self, source_name, index, decision_sha256, stage):
        import os
        import time
        from data.sglang_backlog import rollout_one, summarize_results, circuit_breaker
        from data.sglang_backlog_fast import continuous_queue, teacher_messages
        from data.expansion_task_normalization import prepare_teacher_task
        from data.expansion_source_probes import classify_source_prefix
        from data.expansion_retry_diagnostic import response_token_ids, ensure_complete_response, RawCaptureError
        from data.qwen38_pilot_sampling import sampling_kwargs
        volume.reload()
        manifest, _, allowed = load_gate(decision_sha256, stage)
        if source_name not in allowed:
            raise ValueError('Source is not approved for this generation stage')
        source, = [s for s in manifest['sources'] if s['source'] == source_name]
        if not 0 <= index < len(source['chunks']) or (stage == 'probe') != (index == 0):
            raise ValueError('Chunk is outside the reviewed stage')
        spec = source['chunks'][index]
        tasks = list(checked_rows(spec))  # Exhaust hashes/counts before any GPU request.
        if len(tasks) > 512:
            raise ValueError('Unbounded generation chunk')
        source_root = ROOT / 'outputs' / source_name
        output = source_root / f'chunk-{index:05d}'
        report_path = output / 'report.json'
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if (previous['input_sha256'] != spec['sha256'] or previous['rows'] != spec['rows']
                    or previous['backlog_manifest_sha256'] != file_sha(ROOT / 'manifest.json')):
                raise ValueError('Saved report belongs to changed chunk')
            if previous['status'] in ('complete', 'held', 'needs_reconciliation', 'attempts_exhausted'):
                return previous
        if (source_root / 'HOLD.json').exists():
            raise ValueError('Source remains held for review')
        if index:
            prior = source_root / f'chunk-{index-1:05d}' / 'report.json'
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                raise ValueError('Previous source chunk is incomplete or held')
        binding = {'source': source_name, 'chunk': index, 'input_sha256': spec['sha256'],
            'effective_task_ids_sha256': spec['task_ids_sha256'], 'decision_sha256': decision_sha256,
            'backlog_manifest_sha256': file_sha(ROOT / 'manifest.json'), 'stage': stage}
        identity = {'function_call_id': modal.current_function_call_id(), 'input_id': modal.current_input_id()}
        output.mkdir(parents=True, exist_ok=True)
        result_root = output / 'results'
        result_root.mkdir(exist_ok=True)
        state = read_chunk_state(output, tasks, binding, identity)
        results = list(state['committed'].values())
        begin, last_commit = time.monotonic(), time.monotonic()
        stop_reason = circuit_breaker(results)
        for interrupted in state['interrupted_attempts']:
            path = output / 'attempts' / interrupted['task_id'] / f"attempt-{interrupted['attempt']:02d}" / 'interrupted.json'
            atomic_json(path, {**interrupted, 'identity': identity, 'binding': binding})
        if state['status'] in ('needs_reconciliation', 'exhausted'):
            status = 'attempts_exhausted' if state['status'] == 'exhausted' else 'needs_reconciliation'
            report = {**binding, 'status': status, 'rows': len(tasks), 'completed': len(results),
                'unattempted': len(tasks) - len(results) - len(state['unresolved_ids']),
                'unresolved_attempt_ids': state['unresolved_ids'], 'exhausted_ids': state['exhausted_ids'],
                'counts': summarize_results(results), 'subgroups': subgroup_counts(results, source),
                'approved_for_release': False}
            atomic_json(report_path, report)
            atomic_json(source_root / 'HOLD.json', report)
            volume.commit()
            return report
        atomic_json(output / 'owner.json', {'identity': identity, 'binding': binding, 'finished': False,
            'started_at': time.time(), 'recovering_interrupted_attempts': state['interrupted_attempts']})
        volume.commit()
        attempts = {item['task']['task_id']: item for item in state['pending']}
        if attempts and not stop_reason:
            self.start_servers()
        choices_path = Path(manifest['maud_ontology_path'])
        if file_sha(choices_path) != manifest['maud_ontology_sha256']:
            raise ValueError('MAUD ontology changed')
        choices = json.loads(choices_path.read_text()) if source_name == 'maud' else {}
        normalized = {key: prepare_teacher_task(item['task'], choices) for key, item in attempts.items()}
        provenance = {**manifest['generation_config'], **binding,
            'server_run': str(getattr(self, 'server_root', '')), 'judge_model': MODEL, 'judge_revision': REVISION,
            'judge_role': 'same-model heuristic; independent source review required for continuation'}
        def reserve(batch):
            with (output / 'attempts.jsonl').open('a') as handle:
                for task in batch:
                    key = task['task_id']
                    handle.write(json.dumps({'task_id': key, 'attempt': attempts[key]['attempt'],
                        'task_sha256': sha(json.dumps(task, sort_keys=True).encode()),
                        'identity': identity, 'binding': binding, 'reserved_at': time.time()}) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            volume.commit()
        def run(raw_task, replica):
            key, records = raw_task['task_id'], []
            task, attempt = normalized[key], attempts[key]['attempt']
            def request(messages, tools, phase):
                kwargs = {'model': MODEL, 'messages': teacher_messages(messages, source_name, phase),
                          **sampling_kwargs('recommended')}
                kwargs['extra_body']['return_token_ids'] = True
                if tools:
                    kwargs.update(tools=list(tools), tool_choice='auto')
                entry = {'phase': phase, 'replica': replica, 'request': kwargs, 'started_at': time.time()}
                records.append(entry)
                start = time.monotonic()
                try:
                    response = self.servers.clients[replica].chat.completions.create(**kwargs)
                    payload = response.model_dump(mode='json')
                    entry['response'] = payload
                    ids = response_token_ids(payload['choices'][0])
                    try:
                        entry['raw_generated_text'] = self.servers.tokenizer.decode(ids, skip_special_tokens=False)
                    except Exception as exc:
                        raise RawCaptureError('Cannot decode raw model output') from exc
                    ensure_complete_response(response.choices[0].finish_reason, phase)
                    return payload['choices'][0]['message']
                except Exception as exc:
                    entry['error'] = {'type': type(exc).__name__, 'message': str(exc),
                        'status_code': getattr(exc, 'status_code', None), 'body': getattr(exc, 'body', None),
                        'request_id': getattr(exc, 'request_id', None)}
                    raise
                finally:
                    entry['seconds'] = time.monotonic() - start
                    entry['finished_at'] = time.time()
            result = rollout_one(task, source_name, request, provenance)
            attempt_root = output / 'attempts' / key / f'attempt-{attempt:02d}'
            atomic_json(attempt_root / 'raw.json', records)
            result.update(attempt=attempt, task_sha256=sha(json.dumps(raw_task, sort_keys=True).encode()),
                raw_responses_sha256=file_sha(attempt_root / 'raw.json'), replica=replica,
                interrupted_prior_attempts=attempts[key]['interrupted_prior_attempts'],
                source_probe_stratum=classify_source_prefix(raw_task, source=source_name)
                    if source_name in ('lex_glue', 'synthetic') else raw_task['_attempt_provenance']['kind'])
            result['token_usage'] = {name: sum((r.get('response', {}).get('usage') or {}).get(name, 0) or 0
                for r in records) for name in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            return result
        def save(result):
            nonlocal last_commit
            path = result_root / (result['task_id'] + '.json')
            if path.exists():
                raise ValueError('Refusing to overwrite a committed final task result')
            atomic_json(path, result)
            results.append(result)
            if len(results) % 16 == 0 or time.monotonic() - last_commit >= 30:
                progress = {**binding, 'rows': len(tasks), 'counts': summarize_results(results),
                    'subgroups': subgroup_counts(results, source), 'seconds_this_call': time.monotonic() - begin}
                atomic_json(output / 'progress.json', progress)
                volume.commit()
                last_commit = time.monotonic()
                print(json.dumps({'source': source_name, 'chunk': index, 'counts': progress['counts']}), flush=True)
        queue = continuous_queue([item['task'] for item in state['pending']], run=run, reserve=reserve,
            save=save, stop=lambda _: circuit_breaker(results)) if not stop_reason else {'stopped': True}
        stop_reason = circuit_breaker(results)
        # A normal return is not an infrastructure restart; later missing results require review.
        atomic_json(output / 'owner.json', {'identity': identity, 'binding': binding, 'finished': True,
            'finished_at': time.time(), 'queue': queue})
        final_state = read_chunk_state(output, tasks, binding, identity)
        unresolved = final_state['unresolved_ids']
        status = 'held' if stop_reason else 'needs_reconciliation' if unresolved else 'complete'
        if not unresolved and len(results) != len(tasks):
            status = 'held' if stop_reason else 'needs_reconciliation'
        report = {**binding, 'status': status, 'rows': len(tasks), 'completed': len(results),
            'unattempted': len(tasks) - len(results) - len(unresolved), 'unresolved_attempt_ids': unresolved,
            'counts': summarize_results(results), 'subgroups': subgroup_counts(results, source),
            'stop_reason': stop_reason, 'queue': queue, 'seconds_this_call': time.monotonic() - begin,
            'approved_for_release': False,
            'result_hashes': {r['task_id']: file_sha(result_root / (r['task_id'] + '.json')) for r in results}}
        atomic_json(report_path, report)
        if status != 'complete':
            atomic_json(source_root / 'HOLD.json', report)
        volume.commit()
        return report

    @modal.exit()
    def stop(self):
        if self.servers is not None:
            self.servers.stop()
        volume.commit()


def coordinate(decision_sha256, stage):
    """Drive finite reviewed chunks; durable child IDs survive CPU preemption."""
    import time
    volume.reload()
    manifest, _, allowed = load_gate(decision_sha256, stage)
    identity = {'function_call_id': modal.current_function_call_id(), 'input_id': modal.current_input_id()}
    path = ROOT / 'coordinators' / (stage + '-' + decision_sha256) / 'state.json'
    plan = [(s['source'], c['index']) for index in range(max(len(s['chunks']) for s in manifest['sources']))
        for s in manifest['sources'] if s['source'] in allowed
        for c in s['chunks'] if c['index'] == index and (stage == 'probe') == (index == 0)]
    binding = {'decision_sha256': decision_sha256, 'stage': stage,
        'manifest_sha256': file_sha(ROOT / 'manifest.json'), 'plan_sha256': sha(json.dumps(plan).encode())}
    state = json.loads(path.read_text()) if path.exists() else {'binding': binding, 'entries': {}, 'status': 'running'}
    if state['binding'] != binding:
        raise ValueError('Coordinator plan/review binding changed')
    state.update(status='running', current_identity=identity)
    atomic_json(path, state)
    volume.commit()
    generator, started = Generator(), time.monotonic()
    def persist():
        state['updated_at'] = time.time()
        atomic_json(path, state)
        volume.commit()
    def hold(source, index, detail):
        item = {'source': source, 'chunk': index, 'status': 'dispatch_needs_reconciliation',
                'detail': detail, 'coordinator': str(path), 'approved_for_release': False}
        atomic_json(ROOT / 'outputs' / source / 'HOLD.json', item)
        print(json.dumps(item), flush=True)
    for source, index in plan:
        volume.reload()
        source_root = ROOT / 'outputs' / source
        report_path = source_root / f'chunk-{index:05d}' / 'report.json'
        key = source + ':' + str(index)
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report['status'] in ('complete', 'held', 'needs_reconciliation', 'attempts_exhausted'):
                state['entries'][key] = {**state['entries'].get(key, {}), 'status': report['status'],
                                         'report_sha256': file_sha(report_path)}
                continue
        if (source_root / 'HOLD.json').exists():
            continue
        if index:
            prior = source_root / f'chunk-{index-1:05d}' / 'report.json'
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                continue
        entry = state['entries'].get(key)
        if not entry and time.monotonic() - started >= 18 * 3600:
            # Leave six hours for the current bounded GPU call under the 24h CPU cap.
            state['status'] = 'continuation_required'
            persist()
            return {'status': state['status'], 'state_path': str(path), 'binding': binding}
        if entry and not entry.get('child_call_id'):
            # A preemption in the spawn/commit window must not create duplicate children.
            owner_path = source_root / f'chunk-{index:05d}' / 'owner.json'
            child_id = None
            if owner_path.exists():
                owner = json.loads(owner_path.read_text())
                if (owner['binding']['source'] == source and owner['binding']['chunk'] == index
                        and owner['binding']['decision_sha256'] == decision_sha256):
                    child_id = owner['identity']['function_call_id']
            if child_id is None and entry.get('coordinator_identity') == identity:
                known = {r.get('child_call_id') for r in state['entries'].values()}
                def walk(nodes):
                    for node in nodes:
                        yield node
                        yield from walk(node.children)
                candidates = {node.function_call_id for node in walk(
                    modal.FunctionCall.from_id(identity['function_call_id']).get_call_graph())
                    if node.function_name.endswith('chunk') and node.function_call_id not in known}
                if len(candidates) == 1:
                    child_id = candidates.pop()
            if child_id is None:
                hold(source, index, 'Interrupted dispatch has no unambiguous durable child call ID')
                entry['status'] = 'dispatch_needs_reconciliation'
                persist()
                continue
            entry['child_call_id'] = child_id
            persist()
        if not entry:
            entry = {'status': 'dispatching', 'source': source, 'chunk': index,
                     'coordinator_identity': identity, 'started_at': time.time()}
            state['entries'][key] = entry
            persist()  # Durable intent before the side effect.
            try:
                child = generator.chunk.spawn(source, index, decision_sha256, stage)
                entry['child_call_id'] = child.object_id
                entry['status'] = 'running'
                persist()
            except Exception as exc:
                hold(source, index, f'Dispatch outcome is unclassified: {type(exc).__name__}: {exc}')
                entry['status'] = 'dispatch_needs_reconciliation'
                persist()
                continue
        child = modal.FunctionCall.from_id(entry['child_call_id'])
        last_progress = None
        while True:
            try:
                report = child.get(timeout=60)
                entry.update(status=report['status'], completed=report['completed'], counts=report['counts'])
                volume.reload()
                entry['report_sha256'] = file_sha(report_path)
                persist()
                print(json.dumps({'source': source, 'chunk': index, 'status': report['status'],
                                  'counts': report['counts']}), flush=True)
                break
            except TimeoutError:
                volume.reload()
                progress_path = source_root / f'chunk-{index:05d}' / 'progress.json'
                if progress_path.exists():
                    progress = json.loads(progress_path.read_text()).get('counts')
                    if progress != last_progress:
                        print(json.dumps({'source': source, 'chunk': index, 'progress': progress}), flush=True)
                        last_progress = progress
            except Exception as exc:
                hold(source, index, f'Child failed without a terminal report: {type(exc).__name__}: {exc}')
                entry.update(status='dispatch_needs_reconciliation', error={'type': type(exc).__name__, 'message': str(exc)})
                persist()
                break
    state['status'] = 'reviewed_plan_drained'
    persist()
    return {'status': state['status'], 'state_path': str(path), 'binding': binding,
            'entries': state['entries'], 'approved_for_release': False}


@app.function(**cpu_options, cpu=4, memory=16384, timeout=86400, max_containers=1, retries=0)
def dispatch_source_probes(decision_sha256: str):
    return coordinate(decision_sha256, 'probe')


@app.function(**cpu_options, cpu=4, memory=16384, timeout=86400, max_containers=1, retries=0)
def continue_approved_sources(decision_sha256: str):
    return coordinate(decision_sha256, 'continuation')


@app.function(**cpu_options, cpu=2, memory=8192, timeout=1800)
def status():
    from data.sglang_backlog import summarize_results
    volume.reload()
    if not (ROOT / 'manifest.json').exists():
        return {'prepared': False, 'output': str(ROOT)}
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    output = {'prepared': True, 'pending_rows': manifest['pending_rows'], 'sources': {},
              'probe_decision_exists': (ROOT / 'probe-decision.json').exists(),
              'continuation_decision_exists': (ROOT / 'continuation-decision.json').exists()}
    for source in manifest['sources']:
        rows, chunks, unresolved = [], [], set()
        for spec in source['chunks']:
            directory = ROOT / 'outputs' / source['source'] / f"chunk-{spec['index']:05d}"
            results = [json.loads(path.read_text()) for path in (directory / 'results').glob('*.json')]
            journal = list(read_jsonl(directory / 'attempts.jsonl')) if (directory / 'attempts.jsonl').exists() else []
            missing = {r['task_id'] for r in journal} - {r['task_id'] for r in results}
            unresolved.update(missing)
            rows.extend(results)
            report_path = directory / 'report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if journal or report or results:
                chunks.append({'chunk': spec['index'], 'status': report.get('status', 'active_or_interrupted'),
                    'completed': len(results), 'unresolved_or_inflight': len(missing), 'attempt_records': len(journal)})
        if len({r['task_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate final result across source chunks')
        output['sources'][source['source']] = {'rows': source['rows'], 'counts': summarize_results(rows),
            'unresolved_or_inflight': len(unresolved), 'unattempted': source['rows'] - len(rows) - len(unresolved),
            'held': (ROOT / 'outputs' / source['source'] / 'HOLD.json').exists(), 'chunks': chunks,
            'subgroups': subgroup_counts(rows, source)}
    return output
