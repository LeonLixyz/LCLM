"""Prepared-only by default. GPU dispatch requires a separately reviewed scale decision.

Deploy an immutable snapshot; spawn tests/prepare on this app. The root agent may
later call dispatch(decision_sha256, max_chunks), after reviewing completed pilot
evidence and writing scale-decision.json. This module never writes that decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from data.sglang_backlog import (
    MODEL, REVISION, IMAGE, SOURCES, CONCURRENCY, EXPECTED_ORIGINAL, EXPECTED_PENDING,
    sha, file_sha, atomic_json, generation_config, pilot_selection,
)

APP_NAME = 'lclm-sglang27-backlog-20260909-v1'
ROOT = Path('/data/stage3-build-20260906/sglang27-backlog-20260909-v1')
PREPARED = Path('/data/stage3-agent/real-expansion/pilots/retry-and-remaining-qwen38-27b-v2-tasks')
PILOT = Path('/data/stage3-build-20260906/sglang-27b-failed-1k-20260909-v1')
SPLIT = Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json')
app = modal.App(APP_NAME)
volume = modal.Volume.from_name('lclm-stage3-data')
cache = modal.Volume.from_name('lclm-hf-cache')
ignore = ['.git', '.venv', '__pycache__', '**/__pycache__/**', '**/*.pyc', '_modal_run']
cpu_image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('pytest==8.4.2', 'datasets==3.6.0', 'transformers==4.57.1', 'jinja2>=3.1')
    .env({'PYTHONPATH': '/opt/lclm'})
    .add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))
gpu_image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/opt/lclm',
          'HF_XET_HIGH_PERFORMANCE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))
cpu_options = dict(image=cpu_image, volumes={'/data': volume})


@app.function(**cpu_options, cpu=2, memory=8192, timeout=1800)
def tests():
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '-q',
        'tests/test_sglang_backlog.py', 'tests/test_expansion_retry_diagnostic.py',
        'tests/test_expansion_retry.py', 'tests/test_harvest_expansion_trace.py',
        'tests/test_expansion_semantic_review.py', 'tests/test_teacher_transition.py'],
        cwd='/opt/lclm', text=True, capture_output=True)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('Backlog CPU test suite failed')
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {'status': 'passed', 'output': result.stdout, 'generation_config': generation_config()}
    atomic_json(ROOT / 'tests.json', report)
    volume.commit()
    return report


@app.function(**cpu_options, cpu=4, memory=16384, timeout=14400, max_containers=4)
def prepare_one(source: str):
    from data.sglang_backlog import prepare_source
    volume.reload()
    try:
        report = prepare_source(source, PREPARED, PILOT, ROOT / 'inputs', SPLIT)
        volume.commit()
        return {'source': source, 'rows': report['rows'], 'counts': report['counts'],
                'excluded_pilot': len(report['excluded_pilot_ids']), 'chunks': len(report['chunks'])}
    except Exception:
        volume.commit()
        raise


@app.function(**cpu_options, cpu=4, memory=16384, timeout=14400)
def finalize_preparation():
    from collections import Counter
    from data.sglang_backlog import read_jsonl
    volume.reload()
    test_report = json.loads((ROOT / 'tests.json').read_text())
    if test_report['status'] != 'passed' or test_report['generation_config'] != generation_config():
        raise ValueError('Tests do not cover current backlog code/configuration')
    _, pilot_ids, pilot_sha = pilot_selection(PILOT)
    sources, counts, original, seen = [], Counter(), Counter(), set()
    for source in SOURCES:
        path = ROOT / 'inputs' / source / 'prepared.json'
        report = json.loads(path.read_text())
        if report['binding']['pilot_manifest_sha256'] != pilot_sha or report['binding']['generation_config'] != generation_config():
            raise ValueError('Source preparation has changed pilot/configuration')
        counts.update(report['counts'])
        original.update({k: report['original_source_report']['counts'].get(k, 0) for k in EXPECTED_ORIGINAL})
        for chunk in report['chunks']:
            chunk_path = path.parent / chunk['name']
            if file_sha(chunk_path) != chunk['sha256']:
                raise ValueError('Prepared chunk bytes changed')
            rows = 0
            for row in read_jsonl(chunk_path):
                identifier = row['task_id']
                if identifier in seen or identifier in pilot_ids:
                    raise ValueError('Duplicate task or pilot overlap across source chunks')
                seen.add(identifier)
                rows += 1
            if rows != chunk['rows']:
                raise ValueError('Prepared chunk row count changed')
        sources.append({'source': source, 'rows': report['rows'], 'counts': report['counts'],
            'report_sha256': file_sha(path), 'chunks': report['chunks'],
            'retained_prior_outputs': report['retained_prior_outputs']})
    if dict(original) != EXPECTED_ORIGINAL or dict(counts) != EXPECTED_PENDING or len(seen) != 273183:
        raise ValueError(f'Unexpected backlog accounting: {dict(original)}, {dict(counts)}, {len(seen)}')
    ontology_source = PREPARED / 'maud-choices.json'
    ontology = ROOT / 'inputs' / 'maud-choices.json'
    raw = ontology_source.read_bytes()
    if ontology.exists() and ontology.read_bytes() != raw:
        raise ValueError('Frozen MAUD ontology changed')
    ontology.write_bytes(raw)
    manifest = {'status': 'prepared_pending_pilot_review', 'pending_rows': len(seen),
        'pending_counts': dict(counts), 'original_counts': dict(original), 'excluded_pilot_attempts': 1000,
        'sources': sources, 'generation_config': generation_config(),
        'parent_retry_manifest_sha256': file_sha(PREPARED / 'retry-manifest.json'),
        'pilot_root': str(PILOT), 'pilot_manifest_sha256': pilot_sha,
        'maud_ontology_sha256': sha(raw), 'pubmed_split_sha256': file_sha(SPLIT),
        'selection': 'Frozen prior rejects and unattempted tasks; every reserved pilot ID excluded, including rejected/failed pilot outcomes.',
        'prior_accepts': 'Preserved at original235B output paths listed per source; not merged or relabeled.',
        'pilot_candidates': 'Preserved separately in pilot results; never automatically promoted for training.',
        'unknown_attempts': 'Started without persisted result requires explicit reconciliation; neither automatically retried nor counted completed.',
        'approved_for_generation': False, 'approved_for_release': False}
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing backlog manifest has changed')
    atomic_json(path, manifest)
    volume.commit()
    return {'manifest_sha256': file_sha(path), 'pending_rows': len(seen),
            'pending_counts': dict(counts), 'chunks': sum(len(s['chunks']) for s in sources), 'output': str(ROOT)}


@app.function(**cpu_options, cpu=1, memory=2048, timeout=86400, max_containers=1)
def prepare():
    tests.remote()
    for report in prepare_one.map(SOURCES, order_outputs=False):
        print(json.dumps(report), flush=True)
    return finalize_preparation.remote()


def load_gate(decision_sha256):
    """Read-only authorization check, also performed before any GPU dispatch."""
    from data.sglang_backlog import validate_scale_decision, read_jsonl
    path = ROOT / 'scale-decision.json'
    if not path.exists() or file_sha(path) != decision_sha256:
        raise ValueError('Missing or changed root-reviewed scale decision')
    decision = json.loads(path.read_text())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['generation_config'] != generation_config():
        raise ValueError('Prepared backlog code/configuration differs from runtime')
    _, pilot_ids, pilot_sha = pilot_selection(PILOT)
    report_path, results_path = PILOT / 'report.json', PILOT / 'results.jsonl'
    if not report_path.exists() or not results_path.exists():
        raise ValueError('Pilot has not completed')
    validate_scale_decision(decision, manifest, json.loads(report_path.read_text()), pilot_ids,
        list(read_jsonl(results_path)), backlog_sha=file_sha(ROOT / 'manifest.json'),
        pilot_manifest_sha=pilot_sha, pilot_results_sha=file_sha(results_path),
        pilot_report_sha=file_sha(report_path))
    if file_sha(ROOT / 'inputs' / 'maud-choices.json') != manifest['maud_ontology_sha256'] or file_sha(SPLIT) != manifest['pubmed_split_sha256']:
        raise ValueError('MAUD ontology or PubMed training split changed')
    return manifest, decision


@app.function(**cpu_options, cpu=4, memory=16384, timeout=1800)
def check_gate(decision_sha256: str):
    volume.reload()
    manifest, _ = load_gate(decision_sha256)
    return {'ready_for_generation': True, 'pending_rows': manifest['pending_rows'],
            'decision_sha256': decision_sha256, 'approved_for_release': False}


@app.cls(image=gpu_image, gpu='H200:8', cpu=16, memory=98304, timeout=21600,
         max_containers=1, scaledown_window=90, retries=0,
         volumes={'/data': volume, '/cache': cache}, secrets=[modal.Secret.from_name('huggingface')])
class Generator:
    @modal.enter()
    def enter(self):
        # Model startup is lazy, after the per-call review gate passes.
        self.process = None
        self.client = None

    def start_server(self):
        import subprocess
        import time
        import urllib.request
        from importlib.metadata import version
        from openai import OpenAI
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponseChoice
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError('SGLang server exited; restart container after inspection')
            return
        if 'return_token_ids' not in ChatCompletionRequest.model_fields or 'response_token_ids' not in ChatCompletionResponseChoice.model_fields:
            raise RuntimeError('Pinned SGLang runtime cannot capture pre-parser tokens')
        run_root = ROOT / 'server-runs' / modal.current_function_call_id()
        run_root.mkdir(parents=True, exist_ok=True)
        command = ['python3', '-m', 'sglang.launch_server', '--model-path', MODEL,
            '--revision', REVISION, '--served-model-name', MODEL, '--host', '127.0.0.1', '--port', '8000',
            '--tp', '8', '--context-length', '32768', '--mem-fraction-static', '0.80',
            '--max-running-requests', '32', '--tool-call-parser', 'qwen3_coder',
            '--reasoning-parser', 'qwen3', '--disable-cuda-graph']
        atomic_json(run_root / 'runtime.json', {'command': command, 'image': IMAGE,
            'packages': {p: version(p) for p in ['sglang', 'transformers', 'torch', 'openai']}})
        with (run_root / 'server.log').open('w') as log:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd='/opt/lclm')
        deadline, last_commit = time.monotonic() + 5400, time.monotonic()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError('SGLang startup failed; inspect ' + str(run_root))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('SGLang startup deadline exceeded')
            if time.monotonic() - last_commit > 30:
                volume.commit()
                last_commit = time.monotonic()
            time.sleep(3)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=0)
        self.server_run = str(run_root)
        atomic_json(run_root / 'ready.json', {'at': time.time()})
        volume.commit()

    @modal.method()
    def chunk(self, source: str, index: int, decision_sha256: str):
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from data.sglang_backlog import read_jsonl, attempt_state, rollout_one, summarize_results, circuit_breaker
        from data.expansion_task_normalization import prepare_teacher_task
        from data.synthetic_expansion_agent import messages_for_openai_api
        from data.qwen38_pilot_sampling import sampling_kwargs
        from data.expansion_retry_diagnostic import response_token_ids, ensure_complete_response, RawCaptureError
        volume.reload()
        manifest, _ = load_gate(decision_sha256)
        specs = [s for s in manifest['sources'] if s['source'] == source]
        if len(specs) != 1 or not 0 <= index < len(specs[0]['chunks']):
            raise ValueError('Unknown source chunk')
        spec = specs[0]['chunks'][index]
        source_root = ROOT / 'outputs' / source
        output = source_root / f'chunk-{index:05d}'
        hold = source_root / 'HOLD.json'
        if hold.exists():
            raise ValueError('Source circuit breaker requires review: ' + str(hold))
        if index:
            prior = source_root / f'chunk-{index-1:05d}' / 'report.json'
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                raise ValueError('Previous source chunk is incomplete/unresolved')
        input_path = ROOT / 'inputs' / source / spec['name']
        if file_sha(input_path) != spec['sha256']:
            raise ValueError('Chunk input hash changed')
        tasks = list(read_jsonl(input_path))
        if len(tasks) != spec['rows']:
            raise ValueError('Chunk row count changed')
        binding = {'source': source, 'chunk': index, 'input_sha256': spec['sha256'],
                   'decision_sha256': decision_sha256, 'backlog_manifest_sha256': file_sha(ROOT / 'manifest.json')}
        output.mkdir(parents=True, exist_ok=True)
        bind_path = output / 'binding.json'
        if bind_path.exists() and json.loads(bind_path.read_text()) != binding:
            raise ValueError('Chunk checkpoint has different inputs/review')
        atomic_json(bind_path, binding)
        started_path = output / 'started.jsonl'
        result_root, raw_root = output / 'results', output / 'raw'
        result_root.mkdir(exist_ok=True)
        raw_root.mkdir(exist_ok=True)
        results = [json.loads(p.read_text()) for p in sorted(result_root.glob('*.json'))]
        for row in results:
            if file_sha(raw_root / (row['task_id'] + '.json')) != row['raw_responses_sha256']:
                raise ValueError('Saved task raw response hash changed')
        state = attempt_state(tasks, list(read_jsonl(started_path)) if started_path.exists() else [], results)
        choices = json.loads((ROOT / 'inputs' / 'maud-choices.json').read_text()) if source == 'maud' else {}
        normalized = {task['task_id']: prepare_teacher_task(task, choices) for task in state['pending']}
        stop_reason = circuit_breaker(results)
        begin = time.monotonic()
        if state['pending'] and not stop_reason:
            self.start_server()
        provenance = {**manifest['generation_config'], **binding,
            'server_run': getattr(self, 'server_run', None),
            'judge_model': MODEL, 'judge_revision': REVISION,
            'judge_role': 'same-model heuristic; no release approval'}
        def run(raw_task):
            records = []
            task = normalized[raw_task['task_id']]
            def request(messages, tools, phase):
                kwargs = {'model': MODEL, 'messages': messages_for_openai_api(messages), **sampling_kwargs('recommended')}
                kwargs['extra_body']['return_token_ids'] = True
                if tools:
                    kwargs.update(tools=list(tools), tool_choice='auto')
                entry = {'phase': phase, 'request': kwargs}
                records.append(entry)
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    entry['error'] = {'type': type(exc).__name__, 'message': str(exc),
                        'status_code': getattr(exc, 'status_code', None), 'body': getattr(exc, 'body', None),
                        'request_id': getattr(exc, 'request_id', None)}
                    raise
                payload = response.model_dump(mode='json')
                entry['response'] = payload
                ids = response_token_ids(payload['choices'][0])
                try:
                    entry['raw_generated_text'] = self.tokenizer.decode(ids, skip_special_tokens=False)
                except Exception as exc:
                    raise RawCaptureError('Cannot decode raw response tokens') from exc
                ensure_complete_response(response.choices[0].finish_reason, phase)
                return payload['choices'][0]['message']
            result = rollout_one(task, source, request, provenance)
            raw = json.dumps(records, ensure_ascii=False).encode()
            raw_path = raw_root / (task['task_id'] + '.json')
            raw_path.write_bytes(raw)
            result['raw_responses_sha256'] = sha(raw)
            result['token_usage'] = {key: sum((r.get('response', {}).get('usage') or {}).get(key, 0) or 0
                for r in records) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            return result
        pending = state['pending']
        # Commit reservations BEFORE submission. A lost result is surfaced as an
        # unresolved attempt on continuation, not retried or counted complete.
        for offset in range(0, len(pending), CONCURRENCY):
            if stop_reason:
                break
            batch = pending[offset:offset + CONCURRENCY]
            with started_path.open('a') as handle:
                for task in batch:
                    handle.write(json.dumps({'task_id': task['task_id'],
                        'task_sha256': sha(json.dumps(task, sort_keys=True).encode()),
                        'reserved_at': time.time(), 'function_call_id': modal.current_function_call_id()}) + '\n')
                handle.flush()
                import os
                os.fsync(handle.fileno())
            volume.commit()
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                for future in as_completed([pool.submit(run, task) for task in batch]):
                    result = future.result()
                    atomic_json(result_root / (result['task_id'] + '.json'), result)
                    results.append(result)
            stop_reason = circuit_breaker(results)
            progress = {**binding, 'counts': summarize_results(results), 'rows': len(tasks),
                        'elapsed_this_call': time.monotonic() - begin, 'stop_reason': stop_reason}
            atomic_json(output / 'progress.json', progress)
            volume.commit()
            print(json.dumps({'source': source, 'chunk': index, 'counts': progress['counts']}), flush=True)
        state = attempt_state(tasks, list(read_jsonl(started_path)) if started_path.exists() else [], results)
        status = 'held' if stop_reason else 'needs_reconciliation' if state['unresolved'] else 'complete'
        report = {**binding, 'status': status, 'rows': len(tasks), 'completed': len(results),
            'unattempted': len(state['pending']), 'unresolved_attempt_ids': state['unresolved'],
            'counts': summarize_results(results), 'stop_reason': stop_reason,
            'seconds_this_call': time.monotonic() - begin, 'approved_for_release': False,
            'result_hashes': {r['task_id']: file_sha(result_root / (r['task_id'] + '.json')) for r in results}}
        atomic_json(output / 'report.json', report)
        if stop_reason:
            atomic_json(hold, report)
        volume.commit()
        return report

    @modal.exit()
    def stop(self):
        import subprocess
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        volume.commit()


@app.function(**cpu_options, cpu=4, memory=16384, timeout=86400, max_containers=1)
def dispatch(decision_sha256: str, max_chunks: int = 8):
    """Bounded continuation; root must supply a reviewed decision before this call."""
    if not 1 <= max_chunks <= 32:
        raise ValueError('Each dispatch must request1..32 bounded chunks')
    volume.reload()
    manifest, _ = load_gate(decision_sha256)
    generator = Generator()
    completed = []
    # Cover each source's first64 before continuing large sources.
    for index in range(max(len(s['chunks']) for s in manifest['sources'])):
        for source in manifest['sources']:
            if index >= len(source['chunks']):
                continue
            source_root = ROOT / 'outputs' / source['source']
            report_path = source_root / f'chunk-{index:05d}' / 'report.json'
            if (source_root / 'HOLD.json').exists():
                continue
            if report_path.exists():
                previous = json.loads(report_path.read_text())
                if previous['status'] in ('complete', 'held', 'needs_reconciliation'):
                    continue
            if index:
                prior = source_root / f'chunk-{index-1:05d}' / 'report.json'
                if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                    continue
            report = generator.chunk.remote(source['source'], index, decision_sha256)
            completed.append({'source': source['source'], 'chunk': index, 'status': report['status'], 'counts': report['counts']})
            volume.reload()
            if len(completed) >= max_chunks:
                return {'dispatched_chunks': completed, 'bounded_dispatch_complete': True}
    return {'dispatched_chunks': completed, 'bounded_dispatch_complete': True}


@app.function(**cpu_options, cpu=2, memory=8192, timeout=1800)
def status():
    volume.reload()
    from collections import Counter
    result = {'prepared': (ROOT / 'manifest.json').exists(), 'generation_decision_exists': (ROOT / 'scale-decision.json').exists(), 'sources': {}}
    for source in SOURCES:
        root = ROOT / 'outputs' / source
        reports = [json.loads(p.read_text()) for p in sorted(root.glob('chunk-*/report.json'))]
        counts = Counter()
        for report in reports:
            counts.update(report['counts'])
        result['sources'][source] = {'reported_chunks': len(reports), 'counts': dict(counts),
            'unresolved_attempts': sum(len(r['unresolved_attempt_ids']) for r in reports),
            'held': (root / 'HOLD.json').exists(),
            'reports': [{'chunk': r['chunk'], 'status': r['status'], 'completed': r['completed'], 'unattempted': r['unattempted']} for r in reports]}
    return result
