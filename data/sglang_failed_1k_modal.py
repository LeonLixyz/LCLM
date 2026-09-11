"""Exactly 1,000 prior failures, Qwen3.8-27B on SGLang; diagnostic outputs only.

Prepare: modal run -m data.sglang_failed_1k_modal
Deploy: modal deploy -m data.sglang_failed_1k_modal
Launch the deployed ``pilot`` once with Function.from_name(...).spawn().
"""
import hashlib
import json
from pathlib import Path

import modal
from data.stage3_full_modal import image as cpu_image, volume, ROOT

APP = 'lclm-sglang-27b-failed-1k-20260909'
MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
IMAGE = 'lmsysorg/sglang@sha256:b91d664a8e4825afc16ab831c6035a6c88ac20ef8bd26da4fe2b9813a9f44376'
PREPARED = Path('/data/stage3-agent/real-expansion/pilots/retry-and-remaining-qwen38-27b-v2-tasks')
OUTPUT = ROOT/'sglang-27b-failed-1k-20260909-v1'
app = modal.App(APP)
cache = modal.Volume.from_name('lclm-hf-cache')
image = (modal.Image.from_registry(IMAGE).entrypoint([])
         .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/opt/lclm',
               'HF_XET_HIGH_PERFORMANCE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir('.', '/opt/lclm', copy=True, ignore=[
             '.git', '.venv', '__pycache__', '**/__pycache__/**', '**/*.pyc', '_modal_run']))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=1800, volumes={'/data': volume})
def plan():
    import subprocess
    from data.expansion_retry_diagnostic import fair_quotas
    subprocess.run(['python', '-m', 'pytest', '-q',
        'tests/test_expansion_retry_diagnostic.py', 'tests/test_expansion_retry.py',
        'tests/test_harvest_expansion_trace.py', 'tests/test_qwen38_pilot_sampling.py',
        'tests/test_expansion_semantic_review.py', 'tests/test_clean_agent_trajectories.py'],
        cwd='/opt/lclm', check=True)
    volume.reload()
    parent = json.loads((PREPARED/'retry-manifest.json').read_text())
    return fair_quotas({r['source']: r['counts'].get('retry_rejected', 0)
                       for r in parent['sources']}, 1000)


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=7200,
              max_containers=4, volumes={'/data': volume})
def prepare_source(source: str, count: int):
    from collections import Counter
    from data.expansion_retry_diagnostic import select_failures
    from data.expansion_task_normalization import prepare_teacher_task
    from data.prepare_expansion_retry import validate_prepared
    volume.reload()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT/(source+'.inputs.json')
    marker = OUTPUT/(source+'.prepared.json')
    parent = json.loads((PREPARED/(source+'.build.json')).read_text())
    validate_prepared(parent, PREPARED)
    expected = parent['prepared_files'][source+'.tasks.jsonl']
    if marker.exists():
        report = json.loads(marker.read_text())
        if report['count'] != count or report['prepared_source_sha256'] != expected or sha(target.read_bytes()) != report['inputs_sha256']:
            raise ValueError('Existing selection differs')
        return report
    digest = hashlib.sha256()
    def rows():
        with (PREPARED/(source+'.tasks.jsonl')).open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    raise ValueError('Incomplete task line')
                digest.update(line)
                yield json.loads(line)
    selected, available, selected_counts = select_failures(rows(), count)
    if digest.hexdigest() != expected:
        raise ValueError('Prepared task file hash changed')
    baseline = {}
    wanted = {r['task_id'] for r in selected}
    parent_path = Path(parent['previous_root'])/(source+'.rejected.jsonl')
    with parent_path.open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row['task_id'] not in wanted:
                continue
            if row['verification']['accepted'] is not False or row['task_id'] in baseline:
                raise ValueError('Invalid prior failure')
            baseline[row['task_id']] = {
                'row_sha256': sha(line), 'file': str(parent_path),
                'verification': row['verification'],
                'assistant_messages': [m for m in row.get('messages', []) if m['role'] == 'assistant'],
                'error': row.get('error'), 'model': row.get('model'),
                'model_revision': row.get('model_revision')}
    if set(baseline) != wanted:
        raise ValueError('Selected failures missing from saved parent')
    choices = json.loads((PREPARED/'maud-choices.json').read_text()) if source == 'maud' else {}
    cases = []
    for original in selected:
        previous = baseline[original['task_id']]
        if original['_attempt_provenance']['previous_reason'] != previous['verification']['reason']:
            raise ValueError('Prior failure reason changed')
        if source == 'pubmedqa_labeled':
            from data.pubmedqa_split import training_ids
            splits = json.loads(Path('/data/stage3-agent/real-expansion/sources/pubmedqa_labeled/official-splits/split-manifest.json').read_text())
            if original['source_row_id'] not in training_ids(splits):
                raise ValueError('PubMedQA non-training task')
        task = prepare_teacher_task(original, choices)
        cases.append({'source': source, 'task': task, 'baseline': previous})
    raw = json.dumps(cases, ensure_ascii=False).encode()
    target.write_bytes(raw)
    report = {'source': source, 'count': len(cases), 'inputs_sha256': sha(raw),
              'prepared_source_sha256': expected, 'available_failure_reasons': available,
              'selected_failure_reasons': selected_counts, 'approved_for_release': False}
    write_json(marker, report)
    volume.commit()
    print(json.dumps(report), flush=True)
    return report


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=600, volumes={'/data': volume})
def finalize(quotas: dict):
    from data.qwen38_pilot_sampling import sampling_kwargs
    from data.synthetic_expansion_agent import TEACHER_SYSTEM_PROMPT, EXPAND_TOOL
    volume.reload()
    reports = [json.loads((OUTPUT/(s+'.prepared.json')).read_text()) for s in sorted(quotas)]
    ids = set()
    for r in reports:
        raw = (OUTPUT/(r['source']+'.inputs.json')).read_bytes()
        if sha(raw) != r['inputs_sha256'] or r['count'] != quotas[r['source']]:
            raise ValueError('Selection changed')
        for case in json.loads(raw):
            identifier = case['task']['task_id']
            if identifier in ids or case['baseline']['verification']['accepted'] is not False:
                raise ValueError('Duplicate or non-failed pilot case')
            ids.add(identifier)
    if len(ids) != 1000:
        raise ValueError('Pilot must contain exactly 1000 failures')
    manifest = {'rows': 1000, 'model': MODEL, 'model_revision': REVISION, 'image': IMAGE,
        'sources': reports, 'sampling': sampling_kwargs('recommended'), 'max_tool_calls': 16,
        'context_length': 32768, 'concurrency': 16,
        'teacher_system_prompt_sha256': sha(TEACHER_SYSTEM_PROMPT.encode()),
        'tool_schema_sha256': sha(json.dumps(EXPAND_TOOL, sort_keys=True).encode()),
        'selection': 'Only saved rejected/failed tasks; balanced across sources and failure reasons; stable hash within each stratum. Not a population accuracy estimate.',
        'comparison': 'New teacher, serving engine and recommended sampling (235B temperature0 versus27B temperature0.7/presence_penalty1.5); same task/prompt/verification rules. Does not isolate the effect of engine, sampling or model.',
        'raw_capture': 'Full requests/responses and decoded response_token_ids before tool parsing.',
        'judge': 'Same production semantic checks, now 27B; heuristic self-judge, not release approval.',
        'approved_for_release': False}
    path = OUTPUT/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing manifest changed')
    write_json(path, manifest)
    volume.commit()
    return {'rows': 1000, 'quotas': quotas, 'manifest_sha256': sha(path.read_bytes()), 'output': str(OUTPUT)}


@app.function(image=image, cpu=2, memory=8192, timeout=600)
def inspect_runtime():
    """Check capture/API compatibility before allocating the model-serving GPUs."""
    from importlib.metadata import version
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponseChoice
    fields = {'request': sorted(ChatCompletionRequest.model_fields),
              'choice': sorted(ChatCompletionResponseChoice.model_fields)}
    if 'return_token_ids' not in fields['request'] or 'response_token_ids' not in fields['choice']:
        raise RuntimeError('Pinned SGLang runtime lacks pre-parser token capture')
    return {'packages': {p: version(p) for p in ['sglang', 'transformers', 'torch', 'openai']},
            'return_token_ids_supported': True, 'response_token_ids_supported': True}


@app.function(image=image, gpu='H200:8', cpu=16, memory=98304, timeout=21600,
              max_containers=1, scaledown_window=60,
              volumes={'/data': volume, '/cache': cache},
              secrets=[modal.Secret.from_name('huggingface')])
def pilot():
    import os
    import subprocess
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from importlib.metadata import version
    from openai import OpenAI
    from transformers import AutoTokenizer
    from data.synthetic_expansion_agent import run_agent_rollout, messages_for_openai_api
    from data.clean_agent_trajectories import strip_cot
    from data.harvest_expansion_trace import harvest_training_messages
    from data.real_expansion_agent import verify_real_trace
    from data.expansion_semantic_review import apply_semantic_review
    from data.expansion_retry_diagnostic import summarize, response_token_ids, ensure_complete_response, RawCaptureError
    from data.qwen38_pilot_sampling import sampling_kwargs
    from data.synthetic_expansion_agent import TEACHER_SYSTEM_PROMPT, EXPAND_TOOL

    volume.reload()
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    if manifest['model_revision'] != REVISION or manifest['image'] != IMAGE:
        raise ValueError('Pilot runtime does not match manifest')
    if (manifest['sampling'] != sampling_kwargs('recommended') or
            manifest['teacher_system_prompt_sha256'] != sha(TEACHER_SYSTEM_PROMPT.encode()) or
            manifest['tool_schema_sha256'] != sha(json.dumps(EXPAND_TOOL, sort_keys=True).encode()) or
            manifest['max_tool_calls'] != 16 or manifest['context_length'] != 32768 or
            manifest['concurrency'] != 16):
        raise ValueError('Pilot prompt/tool/sampling/budget configuration changed')
    if (OUTPUT/'report.json').exists():
        return json.loads((OUTPUT/'report.json').read_text())
    if (OUTPUT/'started.json').exists():
        raise ValueError('Pilot already started; inspect existing run before resuming')
    cases = []
    for report in manifest['sources']:
        raw = (OUTPUT/(report['source']+'.inputs.json')).read_bytes()
        if sha(raw) != report['inputs_sha256']:
            raise ValueError('Frozen inputs changed')
        cases.extend(json.loads(raw))
    if len(cases) != 1000 or len({c['task']['task_id'] for c in cases}) != 1000:
        raise ValueError('Pilot is not exactly 1000 unique tasks')
    # Interleave sources so early progress covers the entire diagnostic set.
    cases.sort(key=lambda c: sha(('run-order:' + c['task']['task_id']).encode()))
    raw_dir = OUTPUT/'raw-responses'; raw_dir.mkdir(exist_ok=True)
    write_json(OUTPUT/'started.json', {'at': time.time(), 'manifest_sha256': sha((OUTPUT/'manifest.json').read_bytes())})
    command = ['python3', '-m', 'sglang.launch_server', '--model-path', MODEL,
        '--revision', REVISION, '--served-model-name', MODEL, '--host', '127.0.0.1', '--port', '8000',
        '--tp', '8', '--context-length', '32768', '--mem-fraction-static', '0.80',
        '--max-running-requests', '32', '--tool-call-parser', 'qwen3_coder',
        '--reasoning-parser', 'qwen3', '--disable-cuda-graph']
    write_json(OUTPUT/'server-command.json', command)
    write_json(OUTPUT/'runtime.json', {'image': IMAGE, 'packages': {p: version(p) for p in ['sglang', 'transformers', 'torch', 'openai']}})
    volume.commit()
    process = None
    try:
        with (OUTPUT/'server.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd='/opt/lclm')
        deadline = time.monotonic() + 5400
        last_commit = time.monotonic()
        while True:
            if process.poll() is not None:
                raise RuntimeError('SGLang exited; inspect server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('SGLang model startup timeout')
            if time.monotonic() - last_commit > 30:
                volume.commit(); last_commit = time.monotonic()
            time.sleep(3)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=0)
        sampling = sampling_kwargs('recommended')
        write_json(OUTPUT/'ready.json', {'at': time.time()}); volume.commit()

        def run(case):
            task = case['task']; records = []; trace = None; phase = 'rollout'
            started = time.monotonic()
            result = {'task_id': task['task_id'], 'source': case['source'],
                'previous_reason': case['baseline']['verification']['reason'],
                'baseline': case['baseline'], 'approved_for_release': False}
            def request(messages, tools=None):
                kwargs = {'model': MODEL, 'messages': messages_for_openai_api(messages), **sampling}
                kwargs['extra_body'] = {**sampling['extra_body'], 'return_token_ids': True}
                if tools:
                    kwargs.update(tools=list(tools), tool_choice='auto')
                entry = {'phase': phase, 'request': kwargs}; records.append(entry)
                try:
                    response = client.chat.completions.create(**kwargs)
                except Exception as exc:
                    entry['error'] = {'type': type(exc).__name__, 'message': str(exc),
                        'status_code': getattr(exc, 'status_code', None),
                        'body': getattr(exc, 'body', None), 'request_id': getattr(exc, 'request_id', None)}
                    raise
                payload = response.model_dump(mode='json'); entry['response'] = payload
                ids = response_token_ids(payload['choices'][0])
                try:
                    entry['raw_generated_text'] = tokenizer.decode(ids, skip_special_tokens=False)
                except Exception as exc:
                    raise RawCaptureError('Cannot decode pre-parser output tokens') from exc
                ensure_complete_response(response.choices[0].finish_reason, phase)
                return response.choices[0].message
            def complete(messages, tools):
                message = request(messages, tools)
                return {'content': strip_cot(message.content),
                        'tool_calls': [c.model_dump() for c in (message.tool_calls or [])]}
            try:
                trace = run_agent_rollout(task, complete, max_tool_calls=16)
                result['tool_call_count'] = trace['tool_call_count']
                result['rollout_failure_reason'] = trace['rollout_failure_reason']
                if trace['rollout_failure_reason']:
                    result['verification'] = trace['verification']
                else:
                    phase = 'harvest'
                    trace['messages'], trace['harvesting'] = harvest_training_messages(trace['messages'])
                    phase = 'rule_verification'
                    trace['verification'] = verify_real_trace(task, trace['messages'])
                    result['rule_verification'] = trace['verification']
                    phase = 'judge'
                    trace['verification'] = apply_semantic_review(case['source'], trace,
                        lambda messages: request(messages).content)
                    result['verification'] = trace['verification']
                trace.update(model=MODEL, model_revision=REVISION,
                    source_dataset=task['source_dataset'], source_row_id=task['source_row_id'])
            except Exception as exc:
                result['error'] = {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)}
                result['verification'] = {'accepted': False,
                    'reason': 'diagnostic_error:' + phase + ':' + type(exc).__name__}
                if trace is not None:
                    trace['verification'] = result['verification']
            if trace is not None:
                # Keep partial/failed messages intact for diagnosis; never export these as approved rows.
                result['trace'] = trace
            raw = json.dumps(records, ensure_ascii=False).encode()
            (raw_dir/(task['task_id']+'.json')).write_bytes(raw)
            result.update(raw_responses_sha256=sha(raw), requests=len(records), seconds=time.monotonic()-started)
            return result

        results = []
        started = time.monotonic()
        with (OUTPUT/'results.jsonl').open('x') as out:
            def save(result):
                nonlocal last_commit
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False)+'\n'); out.flush(); os.fsync(out.fileno())
                if len(results) == 1 or len(results) % 20 == 0 or time.monotonic()-last_commit > 30:
                    progress = {**summarize(results), 'total': 1000, 'elapsed_seconds': time.monotonic()-started}
                    write_json(OUTPUT/'progress.json', progress); volume.commit(); last_commit = time.monotonic()
                    print(json.dumps({'completed': len(results), 'counts': progress['counts']}), flush=True)
            # The first requested task doubles as the API/token-capture integration check.
            first = run(cases[0]); save(first)
            if (first.get('error', {}).get('phase') == 'rollout' and
                    first['error']['type'] != 'TruncatedResponseError'):
                raise RuntimeError('First-task API integration failed; inspect before submitting remaining tasks')
            with ThreadPoolExecutor(max_workers=16) as pool:
                for future in as_completed([pool.submit(run, case) for case in cases[1:]]):
                    save(future.result())
        report = {**summarize(results), 'status': 'complete_diagnostic_pending_review',
            'manifest_sha256': sha((OUTPUT/'manifest.json').read_bytes()),
            'results_sha256': sha((OUTPUT/'results.jsonl').read_bytes()),
            'elapsed_seconds': time.monotonic()-started, 'approved_for_release': False,
            'limitations': manifest['selection'] + ' ' + manifest['comparison']}
        write_json(OUTPUT/'report.json', report); volume.commit()
        return report
    except Exception as exc:
        write_json(OUTPUT/'failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        volume.commit()
        raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        volume.commit()


@app.local_entrypoint()
def main():
    quotas = plan.remote()
    print(json.dumps({'quotas': quotas}), flush=True)
    for result in prepare_source.starmap(sorted(quotas.items()), order_outputs=False):
        print(json.dumps(result), flush=True)
    print(json.dumps(finalize.remote(quotas)), flush=True)
