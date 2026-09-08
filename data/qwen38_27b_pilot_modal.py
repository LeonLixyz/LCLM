"""Isolated 24-task non-thinking teacher comparison. No release or scale option."""
import hashlib
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image as cpu_image, volume, ROOT

APP = 'lclm-qwen38-27b-teacher-pilot-v1'
MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
OUTPUT = ROOT/'qwen38-27b-teacher-pilot-v1'
BASE = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
TASKS = BASE.parent/'full-20260906-v3'
SOURCES = ('billsum', 'finqa', 'tatqa', 'contract_nli', 'faithdial', 'clapnq')
app = modal.App(APP)
cache = modal.Volume.from_name('lclm-hf-cache')
image = (modal.Image.from_registry('vllm/vllm-openai:v0.28.0').entrypoint([])
         .uv_pip_install('transformers==5.16.1', 'openai>=2,<3')
         .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/opt/lclm',
               'HF_XET_HIGH_PERFORMANCE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir('.', '/opt/lclm', copy=True,
                        ignore=['.git', '.venv', '__pycache__', '**/__pycache__/**', '**/*.pyc', '_modal_run']))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@app.function(image=cpu_image, cpu=2, memory=8192, timeout=1800, volumes={'/data': volume})
def prepare():
    import subprocess
    from data.expansion_task_normalization import prepare_teacher_task
    subprocess.run(['python', '-m', 'pytest', '-q',
                    '/opt/lclm/tests/test_harvest_expansion_trace.py',
                    '/opt/lclm/tests/test_clean_agent_trajectories.py'], check=True)
    volume.reload()
    if (OUTPUT/'inputs.json').exists():
        payload = (OUTPUT/'inputs.json').read_bytes()
        manifest = json.loads((OUTPUT/'manifest.json').read_text())
        if sha(payload) != manifest['inputs_sha256'] or manifest['model_revision'] != REVISION:
            raise ValueError('Changed existing pilot inputs')
        return manifest
    cases = []
    for source in SOURCES:
        picked = {}
        for accepted in (True, False):
            path = BASE/(source+('.accepted.jsonl' if accepted else '.rejected.jsonl'))
            with path.open('rb') as stream:
                for _ in range(2):
                    line = next(stream)
                    if not line.endswith(b'\n'): raise ValueError('Incomplete baseline row')
                    row = json.loads(line)
                    if row['verification']['accepted'] is not accepted or row['task_id'] in picked:
                        raise ValueError('Invalid baseline bucket')
                    picked[row['task_id']] = {'source': source, 'baseline': row,
                        'baseline_row_sha256': sha(line), 'baseline_file': str(path)}
        found = set()
        with (TASKS/(source+'.tasks.jsonl')).open('rb') as stream:
            for line in stream:
                task = json.loads(line)
                if task['task_id'] not in picked: continue
                if task['task_id'] in found: raise ValueError('Duplicate prepared task')
                found.add(task['task_id'])
                task = prepare_teacher_task(task)
                case = picked[task['task_id']]
                if case['baseline'].get('task', task['question']) != task['question']:
                    raise ValueError('Baseline/prepared question differs')
                cases.append({**case, 'task': task, 'prepared_row_sha256': sha(line)})
        if found != picked.keys(): raise ValueError('Missing matched tasks')
    if len(cases) != 24 or len({c['task']['task_id'] for c in cases}) != 24:
        raise ValueError('Invalid pilot size or identity')
    raw = json.dumps(cases, ensure_ascii=False, indent=2).encode()
    manifest = {'status': 'prepared', 'model': MODEL, 'model_revision': REVISION,
        'rows': 24, 'sources': SOURCES, 'inputs_sha256': sha(raw),
        'selection': 'First two saved accepted and two rejected records per source; diagnostic balanced sample, not representative yield estimate.',
        'temperature': 0, 'enable_thinking': False, 'max_tokens_per_turn': 2048,
        'max_tool_calls': 16, 'concurrency': 4, 'vllm': '0.28.0', 'transformers': '5.16.1',
        'sampling_scope': 'Temperature and token/tool budgets match existing 235B run for a controlled comparison.',
        'judge': 'No new self-judge. Fixed-answer rules plus pending manual review of free-form answers.',
        'approved_for_release': False, 'training_rows_modified': False}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'inputs.json').write_bytes(raw)
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.function(image=image, gpu='H200:8', cpu=16, memory=65536, timeout=5400,
              max_containers=1, scaledown_window=60,
              volumes={'/data': volume, '/cache': cache},
              secrets=[modal.Secret.from_name('huggingface')])
def pilot():
    import os
    import subprocess
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from data.synthetic_expansion_agent import run_agent_rollout, messages_for_openai_api
    from data.clean_agent_trajectories import strip_cot
    from data.harvest_expansion_trace import harvest_training_messages
    from data.real_expansion_agent import verify_real_trace
    volume.reload()
    raw = (OUTPUT/'inputs.json').read_bytes()
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    if sha(raw) != manifest['inputs_sha256'] or manifest['model_revision'] != REVISION:
        raise ValueError('Missing/stale input preparation')
    cases = json.loads(raw)
    if len(cases) != 24: raise ValueError('Unbounded pilot input')
    if (OUTPUT/'report.json').exists(): return json.loads((OUTPUT/'report.json').read_text())
    results_path = OUTPUT/'results.jsonl'
    if results_path.exists(): raise ValueError('Partial pilot exists; inspect before explicit recovery')
    raw_dir = OUTPUT/'raw-responses'; raw_dir.mkdir(exist_ok=True)
    command = ['vllm', 'serve', MODEL, '--revision', REVISION, '--served-model-name', MODEL,
               '--host', '127.0.0.1', '--port', '8000', '--tensor-parallel-size', '8',
               '--max-model-len', '32768', '--max-num-seqs', '16', '--gpu-memory-utilization', '0.85',
               '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
               '--reasoning-parser', 'qwen3', '--enforce-eager']
    (OUTPUT/'server-command.json').write_text(json.dumps(command)); volume.commit()
    with (OUTPUT/'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+3600
            while True:
                if process.poll() is not None: raise RuntimeError('Pilot server exited; inspect server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                        if response.status == 200: break
                except (OSError, TimeoutError): pass
                if time.monotonic() > deadline: raise TimeoutError('Pilot startup timeout')
                time.sleep(5)
            client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=180, max_retries=1)
            def run(case):
                records = []; task = case['task']; phase = 'rollout'
                result = {'task_id': task['task_id'], 'source': case['source'],
                          'baseline_verification': case['baseline']['verification'],
                          'approved_for_release': False}
                started = time.monotonic()
                def complete(messages, tools):
                    request = {'model': MODEL, 'messages': messages_for_openai_api(messages),
                        'temperature': 0, 'max_tokens': 2048,
                        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
                    if tools: request.update(tools=list(tools), tool_choice='auto')
                    entry = {'request': request}; records.append(entry)
                    response = client.chat.completions.create(**request)
                    entry['response'] = response.model_dump(mode='json')
                    if response.choices[0].finish_reason == 'length': raise ValueError('Truncated model response')
                    message = response.choices[0].message
                    return {'content': strip_cot(message.content),
                            'tool_calls': [c.model_dump() for c in (message.tool_calls or [])]}
                try:
                    trace = run_agent_rollout(task, complete, max_tool_calls=16)
                    result['rollout_failure_reason'] = trace['rollout_failure_reason']
                    if trace['rollout_failure_reason']:
                        result['diagnostic_trace'] = trace
                    else:
                        phase = 'harvest'
                        trace['messages'], trace['harvesting'] = harvest_training_messages(trace['messages'])
                        phase = 'rule_verification'
                        result['rule_verification'] = verify_real_trace(task, trace['messages'])
                        trace.update(model=MODEL, model_revision=REVISION, source_dataset=task['source_dataset'],
                            source_row_id=task['source_row_id'], generation={'enable_thinking': False,
                            'teacher_prompt_saved_in_training_messages': False})
                        trace['verification'] = {'accepted': False, 'reason': 'diagnostic_pending_manual_review'}
                        result['diagnostic_trace'] = trace
                except Exception as exc:
                    result['error'] = {'phase': phase, 'type': type(exc).__name__, 'message': str(exc)}
                raw_payload = json.dumps(records, ensure_ascii=False, indent=2).encode()
                (raw_dir/(task['task_id']+'.json')).write_bytes(raw_payload)
                result.update(raw_responses_sha256=sha(raw_payload), seconds=time.monotonic()-started)
                return result
            results = []
            with results_path.open('a') as out, ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(run, case) for case in cases]):
                    result = future.result(); results.append(result)
                    out.write(json.dumps(result, ensure_ascii=False)+'\n'); out.flush(); os.fsync(out.fileno())
                    (OUTPUT/'progress.json').write_text(json.dumps({'completed': len(results), 'total': 24}))
                    volume.commit()
            report = {'status': 'awaiting_manual_comparison', 'completed': len(results),
                      'errors': sum('error' in r for r in results),
                      'rollout_failures': sum(bool(r.get('rollout_failure_reason')) for r in results),
                      'inputs_sha256': manifest['inputs_sha256'],
                      'results_sha256': sha(results_path.read_bytes()), 'approved_for_release': False,
                      'limits': 'Balanced diagnostic sample; free-form rule scores are not semantic acceptance. No self-judge or production model switch.'}
            (OUTPUT/'report.json').write_text(json.dumps(report, indent=2)); volume.commit()
            return report
        except Exception as exc:
            (OUTPUT/'failure.json').write_text(json.dumps({'error': str(exc), 'type': type(exc).__name__}))
            volume.commit()
            raise
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            volume.commit()


@app.local_entrypoint()
def main():
    print(json.dumps(prepare.remote(), indent=2))
