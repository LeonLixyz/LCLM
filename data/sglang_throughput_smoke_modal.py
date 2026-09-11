"""Exactly128 captured first-turn replays on eight TP1 BF16 replicas; no data export."""
import json
from pathlib import Path
import modal

from data.sglang_backlog import MODEL, REVISION, IMAGE, sha, file_sha, atomic_json

APP_NAME = 'lclm-sglang27-tp1x8-smoke-20260909-v1'
ROOT = Path('/data/stage3-build-20260906/sglang27-tp1x8-smoke-20260909-v1')
PILOT = Path('/data/stage3-build-20260906/sglang-27b-failed-1k-20260909-v1')
app = modal.App(APP_NAME)
volume = modal.Volume.from_name('lclm-stage3-data')
cache = modal.Volume.from_name('lclm-hf-cache')
CODE_FILES = ['sglang_throughput_smoke.py', 'sglang_throughput_smoke_modal.py',
              'synthetic_expansion_agent.py', 'expansion_retry_diagnostic.py', 'sglang_backlog.py']


def code_hashes():
    return {name: file_sha(Path(__file__).parent / name) for name in CODE_FILES}


ignore = ['.git', '.venv', '__pycache__', '**/__pycache__/**', '**/*.pyc', '_modal_run']
cpu_image = (modal.Image.debian_slim(python_version='3.11').pip_install('pytest==8.4.2')
    .env({'PYTHONPATH': '/opt/lclm'}).add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))
gpu_image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/opt/lclm', 'HF_XET_HIGH_PERFORMANCE': '1',
          'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'})
    .add_local_dir('.', '/opt/lclm', copy=True, ignore=ignore))


@app.function(image=cpu_image, cpu=4, memory=16384, timeout=1800, volumes={'/data': volume})
def prepare():
    import subprocess
    from data.sglang_throughput_smoke import first_request
    result = subprocess.run(['python', '-m', 'pytest', '-q', 'tests/test_sglang_throughput_smoke.py'],
        cwd='/opt/lclm', check=True, text=True, capture_output=True)
    volume.reload()
    output = ROOT / 'inputs.json'
    if output.exists():
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        if file_sha(output) != manifest['inputs_sha256'] or manifest['code_sha256'] != code_hashes():
            raise ValueError('Frozen smoke inputs changed')
        return manifest
    # Read one committed snapshot; the parent pilot may continue independently.
    raw_results = (PILOT / 'results.jsonl').read_bytes()
    if not raw_results.endswith(b'\n'):
        raise ValueError('Pilot result snapshot ends in a partial line')
    rows = [json.loads(line) for line in raw_results.splitlines()]
    if len(rows) < 128:
        raise ValueError('Need at least128 already captured pilot tasks')
    selected = rows[:128]
    if len({r['task_id'] for r in selected}) != 128:
        raise ValueError('Duplicate replay task ID')
    segment_ids = {}
    pilot_manifest = json.loads((PILOT / 'manifest.json').read_text())
    wanted = {r['task_id'] for r in selected}
    for source in pilot_manifest['sources']:
        path = PILOT / (source['source'] + '.inputs.json')
        if file_sha(path) != source['inputs_sha256']:
            raise ValueError('Frozen pilot task inputs changed')
        for case in json.loads(path.read_text()):
            if case['task']['task_id'] in wanted:
                segment_ids[case['task']['task_id']] = [s['segment_id'] for s in case['task']['segments']]
    cases = []
    for row in selected:
        path = PILOT / 'raw-responses' / (row['task_id'] + '.json')
        if file_sha(path) != row['raw_responses_sha256']:
            raise ValueError('Captured request/response bytes changed')
        records = json.loads(path.read_text())
        request = first_request(records)
        if request['model'] != MODEL:
            raise ValueError('Captured request uses a different model')
        cases.append({'task_id': row['task_id'], 'source': row['source'], 'request': request,
            'valid_segment_ids': segment_ids[row['task_id']], 'prior_first_response': records[0]['response'],
            'captured_raw_sha256': row['raw_responses_sha256']})
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(cases, ensure_ascii=False).encode()
    output.write_bytes(raw)
    manifest = {'requests': 128, 'inputs_sha256': sha(raw), 'model': MODEL, 'model_revision': REVISION,
        'image': IMAGE, 'code_sha256': code_hashes(), 'layout': '8 independent TP1 replicas; one visible H200 each',
        'dtype': 'bfloat16', 'cuda_graphs': 'enabled; max batch16',
        'max_running_requests_per_replica': 16, 'max_mamba_cache_size_per_replica': 80,
        'selection': 'First128 committed completed pilot tasks; only their already captured first-turn requests are replayed once.',
        'pilot_manifest_sha256': file_sha(PILOT / 'manifest.json'),
        'pilot_results_snapshot_sha256': sha(raw_results), 'tests': result.stdout,
        'training_rows_generated': 0, 'approved_for_release': False,
        'comparison_limit': 'First-turn replay latency cannot be compared to earlier whole-task latency. Absolute serving throughput only; different completion lengths and source-selection bias apply.'}
    atomic_json(ROOT / 'manifest.json', manifest)
    volume.commit()
    return manifest


@app.function(image=gpu_image, gpu='H200:8', cpu=32, memory=262144, timeout=3600,
              max_containers=1, retries=0, scaledown_window=30,
              volumes={'/data': volume, '/cache': cache}, secrets=[modal.Secret.from_name('huggingface')])
def smoke():
    import os
    import re
    import signal
    import subprocess
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from importlib.metadata import version
    from openai import OpenAI
    from transformers import AutoTokenizer
    from data.sglang_throughput_smoke import inspect_response, graph_capture_lines, report_requests
    from data.expansion_retry_diagnostic import response_token_ids
    volume.reload()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if (file_sha(ROOT / 'inputs.json') != manifest['inputs_sha256'] or manifest['requests'] != 128
            or manifest['code_sha256'] != code_hashes()):
        raise ValueError('Changed/unbounded replay inputs')
    if (ROOT / 'started.json').exists():
        raise ValueError('Smoke already started; inspect existing results before any rerun')
    cases = json.loads((ROOT / 'inputs.json').read_text())
    if len(cases) != 128 or len({c['task_id'] for c in cases}) != 128:
        raise ValueError('Smoke must contain128 distinct already-captured tasks')
    atomic_json(ROOT / 'started.json', {'at': time.time(), 'function_call_id': modal.current_function_call_id(),
        'manifest_sha256': file_sha(ROOT / 'manifest.json')})
    atomic_json(ROOT / 'runtime.json', {p: version(p) for p in ['sglang', 'transformers', 'torch', 'openai']})
    volume.commit()
    processes = [None] * 8
    clients = [None] * 8
    overall_start = time.monotonic()
    deadline = overall_start + 3300
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    results = []
    def launch(index):
        command = ['python3', '-m', 'sglang.launch_server', '--model-path', MODEL,
            '--revision', REVISION, '--served-model-name', MODEL, '--host', '127.0.0.1',
            '--port', str(8000 + index), '--tp', '1', '--dtype', 'bfloat16',
            '--context-length', '32768', '--mem-fraction-static', '0.80',
            '--max-running-requests', '16', '--cuda-graph-max-bs', '16',
            '--max-mamba-cache-size', '80', '--chunked-prefill-size', '32768',
            '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(index)}
        atomic_json(ROOT / f'server-{index}.command.json', {'command': command, 'CUDA_VISIBLE_DEVICES': str(index)})
        start = time.monotonic()
        last_progress = start
        with (ROOT / f'server-{index}.log').open('w') as log:
            processes[index] = subprocess.Popen(command, env=env, cwd='/opt/lclm',
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            if processes[index].poll() is not None:
                raise RuntimeError(f'Replica{index} exited; inspect server-{index}.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{8000 + index}/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > min(deadline, start + 1200):
                raise TimeoutError(f'Replica{index} startup timeout')
            if time.monotonic() - last_progress >= 60:
                lines = (ROOT / f'server-{index}.log').read_text(errors='replace').splitlines()
                print(json.dumps({'replica': index, 'startup_seconds': time.monotonic() - start,
                    'recent_log': lines[-2:]}), flush=True)
                last_progress = time.monotonic()
            time.sleep(2)
        clients[index] = OpenAI(api_key='not-needed', base_url=f'http://127.0.0.1:{8000 + index}/v1', timeout=300, max_retries=0)
        lines = graph_capture_lines((ROOT / f'server-{index}.log').read_text())
        if not lines:
            raise RuntimeError(f'Replica{index} ready but CUDA graph capture not confirmed in log')
        report = {'replica': index, 'startup_seconds': time.monotonic() - start,
            'graph_capture_lines': lines[-12:], 'ready': True}
        atomic_json(ROOT / f'server-{index}.ready.json', report)
        print(json.dumps({'replica': index, 'startup_seconds': report['startup_seconds'], 'ready': True}), flush=True)
        return report
    def request(case, index, phase):
        start = time.monotonic()
        row = {'task_id': case['task_id'], 'source': case['source'], 'replica': index, 'phase': phase,
               'request': case['request'], 'captured_raw_sha256': case['captured_raw_sha256'],
               'started_at': time.time()}
        stage = 'http_request'
        try:
            response = clients[index].chat.completions.create(**case['request'])
            payload = response.model_dump(mode='json')
            row['response'] = payload
            stage = 'raw_token_capture'
            raw_text = tokenizer.decode(response_token_ids(payload['choices'][0]), skip_special_tokens=False)
            row['raw_generated_text'] = raw_text
            row['generated_reasoning_markers'] = bool(re.search(r'</?(?:think|analysis)>', raw_text, re.I))
            stage = 'native_parser_validation'
            inspection = inspect_response(payload, set(case['valid_segment_ids']))
            inspection.pop('output_token_ids')
            row['inspection'] = inspection
        except Exception as exc:
            row['error'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc),
                'status_code': getattr(exc, 'status_code', None), 'body': getattr(exc, 'body', None)}
        row['seconds'] = time.monotonic() - start
        row['finished_at'] = time.time()
        atomic_json(ROOT / 'responses' / (case['task_id'] + '.json'), row)
        return row
    try:
        first_server = launch(0)
        boot = request(cases[0], 0, 'single_replica_boot_check')
        results.append(boot)
        atomic_json(ROOT / 'boot-check.json', boot)
        volume.commit()
        if boot.get('error'):
            raise RuntimeError('TP1 boot request failed; remaining127 replays were not submitted')
        with ThreadPoolExecutor(max_workers=7) as pool:
            servers = [first_server, *list(pool.map(launch, range(1, 8)))]
        volume.commit()
        startup_seconds = time.monotonic() - overall_start
        atomic_json(ROOT / 'all-ready.json', {'startup_and_boot_check_seconds': startup_seconds, 'servers': servers})
        volume.commit()
        wave_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=128) as pool:
            futures = [pool.submit(request, case, number % 8, 'eight_replica_wave')
                       for number, case in enumerate(cases[1:], 1)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.monotonic() - wave_start
        wave = [r for r in results if r['phase'] == 'eight_replica_wave']
        report = {'status': 'complete_serving_smoke', 'layout': manifest['layout'],
            'manifest_sha256': file_sha(ROOT / 'manifest.json'), 'total_replayed_requests': len(results),
            'boot_check_seconds': boot['seconds'], 'startup_and_boot_check_seconds': startup_seconds,
            'eight_replica_wave': report_requests(wave, elapsed),
            'all_requests': report_requests(results, time.monotonic() - overall_start),
            'result_sha256': {r['task_id']: file_sha(ROOT / 'responses' / (r['task_id'] + '.json')) for r in results},
            'training_rows_generated': 0, 'approved_for_release': False, 'limitations': manifest['comparison_limit']}
        if len(results) != 128:
            raise RuntimeError('Replay accounting incomplete')
        atomic_json(ROOT / 'report.json', report)
        volume.commit()
        return report
    except Exception as exc:
        atomic_json(ROOT / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'completed_replays': len(results)})
        volume.commit()
        raise
    finally:
        for process in processes:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        volume.commit()
