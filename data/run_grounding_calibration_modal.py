"""Bounded 58-row diagnostic only. Deploy, inspect status, then spawn once.

No source replacement, training data rewrite or full-source option is exposed.
"""
import hashlib
import json
import os
from pathlib import Path

import modal
from data.generate_real_expansion_modal import (
    image, data_volume, hf_cache_volume, CACHE_ROOT, PROJECT_ROOT,
    MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME)

APP_NAME = 'lclm-grounding-claims-pilot-v3'
app = modal.App(APP_NAME)
ROOT = Path('/data/stage3-build-20260906')
INPUT = ROOT/'grounding-calibration-v1'
OUTPUT = ROOT/'grounding-calibration-claims-v3'
EXPECTED_SHA = 'f1caf75d1e4716aaa8cb49c710b13cd1fa15726fdf6103405dbc4ae3dc0a2eac'


def load_inputs():
    from data.grounding_claim_review import primary_evidence, answer_sentences
    manifest_bytes = (INPUT/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes); content = (INPUT/'candidates.jsonl').read_bytes()
    if (manifest['status'] != 'prepared' or manifest['rows'] != 58
            or manifest['selected_rows_sha256'] != EXPECTED_SHA
            or hashlib.sha256(content).hexdigest() != EXPECTED_SHA):
        raise ValueError('Diagnostic input hash/count mismatch')
    rows = [json.loads(line) for line in content.splitlines()]
    ids = {r['task_id'] for r in rows}; expected = {r['task_id'] for r in manifest['candidates']}
    controls = manifest['calibration_controls']
    if (len(rows) != 58 or len(ids) != 58 or ids != expected or len(expected) != len(manifest['candidates'])
            or len(controls) != 4 or not set(controls) <= ids
            or sorted(r['expected_keep'] for r in controls.values()) != [False, False, True, True]):
        raise ValueError('Diagnostic membership/control mismatch')
    for row in rows:
        primary_evidence(row); answer_sentences(row)
    return rows, manifest, hashlib.sha256(manifest_bytes).hexdigest()


@app.function(image=image, gpu='H200:8', cpu=16, memory=65536, timeout=7200,
              max_containers=1, scaledown_window=60,
              volumes={'/data': data_volume, CACHE_ROOT: hf_cache_volume},
              secrets=[modal.Secret.from_name('huggingface')])
def pilot():
    import subprocess
    import time
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from data.grounding_claim_review import VERSION, review_claims
    data_volume.reload()
    rows, inputs, input_manifest_sha = load_inputs()
    manifest = {'version': VERSION, 'rows': 58, 'input_manifest_sha256': input_manifest_sha,
                'input_rows_sha256': EXPECTED_SHA, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                'protocol_sha256': hashlib.sha256((PROJECT_ROOT/'data/grounding_claim_review.py').read_bytes()).hexdigest(),
                'temperature': 0, 'max_tokens': 1536, 'concurrency': 8, 'enable_thinking': False,
                'training_rows_modified': False, 'control_labels_sent_to_judge': False}
    OUTPUT.mkdir(exist_ok=True); manifest_path = OUTPUT/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Changed diagnostic protocol/input: use a new output version')
    else:
        if list(OUTPUT.iterdir()): raise ValueError('Unexpected partial diagnostic output')
        manifest_path.write_text(json.dumps(manifest, indent=2)); data_volume.commit()
    completed = {}; path = OUTPUT/'decisions.jsonl'; ids = {r['task_id'] for r in rows}
    if path.exists():
        for line in path.read_text().splitlines():
            result = json.loads(line)
            if result['task_id'] in completed or result['task_id'] not in ids:
                raise ValueError('Invalid diagnostic checkpoint IDs')
            completed[result['task_id']] = result
    pending = [r for r in rows if r['task_id'] not in completed]
    if pending:
        command = ['vllm', 'serve', MODEL_ID, '--revision', MODEL_REVISION,
                   '--served-model-name', SERVED_MODEL_NAME, '--host', '127.0.0.1', '--port', '8000',
                   '--tensor-parallel-size', '8', '--max-model-len', '32768',
                   '--gpu-memory-utilization', '0.90', '--safetensors-load-strategy', 'prefetch',
                   '--enforce-eager', '--uvicorn-log-level', 'warning']
        process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        try:
            deadline = time.monotonic()+5400
            while True:
                if process.poll() is not None: raise RuntimeError('Diagnostic model server exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                        if response.status == 200: break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() > deadline: raise TimeoutError('Diagnostic server startup timed out')
                time.sleep(5)
            client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=2)
            def complete(messages):
                response = client.chat.completions.create(model=SERVED_MODEL_NAME, messages=messages,
                    temperature=0, max_tokens=1536, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                choice = response.choices[0]
                if choice.finish_reason != 'stop': raise ValueError('Incomplete diagnostic judge response')
                return choice.message.content
            def run(row):
                try: return review_claims(row, complete)
                except Exception as exc:
                    return {'task_id': row['task_id'], 'keep': False, 'error': type(exc).__name__+': '+str(exc)}
            with ThreadPoolExecutor(max_workers=8) as pool, path.open('a') as stream:
                for future in as_completed([pool.submit(run, row) for row in pending]):
                    result = future.result(); completed[result['task_id']] = result
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    data_volume.commit(); print('Diagnostic completed', len(completed), 'of 58', flush=True)
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=30)
    calibration = {task_id: {'expected_keep': control['expected_keep'],
                   'actual_keep': completed[task_id]['keep'],
                   'passed': 'error' not in completed[task_id] and completed[task_id]['keep'] == control['expected_keep']}
                   for task_id, control in inputs['calibration_controls'].items()}
    report = {'status': 'complete', 'manifest': manifest, 'rows': len(completed),
              'kept': sum(r['keep'] for r in completed.values()),
              'errors': sum('error' in r for r in completed.values()), 'controls': calibration,
              'calibration_passed': all(r['passed'] for r in calibration.values()),
              'approved_for_full_source_review': False, 'approved_for_release': False,
              'scope': 'Diagnostic decisions only. Manual control and sampled keep/reject inspection required before scale. Original traces unchanged.'}
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2)); data_volume.commit()
    return report
