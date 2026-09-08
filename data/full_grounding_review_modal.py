"""Review every previously accepted held-source row, without modifying datasets."""
import hashlib
import json
import os
from pathlib import Path
import modal
from data.generate_real_expansion_modal import (
    image, data_volume, hf_cache_volume, CACHE_ROOT, PROJECT_ROOT, MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME)
from data.stage3_full_modal import image as cpu_image
from data.grounding_full_review_gate import TARGETS, validate_scale_approval

ROOT = Path('/data/stage3-build-20260906')
CALIBRATION = ROOT/'grounding-calibration-claims-v3'
GENERATED = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
OUTPUT = ROOT/'full-grounding-review-v3'
APP_NAME = 'lclm-held-sources-grounding-review-v3'
app = modal.App(APP_NAME)


def load_job():
    def read(path): return json.loads(path.read_text())
    report = read(CALIBRATION/'report.json')
    decision_bytes = (CALIBRATION/'decisions.jsonl').read_bytes()
    decisions = [json.loads(line) for line in decision_bytes.splitlines()]
    review = read(CALIBRATION/'manual-review.json')
    protocol_sha = hashlib.sha256((PROJECT_ROOT/'data/grounding_claim_review.py').read_bytes()).hexdigest()
    decision_sha = hashlib.sha256(decision_bytes).hexdigest()
    validate_scale_approval(report, decisions, review, read(CALIBRATION/'manual-samples.json'), decision_sha, protocol_sha)
    items = []; seen = set(); sources = []
    for source, count in TARGETS.items():
        audit = read(ROOT/'full-expansion-format-audit'/f'{source}.json')
        content = (GENERATED/f'{source}.accepted.jsonl').read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if audit['status'] != 'passed' or audit['rows'] != count or digest != audit['accepted_file_sha256']:
            raise ValueError('Changed/incomplete held source input: '+source)
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != count: raise ValueError('Held source count mismatch')
        for row in rows:
            if row['task_id'] in seen or row['verification']['accepted'] is not True:
                raise ValueError('Duplicate/unverified held source row')
            seen.add(row['task_id']); items.append((source, row))
        sources.append({'source': source, 'rows': count, 'accepted_file_sha256': digest,
                        'path': str(GENERATED/f'{source}.accepted.jsonl')})
    manifest = {'version': 'full-grounding-review-v3', 'candidate_rows': len(items), 'sources': sources,
                'protocol_sha256': protocol_sha, 'calibration_decisions_sha256': decision_sha,
                'manual_review_sha256': hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest(),
                'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'enable_thinking': False,
                'temperature': 0, 'max_tokens': 1536, 'concurrency': 8,
                'training_rows_modified': False, 'purpose': 'Review decisions only; no dataset publication or filtering yet'}
    return items, manifest


@app.function(image=cpu_image, cpu=2, memory=16384, timeout=3600, volumes={'/data': data_volume})
def preflight():
    items, manifest = load_job()
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'preflight.json').write_text(json.dumps(manifest, indent=2)); data_volume.commit()
    return {'status': 'passed', 'rows': len(items), 'manifest': manifest}


@app.function(image=image, gpu='H200:8', cpu=16, memory=65536, timeout=43200,
              max_containers=1, scaledown_window=60,
              volumes={'/data': data_volume, CACHE_ROOT: hf_cache_volume},
              secrets=[modal.Secret.from_name('huggingface')])
def full():
    import subprocess
    import time
    import urllib.error
    import urllib.request
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from data.grounding_claim_review import review_claims
    data_volume.reload(); items, manifest = load_job()
    OUTPUT.mkdir(exist_ok=True); manifest_path = OUTPUT/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Changed full-review inputs/protocol; do not mix output versions')
    decisions_path = OUTPUT/'decisions.jsonl'
    if not manifest_path.exists():
        if decisions_path.exists(): raise ValueError('Unbound partial review decisions')
        manifest_path.write_text(json.dumps(manifest, indent=2)); data_volume.commit()
    expected = {row['task_id']: source for source, row in items}; completed = {}
    if decisions_path.exists():
        with decisions_path.open() as stream:
            for line in stream:
                if not line.endswith('\n'): raise ValueError('Truncated review checkpoint')
                result = json.loads(line); task_id = result['task_id']
                if (task_id in completed or expected.get(task_id) != result.get('source')
                        or type(result.get('keep')) is not bool or (result['keep'] and 'error' in result)):
                    raise ValueError('Invalid review checkpoint')
                completed[task_id] = result
    def snapshot(status):
        counts = {s: Counter() for s in TARGETS}
        for result in completed.values():
            counts[result['source']]['error' if 'error' in result else 'kept' if result['keep'] else 'rejected'] += 1
        return {'status': status, 'manifest': manifest, 'reviewed': len(completed),
                'remaining': len(items)-len(completed), 'counts': {s: dict(c) for s, c in counts.items()},
                'approved_for_release': False, 'training_rows_modified': False}
    pending = [(source, row) for source, row in items if row['task_id'] not in completed]
    if pending:
        process = subprocess.Popen(['vllm', 'serve', MODEL_ID, '--revision', MODEL_REVISION,
            '--served-model-name', SERVED_MODEL_NAME, '--host', '127.0.0.1', '--port', '8000',
            '--tensor-parallel-size', '8', '--max-model-len', '32768', '--gpu-memory-utilization', '0.90',
            '--safetensors-load-strategy', 'prefetch', '--enforce-eager', '--uvicorn-log-level', 'warning'], cwd=PROJECT_ROOT)
        try:
            deadline = time.monotonic()+5400
            while True:
                if process.poll() is not None: raise RuntimeError('Review server exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                        if response.status == 200: break
                except (urllib.error.URLError, TimeoutError, ConnectionError): pass
                if time.monotonic() > deadline: raise TimeoutError('Review server startup timeout')
                time.sleep(5)
            client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=2)
            def complete(messages):
                response = client.chat.completions.create(model=SERVED_MODEL_NAME, messages=messages,
                    temperature=0, max_tokens=1536, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                if response.choices[0].finish_reason != 'stop': raise ValueError('Incomplete judge response')
                return response.choices[0].message.content
            def run(item):
                source, row = item
                try: return {'source': source, **review_claims(row, complete)}
                except Exception as exc:
                    return {'source': source, 'task_id': row['task_id'], 'keep': False,
                            'error': type(exc).__name__+': '+str(exc)}
            with ThreadPoolExecutor(max_workers=8) as pool, decisions_path.open('a') as stream:
                for future in as_completed([pool.submit(run, item) for item in pending]):
                    result = future.result(); completed[result['task_id']] = result
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                    if len(completed) % 50 == 0:
                        stream.flush(); os.fsync(stream.fileno())
                        (OUTPUT/'progress.json').write_text(json.dumps(snapshot('running'), indent=2)); data_volume.commit()
                        print('Reviewed', len(completed), 'of', len(items), flush=True)
                stream.flush(); os.fsync(stream.fileno())
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=30)
    if set(completed) != set(expected): raise ValueError('Incomplete full review')
    report = snapshot('complete')
    report['decisions_sha256'] = hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2)); data_volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(preflight.remote(), indent=2))
