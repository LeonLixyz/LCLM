"""Bounded 88-row diagnostic; no full-source, filtering, or publication option."""
import hashlib
import json
import os
from pathlib import Path
import modal
from data.generate_real_expansion_modal import (
    image, data_volume, hf_cache_volume, CACHE_ROOT, PROJECT_ROOT, MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME)
from data.stage3_full_modal import image as cpu_image

ROOT = Path('/data/stage3-build-20260906')
BASE = ROOT/'label-grounding-calibration-v1'
INPUT = ROOT/'question-grounding-calibration-v3'
OUTPUT = ROOT/'question-grounding-claims-v1'
MAIN = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v6')
MD = MAIN.parent/'multidoc2dial-chronological-v1-full'
BASE_SHA = 'd4006192bc68b5c15e39c78b9da65216cb394d3af6f40fbf5d748e1a272992a9'
MD_SHA = 'a99104742ebbe657963ba58cb976c0eb223c7fbc71b48ce7d060f8b389d44bfa'
MD_CONTROLS = {'rea4-md2d-873fbe82c90af128e0f9775e': False, 'rea4-md2d-6dbabd8d53116f6cfd91c521': True}
app = modal.App('lclm-question-grounding-pilot-v1')


def sha(content): return hashlib.sha256(content).hexdigest()


def load_inputs():
    from data.grounding_claim_review import primary_evidence, answer_sentences
    base_manifest = json.loads((BASE/'manifest.json').read_text())
    content = (BASE/'candidates.jsonl').read_bytes()
    if (sha(content) != BASE_SHA or base_manifest['selected_rows_sha256'] != BASE_SHA
            or base_manifest['status'] != 'prepared' or base_manifest['rows'] != 70):
        raise ValueError('Changed base diagnostic input')
    selected = {}; membership = {}; controls = dict(base_manifest['calibration_controls'])
    for line in content.splitlines(keepends=True):
        row = json.loads(line); task_id = row['task_id']
        if not line.endswith(b'\n') or task_id in selected: raise ValueError('Invalid base diagnostic row')
        selected[task_id] = line
    for entry in base_manifest['candidates']:
        if entry['task_id'] in membership: raise ValueError('Duplicate diagnostic membership')
        membership[entry['task_id']] = entry
    if (set(selected) != set(membership) or len(selected) != 70 or len(controls) != 6
            or not set(controls) <= set(selected)
            or sorted(x['expected_keep'] for x in controls.values()) != [False]*3+[True]*3):
        raise ValueError('Base diagnostic membership/control mismatch')
    receipts = base_manifest['sources']
    if len(receipts) != 2 or {r['source']: r['selected_rows'] for r in receipts} != {'maud': 18, 'tatqa': 52}:
        raise ValueError('Unexpected base source coverage')
    for receipt in receipts:
        review = json.loads((ROOT/'full-expansion-manual-review-samples'/f"{receipt['source']}.review.json").read_text())
        if (review['status'] != 'sample_review_failed'
                or sha(json.dumps(review, sort_keys=True).encode()) != receipt['manual_review_sha256']):
            raise ValueError('Changed source hold')
        known = review['reviewed_examples'] + review.get('additional_reviewed_examples', [])
        source_controls = {r['task_id']: {'source': receipt['source'], 'expected_keep': r['result'] == 'pass',
                                        'manual_evidence': r['evidence']} for r in known}
        if source_controls != {k: c for k, c in controls.items() if c['source'] == receipt['source']}:
            raise ValueError('Base control expectations differ from manual reviews')
        digest = hashlib.sha256()
        with (MAIN/f"{receipt['source']}.accepted.jsonl").open('rb') as stream:
            while chunk := stream.read(8*1024*1024): digest.update(chunk)
        if digest.hexdigest() != receipt['input_sha256']: raise ValueError('Changed original source bytes')
    review = json.loads((MD/'release-review.json').read_text())
    audit = json.loads((MD/'format-audit/multidoc2dial.json').read_text())
    expected = {r['task_id']: r for r in review['reviewed_examples']}
    if (review['approved_for_release'] is not False or review['status'] != 'sample_review_failed'
            or audit['status'] != 'passed' or audit['rows'] != 16158
            or audit['accepted_file_sha256'] != MD_SHA or review['accepted_file_sha256'] != MD_SHA
            or {k: r['result'] == 'pass' for k, r in expected.items()} != MD_CONTROLS):
        raise ValueError('Missing/stale corrected MD controls')
    digest = hashlib.sha256(); seen = set(); buckets = {'single': [], 'multi': []}
    with (MD/'multidoc2dial.accepted.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line); row = json.loads(line); task_id = row['task_id']
            if not line.endswith(b'\n') or task_id in seen or row['verification']['accepted'] is not True:
                raise ValueError('Invalid corrected source row')
            seen.add(task_id)
            if task_id not in MD_CONTROLS:
                segments = {c['function']['arguments']['segment_id'] for m in row['messages'] for c in m.get('tool_calls', [])}
                category = 'single' if len(segments) == 1 else 'multi'
                score = sha(('question-calibration-v1:'+task_id).encode())
                buckets[category].append((score, task_id, line)); buckets[category].sort()
                if len(buckets[category]) > 8: buckets[category].pop()
            if task_id in MD_CONTROLS:
                if task_id in selected: raise ValueError('Duplicate cross-source control')
                selected[task_id] = line
                membership[task_id] = {'task_id': task_id, 'source': 'multidoc2dial', 'category': expected[task_id]['category']}
    if digest.hexdigest() != MD_SHA or len(seen) != 16158 or not set(MD_CONTROLS) <= set(selected):
        raise ValueError('Corrected source bytes/count/control mismatch')
    for category, bucket in buckets.items():
        if len(bucket) != 8: raise ValueError('Missing fresh corrected-source sample coverage')
        for _, task_id, line in bucket:
            if task_id in selected: raise ValueError('Duplicate fresh source sample')
            selected[task_id] = line
            membership[task_id] = {'task_id': task_id, 'source': 'multidoc2dial', 'category': category}
    controls.update({k: {'source': 'multidoc2dial', 'expected_keep': value,
                        'manual_evidence': expected[k]['evidence']} for k, value in MD_CONTROLS.items()})
    rows_bytes = b''.join(selected[k] for k in sorted(selected))
    rows = [json.loads(selected[k]) for k in sorted(selected)]
    if len(rows) != 88: raise ValueError('Diagnostic must have exactly 88 rows')
    for row in rows: primary_evidence(row); answer_sentences(row)
    manifest = {'status': 'prepared', 'version': 'question-grounding-calibration-v3', 'rows': 88,
        'selected_rows_sha256': sha(rows_bytes), 'parent_rows_sha256': BASE_SHA,
        'parent_manifest_sha256': sha((BASE/'manifest.json').read_bytes()),
        'sources': receipts + [{'source': 'multidoc2dial', 'selected_rows': 18,
            'input_sha256': MD_SHA, 'accepted_rows_scanned': 16158,
            'manual_review_sha256': sha(json.dumps(review, sort_keys=True).encode())}],
        'candidates': [membership[k] for k in sorted(membership)], 'calibration_controls': controls,
        'approved_for_release': False,
        'selection': 'Original 70 MAUD/TAT-QA diagnostic rows plus two corrected-MD controls and eight new SHA256(question-calibration-v1:task_id) samples per single/multi bucket, excluding controls',
        'instruction_separation': 'Control expectations and references are evaluator metadata only; never sent to judge or training.'}
    return rows, manifest, rows_bytes


@app.function(image=cpu_image, cpu=2, memory=16384, timeout=3600, volumes={'/data': data_volume})
def preflight():
    from transformers import AutoTokenizer
    from data.stage3_tokenizers import DECODER, DECODER_REVISION
    from data.question_grounding_review import primary_evidence, answer_sentences, messages, CLAIM_INSTRUCTION, FIT_INSTRUCTION
    rows, manifest, content = load_inputs()
    tokenizer = AutoTokenizer.from_pretrained(DECODER, revision=DECODER_REVISION)
    largest = 0
    for row in rows:
        answer, sentences = answer_sentences(row); evidence = primary_evidence(row)
        prompts = [messages(CLAIM_INSTRUCTION, row['task'], answer, evidence, s) for s in sentences]
        prompts.append(messages(FIT_INSTRUCTION, row['task'], answer, evidence))
        for prompt in prompts:
            largest = max(largest, len(tokenizer.apply_chat_template(prompt, tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)))
    if largest + 2048 > 32768: raise ValueError('Diagnostic prompt exceeds teacher context budget')
    INPUT.mkdir(exist_ok=True)
    if (INPUT/'manifest.json').exists():
        if json.loads((INPUT/'manifest.json').read_text()) != manifest or (INPUT/'candidates.jsonl').read_bytes() != content:
            raise ValueError('Prepared diagnostic differs; use new version')
    else:
        if list(INPUT.iterdir()): raise ValueError('Partial diagnostic needs inspection')
        (INPUT/'candidates.jsonl').write_bytes(content)
        (INPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)); data_volume.commit()
    result = {'status': 'passed', 'rows': len(rows), 'controls': len(manifest['calibration_controls']),
        'selected_rows_sha256': manifest['selected_rows_sha256'], 'max_prompt_tokens': largest,
        'decoder_tokenizer_revision': DECODER_REVISION,
        'protocol_files': {f: sha((PROJECT_ROOT/'data'/f).read_bytes()) for f in
                           ('question_grounding_review.py', 'grounding_claim_review.py')}}
    (INPUT/'preflight.json').write_text(json.dumps(result, indent=2)); data_volume.commit()
    return result


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
    from data.question_grounding_review import VERSION, review_question
    data_volume.reload(); rows, inputs, content = load_inputs()
    if json.loads((INPUT/'manifest.json').read_text()) != inputs or (INPUT/'candidates.jsonl').read_bytes() != content:
        raise ValueError('Prepared CPU preflight required')
    manifest = {'version': VERSION, 'rows': 88, 'input_rows_sha256': inputs['selected_rows_sha256'],
        'input_manifest_sha256': sha((INPUT/'manifest.json').read_bytes()),
        'model': MODEL_ID, 'model_revision': MODEL_REVISION,
        'protocol_files': {f: sha((PROJECT_ROOT/'data'/f).read_bytes()) for f in
                           ('question_grounding_review.py', 'grounding_claim_review.py')},
        'temperature': 0, 'max_tokens': 2048, 'concurrency': 8, 'enable_thinking': False,
        'training_rows_modified': False, 'control_labels_sent_to_judge': False,
        'reference_answers_sent_to_judge': False}
    preflight_report = json.loads((INPUT/'preflight.json').read_text())
    if (preflight_report['status'] != 'passed' or preflight_report['rows'] != 88
            or preflight_report['selected_rows_sha256'] != inputs['selected_rows_sha256']
            or preflight_report['protocol_files'] != manifest['protocol_files']
            or preflight_report['max_prompt_tokens']+2048 > 32768):
        raise ValueError('Stale/failed tokenizer protocol preflight')
    OUTPUT.mkdir(exist_ok=True); manifest_path = OUTPUT/'manifest.json'; path = OUTPUT/'decisions.jsonl'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest: raise ValueError('Changed diagnostic version')
    else:
        if list(OUTPUT.iterdir()): raise ValueError('Partial output without manifest')
        manifest_path.write_text(json.dumps(manifest, indent=2)); data_volume.commit()
    completed = {}; expected = {r['task_id'] for r in rows}
    if path.exists():
        for line in path.read_text().splitlines(keepends=True):
            r = json.loads(line)
            if (not line.endswith('\n') or r['task_id'] in completed or r['task_id'] not in expected
                    or type(r.get('keep')) is not bool or (r['keep'] and 'error' in r)):
                raise ValueError('Invalid diagnostic checkpoint')
            completed[r['task_id']] = r
    pending = [r for r in rows if r['task_id'] not in completed]
    if pending:
        process = subprocess.Popen(['vllm', 'serve', MODEL_ID, '--revision', MODEL_REVISION,
            '--served-model-name', SERVED_MODEL_NAME, '--host', '127.0.0.1', '--port', '8000',
            '--tensor-parallel-size', '8', '--max-model-len', '32768', '--gpu-memory-utilization', '0.90',
            '--safetensors-load-strategy', 'prefetch', '--enforce-eager', '--uvicorn-log-level', 'warning'], cwd=PROJECT_ROOT)
        try:
            deadline = time.monotonic()+5400
            while True:
                if process.poll() is not None: raise RuntimeError('Diagnostic server exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5) as response:
                        if response.status == 200: break
                except (urllib.error.URLError, TimeoutError, ConnectionError): pass
                if time.monotonic() > deadline: raise TimeoutError('Diagnostic server startup timeout')
                time.sleep(5)
            client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=2)
            def complete(messages):
                response = client.chat.completions.create(model=SERVED_MODEL_NAME, messages=messages,
                    temperature=0, max_tokens=2048, extra_body={'chat_template_kwargs': {'enable_thinking': False}})
                if response.choices[0].finish_reason != 'stop': raise ValueError('Incomplete diagnostic answer')
                return response.choices[0].message.content
            def run(row):
                try: return review_question(row, complete)
                except Exception as exc:
                    return {'task_id': row['task_id'], 'keep': False, 'error': type(exc).__name__+': '+str(exc)}
            with ThreadPoolExecutor(max_workers=8) as pool, path.open('a') as stream:
                for future in as_completed([pool.submit(run, row) for row in pending]):
                    r = future.result(); completed[r['task_id']] = r
                    stream.write(json.dumps(r, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    data_volume.commit(); print('Diagnostic reviewed', len(completed), 'of 88', flush=True)
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=30)
    if set(completed) != expected: raise ValueError('Incomplete diagnostic result set')
    controls = {k: {'expected_keep': c['expected_keep'], 'actual_keep': completed[k]['keep'],
                   'passed': 'error' not in completed[k] and completed[k]['keep'] == c['expected_keep']}
                for k, c in inputs['calibration_controls'].items()}
    report = {'status': 'complete', 'manifest': manifest, 'rows': len(completed),
        'kept': sum(r['keep'] for r in completed.values()), 'errors': sum('error' in r for r in completed.values()),
        'controls': controls, 'calibration_passed': all(c['passed'] for c in controls.values()),
        'decisions_sha256': sha(path.read_bytes()), 'approved_for_full_source_review': False,
        'approved_for_release': False, 'scope': 'Bounded diagnostic only; manual review required before scale'}
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2)); data_volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(preflight.remote(), indent=2))
