"""Separate 88-row V2 diagnostic plus three V1 format probes; no scale option."""
import json
import os
from pathlib import Path
import modal
from data.run_question_grounding_calibration_modal import (
    load_inputs, INPUT, ROOT, sha, image, cpu_image, data_volume, hf_cache_volume,
    CACHE_ROOT, PROJECT_ROOT, MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME)

OUTPUT = ROOT/'question-grounding-structured-v2'
V1 = ROOT/'question-grounding-claims-v1'
V1_DECISIONS_SHA = 'cbb5c41d70f514e40412ce2cb21b9af6a2bdaf8f8cfc1154c10fe2b2a9ba9830'
FILES = ('question_grounding_structured.py', 'question_grounding_review.py',
         'grounding_claim_review.py', 'run_question_grounding_calibration_modal.py',
         'run_question_structured_calibration_modal.py')
app = modal.App('lclm-question-grounding-structured-v2')


def verified_inputs():
    rows, inputs, content = load_inputs()
    if ((INPUT/'candidates.jsonl').read_bytes() != content
            or json.loads((INPUT/'manifest.json').read_text()) != inputs):
        raise ValueError('Original immutable 88-row inputs differ')
    original = (V1/'decisions.jsonl').read_bytes()
    if sha(original) != V1_DECISIONS_SHA: raise ValueError('Changed V1 diagnostic decisions')
    decisions = {r['task_id']: r for r in map(json.loads, original.splitlines())}
    probes = sorted(k for k in inputs['calibration_controls'] if 'error' in decisions[k])[:3]
    if len(probes) != 3: raise ValueError('Missing V1 error probes')
    return rows, inputs, probes


def identity(inputs, probes):
    from data.question_grounding_structured import VERSION
    return {'version': VERSION, 'rows': 88, 'input_rows_sha256': inputs['selected_rows_sha256'],
        'input_manifest_sha256': sha((INPUT/'manifest.json').read_bytes()),
        'protocol_files': {f: sha((PROJECT_ROOT/'data'/f).read_bytes()) for f in FILES},
        'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'temperature': 0,
        'max_tokens': 2048, 'concurrency': 8, 'enable_thinking': False,
        'response_format': 'json_schema', 'v1_probe_ids': probes, 'v1_decisions_sha256': V1_DECISIONS_SHA,
        'training_rows_modified': False, 'control_labels_sent_to_judge': False,
        'reference_answers_sent_to_judge': False, 'raw_responses_saved': True}


@app.function(image=cpu_image, cpu=2, memory=16384, timeout=3600, volumes={'/data': data_volume})
def preflight():
    import subprocess
    from transformers import AutoTokenizer
    from data.stage3_tokenizers import DECODER, DECODER_REVISION
    from data.question_grounding_structured import request_messages
    from data.question_grounding_review import messages, CLAIM_INSTRUCTION, FIT_INSTRUCTION
    from data.grounding_claim_review import primary_evidence, answer_sentences
    subprocess.run(['python', '-m', 'pytest', '-q',
        'tests/test_question_grounding_structured.py', 'tests/test_question_grounding_review.py',
        'tests/test_grounding_claim_review.py', 'tests/test_grounding_full_review_gate.py',
        'tests/test_expansion_release_selection.py', 'tests/test_accounting_answer_review.py',
        'tests/test_stage3_release_checks.py', 'tests/test_multidoc2dial_dialogue.py'],
        cwd=PROJECT_ROOT, check=True)
    data_volume.reload(); rows, inputs, probes = verified_inputs(); manifest = identity(inputs, probes)
    tokenizer = AutoTokenizer.from_pretrained(DECODER, revision=DECODER_REVISION)
    prompts = [request_messages(row) for row in rows]
    for row in rows:
        if row['task_id'] not in probes: continue
        answer, sentences = answer_sentences(row); evidence = primary_evidence(row)
        prompts.extend(messages(CLAIM_INSTRUCTION, row['task'], answer, evidence, s) for s in sentences)
        prompts.append(messages(FIT_INSTRUCTION, row['task'], answer, evidence))
    maximum = max(len(tokenizer.apply_chat_template(m, tokenize=True,
                      add_generation_prompt=True, enable_thinking=False)) for m in prompts)
    if maximum+2048 > 32768: raise ValueError('Prompt budget exceeded')
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/'manifest.json').exists():
        if json.loads((OUTPUT/'manifest.json').read_text()) != manifest:
            raise ValueError('Changed diagnostic; create a new version')
    else:
        if list(OUTPUT.iterdir()): raise ValueError('Partial output without manifest')
        (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = {'status': 'passed', 'manifest': manifest, 'controls': 8,
              'max_prompt_tokens': maximum, 'decoder_tokenizer_revision': DECODER_REVISION}
    (OUTPUT/'preflight.json').write_text(json.dumps(report, indent=2)); data_volume.commit()
    return report


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
    from data.question_grounding_structured import review_structured
    from data.question_grounding_review import review_question
    data_volume.reload(); rows, inputs, probes = verified_inputs(); manifest = identity(inputs, probes)
    preflight_report = json.loads((OUTPUT/'preflight.json').read_text())
    if (json.loads((OUTPUT/'manifest.json').read_text()) != manifest
            or preflight_report['status'] != 'passed' or preflight_report['manifest'] != manifest
            or preflight_report['max_prompt_tokens']+2048 > 32768):
        raise ValueError('Missing/stale preflight')
    raw_dir = OUTPUT/'raw-responses'; raw_dir.mkdir(exist_ok=True)
    path = OUTPUT/'decisions.jsonl'; expected = {r['task_id'] for r in rows}; completed = {}
    if path.exists():
        for line in path.read_text().splitlines(keepends=True):
            r = json.loads(line)
            if (not line.endswith('\n') or r['task_id'] not in expected or r['task_id'] in completed
                    or type(r.get('keep')) is not bool or ('error' in r and r['keep'])
                    or sha((raw_dir/(r['task_id']+'.json')).read_bytes()) != r['raw_responses_sha256']):
                raise ValueError('Invalid diagnostic checkpoint')
            completed[r['task_id']] = r
    pending = [r for r in rows if r['task_id'] not in completed]
    probe_pending = [r for r in rows if r['task_id'] in probes
                     and not (OUTPUT/('v1-probe-'+r['task_id']+'.json')).exists()]
    if pending or probe_pending:
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
                if time.monotonic() > deadline: raise TimeoutError('Diagnostic startup timeout')
                time.sleep(5)
            client = OpenAI(api_key='not-needed', base_url='http://127.0.0.1:8000/v1', timeout=300, max_retries=2)
            def run(row, probe=False):
                records = []
                def complete(messages, schema=None):
                    request = {'model': SERVED_MODEL_NAME, 'messages': messages, 'temperature': 0,
                        'max_tokens': 2048, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
                    if schema is not None:
                        request['response_format'] = {'type': 'json_schema',
                            'json_schema': {'name': 'grounding_review', 'schema': schema}}
                    record = {'request': request}; records.append(record)
                    try:
                        response = client.chat.completions.create(**request)
                        record['response'] = response.model_dump(mode='json')
                        if response.choices[0].finish_reason != 'stop': raise ValueError('Incomplete diagnostic answer')
                        return response.choices[0].message.content
                    except Exception as exc:
                        record['error'] = type(exc).__name__+': '+str(exc); raise
                try: result = (review_question if probe else review_structured)(row, complete)
                except Exception as exc:
                    result = {'task_id': row['task_id'], 'keep': False, 'error': type(exc).__name__+': '+str(exc)}
                return result, records
            for row in probe_pending:
                result, records = run(row, probe=True)
                (OUTPUT/('v1-probe-'+row['task_id']+'.json')).write_text(json.dumps({
                    'mode': 'unconstrained-v1-reproduction', 'result': result, 'requests_and_responses': records}, indent=2))
                data_volume.commit()
            with ThreadPoolExecutor(max_workers=8) as pool, path.open('a') as stream:
                for future in as_completed([pool.submit(run, row) for row in pending]):
                    result, records = future.result()
                    raw = json.dumps(records, indent=2, ensure_ascii=False).encode()
                    raw_path = raw_dir/(result['task_id']+'.json')
                    with raw_path.open('wb') as raw_stream:
                        raw_stream.write(raw); raw_stream.flush(); os.fsync(raw_stream.fileno())
                    result['raw_responses_sha256'] = sha(raw)
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    completed[result['task_id']] = result; data_volume.commit()
                    print('Structured diagnostic reviewed', len(completed), 'of 88', flush=True)
        finally:
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=30)
    if set(completed) != expected: raise ValueError('Incomplete diagnostic')
    controls = {k: {'expected_keep': c['expected_keep'], 'actual_keep': completed[k]['keep'],
        'passed': 'error' not in completed[k] and completed[k]['keep'] == c['expected_keep']}
        for k, c in inputs['calibration_controls'].items()}
    report = {'status': 'complete', 'manifest': manifest, 'rows': len(completed),
        'kept': sum(r['keep'] for r in completed.values()), 'errors': sum('error' in r for r in completed.values()),
        'controls': controls, 'calibration_passed': all(c['passed'] for c in controls.values()),
        'decisions_sha256': sha(path.read_bytes()), 'approved_for_full_source_review': False,
        'approved_for_release': False, 'scope': 'Bounded diagnostic; fresh manual review still required'}
    (OUTPUT/'report.json').write_text(json.dumps(report, indent=2)); data_volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(preflight.remote(), indent=2))
