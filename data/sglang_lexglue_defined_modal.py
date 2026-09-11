"""New corrected LexGLUE snapshot; deploy/CPU prepare never starts generation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import modal
from data.sglang_backlog import MODEL,REVISION,IMAGE,atomic_json,file_sha,sha,read_jsonl
from data.sglang_backlog_fast import checked_rows
from data.sglang_backlog_storage import compact_completion,predecessor_path,ORIGIN_MANIFEST_SHA256,ORIGIN_DECISION_SHA256
from data.sglang_lexglue_defined import generation_config,verify_settings,prepare_chunk,validate_decision,EXPECTED_ROWS,EXPECTED_PROBES,PARENT_MANIFEST_SHA256,ONTOLOGY_PATH
from data.lexglue_task_definition_correction import VERSION,DEFINITIONS_SHA256,DEFINITIONS_PATH

APP_NAME='lclm-sglang27-lexglue-defined-20260909-v4'
ROOT=Path('/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4')
ORIGIN=Path('/data/stage3-build-20260906/sglang27-backlog-fast-20260909-v2')
app=modal.App(APP_NAME)
inputs=modal.Volume.from_name('lclm-stage3-data',create_if_missing=False)
volume=modal.Volume.from_name('lclm-stage3-agent-outputs-v2-20260909',create_if_missing=False,version=2)
cache=modal.Volume.from_name('lclm-hf-cache')
ignore=['.git','.venv','__pycache__','**/__pycache__/**','**/*.pyc','_modal_run']
cpu_image=(modal.Image.debian_slim(python_version='3.11').pip_install('pytest==8.4.2')
    .env({'PYTHONPATH':'/opt/lclm'}).add_local_dir('.','/opt/lclm',copy=True,ignore=ignore))
gpu_image=(modal.Image.from_registry(IMAGE).entrypoint([])
    .env({'HF_HOME':'/cache/huggingface','PYTHONPATH':'/opt/lclm','HF_XET_HIGH_PERFORMANCE':'1',
          'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1'})
    .add_local_dir('.','/opt/lclm',copy=True,ignore=ignore))
cpu_options=dict(image=cpu_image,volumes={'/data':inputs,'/runs':volume})


def parent():
    path=ORIGIN/'manifest.json'
    if file_sha(path)!=PARENT_MANIFEST_SHA256:raise ValueError('Original LexGLUE task selection changed')
    original=json.loads(path.read_text())
    source,=[s for s in original['sources'] if s['source']=='lex_glue']
    if source['rows']!=EXPECTED_ROWS or source['probe_rows']!=EXPECTED_PROBES:
        raise ValueError('Unexpected original LexGLUE task count')
    return original,source


@app.function(**cpu_options,cpu=2,memory=8192,timeout=1800)
def tests():
    import subprocess
    ontology=Path(ONTOLOGY_PATH);raw=DEFINITIONS_PATH.read_bytes()
    if sha(raw)!=DEFINITIONS_SHA256:raise ValueError('Reviewed ontology bytes changed')
    ontology.parent.mkdir(parents=True,exist_ok=True)
    if ontology.exists() and ontology.read_bytes()!=raw:raise ValueError('Shared ontology path contains different bytes')
    if not ontology.exists():ontology.write_bytes(raw)
    volume.commit()
    result=subprocess.run(['python','-m','pytest','-q','tests/test_sglang_lexglue_defined.py',
        'tests/test_lexglue_task_definition_correction.py','tests/test_sglang_backlog_storage.py',
        'tests/test_sglang_backlog_fast.py'],cwd='/opt/lclm',capture_output=True,text=True)
    print(result.stdout,result.stderr,flush=True)
    if result.returncode:raise RuntimeError('Corrected LexGLUE CPU tests failed')
    report={'status':'passed','output':result.stdout,'generation_config':generation_config()}
    atomic_json(ROOT/'tests.json',report);volume.commit();return report


@app.function(**cpu_options,cpu=2,memory=8192,timeout=3600,max_containers=4,retries=0)
def prepare_one(index:int):
    inputs.reload();volume.reload();_,source=parent()
    spec=source['chunks'][index]
    report=prepare_chunk(spec,ROOT/'inputs'/f'chunk-{index:05d}')
    volume.commit()
    return {'index':index,'rows':report['rows'],'modified':report['modified'],'configs':report['configs']}


@app.function(**cpu_options,cpu=4,memory=16384,timeout=3600)
def finalize_preparation():
    from collections import Counter
    inputs.reload();volume.reload();original,source=parent()
    test=json.loads((ROOT/'tests.json').read_text())
    if test['status']!='passed' or test['generation_config']!=generation_config():
        raise ValueError('Tests do not cover corrected generation code')
    verify_settings(generation_config())
    seen=set();counts=Counter();modified=0;chunks=[];reports=[]
    for index,spec in enumerate(source['chunks']):
        path=ROOT/'inputs'/f'chunk-{index:05d}'/'report.json'
        report=json.loads(path.read_text())
        if report['binding']!={'parent_chunk':spec,'generation_config':generation_config()}:
            raise ValueError('Corrected input preparation binding changed')
        if file_sha(report['chunk']['path'])!=report['chunk']['sha256']:
            raise ValueError('Corrected input bytes changed')
        ids=set(report['task_ids'])
        if len(ids)!=report['rows'] or seen&ids or report['chunk']['task_ids_sha256']!=spec['task_ids_sha256']:
            raise ValueError('Corrected IDs overlap or differ from original source')
        seen.update(ids);counts.update(report['configs']);modified+=report['modified'];chunks.append(report['chunk'])
        reports.append({'path':str(path),'sha256':file_sha(path),'coverage':report['coverage']})
        if index%32==0:print(json.dumps({'verified_chunks':index+1,'rows':len(seen)}),flush=True)
    if len(seen)!=EXPECTED_ROWS or sum(c['rows'] for c in chunks)!=EXPECTED_ROWS:
        raise ValueError('Corrected source must preserve exactly186444 pending IDs')
    ontology=Path(ONTOLOGY_PATH)
    raw=DEFINITIONS_PATH.read_bytes()
    if sha(raw)!=DEFINITIONS_SHA256:raise ValueError('Ontology bundle changed')
    if ontology.exists() and ontology.read_bytes()!=raw:raise ValueError('Frozen ontology changed')
    ontology.write_bytes(raw)
    current=copy.deepcopy(source);current.update(origin_pending_rows=source['rows'],rows=source['rows']-source['probe_rows'],
        chunks=chunks,original_coverage=source['coverage'],corrected_input_reports=reports,
        task_version=VERSION,configs=dict(counts),modified_rows=modified,
        counts={'previously_unattempted':EXPECTED_ROWS-EXPECTED_PROBES})
    manifest={'status':'prepared_pending_corrected_probe_review','task_version':VERSION,
        'ontology_artifact':{'path':str(ontology),'sha256':DEFINITIONS_SHA256},
        'sources':[current],'pending_rows':EXPECTED_ROWS-EXPECTED_PROBES,'probe_rows':EXPECTED_PROBES,
        'original_pending_rows':EXPECTED_ROWS,'pending_counts':{'previously_unattempted':EXPECTED_ROWS-EXPECTED_PROBES},'remaining_task_ids_sha256':sha(json.dumps(sorted(seen-set(source['probe_ids']))).encode()),
        'all_corrected_task_ids_sha256':sha(json.dumps(sorted(seen)).encode()),'modified_rows':modified,'configs':dict(counts),
        'generation_config':generation_config(),'original_probe_manifest_path':str(ORIGIN/'manifest.json'),
        'origin_probe_manifest_path':str(ORIGIN/'manifest.json'),'origin_probe_manifest_sha256':PARENT_MANIFEST_SHA256,
        'original_diagnostic_probes':'Opaque-v3 results are preserved separately and cannot be used as corrected results.',
        'old_completed_lexglue_outputs_selected_for_training':False,
        'maud_ontology_path':original['maud_ontology_path'],'maud_ontology_sha256':original['maud_ontology_sha256'],
        'approved_for_generation':False,'approved_for_release':False}
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Corrected manifest changed')
    atomic_json(path,manifest);volume.commit()
    return {'manifest_sha256':file_sha(path),'output':str(ROOT),'task_version':VERSION,
        'ontology_artifact':manifest['ontology_artifact'],'all_rows':EXPECTED_ROWS,'probe_rows':EXPECTED_PROBES,
        'continuation_rows':EXPECTED_ROWS-EXPECTED_PROBES,'modified_rows':modified,'configs':dict(counts)}


@app.function(**cpu_options,cpu=1,memory=2048,timeout=86400,retries=0)
def prepare():
    _,source=parent()
    for report in prepare_one.map(range(len(source['chunks'])),order_outputs=False):
        print(json.dumps(report),flush=True)
    return finalize_preparation.remote()


def load_gate(decision_sha256,stage):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if manifest['generation_config']!=generation_config():raise ValueError('Corrected generation config changed')
    if file_sha(manifest['ontology_artifact']['path'])!=DEFINITIONS_SHA256:
        raise ValueError('Corrected ontology artifact changed')
    parent()
    path=ROOT/('probe-decision.json' if stage=='probe' else 'continuation-decision.json')
    if file_sha(path)!=decision_sha256:raise ValueError('Root corrected-source decision changed')
    decision=json.loads(path.read_text());probe_sha=probe_report_sha=None;probe_report=None
    if stage=='continuation':
        probe_sha=file_sha(ROOT/'probe-decision.json');load_gate(probe_sha,'probe')
        report_path=ROOT/'outputs/lex_glue/chunk-00000/report.json'
        probe_report_sha=file_sha(report_path);probe_report=json.loads(report_path.read_text())
    allowed=validate_decision(decision,manifest,stage=stage,manifest_sha=file_sha(ROOT/'manifest.json'),
        probe_decision_sha=probe_sha,probe_report=probe_report,probe_report_sha=probe_report_sha)
    return manifest,decision,allowed


def reused_probe_routes():return {}


def probe_path(source,reused):return ROOT/'outputs'/source/'chunk-00000'/'report.json'


@app.function(**cpu_options,cpu=4,memory=16384,timeout=1800)
def check_gate(decision_sha256:str,stage:str):
    inputs.reload();volume.reload();manifest,_,allowed=load_gate(decision_sha256,stage)
    return {'ready_for_generation':True,'stage':stage,'allowed_sources':sorted(allowed),
        'task_version':VERSION,'pending_rows':manifest['pending_rows'],'approved_for_release':False}


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
         volumes={'/data': inputs, '/runs': volume, '/cache': cache}, secrets=[modal.Secret.from_name('huggingface')])
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
        self.local_server_root = Path('/tmp/sglang-server-runs') / self.server_root.name
        self.servers = Replicas(self.local_server_root, self.snapshot_logs)
        self.servers.start()

    def snapshot_logs(self):
        import os
        if getattr(self, 'local_server_root', None) is not None:
            self.server_root.mkdir(parents=True, exist_ok=True)
            for source in self.local_server_root.iterdir():
                if source.is_file():
                    destination = self.server_root / source.name
                    temporary = destination.with_name(destination.name + '.snapshot.tmp')
                    # Source is an active LOCAL server log; output descriptors close before commit/reload.
                    with source.open('rb') as reader, temporary.open('wb') as writer:
                        import shutil
                        shutil.copyfileobj(reader, writer)
                        writer.flush()
                        os.fsync(writer.fileno())
                    temporary.replace(destination)
        volume.commit()

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
            prior = predecessor_path(source_name, index, origin_root=ORIGIN, output_root=ROOT, reused_probes=reused_probe_routes())
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                raise ValueError('Previous source chunk is incomplete or held')
        binding = {'source': source_name, 'chunk': index, 'input_sha256': spec['sha256'],
            'effective_task_ids_sha256': spec['task_ids_sha256'], 'decision_sha256': decision_sha256,
            'backlog_manifest_sha256': file_sha(ROOT / 'manifest.json'), 'stage': stage,
            'origin_probe_manifest_sha256': ORIGIN_MANIFEST_SHA256,
            'origin_probe_decision_sha256': ORIGIN_DECISION_SHA256,
            'task_version': VERSION, 'ontology_artifact': manifest['ontology_artifact'],
            'corrected_probe_decision_sha256': file_sha(ROOT / 'probe-decision.json')}
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
                'approved_for_release': False,
                'result_hashes': {r['task_id']: file_sha(result_root / (r['task_id'] + '.json')) for r in results}}
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
        self.snapshot_logs()
        return report

    @modal.exit()
    def stop(self):
        if self.servers is not None:
            self.servers.stop()
        volume.commit()


def report_path_for(source, index, reused):
    return probe_path(source, reused) if index == 0 else ROOT / 'outputs' / source / f'chunk-{index:05d}' / 'report.json'


def finish_report(manifest, stage, allowed, state, path):
    reused = reused_probe_routes()
    chunks, holds = {}, {}
    for source in manifest['sources']:
        name = source['source']
        records = []
        for spec in source['chunks']:
            if (stage == 'probe') != (spec['index'] == 0):
                continue
            report_path = report_path_for(name, spec['index'], reused)
            if not report_path.exists():
                directory = report_path.parent
                journal_path = directory / 'attempts.jsonl'
                result_files = sorted((directory / 'results').glob('*.json'))
                journal = list(read_jsonl(journal_path)) if journal_path.exists() else []
                if not journal and not result_files:
                    continue
                from data.sglang_backlog import summarize_results
                results = [json.loads(p.read_text()) for p in result_files]
                expected = {task['task_id'] for task in checked_rows(spec)}
                completed = {r['task_id'] for r in results}
                reserved = {r['task_id'] for r in journal}
                if len(completed) != len(results) or not completed <= reserved <= expected:
                    raise ValueError('Interrupted chunk has duplicate or unknown attempts/results')
                report = {'status': 'interrupted_without_terminal_report', 'source': name,
                    'chunk': spec['index'], 'input_sha256': spec['sha256'], 'rows': spec['rows'],
                    'completed': len(results), 'counts': summarize_results(results),
                    'unresolved_attempt_ids': sorted(reserved-completed),
                    'unattempted': len(expected-reserved),
                    'result_hashes': {r['task_id']: file_sha(directory / 'results' / (r['task_id']+'.json')) for r in results}}
                report_path = directory / 'interrupted-accounting.json'
                atomic_json(report_path, report)
            report = json.loads(report_path.read_text())
            if report['input_sha256'] != spec['sha256'] or report['rows'] != spec['rows']:
                raise ValueError('Terminal report input mismatch')
            directory = report_path.parent
            for key, digest in report.get('result_hashes', {}).items():
                result_path = directory / 'results' / (key + '.json')
                if file_sha(result_path) != digest:
                    raise ValueError('Terminal result checksum changed')
            if len(report.get('result_hashes', {})) != report['completed']:
                raise ValueError('Terminal report lacks complete committed-result hashes')
            records.append({'index': spec['index'], 'status': report['status'],
                'report_path': str(report_path), 'report_sha256': file_sha(report_path),
                'result_directory': str(directory / 'results'), 'raw_attempt_directory': str(directory / 'attempts'),
                'input_path': spec['path'], 'input_sha256': spec['sha256'],
                'effective_task_ids_sha256': spec['task_ids_sha256'],
                'completed': report['completed'], 'counts': report['counts'],
                'unresolved': len(report.get('unresolved_attempt_ids', [])),
                'reused_original_probe': spec['index'] == 0 and name in reused})
        chunks[name] = records
        hold_path = ROOT / 'outputs' / name / 'HOLD.json'
        if hold_path.exists():
            holds[name] = {'path': str(hold_path), 'sha256': file_sha(hold_path),
                          'report': json.loads(hold_path.read_text())}
    report = compact_completion(manifest, stage=stage,
        allowed=allowed | (set(reused) if stage == 'probe' else set()), chunks=chunks,
        holds=holds, binding=state['binding'], coordinator=str(path), handoffs=state.get('handoffs', []))
    report['origin_probe_manifest_sha256'] = ORIGIN_MANIFEST_SHA256
    report['origin_probe_decision_sha256'] = ORIGIN_DECISION_SHA256
    report['corrected_probe_decision_sha256'] = file_sha(ROOT / 'probe-decision.json')
    report['remaining_task_ids_sha256'] = manifest['remaining_task_ids_sha256']
    report['task_version'] = VERSION
    report['ontology_artifact'] = manifest['ontology_artifact']
    terminal = path.parent / 'completion-report.json'
    if terminal.exists() and json.loads(terminal.read_text()) != report:
        raise ValueError('Refusing to change a terminal completion report')
    atomic_json(terminal, report)
    state.update(status=report['status'], completion_report_path=str(terminal),
                 completion_report_sha256=file_sha(terminal))
    atomic_json(path, state)
    volume.commit()
    return {key: state[key] for key in ('status', 'completion_report_path', 'completion_report_sha256')}


def coordinate(decision_sha256, stage):
    """Finite automatic handoff; known in-flight calls are awaited by the successor."""
    import time
    from data.sglang_backlog_storage import resolve_recorded_successor
    inputs.reload()
    volume.reload()
    manifest, _, allowed = load_gate(decision_sha256, stage)
    reused = reused_probe_routes()
    identity = {'function_call_id': modal.current_function_call_id(), 'input_id': modal.current_input_id()}
    path = ROOT / 'coordinators' / (stage + '-' + decision_sha256) / 'state.json'
    plan = [(s['source'], c['index']) for index in range(max(len(s['chunks']) for s in manifest['sources']))
        for s in manifest['sources'] if s['source'] in allowed
        for c in s['chunks'] if c['index'] == index and (stage == 'probe') == (index == 0)]
    binding = {'decision_sha256': decision_sha256, 'stage': stage,
        'manifest_sha256': file_sha(ROOT / 'manifest.json'), 'plan_sha256': sha(json.dumps(plan).encode())}
    state = json.loads(path.read_text()) if path.exists() else {'binding': binding, 'entries': {}, 'status': 'running', 'handoffs': []}
    if state['binding'] != binding:
        raise ValueError('Coordinator plan/review binding changed')
    if state.get('completion_report_path'):
        if file_sha(state['completion_report_path']) != state['completion_report_sha256']:
            raise ValueError('Terminal accounting changed')
        return {key: state[key] for key in ('status', 'completion_report_path', 'completion_report_sha256')}

    def persist():
        state['updated_at'] = time.time()
        atomic_json(path, state)
        volume.commit()

    def child_ids(call_id, suffix):
        def walk(nodes):
            for node in nodes:
                yield node
                yield from walk(node.children)
        return {node.function_call_id for node in walk(modal.FunctionCall.from_id(call_id).get_call_graph())
                if node.function_name.endswith(suffix)}

    # The previous coordinator may have been preempted immediately after spawning us.
    if state.get('handoffs'):
        last = state['handoffs'][-1]
        if last.get('status') != 'successor_started':
            successor_id = resolve_recorded_successor(last, parent_identity=last['parent_identity'],
                child_call_ids=child_ids(last['parent_identity']['function_call_id'], 'drive_plan'))
            last['successor_call_id'] = successor_id
            if identity == last['parent_identity']:
                persist()
                return {'status': 'handed_off', 'successor_call_id': successor_id, 'state_path': str(path)}
            if identity['function_call_id'] != successor_id:
                raise ValueError('Different coordinator cannot claim an approved handoff')
            last.update(status='successor_started', successor_identity=identity)
    state.update(status='running', current_identity=identity)
    persist()
    generator, started = Generator(), time.monotonic()

    def handoff_if_due():
        if time.monotonic() - started < 17 * 3600:
            return None
        if len(state['handoffs']) >= len(plan) + 1:
            raise RuntimeError('Finite coordinator handoff bound exhausted')
        item = {'parent_identity': identity, 'status': 'dispatching', 'started_at': time.time(),
                'known_successor_call_ids': sorted(child_ids(identity['function_call_id'], 'drive_plan')),
                'plan_sha256': binding['plan_sha256']}
        state['handoffs'].append(item)
        persist()
        successor = drive_plan.spawn(decision_sha256, stage)
        item.update(successor_call_id=successor.object_id, status='spawned')
        state['status'] = 'handed_off'
        persist()
        return {'status': 'handed_off', 'successor_call_id': successor.object_id,
                'state_path': str(path), 'binding': binding}

    def hold(source, index, detail):
        item = {'source': source, 'chunk': index, 'status': 'dispatch_needs_reconciliation',
                'detail': detail, 'coordinator': str(path), 'approved_for_release': False}
        atomic_json(ROOT / 'outputs' / source / 'HOLD.json', item)
        print(json.dumps(item), flush=True)

    for source, index in plan:
        volume.reload()
        source_root = ROOT / 'outputs' / source
        report_path = report_path_for(source, index, reused)
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
            prior = predecessor_path(source, index, origin_root=ORIGIN, output_root=ROOT, reused_probes=reused)
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'complete':
                continue
        next_coordinator = handoff_if_due()
        if next_coordinator:
            return next_coordinator
        entry = state['entries'].get(key)
        if entry and not entry.get('child_call_id'):
            owner_path = source_root / f'chunk-{index:05d}' / 'owner.json'
            child_id = None
            if owner_path.exists():
                owner = json.loads(owner_path.read_text())
                if (owner['binding']['source'] == source and owner['binding']['chunk'] == index
                        and owner['binding']['decision_sha256'] == decision_sha256):
                    child_id = owner['identity']['function_call_id']
            if child_id is None:
                known = {r.get('child_call_id') for r in state['entries'].values()}
                candidates = child_ids(entry['coordinator_identity']['function_call_id'], 'chunk') - known
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
            persist()
            try:
                child = generator.chunk.spawn(source, index, decision_sha256, stage)
                entry.update(child_call_id=child.object_id, status='running')
                persist()
            except Exception as exc:
                hold(source, index, f'Dispatch outcome is unclassified: {type(exc).__name__}: {exc}')
                entry['status'] = 'dispatch_needs_reconciliation'
                persist()
                continue
        child = modal.FunctionCall.from_id(entry['child_call_id'])
        last_progress = None
        while True:
            next_coordinator = handoff_if_due()
            if next_coordinator:
                return next_coordinator
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
    return finish_report(manifest, stage, allowed, state, path)


@app.function(**cpu_options, cpu=4, memory=16384, timeout=86400, max_containers=1, retries=0)
def drive_plan(decision_sha256: str, stage: str):
    return coordinate(decision_sha256, stage)


@app.function(**cpu_options, cpu=2, memory=8192, timeout=1800)
def status():
    volume.reload()
    if not (ROOT / 'manifest.json').exists():
        return {'prepared': False, 'output': str(ROOT)}
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    output = {'prepared': True, 'pending_rows': manifest['pending_rows'], 'sources': {}, 'coordinators': []}
    for source in manifest['sources']:
        name, records = source['source'], []
        for spec in source['chunks']:
            directory = ROOT / 'outputs' / name / f"chunk-{spec['index']:05d}"
            path = directory / 'report.json'
            if path.exists():
                report = json.loads(path.read_text())
                records.append({key: report.get(key) for key in ('chunk', 'status', 'completed', 'counts', 'subgroups')})
        output['sources'][name] = {'continuation_rows': source['rows'], 'probe_rows': source['probe_rows'],
            'held': (ROOT / 'outputs' / name / 'HOLD.json').exists(), 'chunks': records}
    for path in (ROOT / 'coordinators').glob('*/state.json'):
        state = json.loads(path.read_text())
        output['coordinators'].append({'path': str(path), 'status': state['status'],
            'entries': len(state['entries']), 'handoffs': state.get('handoffs', []),
            'completion_report_path': state.get('completion_report_path'),
            'completion_report_sha256': state.get('completion_report_sha256')})
    return output
