"""Latest user routing:235B first attempts,27B retries only. Separate outputs."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image as cpu_image, volume
from data.generate_real_expansion_modal import image as image235, MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME
from data.qwen38_27b_pilot_modal import image as image27, cache, MODEL, REVISION
from data.prepare_expansion_retry import SOURCES

app=modal.App('lclm-expansion-235first-27retry-v3')
BASE=Path('/data/stage3-agent/real-expansion/pilots')
PREPARED=BASE/'retry-and-remaining-qwen38-27b-v2-tasks'
ROUTES=BASE/'235first-27retry-v3-tasks'
CONFIG={'model_id':MODEL,'revision':REVISION,'sampling_profile':'recommended'}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=7200,max_containers=4,volumes={'/data':volume})
def prepare(source: str):
    from data.split_expansion_routes import split_source
    if source not in SOURCES:raise ValueError('Unknown source')
    volume.reload()
    report=split_source(source,PREPARED,ROUTES);volume.commit()
    return {'source':source,'routes':[{k:r[k] for k in ('route','tasks')} for r in report['routes']]}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=7200,volumes={'/data':volume})
def finalize():
    from data.prepare_expansion_retry import validate_prepared
    from data.prepare_teacher_transition import file_sha
    volume.reload()
    # A completed fresh parent audit is mandatory before either GPU route.
    combined=json.loads((PREPARED/'retry-manifest.json').read_text())
    for report in combined['sources']:validate_prepared(report,PREPARED)
    reports=[json.loads((ROUTES/(s+'.routes.json')).read_text()) for s in SOURCES]
    result={}
    for key in ('first235','retry27'):
        entries=[dict(source=r['source'],**x) for r in reports for x in r['routes'] if x['route']==key]
        for x in entries:
            if file_sha(ROUTES/x['path'])!=x['sha256']:raise ValueError('Route hash mismatch')
        manifest={'route':key,'sources':entries,'tasks':sum(x['tasks'] for x in entries),
            'parent_manifest_sha256':file_sha(PREPARED/'retry-manifest.json'),
            'model':MODEL_ID if key=='first235' else MODEL,
            'revision':MODEL_REVISION if key=='first235' else REVISION,
            'approved_for_release':False}
        if key=='retry27':manifest['maud_ontology_sha256']=file_sha(ROUTES/key/'maud-choices.json')
        path=ROUTES/key/'route-manifest.json'
        if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Route manifest changed')
        path.write_text(json.dumps(manifest,indent=2));result[key]=manifest['tasks']
    volume.commit();return result


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=1800)
def tests():
    import subprocess
    subprocess.run(['python','-m','pytest','-q','tests/test_split_expansion_routes.py',
        'tests/test_expansion_retry.py','tests/test_teacher_transition.py',
        'tests/test_expansion_corrective_paths.py','tests/test_harvest_expansion_trace.py',
        'tests/test_expansion_semantic_review.py','tests/test_clean_agent_trajectories.py'],
        cwd='/opt/lclm',check=True)


def run_route(key):
    import subprocess,time,urllib.request
    from openai import OpenAI
    from data.full_expansion_rollouts import generate_all
    from data.prepare_teacher_transition import file_sha
    from data.expansion_checkpoint_audit import audit_source
    volume.reload();root=ROUTES/key;output=BASE/('235first-27retry-v3-'+key)
    manifest=json.loads((root/'route-manifest.json').read_text())
    first=key=='first235';model=MODEL_ID if first else MODEL
    revision=MODEL_REVISION if first else REVISION;served=SERVED_MODEL_NAME if first else MODEL
    if manifest['route']!=key or manifest['model']!=model or manifest['revision']!=revision:
        raise ValueError('Teacher/route mismatch')
    if not first and file_sha(root/'maud-choices.json')!=manifest['maud_ontology_sha256']:
        raise ValueError('MAUD ontology changed')
    sources=[]
    for item in manifest['sources']:
        if file_sha(ROUTES/item['path'])!=item['sha256']:raise ValueError('Route task hash mismatch')
        if item['tasks']:
            sources.append(item['source']);audit_source(root/(item['source']+'.tasks.jsonl'),output,item['source'])
    output.mkdir(exist_ok=True)
    command=['vllm','serve',model,'--revision',revision,'--served-model-name',served,
        '--host','127.0.0.1','--port','8000','--tensor-parallel-size','8','--max-model-len','32768',
        '--gpu-memory-utilization','0.90' if first else '0.85','--enforce-eager',
        '--enable-auto-tool-choice','--tool-call-parser','hermes' if first else 'qwen3_coder']
    command+=['--safetensors-load-strategy','prefetch'] if first else ['--reasoning-parser','qwen3','--max-num-seqs','64']
    (output/'server-command.json').write_text(json.dumps(command));volume.commit()
    with (output/'server.log').open('a') as log:
        process=subprocess.Popen(command,cwd='/opt/lclm',stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+5400
            while True:
                if process.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as response:
                        if response.status==200:break
                except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('Model startup timeout')
                time.sleep(5)
            client=OpenAI(api_key='not-needed',base_url='http://127.0.0.1:8000/v1',timeout=300,max_retries=2)
            return generate_all(client,served,revision,root,volume.commit,volume.reload,
                concurrency=32,output_root=output,sources=sources,strict_semantics=True,
                input_provenance={'scope':'not_previously_accepted','route':key,
                    'route_manifest_sha256':file_sha(root/'route-manifest.json')},
                teacher_config=None if first else CONFIG)
        except Exception as exc:
            (output/'last-error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)}));raise
        finally:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            volume.commit()


@app.function(image=image235,gpu='H200:8',cpu=16,memory=65536,timeout=86400,max_containers=1,
              volumes={'/data':volume,'/cache':cache},secrets=[modal.Secret.from_name('huggingface')])
def first235():return run_route('first235')


@app.function(image=image27,gpu='H200:8',cpu=16,memory=65536,timeout=86400,max_containers=1,
              volumes={'/data':volume,'/cache':cache},secrets=[modal.Secret.from_name('huggingface')])
def retry27():return run_route('retry27')


@app.local_entrypoint()
def main(action: str='prepare'):
    if action=='prepare':
        tests.remote()
        for report in prepare.map(SOURCES,order_outputs=False):print(json.dumps(report),flush=True)
        print(json.dumps(finalize.remote()),flush=True)
    elif action=='launch':
        # Run only once against a deployed app, after checking existing calls.
        for name in ('first235','retry27'):
            call=modal.Function.from_name('lclm-expansion-235first-27retry-v3',name).spawn()
            print(name,call.object_id,flush=True)
    else:raise ValueError('Unknown action')
