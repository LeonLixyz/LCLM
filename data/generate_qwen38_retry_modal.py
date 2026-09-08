"""One new nonthinking27B attempt for prior rejects and unattempted tasks."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image as cpu_image, volume
from data.qwen38_27b_pilot_modal import image, cache, MODEL, REVISION, sha
from data.prepare_expansion_retry import SOURCES

app = modal.App('lclm-qwen38-retry-and-remaining-v2')
BASE = Path('/data/stage3-agent/real-expansion/pilots')
TASKS = BASE/'retry-and-remaining-qwen38-27b-v2-tasks'
OUTPUT = BASE/'retry-and-remaining-qwen38-27b-v2'
CONFIG = {'model_id':MODEL,'revision':REVISION,'sampling_profile':'recommended'}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=7200,
              max_containers=4,volumes={'/data':volume})
def prepare(source: str):
    from data.prepare_expansion_retry import prepare_source
    if source not in SOURCES:
        raise ValueError('Unknown source')
    volume.reload()
    parent = BASE/('multidoc2dial-chronological-v1-full' if source=='multidoc2dial' else 'full-20260906-v6')
    original = BASE/('multidoc2dial-chronological-v1-inputs' if source=='multidoc2dial' else 'full-20260906-v3')
    try:
        report = prepare_source(source, original/(source+'.tasks.jsonl'), parent, TASKS)
        volume.commit()
        return {k:report[k] for k in ('source','tasks','counts')}
    except Exception:
        volume.commit()
        raise


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=7200,volumes={'/data':volume})
def finalize():
    from data.prepare_expansion_retry import validate_prepared
    volume.reload()
    reports = [json.loads((TASKS/(s+'.build.json')).read_text()) for s in SOURCES]
    for report in reports:
        validate_prepared(report,TASKS)
    manifest = {'scope':'not_previously_accepted','sources':reports,
        'teacher_config':CONFIG,'user_authorized_retry':True,'approved_for_release':False,
        'old_remaining_call_failed_before_generation':'fc-01M216DEKJDPYYC5PW1C8881N6',
        'old_remaining_failure':'Parent checkpoint hash changed; rebuilt from current audited files.',
        'corrected_multidoc2dial_replaces_original':True}
    path = TASKS/'retry-manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Incompatible existing retry manifest')
    path.write_text(json.dumps(manifest,indent=2))
    OUTPUT.mkdir(exist_ok=True)
    volume.commit()
    return {'tasks':sum(r['tasks'] for r in reports),
        'retry_rejected':sum(r['counts'].get('retry_rejected',0) for r in reports),
        'previously_unattempted':sum(r['counts'].get('previously_unattempted',0) for r in reports),
        'previously_accepted':sum(r['counts'].get('previously_accepted',0) for r in reports),
        'manifest_sha256':sha(path.read_bytes())}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=1800)
def tests():
    import subprocess
    subprocess.run(['python','-m','pytest','-q','tests/test_expansion_retry.py',
        'tests/test_teacher_transition.py','tests/test_qwen38_pilot_sampling.py',
        'tests/test_expansion_corrective_paths.py','tests/test_expansion_checkpoint_audit.py',
        'tests/test_harvest_expansion_trace.py','tests/test_expansion_semantic_review.py',
        'tests/test_clean_agent_trajectories.py'],cwd='/opt/lclm',check=True)


@app.function(image=image,gpu='H200:8',cpu=16,memory=65536,timeout=86400,
              max_containers=1,scaledown_window=60,
              volumes={'/data':volume,'/cache':cache},secrets=[modal.Secret.from_name('huggingface')])
def generate():
    import subprocess
    import time
    import urllib.request
    from openai import OpenAI
    from data.full_expansion_rollouts import generate_all
    from data.prepare_expansion_retry import validate_prepared
    volume.reload()
    path = TASKS/'retry-manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['teacher_config']!=CONFIG or manifest['user_authorized_retry'] is not True:
        raise ValueError('Missing retry authorization')
    # Fail before model startup if either the parent or selected tasks changed.
    for report in manifest['sources']:
        validate_prepared(report,TASKS)
    provenance = {'scope':'not_previously_accepted','retry_manifest_sha256':sha(path.read_bytes()),
                  'approved_for_release':False,'corrected_multidoc2dial_replaces_original':True}
    command = ['vllm','serve',MODEL,'--revision',REVISION,'--served-model-name',MODEL,
        '--host','127.0.0.1','--port','8000','--tensor-parallel-size','8',
        '--max-model-len','32768','--max-num-seqs','64','--gpu-memory-utilization','0.85',
        '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
        '--reasoning-parser','qwen3','--enforce-eager']
    (OUTPUT/'server-command.json').write_text(json.dumps(command));volume.commit()
    with (OUTPUT/'server.log').open('a') as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+3600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('27B server exited; inspect server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5) as r:
                        if r.status==200:break
                except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('27B startup timeout')
                time.sleep(5)
            client=OpenAI(api_key='not-needed',base_url='http://127.0.0.1:8000/v1',
                          timeout=300,max_retries=2)
            return generate_all(client,MODEL,REVISION,TASKS,volume.commit,volume.reload,
                concurrency=32,output_root=OUTPUT,sources=list(SOURCES),strict_semantics=True,
                input_provenance=provenance,teacher_config=CONFIG)
        except Exception as exc:
            (OUTPUT/'last-error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc)}))
            volume.commit();raise
        finally:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            volume.commit()


@app.local_entrypoint()
def main(test_only: bool=False, source: str=''):
    tests.remote()
    if test_only:return
    if source:
        print(json.dumps(prepare.remote(source),indent=2));return
    for result in prepare.map(SOURCES,order_outputs=False):
        print(json.dumps(result),flush=True)
    print(json.dumps(finalize.remote(),indent=2))
