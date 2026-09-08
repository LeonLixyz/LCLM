"""User-authorized teacher switch: only unattempted delta tasks, no release approval."""
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image as cpu_image, volume, ROOT
from data.qwen38_27b_pilot_modal import image, cache, MODEL, REVISION, sha
from data.prepare_teacher_transition import SOURCES

app = modal.App('lclm-qwen38-27b-remaining-v1')
TASKS = Path('/data/stage3-agent/real-expansion/pilots/full-20260906-v3')
PREVIOUS = TASKS.parent/'full-20260906-v6'
DELTA = TASKS.parent/'remaining-qwen38-27b-v1-tasks'
OUTPUT = TASKS.parent/'remaining-qwen38-27b-v1'
CONFIG = {'model_id':MODEL,'revision':REVISION,'sampling_profile':'recommended'}


@app.function(image=cpu_image,cpu=4,memory=16384,timeout=7200,volumes={'/data':volume},
              secrets=[modal.Secret.from_name('huggingface')])
def prepare(test_only: bool = False):
    import subprocess
    from huggingface_hub import hf_hub_download
    from transformers.utils.chat_template_utils import _compile_jinja_template
    from data.prepare_teacher_transition import prepare_remaining
    subprocess.run(['python','-m','pytest','-q',
        'tests/test_teacher_transition.py','tests/test_qwen38_pilot_sampling.py',
        'tests/test_expansion_corrective_paths.py','tests/test_expansion_checkpoint_audit.py',
        'tests/test_harvest_expansion_trace.py','tests/test_expansion_semantic_review.py',
        'tests/test_clean_agent_trajectories.py'],cwd='/opt/lclm',check=True)
    # Both are teacher/training *template* checks, not model inference.
    rendered={}
    for key,model,revision in [('teacher',MODEL,REVISION),
        ('training','Qwen/Qwen3-4B-Instruct-2507','cdbee75f17c01a7cc42f958dc650907174af0554')]:
        if key=='teacher':
            path=hf_hub_download(model,'chat_template.jinja',revision=revision)
            raw=Path(path).read_bytes()
        else:
            path=hf_hub_download(model,'tokenizer_config.json',revision=revision)
            raw=json.loads(Path(path).read_text())['chat_template'].encode()
        template=_compile_jinja_template(raw.decode())
        prompt=template.render(messages=[{'role':'user','content':'Example task'}],
                               tools=[],add_generation_prompt=True,enable_thinking=False)
        full=template.render(messages=[{'role':'user','content':'Example task'},
            {'role':'assistant','content':'FINAL: example'}],tools=[],
            add_generation_prompt=False,enable_thinking=False)
        rendered[key]={'model':model,'revision':revision,'template_sha256':sha(raw),
                       'generation_prompt':prompt,'full_conversation':full}
    assert rendered['teacher']['generation_prompt'].endswith('<think>\n\n</think>\n\n')
    assert '<think>' not in rendered['training']['generation_prompt']
    assert '<think>' not in rendered['training']['full_conversation']
    if test_only:
        return {'tests':'passed','data_modified':False,'templates':rendered}
    volume.reload()
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'template-audit.json').write_text(json.dumps(rendered,indent=2))
    volume.commit()
    manifest=prepare_remaining(TASKS,PREVIOUS,DELTA)
    (OUTPUT/'switch-authorization.json').write_text(json.dumps({
        'user_authorized_27b_for_remaining_generation':True,
        'source_review_still_required':True,'approved_for_release':False,
        'teacher_and_semantic_judge':CONFIG,
        'previous_outputs_preserved':str(PREVIOUS),
        'scope':'Only tasks with no saved previous-teacher accepted or rejected row.'},indent=2))
    volume.commit()
    return {'status':'prepared','transition_sha256':sha((DELTA/'transition-manifest.json').read_bytes()),
            'sources':[{k:s[k] for k in ('source','tasks','persisted','remaining','accepted')}
                       for s in manifest['sources']],
            'template_placeholder_teacher_only':True}


@app.function(image=image,gpu='H200:8',cpu=16,memory=65536,timeout=86400,
              max_containers=1,scaledown_window=60,
              volumes={'/data':volume,'/cache':cache},
              secrets=[modal.Secret.from_name('huggingface')])
def generate():
    import subprocess
    import time
    import urllib.request
    from openai import OpenAI
    from data.full_expansion_rollouts import generate_all
    from data.prepare_teacher_transition import prepare_remaining
    volume.reload()
    authorization=json.loads((OUTPUT/'switch-authorization.json').read_text())
    if (authorization.get('user_authorized_27b_for_remaining_generation') is not True
            or authorization.get('teacher_and_semantic_judge')!=CONFIG):
        raise ValueError('Missing model-switch authorization')
    # Recheck immutable parent/output hashes before every resume.
    prepare_remaining(TASKS,PREVIOUS,DELTA)
    provenance={'transition_manifest_sha256':sha((DELTA/'transition-manifest.json').read_bytes()),
                'previous_root':str(PREVIOUS),'scope':'unattempted_delta_only',
                'requires_composition_with_previous_and_corrected_md':True,
                'approved_for_release':False}
    command=['vllm','serve',MODEL,'--revision',REVISION,'--served-model-name',MODEL,
             '--host','127.0.0.1','--port','8000','--tensor-parallel-size','8',
             '--max-model-len','32768','--max-num-seqs','64',
             '--gpu-memory-utilization','0.85','--enable-auto-tool-choice',
             '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3','--enforce-eager']
    (OUTPUT/'server-command.json').write_text(json.dumps(command));volume.commit()
    with (OUTPUT/'server.log').open('a') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
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
            return generate_all(client,MODEL,REVISION,DELTA,volume.commit,volume.reload,
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
def main(test_only: bool = False):
    print(json.dumps(prepare.remote(test_only),indent=2))
