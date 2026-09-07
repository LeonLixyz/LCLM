"""GPU validation of Stage-3 masking, packing, and empty-memory collectives."""
import json
import subprocess
from pathlib import Path
import modal

app = modal.App('lclm-stage3-gpu-validation-20260906')
volume = modal.Volume.from_name('lclm-stage3-data')
image = (modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel')
    .pip_install('transformers==4.57.1','datasets==3.6.0','pyarrow>=18,<22',
                 'accelerate','peft','torchdata','pytest','einops','omegaconf')
    .env({'PYTHONPATH':'/opt/lclm','TOKENIZERS_PARALLELISM':'false','NCCL_DEBUG':'WARN'})
    .add_local_dir('.', '/opt/lclm', ignore=['.git','.venv','__pycache__','_modal_run']))

@app.function(image=image,gpu='H200:8',cpu=16,memory=65536,timeout=3600,
              volumes={'/data':volume})
def validate(intermediate:bool=False):
    output=Path('/data/stage3-build-20260906/validation')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,command in [
        ('pytest',['python','-m','pytest','tests','-q','--disable-warnings']),
        ('packed_artifacts',['python','scripts/stage3_packed_artifact_audit.py']+([] if intermediate else ['--require-expansion'])),
        ('nccl',['torchrun','--standalone','--nproc_per_node=2','scripts/stage3_nccl_smoke.py']),
        ('fsdp',['torchrun','--standalone','--nproc_per_node=2','scripts/stage3_nccl_smoke.py','--fsdp']),
    ]:
        run=subprocess.run(command,cwd='/opt/lclm',capture_output=True,text=True,timeout=3000)
        (output/f'{name}.log').write_text(run.stdout+'\n'+run.stderr)
        results.append({'check':name,'exit_code':run.returncode,'tail':(run.stdout+'\n'+run.stderr)[-10000:]})
        volume.commit()
    (output/('intermediate-report.json' if intermediate else 'report.json')).write_text(json.dumps(results,indent=2))
    volume.commit()
    return results

@app.local_entrypoint()
def main(intermediate:bool=False):
    print(json.dumps(validate.remote(intermediate),indent=2))
