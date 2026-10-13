"""Run ``compaction.eval_math`` on Modal.

    modal run compaction/modal_app.py --limit 20 --run-name smoke
    modal run compaction/modal_app.py --run-name math-full      # all 5,000 problems, both modes

Results go to ``runs/<run-name>/`` on the Modal volume ``LCLM_MODAL_VOLUME``
(default ``lclm-compaction``). Re-running the same command resumes. The GPU
type is set by ``LCLM_MODAL_GPU`` (default ``H200:8``; one vLLM worker per GPU).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get('LCLM_MODAL_GPU', 'H200:8')

image = (
    modal.Image.from_registry('vllm/vllm-openai:v0.11.0').entrypoint([])
    .run_commands('python3 -m pip install transformers==4.57.1 datasets==3.6.0 accelerate==1.10.1 '
                  'peft==0.17.1 math-verify==0.9.0')
    .env({'PYTHONPATH': '/app', 'HF_HOME': '/mnt/lclm/hf_cache', 'TOKENIZERS_PARALLELISM': 'false',
          'OMP_NUM_THREADS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'})
)
for package in ('latent_context', 'inference', 'utils', 'compaction'):
    image = image.add_local_dir(ROOT / package, f'/app/{package}', ignore=['__pycache__'])

volume = modal.Volume.from_name(os.environ.get('LCLM_MODAL_VOLUME', 'lclm-compaction'), create_if_missing=True)
app = modal.App('lclm-math-compaction')


@app.function(image=image, gpu=GPU, cpu=32, memory=262144, timeout=43200, volumes={'/mnt/lclm': volume})
def evaluate(checkpoint: str, run_name: str, limit: int, modes: str):
    volume.reload()
    out = Path('/mnt/lclm/runs') / run_name
    cmd = [sys.executable, '-m', 'compaction.eval_math', '--checkpoint', checkpoint, '--out', str(out),
           '--modes', modes, '--limit', str(limit)]
    proc = subprocess.Popen(cmd, cwd='/app')
    # Commit periodically so an interrupted run can resume from saved trajectories.
    while proc.poll() is None:
        time.sleep(60)
        volume.commit()
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f'eval_math exited with {proc.returncode}')
    return json.loads((out / 'metrics.json').read_text())


@app.local_entrypoint()
def main(checkpoint: str = 'latent-context/0.6b-4b-LCLM-16x', run_name: str = 'math-compaction',
         limit: int = 0, modes: str = 'full,compaction'):
    metrics = evaluate.remote(checkpoint, run_name, limit, modes)
    print(json.dumps(metrics['modes'], indent=2))
