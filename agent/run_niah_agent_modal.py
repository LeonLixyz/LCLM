"""Run ``agent.run_niah_agent`` on Modal, one multi-GPU container per context length.

    modal run agent/run_niah_agent_modal.py                       # Figure 6: 4k, 8k, 16k
    modal run agent/run_niah_agent_modal.py --context-lengths 4096 \\
        --subtasks niah_single_3 --n-questions 2                   # smoke test

Results are written to the ``lclm-agent`` volume and copied to ``--output-dir``
locally. Set ``LCLM_MODAL_GPU`` (default ``H100:8``) to change the container GPUs;
each GPU runs one independent model replica.
"""
from pathlib import Path
import os

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get('LCLM_MODAL_GPU', 'H100:8')
GPUS = int(GPU.split(':')[1]) if ':' in GPU else 1
DEFAULT_CHECKPOINT = 'latent-context/0.6b-4b-LCLM-16x'

volume = modal.Volume.from_name('lclm-agent', create_if_missing=True)
app = modal.App('lclm-niah-agent')
image = (modal.Image.from_registry('vllm/vllm-openai:v0.11.0')
         .entrypoint([])
         .run_commands('python3 -m pip install transformers==4.57.1 datasets==3.6.0 '
                       'accelerate==1.10.1 peft==0.17.1',
                       'ln -sf /usr/bin/python3 /usr/local/bin/python')
         .env({'PYTHONPATH': '/app', 'HF_HOME': '/vol/hf_cache', 'TOKENIZERS_PARALLELISM': 'false',
               'OMP_NUM_THREADS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}))
for package in ('latent_context', 'inference', 'utils', 'agent'):
    image = image.add_local_dir(REPO/package, f'/app/{package}', ignore=['__pycache__'])


@app.function(image=image, gpu=GPU, cpu=16, memory=196608, volumes={'/vol': volume}, timeout=24*3600)
def evaluate(context_length: int, checkpoint: str, subtasks: str, n_questions: int, run_name: str, gpus: int):
    import subprocess
    import sys
    import time
    output = Path('/vol')/run_name/str(context_length)
    command = [sys.executable, '-m', 'agent.run_niah_agent', '--checkpoint', checkpoint,
               '--context-length', str(context_length), '--subtasks', subtasks,
               '--n-questions', str(n_questions), '--gpus', str(gpus), '--output-dir', str(output)]
    child = subprocess.Popen(command, cwd='/app')
    while child.poll() is None:
        time.sleep(60)
        volume.commit()
    volume.commit()
    if child.returncode:
        raise RuntimeError(f'run_niah_agent exited {child.returncode}')
    return {name: (output/name).read_text() for name in ('config.json', 'episodes.jsonl', 'summary.json')}


@app.local_entrypoint()
def main(context_lengths: str = '4096,8192,16384', checkpoint: str = DEFAULT_CHECKPOINT,
         subtasks: str = ','.join(('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
                                   'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery')),
         n_questions: int = 50, run_name: str = 'niah_agent', output_dir: str = 'runs/niah_agent'):
    lengths = [int(x) for x in context_lengths.split(',')]
    calls = {n: evaluate.spawn(n, checkpoint, subtasks, n_questions, run_name, GPUS) for n in lengths}
    for length, call in calls.items():
        folder = Path(output_dir)/str(length)
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in call.get().items():
            (folder/name).write_text(text)
        print(f'{length}: saved to {folder}', flush=True)
    print(f'Aggregate with: python -m agent.aggregate_niah {output_dir}')
