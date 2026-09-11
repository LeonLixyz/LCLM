"""Validate DeepSpeed optimizer ordering on Modal H200:8."""
import json
import os
import subprocess
from pathlib import Path

import modal
from scripts.stage3_readiness_modal import code, volume

app = modal.App('lclm-deepspeed-ordering-20260910-v1')
OUT = Path('/data/stage3-build-20260906/deepspeed-ordering-20260910-v1')
SUPPORTED_CHECKS = frozenset({'regression', 'norm1', 'norm2', '0', '1', '2', '2cpu', '2fp16', 'real2'})


def validate_checks(stages):
    specs = stages.split(',')
    unsupported = set(specs) - SUPPORTED_CHECKS
    if unsupported:
        raise ValueError(f'Unsupported checks {sorted(unsupported)}; use ZeRO-2 at most.')
    return specs


image = code(modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel')
    .pip_install('transformers==4.57.1', 'datasets==3.6.0', 'pyarrow==21.0.0',
                 'accelerate==1.10.1', 'peft==0.17.1', 'torchdata==0.11.0',
                 'pytest==8.4.2', 'einops', 'omegaconf', 'liger-kernel==0.6.2',
                 'wandb==0.21.1', 'python-dotenv', 'ninja', 'packaging')
    .run_commands('MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation')
    .env({'DS_BUILD_OPS': '0', 'MAX_JOBS': '8'})
    .pip_install('deepspeed==0.17.5'))


@app.function(image=image, gpu='H200:8', cpu=32, memory=262144, timeout=10800,
              volumes={'/data': volume})
def checks(stages='1,2,2cpu,2fp16,0', run_name='zero2-only'):
    import hashlib
    specs = validate_checks(stages)
    volume.reload()
    out = OUT / run_name
    out.mkdir(parents=True, exist_ok=True)
    source_files = ['train/trainer.py', 'train/deepspeed_step.py', 'utils/nan_checks.py',
                    'latent_context/model.py', 'latent_context/encoder.py',
                    'scripts/stage3_deepspeed_smoke.py', 'scripts/stage3_real_training_smoke.py']
    (out / 'source-hashes.json').write_text(json.dumps({p: hashlib.sha256((Path('/opt/lclm') / p).read_bytes()).hexdigest()
                                                     for p in source_files}, indent=2))
    results = []
    for spec in specs:
        env = dict(os.environ)
        if spec == 'regression':
            cmd = ['python', '-m', 'pytest', '-q',
                'tests/test_distributed_dataset_balancing.py', 'tests/test_dynamic_packing_dataset.py',
                'tests/test_packed_file_discovery.py', 'tests/test_trainer_no_memory_optimizer.py',
                'tests/test_processor_target_memory.py', 'tests/test_unpacked_agent_collate.py',
                'tests/test_nan_checks.py', 'tests/test_model_mixed_compression.py',
                'tests/test_distributed_mixed_compression.py', 'tests/test_packed_flash_parity.py']
        elif spec.startswith('real'):
            stage = int(spec.removeprefix('real'))
            config = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2,
                      'gradient_clipping': 1.0, 'bf16': {'enabled': True},
                      'zero_optimization': {'stage': stage, 'overlap_comm': False,
                         'reduce_bucket_size': 50000000, 'allgather_bucket_size': 50000000},
                      'zero_allow_untested_optimizer': True, 'steps_per_print': 100000}
            path = out / f'{spec}-config.json'; path.write_text(json.dumps(config))
            env.update(ACCELERATE_USE_DEEPSPEED='true', ACCELERATE_MIXED_PRECISION='bf16',
                       ACCELERATE_DEEPSPEED_CONFIG_FILE=str(path))
            cmd = ['torchrun', '--standalone', '--nproc_per_node=8', 'scripts/stage3_real_training_smoke.py',
                   '--backend', 'deepspeed', '--report-root', str(out / spec)]
        else:
            stage = int(spec.removeprefix('norm').removesuffix('fp16').removesuffix('cpu'))
            cmd = ['torchrun', '--standalone', '--nproc_per_node=8', 'scripts/stage3_deepspeed_smoke.py',
                   '--stage', str(stage), '--out', str(out)]
            if spec.endswith('cpu'):
                cmd.append('--offload')
            if spec.startswith('norm'):
                cmd.append('--linear')
            if spec.endswith('fp16'):
                cmd.append('--fp16')
        path = out / f'{spec}.log'
        with path.open('w') as stream:
            run = subprocess.run(cmd, cwd='/opt/lclm', env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=3000)
        result = {'stage': spec, 'exit_code': run.returncode, 'tail': path.read_text()[-7000:]}
        results.append(result)
        (out / 'report.json').write_text(json.dumps(results, indent=2)); volume.commit()
        print(json.dumps(result), flush=True)
        if run.returncode:
            break
    return results


@app.local_entrypoint()
def main(stages: str = 'regression,norm1,norm2,1,2,2cpu,2fp16,0,real2', run_name: str = 'zero2-only'):
    validate_checks(stages)  # Reject unsupported requests before allocating GPUs.
    print(json.dumps(checks.remote(stages, run_name), indent=2))
