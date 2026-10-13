"""Run a timing benchmark on one Modal H200.

    modal run benchmarks/timing/modal_app.py --script lclm_hf_scaling \\
        --args "--checkpoint latent-context/0.6b-4b-LCLM-16x"
    modal run benchmarks/timing/modal_app.py --script vllm_ttft --args "--model lclm-16x"
    modal run benchmarks/timing/modal_app.py --script kv_baselines_hf --args "--methods no_press,snapkv"

Arguments in ``--args`` are passed to the script unchanged (do not pass
``--output``). Results are appended to ``results/<script>.jsonl`` on the Modal
volume named by ``LCLM_TIMING_VOLUME`` (default ``lclm-timing``) and
copied to ``runs/timing/<script>.jsonl`` locally. For gated models
(Llama), set ``LCLM_HF_SECRET`` to the name of a Modal secret holding
``HF_TOKEN``.
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

# Inside the container this file is imported from a different location.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
VOLUME_NAME = os.environ.get("LCLM_TIMING_VOLUME", "lclm-timing")
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
SECRETS = [modal.Secret.from_name(os.environ["LCLM_HF_SECRET"])] if os.environ.get("LCLM_HF_SECRET") else []
KVPRESS_COMMIT = "fa7a0dc1d6d5332a8ab00d03c67de7182c6846b0"
ATTENTION_MATCHING_REPO = "https://github.com/adamzweiger/compaction"
ATTENTION_MATCHING_COMMIT = "2020f6d37940fe013e7e7f1ebf409e5d107c5670"
ATTENTION_MATCHING_DIR = "/opt/attention-matching"

ENV = {
    "HF_HOME": "/mnt/timing/hf_cache",
    "PYTHONPATH": "/repo",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "LCLM_TIMING_VOLUME": VOLUME_NAME,  # same volume name inside the container
}
IGNORE = [".git", ".venv", "**/__pycache__", "*.pyc", "runs", "checkpoints", "wandb"]

# vLLM 0.11 (PyTorch 2.8) with the Transformers version used for the paper runs.
base_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
    .entrypoint([])
    .run_commands(
        "ln -sf /usr/bin/python3 /usr/local/bin/python",
        "python3 -m pip install transformers==4.57.1 accelerate==1.10.1 peft==0.17.1",
    )
    .env(ENV)
)
lclm_image = base_image.add_local_dir(REPO, "/repo", ignore=IGNORE)
kv_image = (
    base_image.apt_install("git")
    .run_commands(
        f"python3 -m pip install 'kvpress @ git+https://github.com/NVIDIA/kvpress@{KVPRESS_COMMIT}'",
        "python3 -m pip install flash-attn==2.8.3 --no-build-isolation",
        f"git clone {ATTENTION_MATCHING_REPO} {ATTENTION_MATCHING_DIR}"
        f" && git -C {ATTENTION_MATCHING_DIR} checkout {ATTENTION_MATCHING_COMMIT}",
        f"python3 -m pip install -r {ATTENTION_MATCHING_DIR}/requirements.txt",
    )
    .add_local_dir(REPO, "/repo", ignore=IGNORE)
)

app = modal.App("lclm-timing")


def _run(script: str, args: str) -> str:
    output = Path("/mnt/timing/results") / f"{script}.jsonl"
    cmd = [sys.executable, "-m", f"benchmarks.timing.{script}", *shlex.split(args), "--output", str(output)]
    if script == "kv_baselines_hf" and "--attention-matching-repo" not in args:
        cmd += ["--attention-matching-repo", ATTENTION_MATCHING_DIR]
    print(" ".join(cmd), flush=True)
    child = subprocess.Popen(cmd, cwd="/repo")
    while child.poll() is None:
        VOLUME.commit()
        time.sleep(30)
    VOLUME.commit()
    if child.returncode != 0:
        raise RuntimeError(f"{script} exited with code {child.returncode}")
    return output.read_text() if output.exists() else ""


@app.function(image=lclm_image, gpu="H200", cpu=8, memory=131072, timeout=12 * 3600,
              volumes={"/mnt/timing": VOLUME}, secrets=SECRETS)
def run_lclm(script: str, args: str) -> str:
    return _run(script, args)


@app.function(image=kv_image, gpu="H200", cpu=8, memory=131072, timeout=24 * 3600,
              volumes={"/mnt/timing": VOLUME}, secrets=SECRETS)
def run_kv(script: str, args: str) -> str:
    return _run(script, args)


@app.local_entrypoint()
def main(script: str, args: str = ""):
    if script not in ("lclm_hf_scaling", "vllm_ttft", "kv_baselines_hf"):
        raise SystemExit(f"unknown script {script!r}")
    runner = run_kv if script == "kv_baselines_hf" else run_lclm
    text = runner.remote(script, args)
    local = Path("runs/timing") / f"{script}.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text)
    print(f"Saved {local} ({len(text.splitlines())} records in the volume file)")
