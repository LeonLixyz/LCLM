"""RULER NIAH eval on Modal — single container, vLLM + flash-attn.

Image recipe matches the prior working ``benchmark/run_lclm_sweep_modal.py``
on the main branch: build CUDA 12.5 base, install Python 3.11, ``uv sync``
against pyproject.toml (which uses ``no-build-isolation-package=["flash-attn"]``
so torch is installed first, flash-attn is built against it, then vllm
is resolved consistently).

Workflow per task:
    1. prepare prompts.jsonl from latent-context/ruler-full
    2. subprocess: python -m inference.encode  (HF + flash-attn -> .pt)
    3. subprocess: python -m inference.decode  (vllm reads .pt)
    4. score via ruler_string_match_all

Run:
    modal run --detach _modal_run/lclm_ruler_vllm.py
"""
from __future__ import annotations
import modal, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT = "/app"
VOLUME_PATH = "/vol"

app = modal.App("lclm-ruler-niah")
vol = modal.Volume.from_name("kv-cache-compression", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.5.0-devel-ubuntu22.04")
    .run_commands(
        "apt-get update",
        "apt-get install -y software-properties-common",
        "add-apt-repository ppa:deadsnakes/ppa -y",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-dev python-is-python3",
        "update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2",
        "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2",
        "rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        "apt-get update && apt-get install -y python3-pip git curl bash build-essential"
    )
    .add_local_dir(
        str(REPO),
        f"{PROJECT_ROOT}/",
        copy=True,
        ignore=[
            "__pycache__", ".git", "*.pyc", ".venv", "checkpoints",
            ".cache", "wandb", "*.safetensors", "*.bin", "*.pt",
            "_smoketest_ckpts", "_matrix_ckpts",
            "data/packed_batches", "*.parquet",
        ],
    )
    .run_commands(
        "/bin/bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=/root/.local/bin:$PATH && "
        f"cd {PROJECT_ROOT} && uv sync && "
        f"uv pip install rouge fuzzywuzzy python-Levenshtein jieba pyyaml'",
    )
    .env({
        "PATH": f"{PROJECT_ROOT}/.venv/bin:/root/.local/bin:$PATH",
        "VIRTUAL_ENV": f"{PROJECT_ROOT}/.venv",
        "PYTHONPATH": f"{PROJECT_ROOT}",
        "HF_HOME": f"{VOLUME_PATH}/hf_cache",
    })
)

REPO_ID = "latent-context/0.6b-4b-LCLM-16x"
WORK_DIR = f"{VOLUME_PATH}/ruler_16x_niah"
CTX = 4096

NIAH_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]

BASELINE_16X = {
    "niah_single_1": 96.80, "niah_single_2": 90.00, "niah_single_3": 55.20,
    "niah_multikey_1": 86.40, "niah_multikey_2": 82.20, "niah_multikey_3": 49.00,
    "niah_multivalue": 74.10, "niah_multiquery": 81.45,
}


@app.function(image=image, gpu="H200", timeout=4*60*60,
              volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def run_eval():
    import os, subprocess, json, time
    os.chdir(PROJECT_ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # 1. Prepare prompts.jsonl per task.
    from datasets import load_dataset
    print("Loading latent-context/ruler-full ...", flush=True)
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    rows = ds[list(ds.keys())[0]]
    os.makedirs(WORK_DIR, exist_ok=True)
    for task in NIAH_SUBTASKS:
        cat = f"memwrap/ruler/{task}_{CTX}"
        task_rows = rows.filter(lambda r: r["category"] == cat)
        path = f"{WORK_DIR}/{task}_prompts.jsonl"
        with open(path, "w") as f:
            for r in task_rows:
                gt = (r.get("extra_info") or {}).get("ground_truth") or {}
                answers = gt.get("answers") or r.get("answer") or r.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers]
                f.write(json.dumps({"prompt": r["prompt"], "answers": answers}) + "\n")
        print(f"  [{task}] wrote {len(task_rows)} prompts", flush=True)

    # 2-3. Encode then decode per task. Two subprocesses per task so the
    # HF process can free GPU memory before vLLM grabs the rest.
    print(f"\n>>> Running encode+decode for {len(NIAH_SUBTASKS)} tasks", flush=True)
    for task in NIAH_SUBTASKS:
        t0 = time.time()
        embeds = f"{WORK_DIR}/{task}_embeds.pt"
        completions = f"{WORK_DIR}/{task}_completions.jsonl"

        rc = subprocess.run([
            "python", "-m", "inference.encode",
            "--checkpoint", REPO_ID,
            "--prompts-jsonl", f"{WORK_DIR}/{task}_prompts.jsonl",
            "--out", embeds,
            "--max-encode-batch-size", "128",
        ], env=env).returncode
        if rc != 0:
            raise RuntimeError(f"encode failed for {task} (rc={rc})")

        rc = subprocess.run([
            "python", "-m", "inference.decode",
            "--checkpoint", REPO_ID,
            "--embeds-pt", embeds,
            "--out", completions,
            "--max-tokens", "128",
            "--temperature", "0.0",
        ], env=env).returncode
        if rc != 0:
            raise RuntimeError(f"decode failed for {task} (rc={rc})")
        print(f"  [{task}] done in {time.time()-t0:.1f}s", flush=True)
        vol.commit()

    # 4. Score.
    def _match(a, r): return a.lower() in r.lower()
    def _score_all(ans, resp):
        if not ans: return 0.0
        return sum(1.0 for a in ans if _match(a, resp)) / len(ans) * 100.0

    print(f"\n========== 16x NIAH SUMMARY ==========", flush=True)
    print(f"  {'task':22s} {'NEW':>8s} {'BASELINE':>10s}  {'delta':>8s}", flush=True)
    results = {}
    for task in NIAH_SUBTASKS:
        scores = []
        with open(f"{WORK_DIR}/{task}_completions.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                scores.append(_score_all(rec.get("answers", []), rec["response"]))
        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = avg
        base = BASELINE_16X[task]
        print(f"  {task:22s} {avg:8.2f} {base:10.2f}  {avg-base:+8.2f}", flush=True)
    overall = sum(results.values()) / len(results) if results else 0.0
    baseline_overall = sum(BASELINE_16X.values()) / len(BASELINE_16X)
    print(f"  {'NIAH overall':22s} {overall:8.2f} {baseline_overall:10.2f}  {overall-baseline_overall:+8.2f}", flush=True)

    with open(f"{WORK_DIR}/summary.json", "w") as f:
        json.dump({"per_task": results, "overall": overall, "baseline": BASELINE_16X}, f, indent=2)
    vol.commit()
    return {"per_task": results, "overall": overall}


@app.local_entrypoint()
def main():
    run_eval.remote()
