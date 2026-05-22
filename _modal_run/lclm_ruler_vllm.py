"""RULER NIAH eval on latent-context/0.6b-4b-LCLM-16x via the two-step
encode → decode flow.

Single Modal H200 container. Calls ``python -m inference.encode`` to
produce prompt_embeds.pt, then ``python -m inference.decode`` to run vLLM.
Two distinct Python processes → each gets its own GPU-memory budget (no
HF-encoder-vs-vLLM-KV-cache contention).

Run detached:
    modal run --detach _modal_run/lclm_ruler_vllm.py
"""
from __future__ import annotations
import modal, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("lclm-ruler-two-step")
vol = modal.Volume.from_name("lclm-test")

LCLM_IGNORE = [
    "__pycache__/", ".git/", "*.safetensors", "*.bin", "*.pt",
    "_smoketest_ckpts/", "_matrix_ckpts/", "_modal_run/",
    "data/packed_batches/", "wandb/", ".venv/", "*.parquet",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.9.1", "transformers>=4.45", "peft", "safetensors",
        "huggingface-hub", "datasets", "pyyaml", "numpy",
        "wheel", "packaging", "ninja",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("vllm>=0.7")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

REPO_ID = "latent-context/0.6b-4b-LCLM-16x"
WORK_DIR = "/vol/ruler_16x_two_step"

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


@app.function(image=image, gpu="H200", timeout=4*60*60, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def ruler_niah_two_step():
    import os, json, subprocess, time
    os.chdir("/root/LCLM")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/LCLM"

    from datasets import load_dataset
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    rows = ds[list(ds.keys())[0]]

    os.makedirs(WORK_DIR, exist_ok=True)

    # Materialize one prompts.jsonl per task on the volume; carry the answer
    # list as a meta field that encode.py will pass through to decode.py.
    task_info = {}
    for task in NIAH_SUBTASKS:
        target_cat = f"memwrap/ruler/{task}_4096"
        task_rows = rows.filter(lambda r: r["category"] == target_cat)
        prompts_path = f"{WORK_DIR}/{task}_prompts.jsonl"
        with open(prompts_path, "w") as f:
            for r in task_rows:
                gt = r.get("extra_info", {}).get("ground_truth", {}) if isinstance(r.get("extra_info"), dict) else {}
                answers = gt.get("answers") or r.get("answer") or r.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers]
                f.write(json.dumps({"prompt": r["prompt"], "answers": answers}) + "\n")
        task_info[task] = len(task_rows)
        print(f"  [{task}] wrote {len(task_rows)} prompts → {prompts_path}", flush=True)

    # === Stage 1: encode (own subprocess, releases GPU memory on exit) ===
    print(f"\n>>> Stage 1: encode (LCLM in HF)", flush=True)
    t0 = time.time()
    for task in NIAH_SUBTASKS:
        cmd = [
            "python", "-m", "inference.encode",
            "--checkpoint", REPO_ID,
            "--prompts-jsonl", f"{WORK_DIR}/{task}_prompts.jsonl",
            "--out", f"{WORK_DIR}/{task}_embeds.pt",
            "--max-encode-batch-size", "128",
        ]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            raise RuntimeError(f"encode failed for {task} (rc={rc})")
    vol.commit()
    print(f"  encode total wall time: {time.time()-t0:.1f}s", flush=True)

    # === Stage 2: decode (own subprocess, fresh GPU for vLLM) ===
    print(f"\n>>> Stage 2: decode (vLLM)", flush=True)
    t0 = time.time()
    for task in NIAH_SUBTASKS:
        cmd = [
            "python", "-m", "inference.decode",
            "--checkpoint", REPO_ID,
            "--embeds-pt", f"{WORK_DIR}/{task}_embeds.pt",
            "--out", f"{WORK_DIR}/{task}_completions.jsonl",
            "--max-tokens", "128",
            "--temperature", "0.0",
        ]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            raise RuntimeError(f"decode failed for {task} (rc={rc})")
    vol.commit()
    print(f"  decode total wall time: {time.time()-t0:.1f}s", flush=True)

    # === Score ===
    def _match(a, r): return a.lower() in r.lower()
    def _score_all(ans, resp):
        if not ans: return 0.0
        return sum(1.0 for a in ans if _match(a, resp)) / len(ans) * 100.0

    print(f"\n========== 16x NIAH SUMMARY ==========", flush=True)
    print(f"  {'task':22s} {'NEW':>8s} {'BASELINE':>10s}  {'Δ':>8s}", flush=True)
    results = {}
    for task in NIAH_SUBTASKS:
        scores = []
        with open(f"{WORK_DIR}/{task}_completions.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                scores.append(_score_all(rec["answers"], rec["response"]))
        avg = sum(scores) / len(scores) if scores else 0.0
        base = BASELINE_16X[task]
        results[task] = avg
        print(f"  {task:22s} {avg:8.2f} {base:10.2f}  {avg-base:+8.2f}", flush=True)
    overall = sum(results.values()) / len(results) if results else 0.0
    baseline_overall = sum(BASELINE_16X.values()) / len(BASELINE_16X)
    print(f"  {'NIAH overall':22s} {overall:8.2f} {baseline_overall:10.2f}  {overall-baseline_overall:+8.2f}", flush=True)
    return {"per_task": results, "overall": overall}


@app.local_entrypoint()
def main():
    ruler_niah_two_step.remote()
