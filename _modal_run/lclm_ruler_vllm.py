"""RULER NIAH eval — two Modal containers, .pt hand-off on shared volume.

Encode container: torch + flash-attn + LCLM. Runs ``inference.encode``
    over each task's prompts.jsonl, writes ``<task>_embeds.pt`` to volume.

Decode container: vLLM. Runs ``inference.decode`` over each .pt, writes
    ``<task>_completions.jsonl`` to volume.

Why two containers? Pip's dependency resolution can't keep
flash-attn (built against torch X+cuY) coexisting with vllm (which
needs torch X'+cuY') in a single image. Two images, two pins.

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

# Encode side: torch 2.9.1 + flash-attn. No vllm.
encode_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.9.1", "transformers>=4.45", "peft", "safetensors",
        "huggingface-hub", "datasets", "pyyaml", "numpy",
        "wheel", "packaging", "ninja",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

# Decode side: vllm. No flash-attn (load_model falls back to sdpa for
# the encoder embed-table lookup; vLLM is what handles the decoder
# attention anyway).
decode_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.7", "transformers", "peft", "safetensors",
                 "huggingface-hub", "torch")
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


# --- helpers ----------------------------------------------------------------

def _materialize_prompts(rows, ctx: int, work_dir: str):
    """Write per-task prompts.jsonl with {prompt, answers}."""
    import os, json
    os.makedirs(work_dir, exist_ok=True)
    for task in NIAH_SUBTASKS:
        cat = f"memwrap/ruler/{task}_{ctx}"
        task_rows = rows.filter(lambda r: r["category"] == cat)
        path = f"{work_dir}/{task}_prompts.jsonl"
        with open(path, "w") as f:
            for r in task_rows:
                gt = (r.get("extra_info") or {}).get("ground_truth") or {}
                answers = gt.get("answers") or r.get("answer") or r.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers]
                f.write(json.dumps({"prompt": r["prompt"], "answers": answers}) + "\n")
        print(f"  [{task}] wrote {len(task_rows)} prompts", flush=True)


# --- stage 1: encode ---------------------------------------------------------

@app.function(image=encode_image, gpu="H200", timeout=4*60*60,
              volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def encode_all():
    import os, subprocess, time
    os.chdir("/root/LCLM")
    env = os.environ.copy(); env["PYTHONPATH"] = "/root/LCLM"

    from datasets import load_dataset
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    rows = ds[list(ds.keys())[0]]

    _materialize_prompts(rows, 4096, WORK_DIR)

    print(f"\n>>> Stage 1: encode (HF + flash-attn)", flush=True)
    t_total = time.time()
    for task in NIAH_SUBTASKS:
        rc = subprocess.run([
            "python", "-m", "inference.encode",
            "--checkpoint", REPO_ID,
            "--prompts-jsonl", f"{WORK_DIR}/{task}_prompts.jsonl",
            "--out", f"{WORK_DIR}/{task}_embeds.pt",
            "--max-encode-batch-size", "128",
        ], env=env).returncode
        if rc != 0:
            raise RuntimeError(f"encode failed for {task} (rc={rc})")
    vol.commit()
    print(f"  encode total wall time: {time.time()-t_total:.1f}s", flush=True)
    return {"tasks": NIAH_SUBTASKS}


# --- stage 2: decode + score -------------------------------------------------

@app.function(image=decode_image, gpu="H200", timeout=4*60*60,
              volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def decode_and_score():
    import os, subprocess, json, time
    os.chdir("/root/LCLM")
    env = os.environ.copy(); env["PYTHONPATH"] = "/root/LCLM"

    print(f">>> Stage 2: decode (vLLM)", flush=True)
    t_total = time.time()
    for task in NIAH_SUBTASKS:
        embeds = f"{WORK_DIR}/{task}_embeds.pt"
        if not os.path.isfile(embeds):
            raise RuntimeError(f"missing {embeds}")
        rc = subprocess.run([
            "python", "-m", "inference.decode",
            "--checkpoint", REPO_ID,
            "--embeds-pt", embeds,
            "--out", f"{WORK_DIR}/{task}_completions.jsonl",
            "--max-tokens", "128",
            "--temperature", "0.0",
        ], env=env).returncode
        if rc != 0:
            raise RuntimeError(f"decode failed for {task} (rc={rc})")
    vol.commit()
    print(f"  decode total wall time: {time.time()-t_total:.1f}s", flush=True)

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
                scores.append(_score_all(rec.get("answers", []), rec["response"]))
        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = avg
        base = BASELINE_16X[task]
        print(f"  {task:22s} {avg:8.2f} {base:10.2f}  {avg-base:+8.2f}", flush=True)
    overall = sum(results.values()) / len(results) if results else 0.0
    baseline_overall = sum(BASELINE_16X.values()) / len(BASELINE_16X)
    print(f"  {'NIAH overall':22s} {overall:8.2f} {baseline_overall:10.2f}  {overall-baseline_overall:+8.2f}", flush=True)
    return {"per_task": results, "overall": overall}


# --- driver ------------------------------------------------------------------

@app.function(image=decode_image, cpu=2, timeout=8*60*60, volumes={"/vol": vol})
def orchestrate():
    encode_all.remote()
    return decode_and_score.remote()


@app.local_entrypoint()
def main():
    orchestrate.remote()
