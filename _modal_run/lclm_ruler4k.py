"""RULER 4k eval on the 3 migrated checkpoints.

Loads latent-context/ruler-full memwrap variant, runs all 13 tasks @ 4096-ctx
(500 samples each, 6500 total), scores with the official RULER substring
match, and prints per-task + overall numbers next to the baseline.

Backend: HF inference (matches the baseline run conditions).

Run detached:
    modal run --detach _modal_run/lclm_ruler4k.py
"""
from __future__ import annotations
import modal, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("lclm-ruler4k")
lclm_vol = modal.Volume.from_name("lclm-test")

LCLM_IGNORE = [
    "__pycache__/", ".git/", "*.safetensors", "*.bin", "*.pt",
    "_smoketest_ckpts/", "_matrix_ckpts/", "_modal_run/",
    "data/packed_batches/", "wandb/", ".venv/", "*.parquet",
]

eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.9.1", "transformers>=4.45", "accelerate",
        "peft", "torchdata", "safetensors", "huggingface-hub",
        "pyyaml", "numpy", "datasets",
        "wheel", "packaging", "ninja",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

CASES = ["cs4-causal-mlp-3e5", "cs8-causal-mlp-3e5", "cs16-causal-mlp-3e5"]

# Per the baseline metrics on the volume
BASELINES = {
    "cs4-causal-mlp-3e5":  {"overall": 93.05, "niah_single_1": 99.40, "niah_single_2": 99.60,
                            "niah_single_3": 93.60, "niah_multikey_1": 99.00, "niah_multikey_2": 99.40,
                            "niah_multikey_3": 94.20, "niah_multivalue": 98.25, "niah_multiquery": 99.20,
                            "vt": 94.64, "fwe": 86.73, "cwe": 91.42, "qa_1": 81.80, "qa_2": 72.40},
    "cs8-causal-mlp-3e5":  {"overall": 87.10, "niah_single_1": 98.80, "niah_single_2": 95.40,
                            "niah_single_3": 64.40, "niah_multikey_1": 93.00, "niah_multikey_2": 98.20,
                            "niah_multikey_3": 82.80, "niah_multivalue": 92.40, "niah_multiquery": 91.55,
                            "vt": 87.40, "fwe": 90.60, "cwe": 90.80, "qa_1": 77.60, "qa_2": 69.40},
    "cs16-causal-mlp-3e5": {"overall": 77.07, "niah_single_1": 96.80, "niah_single_2": 90.00,
                            "niah_single_3": 55.20, "niah_multikey_1": 86.40, "niah_multikey_2": 82.20,
                            "niah_multikey_3": 49.00, "niah_multivalue": 74.10, "niah_multiquery": 81.45,
                            "vt": 83.92, "fwe": 88.27, "cwe": 80.54, "qa_1": 71.20, "qa_2": 62.80},
}

SCORING_MAP = {
    "niah_single_1": "all", "niah_single_2": "all", "niah_single_3": "all",
    "niah_multikey_1": "all", "niah_multikey_2": "all", "niah_multikey_3": "all",
    "niah_multivalue": "all", "niah_multiquery": "all",
    "vt": "all", "fwe": "all", "cwe": "all",
    "qa_1": "part", "qa_2": "part",
}


def _ruler_match(ans, resp): return ans.lower() in resp.lower()

def _score(scoring_type, answers, response):
    if scoring_type == "part":
        return 100.0 if any(_ruler_match(a, response) for a in answers) else 0.0
    if not answers: return 0.0
    return sum(1.0 for a in answers if _ruler_match(a, response)) / len(answers) * 100.0


@app.function(image=eval_image, gpu="H200", timeout=2*60*60, volumes={"/lclm": lclm_vol})
def ruler_4k(case_name: str):
    import sys, time, torch
    sys.path.insert(0, "/root/LCLM")
    from datasets import load_dataset
    from inference.hf import load_model

    ckpt = f"/lclm/migrated/{case_name}"
    print(f">>> RULER 4k on {case_name}", flush=True)
    print(f"Loading {ckpt}...", flush=True)
    model, tok, proc = load_model(ckpt, device="cuda", dtype="bf16")
    model.eval()

    # latent-context/ruler-full has separate configs per variant (memwrap / plain).
    print("Loading latent-context/ruler-full (memwrap config)...", flush=True)
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    print(f"  splits in memwrap: {list(ds.keys())}", flush=True)
    split_name = list(ds.keys())[0]
    rows = ds[split_name]
    print(f"  using split '{split_name}': {len(rows)} rows", flush=True)
    print(f"  columns: {rows.column_names}", flush=True)

    # Filter to 4096 entries; the existing dirs used category strings like
    # "memwrap/ruler/niah_single_1_4096". Discover the exact structure.
    cat_field = "category" if "category" in rows.column_names else None
    if cat_field:
        cats = set(rows[cat_field])
        cats_4k = sorted(c for c in cats if c.endswith("_4096"))
        print(f"  4096 categories: {len(cats_4k)} (sample: {cats_4k[:3]})", flush=True)
    else:
        cats_4k = []
        print(f"  no 'category' field; columns are {rows.column_names}", flush=True)

    # Iterate tasks
    results = {}
    overall_scores = []
    start = time.time()
    n_samples = 0
    for task_name, scoring_type in SCORING_MAP.items():
        target_cat = f"memwrap/ruler/{task_name}_4096"
        task_rows = rows.filter(lambda r: r["category"] == target_cat)
        n = len(task_rows)
        print(f"  [{task_name}] {n} samples", flush=True)
        task_scores = []
        for i in range(n):
            row = task_rows[i]
            prompt = row.get("prompt") or row.get("input") or row.get("question")
            gt = row.get("ground_truth") or {}
            answers = gt.get("answers") if isinstance(gt, dict) else None
            if answers is None:
                answers = row.get("answer") or row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]

            # Tokenize + generate (greedy, max 128 new tokens)
            from inference.hf import generate_text
            response = generate_text(model, tok, proc, prompt, device="cuda",
                                     max_tokens=128, temperature=0.0)
            task_scores.append(_score(scoring_type, answers, response))
            n_samples += 1
        avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
        results[task_name] = avg
        overall_scores.append(avg)
        print(f"    -> {avg:.2f}", flush=True)
    elapsed = time.time() - start
    results["__overall__"] = sum(overall_scores) / len(overall_scores)
    results["__runtime__"] = {"seconds": elapsed, "samples": n_samples,
                              "samples_per_second": n_samples / elapsed if elapsed > 0 else 0.0}
    return {"name": case_name, "results": results}


@app.function(image=eval_image, cpu=2, timeout=4*60*60, volumes={"/lclm": lclm_vol})
def orchestrate():
    print("RULER 4k eval on 3 migrated checkpoints (parallel on H200s)...", flush=True)
    results = list(ruler_4k.map(CASES))
    print("\n========== RULER 4k SUMMARY ==========", flush=True)
    for r in results:
        name = r["name"]
        scores = r["results"]
        baseline = BASELINES[name]
        rt = scores.pop("__runtime__", {})
        print(f"\n{name}: ({rt.get('samples', '?')} samples in {rt.get('seconds', 0):.1f}s = {rt.get('samples_per_second', 0):.1f}/s)", flush=True)
        print(f"  {'task':22s} {'NEW':>8s} {'BASELINE':>10s}  {'Δ':>8s}", flush=True)
        for task in list(SCORING_MAP.keys()) + ["__overall__"]:
            base_key = "overall" if task == "__overall__" else task
            new = scores.get(task, 0.0)
            base = baseline.get(base_key, 0.0)
            delta = new - base
            print(f"  {task:22s} {new:8.2f} {base:10.2f}  {delta:+8.2f}", flush=True)
    return results


@app.local_entrypoint()
def main():
    print("Dispatching orchestrate()...", flush=True)
    orchestrate.remote()
