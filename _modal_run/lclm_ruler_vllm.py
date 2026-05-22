"""RULER 4k NIAH eval on latent-context/0.6b-4b-LCLM-16x via vLLM.

Loads latent-context/ruler-full memwrap variant (Qwen-tokenized prompts),
filters to the 8 NIAH subtasks at 4096-token context, runs ALL prompts
through LCLMVLLMDecoder (encoder batched, decoder vLLM continuous-batched),
and reports per-task + overall ruler_string_match_all numbers.

Run detached:
    modal run --detach _modal_run/lclm_ruler_vllm.py
"""
from __future__ import annotations
import modal, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("lclm-ruler-vllm")

LCLM_IGNORE = [
    "__pycache__/", ".git/", "*.safetensors", "*.bin", "*.pt",
    "_smoketest_ckpts/", "_matrix_ckpts/", "_modal_run/",
    "data/packed_batches/", "wandb/", ".venv/", "*.parquet",
]

# Single image with vllm + flash-attn (CUDA-devel base so vLLM can JIT
# its flashinfer kernels at first call).
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.7", "transformers", "peft", "safetensors", "pyyaml",
                 "torchdata", "datasets", "huggingface-hub")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

NIAH_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]

# Baseline cs16 ruler_4096 numbers from the volume (per-task ruler_match%):
BASELINE_16X = {
    "niah_single_1": 96.80, "niah_single_2": 90.00, "niah_single_3": 55.20,
    "niah_multikey_1": 86.40, "niah_multikey_2": 82.20, "niah_multikey_3": 49.00,
    "niah_multivalue": 74.10, "niah_multiquery": 81.45,
}


def _ruler_match(ans: str, resp: str) -> bool:
    return ans.lower() in resp.lower()


def _score_all(answers, response):
    """ruler_string_match_all: mean over reference answers (0..100)."""
    if not answers: return 0.0
    return sum(1.0 for a in answers if _ruler_match(a, response)) / len(answers) * 100.0


@app.function(image=vllm_image, gpu="H200", timeout=2*60*60)
def ruler_niah_vllm():
    import sys, time
    sys.path.insert(0, "/root/LCLM")
    from datasets import load_dataset
    from inference.vllm import LCLMVLLMDecoder

    REPO_ID = "latent-context/0.6b-4b-LCLM-16x"
    print(f">>> Loading {REPO_ID} via LCLMVLLMDecoder...", flush=True)
    runner = LCLMVLLMDecoder(
        checkpoint_path=REPO_ID,
        tensor_parallel_size=1,
        max_encode_batch_size=128,  # ≈128k tokens/encoder pass at W=1024
    )

    print("Loading latent-context/ruler-full (memwrap)...", flush=True)
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    split_name = list(ds.keys())[0]
    rows = ds[split_name]
    print(f"  using split '{split_name}': {len(rows)} rows", flush=True)
    print(f"  columns: {rows.column_names}", flush=True)

    # Build (task, [(prompt_str, answers_list), ...]) for each NIAH subtask at 4096
    results = {}
    total_start = time.time()
    for task in NIAH_SUBTASKS:
        target_cat = f"memwrap/ruler/{task}_4096"
        task_rows = rows.filter(lambda r: r["category"] == target_cat)
        prompts = []
        answer_lists = []
        for row in task_rows:
            prompts.append(row["prompt"])
            gt = row.get("extra_info", {}).get("ground_truth", {}) if isinstance(row.get("extra_info"), dict) else {}
            answers = gt.get("answers") if gt else None
            if answers is None:
                answers = row.get("answer") or row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            answer_lists.append(answers)
        print(f"  [{task}] {len(prompts)} samples", flush=True)

        # vLLM batched generation (encoder batched too — single _process_latent_embeddings call)
        t0 = time.time()
        responses = runner.generate(
            prompts=prompts,
            max_tokens=128,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        rate = len(prompts) / elapsed if elapsed > 0 else 0.0

        # Score
        per_sample = [_score_all(a, r) for a, r in zip(answer_lists, responses)]
        avg = sum(per_sample) / len(per_sample) if per_sample else 0.0
        results[task] = avg
        base = BASELINE_16X.get(task, 0.0)
        delta = avg - base
        print(f"    -> {avg:6.2f}  baseline={base:6.2f}  Δ={delta:+6.2f}  "
              f"({len(prompts)} samples in {elapsed:.1f}s = {rate:.1f}/s)", flush=True)

    overall = sum(results.values()) / len(results) if results else 0.0
    baseline_overall = sum(BASELINE_16X.values()) / len(BASELINE_16X)
    total_elapsed = time.time() - total_start
    print(f"\n========== 16x NIAH SUMMARY (vLLM) ==========", flush=True)
    print(f"  {'task':22s} {'NEW':>8s} {'BASELINE':>10s}  {'Δ':>8s}", flush=True)
    for task in NIAH_SUBTASKS:
        new = results[task]
        base = BASELINE_16X[task]
        print(f"  {task:22s} {new:8.2f} {base:10.2f}  {new-base:+8.2f}", flush=True)
    print(f"  {'NIAH overall':22s} {overall:8.2f} {baseline_overall:10.2f}  {overall-baseline_overall:+8.2f}", flush=True)
    print(f"\n  total wall time: {total_elapsed:.1f}s for {sum(len(rows.filter(lambda r: r['category'] == f'memwrap/ruler/{t}_4096')) for t in NIAH_SUBTASKS)} samples", flush=True)
    return {"per_task": results, "overall": overall, "baseline_overall": baseline_overall}


@app.local_entrypoint()
def main():
    print("Dispatching ruler_niah_vllm()...", flush=True)
    ruler_niah_vllm.remote()
