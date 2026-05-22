"""RULER NIAH eval — pure Python, runs on any single-GPU machine.

Two-step end-to-end driver: prepares per-task prompts.jsonl files,
subprocess-calls ``inference.encode`` (HF encoder), then
``inference.decode`` (vLLM), then scores each task with
ruler_string_match_all (lowercase substring containment).

Usage:
    python -m inference.examples.eval_ruler_niah \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --ctx 4096 \\
        --work-dir ./_ruler_eval

The encoder side and the decoder side run as separate Python processes
so vLLM (which eagerly fills the GPU for its KV cache) never has to
share device memory with the HF encoder. Same pattern as the old
step1/step2 .npy hand-off, just shipping a .pt that holds a list of
prompt_embeds tensors.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

NIAH_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]


def _ruler_match(ans: str, resp: str) -> bool:
    return ans.lower() in resp.lower()


def _score_all(answers, response):
    if not answers:
        return 0.0
    return sum(1.0 for a in answers if _ruler_match(a, response)) / len(answers) * 100.0


def _prepare_prompts(work_dir: Path, ctx: int) -> dict[str, int]:
    """Materialize per-task prompts.jsonl files."""
    from datasets import load_dataset
    ds = load_dataset("latent-context/ruler-full", "memwrap")
    rows = ds[list(ds.keys())[0]]
    counts = {}
    for task in NIAH_SUBTASKS:
        target_cat = f"memwrap/ruler/{task}_{ctx}"
        task_rows = rows.filter(lambda r: r["category"] == target_cat)
        out = work_dir / f"{task}_prompts.jsonl"
        with open(out, "w") as f:
            for r in task_rows:
                gt = (r.get("extra_info") or {}).get("ground_truth") or {}
                answers = gt.get("answers") or r.get("answer") or r.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers]
                f.write(json.dumps({"prompt": r["prompt"], "answers": answers}) + "\n")
        counts[task] = len(task_rows)
        print(f"  [{task}] {len(task_rows)} prompts -> {out}", flush=True)
    return counts


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"command failed (rc={rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LCLM checkpoint dir or HF repo id")
    ap.add_argument("--ctx", type=int, default=4096, help="RULER context length (4096/8192/16384/32768)")
    ap.add_argument("--work-dir", required=True, help="Output directory for prompts/embeds/completions")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-encode-batch-size", type=int, default=128)
    ap.add_argument("--tasks", default=",".join(NIAH_SUBTASKS),
                    help="Comma-separated subset of NIAH subtasks (default: all 8)")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing prompts (ctx={args.ctx}, {len(tasks)} tasks) ===", flush=True)
    _prepare_prompts(work, args.ctx)

    print(f"\n=== Stage 1: encode (HF) ===", flush=True)
    for task in tasks:
        _run([
            sys.executable, "-m", "inference.encode",
            "--checkpoint", args.checkpoint,
            "--prompts-jsonl", str(work / f"{task}_prompts.jsonl"),
            "--out", str(work / f"{task}_embeds.pt"),
            "--max-encode-batch-size", str(args.max_encode_batch_size),
        ])

    print(f"\n=== Stage 2: decode (vLLM) ===", flush=True)
    for task in tasks:
        _run([
            sys.executable, "-m", "inference.decode",
            "--checkpoint", args.checkpoint,
            "--embeds-pt", str(work / f"{task}_embeds.pt"),
            "--out", str(work / f"{task}_completions.jsonl"),
            "--max-tokens", str(args.max_tokens),
            "--temperature", "0.0",
        ])

    print(f"\n=== Scoring (ruler_string_match_all) ===", flush=True)
    print(f"  {'task':22s} {'score':>8s}", flush=True)
    results = {}
    for task in tasks:
        scores = []
        with open(work / f"{task}_completions.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                scores.append(_score_all(rec["answers"], rec["response"]))
        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = avg
        print(f"  {task:22s} {avg:8.2f}", flush=True)
    overall = sum(results.values()) / len(results) if results else 0.0
    print(f"  {'overall':22s} {overall:8.2f}", flush=True)

    with open(work / "summary.json", "w") as f:
        json.dump({"per_task": results, "overall": overall, "ctx": args.ctx,
                   "checkpoint": args.checkpoint}, f, indent=2)


if __name__ == "__main__":
    main()
