"""RULER NIAH eval — runs ``inference.vllm_inference.encode`` + ``inference.vllm_inference.decode`` over
already-prepared per-task prompts.jsonl files and scores each with
ruler_string_match_all (lowercase substring containment).

The prompts.jsonl files are expected to be ``{"prompt": ..., "answers": [...]}``
per line — see ``inference.examples.prepare_ruler_niah`` for one way to
materialize them from the ``latent-context/lclm-eval (config="ruler")`` dataset.

Encode and decode run as separate Python processes so vLLM (which fills
the whole GPU for its KV cache) never has to share device memory with
the HF encoder.

Usage:
    python -m inference.examples.eval_ruler_niah \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --prompts-dir ./ruler_4k_prompts \\
        --work-dir   ./_ruler_4k_eval
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def _ruler_match(ans: str, resp: str) -> bool:
    return ans.lower() in resp.lower()


def _score_all(answers, response):
    if not answers:
        return 0.0
    return sum(1.0 for a in answers if _ruler_match(a, response)) / len(answers) * 100.0


def _run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"command failed (rc={rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LCLM checkpoint dir or HF repo id")
    ap.add_argument("--prompts-dir", required=True,
                    help="Directory of per-task prompts.jsonl files (e.g. niah_single_1.jsonl)")
    ap.add_argument("--work-dir", required=True, help="Output dir (embeds + completions land here)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-encode-batch-size", type=int, default=128)
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(prompts_dir.glob("*.jsonl"))
    if not prompt_files:
        sys.exit(f"No *.jsonl files in {prompts_dir}")
    tasks = [p.stem for p in prompt_files]
    print(f"Found {len(tasks)} task(s): {', '.join(tasks)}", flush=True)

    print(f"\n=== Stage 1: encode (HF) ===", flush=True)
    for task, pf in zip(tasks, prompt_files):
        _run([
            sys.executable, "-m", "inference.vllm_inference.encode",
            "--checkpoint", args.checkpoint,
            "--prompts-jsonl", str(pf),
            "--out", str(work / f"{task}_embeds.pt"),
            "--max-encode-batch-size", str(args.max_encode_batch_size),
        ])

    print(f"\n=== Stage 2: decode (vLLM) ===", flush=True)
    for task in tasks:
        _run([
            sys.executable, "-m", "inference.vllm_inference.decode",
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
                scores.append(_score_all(rec.get("answers", []), rec["response"]))
        avg = sum(scores) / len(scores) if scores else 0.0
        results[task] = avg
        print(f"  {task:22s} {avg:8.2f}", flush=True)
    overall = sum(results.values()) / len(results) if results else 0.0
    print(f"  {'overall':22s} {overall:8.2f}", flush=True)

    with open(work / "summary.json", "w") as f:
        json.dump({"per_task": results, "overall": overall,
                   "checkpoint": args.checkpoint,
                   "prompts_dir": str(prompts_dir)}, f, indent=2)


if __name__ == "__main__":
    main()
