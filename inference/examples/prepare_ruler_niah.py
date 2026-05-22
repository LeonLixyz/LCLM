"""Prepare RULER NIAH prompts.jsonl files from latent-context/ruler-full.

Writes one JSONL per NIAH subtask into ``--out-dir``. Each line is a
record ``{"prompt": "<chat-templated str>", "answers": ["..."]}`` ready
for ``inference.vllm_inference.encode`` (whose ``meta`` pass-through carries the
``answers`` field through to ``inference.vllm_inference.decode``'s output JSONL, where
a scorer can pick them up).

Usage:
    python -m inference.examples.prepare_ruler_niah \\
        --ctx 4096 \\
        --out-dir ./ruler_4k_prompts
"""
import argparse
import json
from pathlib import Path

NIAH_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096,
                    help="RULER context length (4096/8192/16384/32768)")
    ap.add_argument("--out-dir", required=True, help="Output dir for per-task .jsonl files")
    ap.add_argument("--tasks", default=",".join(NIAH_SUBTASKS),
                    help="Comma-separated NIAH subtasks (default: all 8)")
    args = ap.parse_args()

    from datasets import load_dataset

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("latent-context/ruler-full", "memwrap")
    rows = ds[list(ds.keys())[0]]
    print(f"Loaded {len(rows)} memwrap rows", flush=True)

    for task in tasks:
        cat = f"memwrap/ruler/{task}_{args.ctx}"
        task_rows = rows.filter(lambda r: r["category"] == cat)
        path = out / f"{task}.jsonl"
        with open(path, "w") as f:
            for r in task_rows:
                gt = (r.get("extra_info") or {}).get("ground_truth") or {}
                answers = gt.get("answers") or r.get("answer") or r.get("answers") or []
                if isinstance(answers, str):
                    answers = [answers]
                f.write(json.dumps({"prompt": r["prompt"], "answers": answers}) + "\n")
        print(f"  [{task}] {len(task_rows)} prompts -> {path}", flush=True)


if __name__ == "__main__":
    main()
