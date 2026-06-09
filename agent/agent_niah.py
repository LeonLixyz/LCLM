#!/usr/bin/env python3
"""Agent runner over the 8 RULER NIAH subtasks at one context length.

Two-pass per sample (triage → answer) with chunk-level memory compression.
All NIAH tasks score via ``ruler_string_match_all`` (mean over reference
answers), which matches the standard run_eval pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset

from agent import PROJECT_ROOT  # registers repo root on sys.path

from agent._lib import (  # noqa: E402
    aggregate_metrics,
    build_single_block_prompt,
    chunk_by_tokens,
    generate,
    load_agent_model,
    run_multi_round_agent,
    run_multi_round_agent_chat,
    score_response,
)

SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]

# All NIAH subtasks share the same scoring function (mean over reference answers).
SCORING_FN = {t: "ruler_string_match_all" for t in SUBTASKS}

# Per-subtask generation budget. Evidence from v1: niah_multikey_3 emitted the
# correct UUID but truncated mid-string at 64 tokens because the model first
# echoes the 36-char question key. These budgets give ~4-6× the answer length.
MAX_TOKENS = {
    "niah_single_1":   128,
    "niah_single_2":   128,
    "niah_single_3":   128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 192,
    "niah_multikey_3": 192,
    "niah_multivalue": 256,
    "niah_multiquery": 256,
}

# All NIAH subtasks run through the "agent" multi-round expand-or-answer flow:
# triage compressed chunks, then the model decides how many to expand per round.
FLOW = {t: "agent" for t in SUBTASKS}

# Default ceiling on the number of expansion rounds per sample.  Each round the
# model decides freely whether to EXPAND more chunks or commit an ANSWER.
DEFAULT_MAX_ROUNDS = 5

# These upstream datasets ship raw RULER text (input + outputs columns) so the
# agent can chunk + memwrap on its own. The "llama-3.2-tokenizer" suffix only
# means upstream sized the haystack to fit 4096/8192/... llama tokens; we
# retokenize with Qwen at runtime, so the effective Qwen-token count drifts by
# ~5% from the standard run_eval baseline (which uses latent-context/lclm-eval (config="ruler") —
# Qwen-resized — but only ships chat-templated prompts, not raw text).
DATASET_FOR_CTX = {
    4096: "SaylorTwift/RULER-4096-llama-3.2-tokenizer",
    8192: "SaylorTwift/RULER-8192-llama-3.2-tokenizer",
    16384: "SaylorTwift/RULER-16384-llama-3.2-tokenizer",
    32768: "SaylorTwift/RULER-32768-llama-3.2-tokenizer",
}


def parse_input(text: str) -> tuple[str, str, str]:
    """RULER NIAH samples are 3 lines: prefix \\n body (haystack) \\n question.

    Empty body would produce an empty <|memory_start|><|memory_end|> region
    that crashes the model, so we fall back to lenient line-count handling.
    """
    lines = text.split("\n")
    if not lines:
        return "", "", ""
    if len(lines) == 1:
        return "", lines[0], ""
    if len(lines) == 2:
        return "", lines[0], lines[1]
    if len(lines) == 3:
        return lines[0], lines[1], lines[2]
    return lines[0], "\n".join(lines[1:-1]), lines[-1]


def run_one_sample(model, decoder_tok, processor, base_tok, sample: dict, *,
                   subtask: str, chunk_size: int, device: str, do_triage: bool,
                   max_rounds: int = DEFAULT_MAX_ROUNDS) -> dict:
    text = sample["input"]
    answers = sample["outputs"]
    prefix, body, question = parse_input(text)

    flow = FLOW.get(subtask, "agent")
    if not do_triage:
        flow = "compressed"

    if flow == "compressed":
        # Single big memory block, no triage. Matches standard memwrap.
        chunks = [body]
        n_chunks = 1
        prompt = build_single_block_prompt(prefix, body, question)
        from agent._lib import clean_response  # local import to avoid cycle
        raw, ans_stats = generate(model, decoder_tok, processor, prompt,
                                  device=device, max_tokens=MAX_TOKENS.get(subtask, 128),
                                  temperature=0.0, label="ANSWER")
        response = clean_response(raw)
        gt = {"answers": answers, "length": sample.get("length"), "index": sample.get("index")}
        metrics = score_response(response, SCORING_FN[subtask], gt)
        return {
            "subtask": subtask,
            "flow": flow,
            "n_chunks": 1,
            "n_rounds": 0,
            "expanded": [],
            "transcripts": [],
            "response": response,
            "expected": answers,
            "metrics": metrics,
            "answer_stats": ans_stats,
        }

    # Multi-round agent flow: chunk body, let model decide EXPAND/ANSWER each round.
    chunks = chunk_by_tokens(body, base_tok, chunk_size=chunk_size)
    result = run_multi_round_agent(
        model, decoder_tok, processor,
        prefix=prefix, chunks=chunks, question=question,
        max_rounds=max_rounds,
        max_tokens=MAX_TOKENS.get(subtask, 128),
        device=device, label=f"{subtask} ",
    )
    response = result["response"]
    gt = {"answers": answers, "length": sample.get("length"), "index": sample.get("index")}
    metrics = score_response(response, SCORING_FN[subtask], gt)

    return {
        "subtask": subtask,
        "flow": flow,
        "n_chunks": len(chunks),
        "n_rounds": result["n_rounds"],
        "expanded": [i + 1 for i in result["expanded"]],
        "transcripts": result["transcripts"],
        "response": response,
        "expected": answers,
        "metrics": metrics,
        "answer_stats": result["answer_stats"],
    }


def run_subtask(model, decoder_tok, processor, base_tok, *, subtask: str, ctx_len: int,
                output_dir: Path, n_samples: int | None, chunk_size: int,
                device: str, do_triage: bool,
                max_rounds: int = DEFAULT_MAX_ROUNDS) -> dict:
    out_path = output_dir / f"{subtask}.jsonl"
    summary_path = output_dir / f"{subtask}_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'#'*70}\nSUBTASK {subtask} @ {ctx_len}\n{'#'*70}")
    ds = load_dataset(DATASET_FOR_CTX[ctx_len], split=subtask)
    total = min(n_samples, len(ds)) if n_samples else len(ds)

    # Resume from existing per-sample lines
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = [json.loads(line) for line in f if line.strip()]
    start = len(existing)

    out_f = open(out_path, "a")
    per_sample_metrics: list[dict] = [r["metrics"] for r in existing if "metrics" in r]
    try:
        for i in range(start, total):
            sample = ds[i]
            print(f"\n[{subtask}] sample {i+1}/{total}")
            result = run_one_sample(
                model, decoder_tok, processor, base_tok, sample,
                subtask=subtask, chunk_size=chunk_size, device=device, do_triage=do_triage,
                max_rounds=max_rounds,
            )
            result["sample_index"] = i
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            per_sample_metrics.append(result["metrics"])
            avg = aggregate_metrics(per_sample_metrics)
            score_key = next((k for k in avg if k != "__n__"), None)
            if score_key:
                print(f"  running {score_key} = {avg[score_key]:.2f} (n={avg['__n__']})")
    finally:
        out_f.close()

    summary = {
        "subtask": subtask,
        "context_length": ctx_len,
        "scoring_function": SCORING_FN[subtask],
        "metrics": aggregate_metrics(per_sample_metrics),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--context_length", type=int, default=4096, choices=sorted(DATASET_FOR_CTX))
    ap.add_argument("--chunk_size", type=int, default=256,
                    help="Token-budget per chunk (default 256). Sweep across {128,256,512,1024}.")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--subtasks", nargs="+", default=SUBTASKS,
                    help="Restrict to a subset of subtasks (default: all 13)")
    ap.add_argument("--no_triage", action="store_true",
                    help="Force compressed flow; skip multi-round agent.")
    ap.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                    help="Max EXPAND rounds before forced ANSWER (default 5).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir) / f"ctx{args.context_length}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, decoder_tok, processor, base_tok = load_agent_model(
        args.checkpoint, device=args.device, dtype=args.dtype, compression_ratio=16,
    )

    summaries: list[dict] = []
    for st in args.subtasks:
        if st not in SCORING_FN:
            print(f"Unknown subtask {st}, skipping")
            continue
        s = run_subtask(
            model, decoder_tok, processor, base_tok,
            subtask=st, ctx_len=args.context_length, output_dir=output_dir,
            n_samples=args.n_samples, chunk_size=args.chunk_size,
            device=args.device, do_triage=not args.no_triage,
            max_rounds=args.max_rounds,
        )
        summaries.append(s)

    overall = {
        "context_length": args.context_length,
        "subtasks": summaries,
    }
    with open(output_dir / "overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RULER agent — ctx {args.context_length}")
    for s in summaries:
        m = s["metrics"]
        score_key = next((k for k in m if k != "__n__"), "?")
        print(f"  {s['subtask']:20s}  {score_key} = {m.get(score_key, 0):.2f}  (n={m.get('__n__', 0)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
