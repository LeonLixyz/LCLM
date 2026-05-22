"""Stage 1 / 2 — encode prompts into vLLM-ready prompt_embeds tensors.

Loads an LCLM checkpoint in HuggingFace Transformers, runs the encoder +
splice over each prompt, saves the resulting prompt-embedding tensors to
a single .pt file. The companion ``decode.py`` script loads that file and
runs vLLM. Splitting the two halves into separate processes avoids the OOM
you get when vLLM eagerly grabs the entire GPU.

Usage:
    python -m inference.vllm_inference.encode \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --prompts-jsonl prompts.jsonl \\
        --out prompt_embeds.pt

``prompts.jsonl`` may contain one of:
    {"prompt": "<chat-templated string>", ...}            # used as-is
    {"prompt": [{"role": "user", "content": "..."}], ...} # chat-templated by the encoder
extra fields are passed through into the output file's ``meta`` list
(useful for carrying answer lists, sample indices, etc.).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LCLM checkpoint dir or HF repo id")
    ap.add_argument("--prompts-jsonl", required=True, help="JSONL with one record per prompt")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-encode-batch-size", type=int, default=128,
                    help="Encoder mini-batch cap in windows (W=1024 → ~128k tokens at default)")
    args = ap.parse_args()

    # Lazy imports so --help is fast and the import error path is obvious.
    from latent_context import from_pretrained
    from inference.vllm_inference._prompt_embeds import _build_prompt_embeds_batch

    records = [json.loads(line) for line in Path(args.prompts_jsonl).read_text().splitlines() if line.strip()]
    prompts = [r["prompt"] for r in records]
    meta = [{k: v for k, v in r.items() if k != "prompt"} for r in records]
    print(f"Loaded {len(prompts)} prompts from {args.prompts_jsonl}", flush=True)

    print(f"Loading LCLM from {args.checkpoint} ...", flush=True)
    model, _, _ = from_pretrained(args.checkpoint, device=args.device, dtype=args.dtype)
    model.encoder.max_encode_batch_size = max(0, args.max_encode_batch_size)
    model.eval()

    print("Encoding prompts ...", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        embeds_list = _build_prompt_embeds_batch(prompts, model)
    embeds_cpu = [e.float().cpu() for e in embeds_list]
    elapsed = time.time() - t0
    print(f"  {len(prompts)} prompts in {elapsed:.1f}s "
          f"({len(prompts)/elapsed:.1f}/s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeds": embeds_cpu, "meta": meta}, args.out)
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
