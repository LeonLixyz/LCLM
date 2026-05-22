"""Stage 2 / 2 — decode prompt_embeds with vLLM.

Loads only the LCLM *decoder* via vllm.LLM (with ``enable_prompt_embeds=True``),
reads the tensors produced by ``encode.py``, runs batched generation, and
writes per-sample completions to a JSONL.

Usage:
    python -m inference.vllm_inference.decode \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --embeds-pt prompt_embeds.pt \\
        --out completions.jsonl \\
        --max-tokens 128 --temperature 0.0
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch


def _resolve_decoder_path(checkpoint: str) -> str:
    """Local dir → return as-is; HF repo id → snapshot_download first."""
    if os.path.isdir(checkpoint):
        return os.path.join(checkpoint, "decoder")
    if "/" in checkpoint and not checkpoint.startswith("/"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=checkpoint, repo_type="model")
        return os.path.join(local, "decoder")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="LCLM checkpoint dir or HF repo id")
    ap.add_argument("--embeds-pt", required=True, help="Tensor file from encode.py")
    ap.add_argument("--out", required=True, help="Output JSONL with completions")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    decoder_path = _resolve_decoder_path(args.checkpoint)
    print(f"Loading vLLM decoder from {decoder_path} ...", flush=True)
    llm = LLM(
        model=decoder_path,
        enable_prompt_embeds=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"Loading embeds from {args.embeds_pt} ...", flush=True)
    blob = torch.load(args.embeds_pt, map_location="cpu", weights_only=False)
    embeds = blob["embeds"]
    meta = blob.get("meta", [{}] * len(embeds))
    print(f"  {len(embeds)} prompt embeds", flush=True)

    sp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    t0 = time.time()
    outputs = llm.generate(
        [{"prompt_embeds": e} for e in embeds],
        sampling_params=sp,
    )
    elapsed = time.time() - t0
    print(f"Generated {len(outputs)} in {elapsed:.1f}s "
          f"({len(outputs)/elapsed:.1f}/s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for o, m in zip(outputs, meta):
            record = {**m, "response": o.outputs[0].text}
            f.write(json.dumps(record) + "\n")
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
