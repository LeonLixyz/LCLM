"""LCLM TTFT and peak GPU memory vs. context length, Hugging Face decoder.

LCLM side of Figure 4 (4k to 1M tokens) and of the TTFT axis in Figure 1
(4k, 8k, 16k and 64k; see ``BENCHMARK_CONTEXT_TOKENS``).

Workload per context length N: one segment of N encoder tokens (``" the"``
repeated), encoded by the optimized encoder (``inference/optimized_encoder.py``)
into ceil(N / ratio) latents, then decoder prefill over the latents in
Hugging Face Transformers with PyTorch SDPA and logits at every position (as
in the KV-cache baseline harness), then argmax of the last position.

TTFT = encoder + adapter + decoder prefill + first token. Each length runs
``--warmups`` unrecorded trials and ``--repeats`` measured trials; the
reported TTFT is the fastest measured trial and peak memory is the maximum of
``torch.cuda.max_memory_allocated`` over measured trials (reset before each).

    python -m benchmarks.timing.lclm_hf_scaling \\
        --checkpoint latent-context/0.6b-4b-LCLM-16x \\
        --output runs/timing/lclm_hf_scaling.jsonl
"""
import argparse
import gc
import math
import os
import time

from benchmarks.timing.common import (
    CONTEXT_LENGTHS,
    LCLM_CHECKPOINTS,
    append_jsonl,
    environment_info,
    parse_length_list,
    run_trials,
    summarize_trials,
)

LATENT_CHECK_TOKENS = 4096


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=LCLM_CHECKPOINTS[16],
                   help="LCLM checkpoint directory or Hugging Face model id")
    p.add_argument("--lengths", type=parse_length_list, default=list(CONTEXT_LENGTHS),
                   help="comma-separated context lengths, e.g. 4k,64k,1M")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--batch-windows", type=int, default=32,
                   help="encoder windows per forward pass")
    p.add_argument("--no-compile", action="store_true",
                   help="skip torch.compile (keeps Flash SDPA and CUDA graphs)")
    p.add_argument("--decoder-attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--skip-latent-check", action="store_true",
                   help="skip comparing optimized and stock encoder latents at 4k")
    p.add_argument("--expandable-segments", action="store_true",
                   help="set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (used for 4x at 1M)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="runs/timing/lclm_hf_scaling.jsonl")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.expandable_segments:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch

    from inference.hf import load_model
    from inference.optimized_encoder import enable_encoder_optimization, set_attn_implementation

    torch.manual_seed(0)
    model, _, _ = load_model(args.checkpoint, device=args.device, dtype="bf16")
    model.eval()
    set_attn_implementation(model.encoder.embed_model, "sdpa")
    set_attn_implementation(model.decoder, args.decoder_attn)
    encoder, decoder = model.encoder, model.decoder
    ratio = encoder.compression_ratio
    token = encoder.embed_tokenizer.encode(" the", add_special_tokens=False)[-1]
    sync = torch.cuda.synchronize
    base = dict(checkpoint=args.checkpoint, ratio=ratio, batch_windows=args.batch_windows,
                compiled=not args.no_compile, decoder_attn=args.decoder_attn, **environment_info())

    def encode(n):
        return model._process_latent_embeddings([[token] * n])

    reference = None
    if not args.skip_latent_check:
        with torch.inference_mode():
            reference = encode(LATENT_CHECK_TOKENS)[0].float().cpu()

    optimization = enable_encoder_optimization(
        model, compile_encoder=not args.no_compile, batch_windows=args.batch_windows
    )

    def run_once(n, staged=False):
        stages = {}
        start = time.perf_counter()
        latents = encode(n)
        if staged:
            sync()
            stages["encoder_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
        embeds = latents[0].unsqueeze(0)
        assert embeds.shape[1] == math.ceil(n / ratio)
        out = decoder(inputs_embeds=embeds, use_cache=True)
        first_token = int(out.logits[:, -1, :].argmax(-1).item())
        if staged:
            stages["decoder_ms"] = (time.perf_counter() - start) * 1000
        del out, latents, embeds
        return first_token, stages

    with torch.inference_mode():
        if reference is not None:
            optimized = encode(LATENT_CHECK_TOKENS)[0].float().cpu()
            rel = float((optimized - reference).norm() / reference.norm())
            append_jsonl(args.output, dict(base, kind="latent_check",
                                           context_tokens=LATENT_CHECK_TOKENS, relative_l2=rel))
            print(f"latent check @ {LATENT_CHECK_TOKENS}: relative L2 = {rel:.3e}", flush=True)
            del optimized, reference

        for n in args.lengths:
            record = dict(base, kind="measurement", context_tokens=n,
                          decoder_tokens=math.ceil(n / ratio))

            def trial():
                gc.collect()
                sync()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                first_token, _ = run_once(n)
                sync()
                return dict(ttft_ms=(time.perf_counter() - start) * 1000,
                            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                            first_token=first_token)

            try:
                record["trials"] = run_trials(trial, args.warmups, args.repeats)
                _, record["stages"] = run_once(n, staged=True)
                sync()
                summary = summarize_trials(record["trials"])
                record.update(ttft_ms=summary["ttft_ms"],
                              peak_allocated_gib=summary["peak_allocated_gib"], status="ok")
            except torch.cuda.OutOfMemoryError:
                record["status"] = "OOM"
            gc.collect()
            torch.cuda.empty_cache()
            record["graph_captures"] = list(optimization.capture_log)
            append_jsonl(args.output, record)
            print(f"{n:>8} tokens  {record['status']:>3}  "
                  f"ttft={record.get('ttft_ms', float('nan')):.1f} ms  "
                  f"peak={record.get('peak_allocated_gib', float('nan')):.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
