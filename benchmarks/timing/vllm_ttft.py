"""TTFT of LCLMs and uncompressed models served with vLLM (Figure 5).

Every decoder runs in vLLM (BF16, one GPU, batch size 1, piecewise CUDA
graphs, prefix caching disabled) at fixed context lengths: 4k/8k/16k tokens
for RULER and 64k for LongHealth5 and LongBench (``BENCHMARK_CONTEXT_TOKENS``).
The workload is synthetic: the token ``" the"`` repeated, one greedy output
token, no question tokens.

* Uncompressed model: TTFT of a fresh N-token request.
* LCLM: the optimized encoder (``inference/optimized_encoder.py``, 32 windows
  per forward) encodes N real token ids into N/ratio latents, then vLLM
  prefills a dummy-token request of the same length N/ratio. TTFT is the sum
  of the two, taken from the same trial. For LongHealth5 the 64k context is
  five segments of 13,312 x 4 + 12,288 tokens (64 windows in total).
  Engine start-up, encoder compilation/graph capture and the hand-off of
  latents to vLLM are excluded.

Each workload runs one untimed encoder setup call, ``--warmups`` unrecorded
trials and ``--repeats`` measured trials; the reported TTFT is the fastest
measured trial. Run one model per process:

    python -m benchmarks.timing.vllm_ttft --model lclm-16x --output runs/timing/vllm_ttft.jsonl
    python -m benchmarks.timing.vllm_ttft --model qwen3-4b-instruct-2507 --output runs/timing/vllm_ttft.jsonl
"""
import argparse
import gc
import math
import os
import time
from pathlib import Path

from benchmarks.timing.common import (
    BENCHMARK_CONTEXT_TOKENS,
    LCLM_CHECKPOINTS,
    append_jsonl,
    environment_info,
    parse_name_list,
    resolve_checkpoint,
    run_trials,
    summarize_trials,
)

# Qwen3 base models are native to 40,960 positions; the 64k workloads use YaRN.
QWEN3_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}

MODELS = {
    "qwen3-0.6b": dict(repo="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", rope_scaling=QWEN3_YARN),
    "qwen3-1.7b": dict(repo="Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", rope_scaling=QWEN3_YARN),
    "qwen3-4b": dict(repo="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c", rope_scaling=QWEN3_YARN),
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218", rope_scaling=QWEN3_YARN),
    "qwen3-4b-instruct-2507": dict(repo="Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554"),
    "llama3.2-1b-instruct": dict(repo="meta-llama/Llama-3.2-1B-Instruct", revision="9213176726f574b556790deb65791e0c5aa438b6"),
    "llama3.2-3b-instruct": dict(repo="meta-llama/Llama-3.2-3B-Instruct", revision="0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    "llama3.1-8b-instruct": dict(repo="meta-llama/Llama-3.1-8B-Instruct", revision="0e9e39f249a16976918f6564b8830bc894c89659"),
    **{f"lclm-{r}x": dict(repo=repo, ratio=r) for r, repo in LCLM_CHECKPOINTS.items()},
}

# LongHealth5: five documents, 64 encoder windows in total.
SEGMENTS = {"longhealth5": [13312] * 4 + [12288]}
MAX_BATCHED_TOKENS = 8192


def segments_for(benchmark: str) -> list:
    tokens = BENCHMARK_CONTEXT_TOKENS[benchmark]
    segments = SEGMENTS.get(benchmark, [tokens])
    assert sum(segments) == tokens
    return segments


def decoder_tokens(benchmark: str, ratio: int) -> int:
    if not ratio:
        return BENCHMARK_CONTEXT_TOKENS[benchmark]
    return sum(math.ceil(n / ratio) for n in segments_for(benchmark))


def engine_kwargs(model_path: str, revision, lengths, rope_scaling=None) -> dict:
    """vLLM engine settings shared by every model."""
    capture = sorted({1, 2, 4} | {min(n, MAX_BATCHED_TOKENS) for n in lengths})
    kwargs = dict(
        model=model_path,
        revision=revision,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prompt_embeds=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_num_seqs=1,
        max_model_len=max(lengths) + 128,
        gpu_memory_utilization=0.65,  # leaves room for the LCLM encoder in the same process
        enforce_eager=False,
        seed=0,
        compilation_config=dict(cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=capture),
    )
    if rope_scaling:
        kwargs["hf_overrides"] = {"rope_scaling": rope_scaling}
    return kwargs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--checkpoint", default=None,
                   help="override the model's checkpoint (local directory or Hugging Face id)")
    p.add_argument("--benchmarks", type=lambda s: parse_name_list(s, list(BENCHMARK_CONTEXT_TOKENS)),
                   default=list(BENCHMARK_CONTEXT_TOKENS))
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--output", default="runs/timing/vllm_ttft.jsonl")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # CUDA is initialized before the engine starts, so workers must be spawned.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = MODELS[args.model]
    ratio = spec.get("ratio", 0)
    torch.set_num_threads(1)
    torch.manual_seed(0)

    model = None
    if ratio:
        from inference.hf import load_model
        from inference.optimized_encoder import enable_encoder_optimization

        checkpoint = resolve_checkpoint(args.checkpoint or spec["repo"])
        model, tokenizer, _ = load_model(checkpoint, device="cuda:0", dtype="bf16")
        model.eval()
        model.decoder = None  # vLLM serves the decoder
        gc.collect()
        torch.cuda.empty_cache()
        assert model.encoder.compression_ratio == ratio
        enable_encoder_optimization(model, compile_encoder=True, batch_windows=args.batch_windows)
        encoder_token = model.encoder.embed_tokenizer.encode(" the", add_special_tokens=False)[-1]
        engine_model, revision = str(Path(checkpoint) / "decoder"), None
    else:
        engine_model = args.checkpoint or spec["repo"]
        revision = None if args.checkpoint else spec["revision"]
        tokenizer = AutoTokenizer.from_pretrained(engine_model, revision=revision)
    decoder_token = tokenizer.encode(" the", add_special_tokens=False)[-1]

    lengths = {b: decoder_tokens(b, ratio) for b in args.benchmarks}
    kwargs = engine_kwargs(engine_model, revision, lengths.values(), spec.get("rope_scaling"))
    start = time.perf_counter()
    llm = LLM(**kwargs)
    engine_setup_seconds = time.perf_counter() - start
    params = SamplingParams(max_tokens=1, temperature=0, top_p=1, top_k=-1, seed=0,
                            ignore_eos=True, detokenize=False)
    base = dict(model=args.model, checkpoint=args.checkpoint or spec["repo"], ratio=ratio,
                engine_config=kwargs, engine_setup_seconds=engine_setup_seconds,
                vllm=vllm.__version__, **environment_info())

    with torch.inference_mode():
        for benchmark in args.benchmarks:
            n = lengths[benchmark]
            memory = [[encoder_token] * size for size in segments_for(benchmark)] if ratio else []
            prompt = {"prompt_token_ids": [decoder_token] * n}

            def encode():
                latents = model._process_latent_embeddings(memory)
                assert sum(x.shape[0] for x in latents) == n
                return latents

            start = time.perf_counter()
            if ratio:
                encode()  # compile / capture for this shape
                torch.cuda.synchronize()
            encoder_setup_seconds = time.perf_counter() - start

            def trial():
                encoder_ms = 0.0
                torch.cuda.synchronize()
                if ratio:
                    start = time.perf_counter()
                    latents = encode()
                    torch.cuda.synchronize()
                    encoder_ms = (time.perf_counter() - start) * 1000
                    del latents
                start = time.perf_counter()
                output = llm.generate([prompt], params, use_tqdm=False)[0]
                torch.cuda.synchronize()
                decoder_ms = (time.perf_counter() - start) * 1000
                token_ids = output.outputs[0].token_ids
                assert len(token_ids) == 1 and len(output.prompt_token_ids) == n
                cached = getattr(output, "num_cached_tokens", None)
                assert cached in (None, 0), "prefix caching must be off"
                return dict(encoder_ms=encoder_ms, decoder_ms=decoder_ms,
                            ttft_ms=encoder_ms + decoder_ms, first_token=int(token_ids[0]))

            trials = run_trials(trial, args.warmups, args.repeats)
            summary = summarize_trials(trials)
            fastest = summary["fastest_trial"]
            record = dict(base, benchmark=benchmark, context_tokens=BENCHMARK_CONTEXT_TOKENS[benchmark],
                          decoder_tokens=n, memory_segment_tokens=segments_for(benchmark) if ratio else [],
                          encoder_setup_seconds=encoder_setup_seconds, trials=trials,
                          ttft_ms=summary["ttft_ms"], encoder_ms=fastest["encoder_ms"],
                          decoder_ms=fastest["decoder_ms"], median_ttft_ms=summary["median_ttft_ms"])
            append_jsonl(args.output, record)
            print(f"{args.model:>24} {benchmark:>12}  ttft={record['ttft_ms']:.2f} ms  "
                  f"(encoder {record['encoder_ms']:.2f} + decoder {record['decoder_ms']:.2f})", flush=True)


if __name__ == "__main__":
    main()
