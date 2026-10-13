"""KV-cache compression baselines: compression time and peak GPU memory vs. context length.

Baseline side of Figures 1 and 4. Adapted from the KVPress speed-and-memory
notebook (https://github.com/NVIDIA/kvpress/blob/main/notebooks/speed_and_memory.ipynb).
Every method compresses the KV cache of Qwen/Qwen3-4B-Instruct-2507 in
Hugging Face Transformers with FlashAttention-2, batch size 1.

Methods (``--methods``):

* ``no_press``: plain prefill, no compression.
* ``snapkv`` (window 64, kernel 5), ``snapkv_w32_k7`` (window 32, kernel 7;
  the "SnapKV (Self-Study)" setting), ``expected_attention`` (AdaKV +
  Expected Attention), ``kvzip``, ``fastkvzip``: NVIDIA KVPress presses.
* ``attention_matching_fast`` and ``attention_matching``: the official
  Attention Matching code, without and with on-policy queries
  (needs ``--attention-matching-repo``).

Workload per (method, context N): N copies of token id 0, one forward with
``use_cache=True`` inside the press context (logits at every position). For
Attention Matching: KV extraction over the repeat prompt plus compaction.
The compression ratio does not change the timing of these methods.

A fresh model is loaded per context length. After ``--warmups`` unrecorded
trials, the peak-memory counter is reset and ``--repeats`` trials are timed.
The reported time is the fastest trial and peak memory is
``torch.cuda.max_memory_allocated`` over the timed trials. Once a method runs
out of memory, longer contexts are marked OOM without running.

    python -m benchmarks.timing.kv_baselines_hf --methods no_press,snapkv \\
        --output runs/timing/kv_baselines_hf.jsonl
"""
import argparse
import gc
import os
import sys
import time
import traceback
from contextlib import nullcontext
from types import SimpleNamespace

from benchmarks.timing.common import (
    CONTEXT_LENGTHS,
    append_jsonl,
    environment_info,
    parse_length_list,
    parse_name_list,
    run_trials,
    summarize_trials,
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
KVPRESS_METHODS = ["no_press", "snapkv", "snapkv_w32_k7", "expected_attention", "kvzip", "fastkvzip"]
ATTENTION_MATCHING_METHODS = {
    "attention_matching_fast": "highest_attn_keys_rms_nnls2_-3_3_lsq",
    "attention_matching": "highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy",
}
METHODS = KVPRESS_METHODS + list(ATTENTION_MATCHING_METHODS)
# Fraction of KV pairs removed (16x). Does not affect timing of these methods.
COMPRESSION_RATIO = 0.9375
AM_HEAD_BUDGETS = "head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=lambda s: parse_name_list(s, METHODS), default=list(KVPRESS_METHODS),
                   help=f"comma-separated subset of {METHODS}, or 'all'")
    p.add_argument("--lengths", type=parse_length_list, default=list(CONTEXT_LENGTHS))
    p.add_argument("--warmups", type=int, default=30)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--attention-matching-repo", default=None,
                   help="clone of the official Attention Matching repository")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="runs/timing/kv_baselines_hf.jsonl")
    return p


def make_press(method: str):
    from kvpress import AdaKVPress, ExpectedAttentionPress, FastKVzipPress, KVzipPress, SnapKVPress

    if method == "no_press":
        return None
    if method == "snapkv":
        return SnapKVPress(compression_ratio=COMPRESSION_RATIO, window_size=64, kernel_size=5)
    if method == "snapkv_w32_k7":
        return SnapKVPress(compression_ratio=COMPRESSION_RATIO, window_size=32, kernel_size=7)
    if method == "expected_attention":
        return AdaKVPress(ExpectedAttentionPress(compression_ratio=COMPRESSION_RATIO, epsilon=1e-2))
    if method == "kvzip":
        return KVzipPress(compression_ratio=COMPRESSION_RATIO)
    if method == "fastkvzip":
        return FastKVzipPress(compression_ratio=COMPRESSION_RATIO)
    raise ValueError(method)


def load_hf_model(model_name: str, context: int, device: str):
    """Load the decoder with FlashAttention-2.

    Above the native maximum position (262,144), only ``max_position_embeddings``
    is raised: RoPE cost per token does not depend on the scaling factor.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    if context > getattr(config, "max_position_embeddings", 32768):
        config.max_position_embeddings = context
        config.rope_scaling = None
    return AutoModelForCausalLM.from_pretrained(
        model_name, config=config, dtype="bfloat16", attn_implementation="flash_attention_2"
    ).to(device)


def kvpress_trial(method: str, context: int, model) -> float:
    import torch

    device = next(model.parameters()).device
    # Token id 0 everywhere: arange(context) would exceed the vocabulary above ~152k.
    ids = torch.zeros((1, context), dtype=torch.long, device=device)
    press = make_press(method)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        with nullcontext() if press is None else press(model):
            model(ids, use_cache=True, cache_position=torch.arange(context, device=device))
        torch.cuda.synchronize()
    return time.perf_counter() - start


class AttentionMatching:
    """Official Attention Matching code: KV extraction + compaction."""

    def __init__(self, repo: str, model_name: str, method: str, context: int, device: str):
        if not repo or not os.path.isdir(repo):
            raise ValueError("--attention-matching-repo must point to a clone of the official repository")
        self.repo, self.model_name = os.path.abspath(repo), model_name
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from evaluation.utils import load_model_and_tokenizer

        # The repeat-prompt extraction prefills about twice the context.
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name, device=device, max_model_len=max(context * 2, 32768)
        )
        self.model = self.model.to(device)
        self.method_name = ATTENTION_MATCHING_METHODS[method]

    def trial(self, context: int) -> float:
        import torch
        from compaction.compaction_methods.registry import get_compaction_method
        from evaluation.configs.utils import get_method_kwargs, load_algorithm_config, load_query_config
        from evaluation.utils import extract_full_kv_cache

        cwd = os.getcwd()
        os.chdir(self.repo)  # config and head-budget paths are relative to the repo root
        try:
            kwargs = get_method_kwargs(load_algorithm_config("best"), self.method_name)
            kwargs["precomputed_budget_path"] = AM_HEAD_BUDGETS
            method = get_compaction_method(self.method_name, kwargs)
            query_config = load_query_config("repeat")
            text = self.tokenizer.decode([0] * context, skip_special_tokens=False)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.perf_counter()
                seq_len, past_kv, article_idx, formatted, _ = extract_full_kv_cache(
                    self.model, self.tokenizer, text, device="cuda", model_name=self.model_name
                )
                article_target = max(1, int(len(article_idx) * (1 - COMPRESSION_RATIO)))
                method.compact_kv_cache(
                    past_key_values=past_kv, target_size=article_target + (seq_len - len(article_idx)),
                    indices=article_idx, query_config=query_config, model=self.model,
                    tokenizer=self.tokenizer, formatted_context=formatted, compute_stats=False,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            del past_kv
            torch.cuda.empty_cache()
            return elapsed
        finally:
            os.chdir(cwd)


def measure_cell(args, method: str, context: int) -> dict:
    import torch

    if method in ATTENTION_MATCHING_METHODS:
        runner = AttentionMatching(args.attention_matching_repo, args.model, method, context, args.device)
        step = lambda: runner.trial(context)  # noqa: E731
    else:
        runner = SimpleNamespace(model=load_hf_model(args.model, context, args.device))
        step = lambda: kvpress_trial(method, context, runner.model)  # noqa: E731
    try:
        for _ in range(args.warmups):
            step()
        torch.cuda.reset_peak_memory_stats()
        trials = run_trials(lambda: dict(ttft_ms=step() * 1000), 0, args.repeats)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
    finally:
        runner.model.cpu()
        runner.model = None
        gc.collect()
        torch.cuda.empty_cache()
    summary = summarize_trials(trials)
    return dict(trials=trials, ttft_ms=summary["ttft_ms"],
                median_ttft_ms=summary["median_ttft_ms"], peak_allocated_gib=peak_gib)


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch

    env = environment_info()
    for method in args.methods:
        oom = False
        for context in args.lengths:
            record = dict(method=method, model=args.model, context_tokens=context,
                          warmups=args.warmups, repeats=args.repeats, **env)
            if oom:
                record["status"] = "OOM (skipped after shorter context ran out of memory)"
            else:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    record.update(measure_cell(args, method, context), status="ok")
                except torch.cuda.OutOfMemoryError as e:
                    record.update(status="OOM", error=str(e)[:300])
                    oom = True
                except Exception as e:  # e.g. Attention Matching's Cholesky failure at 512k
                    record.update(status="error", error=str(e)[:300], traceback=traceback.format_exc())
                gc.collect()
                torch.cuda.empty_cache()
            append_jsonl(args.output, record)
            print(f"{method:>24} {context:>8}  {record['status'][:5]:>5}  "
                  f"time={record.get('ttft_ms', float('nan')):.1f} ms  "
                  f"peak={record.get('peak_allocated_gib', float('nan')):.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
