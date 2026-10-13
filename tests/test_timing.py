"""CPU checks for benchmarks/timing: argument parsing, trial aggregation, workload setup."""
import argparse

import pytest

from benchmarks.timing import kv_baselines_hf, lclm_hf_scaling, vllm_ttft
from benchmarks.timing.common import (
    BENCHMARK_CONTEXT_TOKENS,
    CONTEXT_LENGTHS,
    parse_length,
    parse_length_list,
    parse_name_list,
    run_trials,
    summarize_trials,
)


@pytest.mark.parametrize(
    "text, expected",
    [("4096", 4096), ("4k", 4096), ("64K", 65536), ("1M", 1048576), (" 16k ", 16384)],
)
def test_parse_length(text, expected):
    assert parse_length(text) == expected


@pytest.mark.parametrize("text", ["", "k", "abc", "0", "-4k", "4g"])
def test_parse_length_rejects(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_length(text)


def test_parse_length_list():
    assert parse_length_list("4k,64k,1M") == [4096, 65536, 1048576]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_length_list(" , ")


def test_parse_name_list():
    choices = ["a", "b", "c"]
    assert parse_name_list("all", choices) == choices
    assert parse_name_list("c,a", choices) == ["c", "a"]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_name_list("a,d", choices)


def test_run_trials_tags_warmups():
    calls = iter(range(10))
    trials = run_trials(lambda: dict(ttft_ms=float(next(calls))), warmups=2, repeats=5)
    assert [t["warmup"] for t in trials] == [True, True, False, False, False, False, False]
    assert [t["ttft_ms"] for t in trials] == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        run_trials(dict, warmups=0, repeats=0)


def test_summary_is_fastest_measured_trial_excluding_warmups():
    trials = [
        dict(warmup=True, ttft_ms=1.0, peak_allocated_gib=50.0),  # faster, but a warmup
        dict(warmup=True, ttft_ms=2.0, peak_allocated_gib=50.0),
        dict(warmup=False, ttft_ms=12.0, peak_allocated_gib=9.0),
        dict(warmup=False, ttft_ms=10.0, peak_allocated_gib=9.5),
        dict(warmup=False, ttft_ms=11.0, peak_allocated_gib=9.1),
    ]
    summary = summarize_trials(trials)
    assert summary["ttft_ms"] == 10.0
    assert summary["fastest_trial_index"] == 3
    assert summary["median_ttft_ms"] == 11.0
    assert summary["measured_trials"] == 3
    assert summary["peak_allocated_gib"] == 9.5  # max over measured trials only


def test_summary_keeps_stages_of_one_trial():
    # The fastest total is reported with its own stages, never the per-stage minima.
    trials = [
        dict(warmup=False, encoder_ms=5.0, decoder_ms=20.0, ttft_ms=25.0),
        dict(warmup=False, encoder_ms=8.0, decoder_ms=15.0, ttft_ms=23.0),
    ]
    fastest = summarize_trials(trials)["fastest_trial"]
    assert (fastest["encoder_ms"], fastest["decoder_ms"]) == (8.0, 15.0)


def test_summary_requires_measured_trials():
    with pytest.raises(ValueError):
        summarize_trials([dict(warmup=True, ttft_ms=1.0)])


def test_lclm_scaling_defaults_match_paper():
    args = lclm_hf_scaling.build_parser().parse_args([])
    assert args.checkpoint == "latent-context/0.6b-4b-LCLM-16x"
    assert args.lengths == CONTEXT_LENGTHS == [4096 * 2**i for i in range(9)]
    assert (args.warmups, args.repeats, args.batch_windows) == (2, 5, 32)
    assert args.decoder_attn == "sdpa" and not args.no_compile
    args = lclm_hf_scaling.build_parser().parse_args(["--lengths", "1M", "--expandable-segments"])
    assert args.lengths == [1048576] and args.expandable_segments


def test_kv_baseline_defaults():
    args = kv_baselines_hf.build_parser().parse_args([])
    assert args.methods == kv_baselines_hf.KVPRESS_METHODS
    assert (args.warmups, args.repeats) == (30, 20)
    assert args.model == "Qwen/Qwen3-4B-Instruct-2507"
    args = kv_baselines_hf.build_parser().parse_args(["--methods", "all"])
    assert "attention_matching" in args.methods and "no_press" in args.methods
    with pytest.raises(SystemExit):
        kv_baselines_hf.build_parser().parse_args(["--methods", "h2o"])


def test_vllm_models_and_workloads():
    assert {"lclm-4x", "lclm-8x", "lclm-16x", "qwen3-4b-instruct-2507", "qwen3-0.6b"} <= set(vllm_ttft.MODELS)
    assert len(vllm_ttft.MODELS) == 11
    assert vllm_ttft.MODELS["lclm-16x"]["repo"] == "latent-context/0.6b-4b-LCLM-16x"
    args = vllm_ttft.build_parser().parse_args(["--model", "lclm-8x"])
    assert args.benchmarks == list(BENCHMARK_CONTEXT_TOKENS)
    assert (args.warmups, args.repeats, args.batch_windows) == (2, 5, 32)

    # LongHealth5: five segments, 64 windows of 1024 tokens.
    assert vllm_ttft.segments_for("longhealth5") == [13312] * 4 + [12288]
    assert sum(vllm_ttft.segments_for("longhealth5")) // 1024 == 64
    for ratio in (4, 8, 16):
        for benchmark, tokens in BENCHMARK_CONTEXT_TOKENS.items():
            assert vllm_ttft.decoder_tokens(benchmark, ratio) == tokens // ratio
    assert vllm_ttft.decoder_tokens("longbench", 0) == 65536


def test_vllm_engine_settings():
    lengths = [4096 // 16, 65536 // 16]
    kwargs = vllm_ttft.engine_kwargs("decoder", None, lengths)
    assert kwargs["max_model_len"] == 4096 + 128
    assert kwargs["enable_prefix_caching"] is False and kwargs["max_num_seqs"] == 1
    assert kwargs["compilation_config"] == dict(
        cudagraph_mode="PIECEWISE", cudagraph_capture_sizes=[1, 2, 4, 256, 4096]
    )
    assert "hf_overrides" not in kwargs
    raw = vllm_ttft.engine_kwargs("Qwen/Qwen3-0.6B", "rev", [65536], vllm_ttft.QWEN3_YARN)
    assert raw["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4, 8192]
    assert raw["hf_overrides"]["rope_scaling"]["factor"] == 4.0
