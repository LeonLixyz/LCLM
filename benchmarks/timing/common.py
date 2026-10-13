"""Shared constants and helpers for the timing benchmarks.

Kept free of GPU imports so argument parsing and trial aggregation can be
unit-tested on CPU.
"""
import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Figure 4 context grid (also the KV-cache baseline grid).
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

LCLM_CHECKPOINTS = {
    4: "latent-context/0.6b-4b-LCLM-4x",
    8: "latent-context/0.6b-4b-LCLM-8x",
    16: "latent-context/0.6b-4b-LCLM-16x",
}

# Fixed context length used for each benchmark's TTFT point (Figures 1 and 5).
BENCHMARK_CONTEXT_TOKENS = {
    "ruler_4096": 4096,
    "ruler_8192": 8192,
    "ruler_16384": 16384,
    "longhealth5": 65536,
    "longbench": 65536,
}

_SUFFIXES = {"k": 1024, "m": 1024 * 1024}


def parse_length(text: str) -> int:
    """Parse ``"4096"``, ``"4k"`` or ``"1M"`` (binary multiples) as a token count."""
    value = text.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("empty length")
    scale = _SUFFIXES.get(value[-1], 1)
    if scale != 1:
        value = value[:-1]
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid length: {text!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"length must be positive: {text!r}")
    return number * scale


def parse_length_list(text: str) -> List[int]:
    """Parse a comma-separated list of lengths, e.g. ``"4k,16k,1M"``."""
    lengths = [parse_length(part) for part in text.split(",") if part.strip()]
    if not lengths:
        raise argparse.ArgumentTypeError("expected at least one length")
    return lengths


def parse_name_list(text: str, choices: Sequence[str]) -> List[str]:
    """Parse a comma-separated subset of ``choices`` (``"all"`` selects every choice)."""
    if text.strip() == "all":
        return list(choices)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in choices]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown choice(s) {unknown}; expected a subset of {list(choices)} or 'all'"
        )
    return names


def run_trials(trial: Callable[[], Dict], warmups: int, repeats: int) -> List[Dict]:
    """Call ``trial`` ``warmups + repeats`` times; tag each result with ``warmup``."""
    if warmups < 0 or repeats < 1:
        raise ValueError("need warmups >= 0 and repeats >= 1")
    results = []
    for index in range(warmups + repeats):
        record = dict(trial())
        record["warmup"] = index < warmups
        results.append(record)
    return results


def summarize_trials(trials: Sequence[Dict], key: str = "ttft_ms") -> Dict:
    """Fastest measured (non-warmup) trial, reported as a whole.

    Stage times (e.g. encoder and decoder) come from that same trial; stage
    minima are never summed across different trials. Peak memory is the
    maximum over measured trials.
    """
    measured = [(i, t) for i, t in enumerate(trials) if not t.get("warmup")]
    if not measured:
        raise ValueError("no measured trials")
    index, fastest = min(measured, key=lambda pair: pair[1][key])
    summary = {
        key: fastest[key],
        f"median_{key}": statistics.median(t[key] for _, t in measured),
        "fastest_trial_index": index,
        "fastest_trial": dict(fastest),
        "measured_trials": len(measured),
    }
    peaks = [t["peak_allocated_gib"] for _, t in measured if "peak_allocated_gib" in t]
    if peaks:
        summary["peak_allocated_gib"] = max(peaks)
    return summary


def append_jsonl(path, record: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def resolve_checkpoint(path_or_repo: str, revision: Optional[str] = None) -> str:
    """Return a local directory for a local path or a Hugging Face model id."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=path_or_repo, revision=revision, repo_type="model")


def environment_info() -> Dict:
    """Library versions and GPU name for result records."""
    import torch
    import transformers

    info = dict(torch=torch.__version__, transformers=transformers.__version__)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
    return info
