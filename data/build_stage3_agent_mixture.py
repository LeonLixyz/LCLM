#!/usr/bin/env python3
"""Build a streaming mixture of stage-3 and native agent-trajectory rows.

Reasoning-heavy stage-3 rows move compression from the prompt to the analysis
part of the target. Agent trajectories are deliberately kept as native
multi-turn messages (canonicalized into ``messages``) and are never decorated
with LCLM memory tags.

The command writes JSON Lines so input and output stay memory bounded.  The
result can be consumed by Hugging Face Datasets with, for example,
``load_dataset("json", data_files="mixture.jsonl")``.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"
MEMORY_PLACEHOLDER = "<|memory|>"
MEMORY_MARKERS = (MEMORY_START, MEMORY_END, MEMORY_PLACEHOLDER)

DEFAULT_STAGE3_DATASET = "leonli66/stage3-final-mixture"
DEFAULT_AGENT_DATASET = "open-thoughts/OpenThoughts-Agent-SFT-100K"
DEFAULT_DECODER_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_FALLBACK_ANSWER_TOKENS = 128
FALLBACK_TOKENIZATION_WINDOW_CHARS = 64 * 1024
DEFAULT_REASONING_COMPRESSION_FRACTION = 0.5
COLDSTART_AGENT_DATASET = (
    "open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K"
)
DEFAULT_REASONING_SUBDATASETS = (
    "reasoning_data",
    "dolci_think",
)
DEFAULT_ANALYSIS_TAG_PAIRS = (("<think>", "</think>"), ("<analysis>", "</analysis>"))
DEFAULT_FINAL_ANSWER_MARKERS = (
    "Final Answer:",
    "Final answer:",
    "FINAL ANSWER:",
    "Answer:",
    "ANSWER:",
)

SCHEMA_KEYS = (
    "schema_version",
    "data_type",
    "source_dataset",
    "sub_dataset",
    "compression_scope",
    "reasoning_split",
    "prompt",
    "compression_prompt",
    "target",
    "messages",
    "conversations",
    "tools",
    "task",
    "trace_source",
    "agent",
    "model",
    "model_provider",
    "result",
    "episode",
    "run_id",
    "trial_name",
    "date",
    "metadata_json",
)


@dataclass(frozen=True)
class ReasoningSplit:
    """A target decomposition that preserves text around the reasoning span."""

    prefix: str
    analysis: str
    suffix: str
    method: str


@dataclass(frozen=True)
class SourceSpec:
    dataset_id: str
    kind: str
    weight: float = 1.0
    max_rows: int | None = None
    repeat: int = 1


def _validate_marker_pairs(
    analysis_tag_pairs: Sequence[tuple[str, str]],
) -> None:
    for pair in analysis_tag_pairs:
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise ValueError(f"analysis tag pairs must contain two nonempty strings: {pair!r}")


def split_reasoning_target(
    target: str,
    *,
    analysis_tag_pairs: Sequence[tuple[str, str]] = DEFAULT_ANALYSIS_TAG_PAIRS,
    final_answer_markers: Sequence[str] = DEFAULT_FINAL_ANSWER_MARKERS,
    fallback_tokenizer: Any | None = None,
    fallback_answer_tokens: int = DEFAULT_FALLBACK_ANSWER_TOKENS,
) -> ReasoningSplit | None:
    """Conservatively identify analysis while retaining a final-answer suffix.

    Explicit analysis tags win.  A tag pair is accepted only when it occurs
    exactly once, encloses non-whitespace text, and is followed by a nonempty
    suffix. Otherwise, the last safe built-in final-answer marker (or an exact
    caller-supplied marker) is used. Generic sentence splitting is intentionally
    not attempted. If no explicit boundary is found and a tokenizer is supplied,
    the final ``fallback_answer_tokens`` tokens remain trainable while the
    preceding text becomes the compressed analysis span.
    """

    if not isinstance(target, str):
        raise TypeError(f"target must be a string, got {type(target).__name__}")
    if fallback_answer_tokens <= 0:
        raise ValueError("fallback_answer_tokens must be positive")
    _validate_marker_pairs(analysis_tag_pairs)

    saw_analysis_tag = False
    for start_tag, end_tag in analysis_tag_pairs:
        saw_analysis_tag |= start_tag in target or end_tag in target
        if target.count(start_tag) != 1 or target.count(end_tag) != 1:
            continue
        start = target.find(start_tag)
        analysis_start = start + len(start_tag)
        end = target.find(end_tag, analysis_start)
        if end < analysis_start:
            continue
        analysis = target[analysis_start:end]
        suffix = target[end + len(end_tag) :]
        if analysis.strip() and suffix.strip():
            return ReasoningSplit(
                prefix=target[:analysis_start],
                analysis=analysis,
                suffix=target[end:],
                method=f"tag:{start_tag}...{end_tag}",
            )

    # A malformed or ambiguous explicit analysis annotation is stronger evidence
    # than an answer-looking line later in the text.  Refuse to guess its span.
    if saw_analysis_tag:
        return _fallback_reasoning_split(
            target,
            tokenizer=fallback_tokenizer,
            answer_tokens=fallback_answer_tokens,
        )

    candidates: list[tuple[int, str]] = []
    for marker in final_answer_markers:
        if not marker:
            raise ValueError("final-answer markers must be nonempty")
        search_from = 0
        while True:
            position = target.find(marker, search_from)
            if position < 0:
                break
            is_default = marker in DEFAULT_FINAL_ANSWER_MARKERS
            at_line_start = position == 0 or target[position - 1] == "\n"
            generic_answer = marker.casefold() == "answer:"
            has_paragraph_break = position >= 2 and target[position - 2 : position] == "\n\n"
            safe_boundary = (
                not is_default
                or (at_line_start and (not generic_answer or has_paragraph_break))
            )
            after_marker = target[position + len(marker) :]
            if safe_boundary and target[:position].strip() and after_marker.strip():
                candidates.append((position, marker))
            search_from = position + len(marker)

    if not candidates:
        # Common reasoning-model outputs use a boxed/"answer is" final line
        # without an explicit analysis tag. Keep that conclusion trainable.
        gsm_final_lines = list(re.finditer(r"(?m)^[ \t]*####[ \t]+\S.*$", target))
        if gsm_final_lines:
            final_line = gsm_final_lines[-1]
            if (
                target[: final_line.start()].strip()
                and not target[final_line.end() :].strip()
            ):
                return ReasoningSplit(
                    prefix="",
                    analysis=target[: final_line.start()],
                    suffix=target[final_line.start() :],
                    method="pattern:gsm8k-final-line",
                )
        answer_lines = list(
            re.finditer(
                r"(?im)^(?:\s*)(?:the\s+)?(?:final\s+)?answer\s+(?:is\b|:)",
                target,
            )
        )
        boxed = target.rfind(r"\boxed{")
        if boxed >= 0:
            boxed_line_start = target.rfind("\n", 0, boxed) + 1
            if target[:boxed_line_start].strip():
                return ReasoningSplit(
                    prefix="",
                    analysis=target[:boxed_line_start],
                    suffix=target[boxed_line_start:],
                    method="pattern:boxed",
                )
        if answer_lines:
            position = answer_lines[-1].start()
            if target[:position].strip() and target[position:].strip():
                return ReasoningSplit(
                    prefix="",
                    analysis=target[:position],
                    suffix=target[position:],
                    method="pattern:answer-line",
                )

        # Programming traces often contain scratch reasoning followed by a final
        # fenced implementation and explanation. Only use a structurally balanced
        # last fence; malformed Markdown fails closed.
        fences = [match.start() for match in re.finditer(r"(?m)^```", target)]
        if len(fences) >= 2 and len(fences) % 2 == 0:
            position = fences[-2]
            if target[:position].strip() and target[position:].strip():
                return ReasoningSplit(
                    prefix="",
                    analysis=target[:position],
                    suffix=target[position:],
                    method="pattern:last-code-fence",
                )
        return _fallback_reasoning_split(
            target,
            tokenizer=fallback_tokenizer,
            answer_tokens=fallback_answer_tokens,
        )

    position, marker = max(candidates, key=lambda candidate: candidate[0])
    return ReasoningSplit(
        prefix="",
        analysis=target[:position],
        suffix=target[position:],
        method=f"marker:{marker}",
    )


def _fallback_reasoning_split(
    target: str,
    *,
    tokenizer: Any | None,
    answer_tokens: int,
) -> ReasoningSplit | None:
    """Split before the final N decoder tokens without rewriting the text.

    Fast-tokenizer offset mappings give a character boundary into the original
    string, avoiding decode/re-encode whitespace changes. Targets of at most N
    tokens stay fully trainable because they have no nonempty analysis prefix.
    """

    if tokenizer is None:
        return None
    # Some legacy rows contain hundreds of millions of characters. The final
    # 64 KiB is vastly larger than a 128-token answer, and bounding tokenization
    # here prevents a malformed row from allocating gigabytes of offset tuples.
    window_start = max(0, len(target) - FALLBACK_TOKENIZATION_WINDOW_CHARS)
    tokenization_window = target[window_start:]
    encoded = tokenizer(
        tokenization_window,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if len(input_ids) <= answer_tokens or offsets is None:
        return None
    boundary = window_start + offsets[-answer_tokens][0]
    if boundary <= 0 or not target[:boundary].strip() or not target[boundary:].strip():
        return None
    return ReasoningSplit(
        prefix="",
        analysis=target[:boundary],
        suffix=target[boundary:],
        method=f"fallback:last-{answer_tokens}-tokens",
    )


def wrap_reasoning_split(split: ReasoningSplit) -> str:
    """Insert exactly one nonempty LCLM memory region into a split target."""

    if not split.analysis.strip():
        raise ValueError("analysis span must contain non-whitespace text")
    return f"{split.prefix}{MEMORY_START}{split.analysis}{MEMORY_END}{split.suffix}"


def reasoning_row_is_compressed(key: str, fraction: float = 0.5) -> bool:
    """Deterministically assign a reasoning row to the compressed mixture arm."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("reasoning compression fraction must be between 0 and 1")
    if fraction == 0.0:
        return False
    if fraction == 1.0:
        return True
    digest = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=8,
        person=b"lclm-cot-mix",
    ).digest()
    sample = int.from_bytes(digest, "big") / float(1 << 64)
    return sample < fraction


def _empty_output_row() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "data_type": None,
        "source_dataset": None,
        "sub_dataset": None,
        "compression_scope": None,
        "reasoning_split": None,
        "prompt": None,
        "compression_prompt": None,
        "target": None,
        "messages": None,
        "conversations": None,
        "tools": None,
        "task": None,
        "trace_source": None,
        "agent": None,
        "model": None,
        "model_provider": None,
        "result": None,
        "episode": None,
        "run_id": None,
        "trial_name": None,
        "date": None,
        "metadata_json": "{}",
    }


def _metadata_json(row: Mapping[str, Any], consumed: set[str]) -> str:
    extras = {key: copy.deepcopy(value) for key, value in row.items() if key not in consumed}
    return json.dumps(extras, ensure_ascii=False, sort_keys=True, default=str)


def _has_memory_marker(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in MEMORY_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _has_memory_marker(key) or _has_memory_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_memory_marker(item) for item in value)
    return False


def _validate_native_trajectory(field: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ValueError(f"agent {field} must be a nonempty list")
    if not all(isinstance(message, Mapping) for message in value):
        raise ValueError(f"agent {field} must contain message mappings")


def transform_stage3_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str = DEFAULT_STAGE3_DATASET,
    reasoning_subdatasets: Iterable[str] = DEFAULT_REASONING_SUBDATASETS,
    analysis_tag_pairs: Sequence[tuple[str, str]] = DEFAULT_ANALYSIS_TAG_PAIRS,
    final_answer_markers: Sequence[str] = DEFAULT_FINAL_ANSWER_MARKERS,
    fallback_tokenizer: Any | None = None,
    fallback_answer_tokens: int = DEFAULT_FALLBACK_ANSWER_TOKENS,
    compress_reasoning: bool = True,
) -> dict[str, Any]:
    """Transform one stage-3 row without accessing a dataset or tokenizer."""

    required = ("prompt", "compression_prompt", "target", "sub_dataset")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"stage-3 row is missing required fields: {', '.join(missing)}")
    if not isinstance(row["target"], str):
        raise TypeError("stage-3 target must be a string")

    output = _empty_output_row()
    output.update(
        {
            "data_type": "stage3",
            "source_dataset": source_dataset,
            "sub_dataset": row["sub_dataset"],
            "prompt": copy.deepcopy(row["prompt"]),
            "compression_prompt": copy.deepcopy(row["compression_prompt"]),
            "target": row["target"],
        }
    )

    selected = row["sub_dataset"] in set(reasoning_subdatasets)
    if selected:
        # A reasoning example never spends compression capacity on the question.
        output["compression_prompt"] = copy.deepcopy(row["prompt"])
        if not compress_reasoning:
            output["compression_scope"] = "none"
            output["reasoning_split"] = "mixture:uncompressed"
        else:
            split = split_reasoning_target(
                row["target"],
                analysis_tag_pairs=analysis_tag_pairs,
                final_answer_markers=final_answer_markers,
                fallback_tokenizer=fallback_tokenizer,
                fallback_answer_tokens=fallback_answer_tokens,
            )
            if split is None:
                output["compression_scope"] = "none"
                output["reasoning_split"] = "none"
            else:
                output["target"] = wrap_reasoning_split(split)
                output["compression_scope"] = "target_analysis"
                output["reasoning_split"] = split.method
    else:
        output["compression_scope"] = (
            "prompt" if _has_memory_marker(row["compression_prompt"]) else "none"
        )
        output["reasoning_split"] = "not_selected"

    consumed = {"prompt", "compression_prompt", "target", "sub_dataset"}
    output["metadata_json"] = _metadata_json(row, consumed)
    return output


def transform_agent_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str = DEFAULT_AGENT_DATASET,
) -> dict[str, Any]:
    """Validate and preserve a native multi-turn agent trajectory."""

    messages = row.get("messages")
    conversations = row.get("conversations")
    if messages is None and conversations is None:
        raise ValueError("agent row must contain native messages or conversations")
    _validate_native_trajectory("messages", messages)
    _validate_native_trajectory("conversations", conversations)
    if messages is not None and conversations is not None and messages != conversations:
        raise ValueError("agent row has conflicting messages and conversations")
    if _has_memory_marker(row):
        raise ValueError("agent row contains an LCLM memory marker")

    output = _empty_output_row()
    output.update(
        {
            "data_type": "agent_trajectory",
            "source_dataset": source_dataset,
            "sub_dataset": row.get("sub_dataset") or row.get("trace_source"),
            "compression_scope": "none",
            "reasoning_split": "not_applicable",
            # Emit one canonical trajectory column. Keeping both populated would
            # make downstream routing ambiguous; OpenThoughts currently uses
            # ``conversations`` while native Qwen exports commonly use ``messages``.
            "messages": copy.deepcopy(messages if messages is not None else conversations),
            "conversations": None,
            "tools": copy.deepcopy(row.get("tools")),
            "task": copy.deepcopy(row.get("task")),
        }
    )

    metadata_fields = {
        "trace_source",
        "agent",
        "model",
        "model_provider",
        "result",
        "episode",
        "run_id",
        "trial_name",
        "date",
    }
    for field in metadata_fields:
        output[field] = copy.deepcopy(row.get(field))

    consumed = {
        "messages",
        "conversations",
        "tools",
        "task",
        "sub_dataset",
        *metadata_fields,
    }
    output["metadata_json"] = _metadata_json(row, consumed)
    return output


def agent_row_is_successful(row: Mapping[str, Any]) -> bool:
    """OpenThoughts records failed rollouts as a nonempty ``result`` value."""
    result = row.get("result")
    return result is None or result == ""


def agent_row_is_primary_trace(row: Mapping[str, Any]) -> bool:
    """Exclude Agent2's derived summary/answer variants unless requested."""
    trace_source = row.get("trace_source")
    return not (
        isinstance(trace_source, str)
        and trace_source.startswith("summarization-")
    )


def weighted_interleave(
    streams: Sequence[Iterable[dict[str, Any]]],
    weights: Sequence[float],
    *,
    seed: int,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Randomly interleave finite or streaming iterables with bounded memory."""

    if len(streams) != len(weights) or not streams:
        raise ValueError("streams and weights must have the same nonzero length")
    if any(weight <= 0 for weight in weights):
        raise ValueError("all mixture weights must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")

    rng = random.Random(seed)
    active = [(iter(stream), float(weight)) for stream, weight in zip(streams, weights)]
    emitted = 0
    while active and (max_rows is None or emitted < max_rows):
        index = rng.choices(
            range(len(active)), weights=[entry[1] for entry in active], k=1
        )[0]
        iterator, _ = active[index]
        try:
            item = next(iterator)
        except StopIteration:
            active.pop(index)
            continue
        emitted += 1
        yield item


def _iter_source(
    spec: SourceSpec,
    *,
    split: str,
    reasoning_subdatasets: set[str],
    analysis_tag_pairs: Sequence[tuple[str, str]],
    final_answer_markers: Sequence[str],
    fallback_tokenizer: Any | None = None,
    fallback_answer_tokens: int = DEFAULT_FALLBACK_ANSWER_TOKENS,
    reasoning_compression_fraction: float = DEFAULT_REASONING_COMPRESSION_FRACTION,
    include_failed_agent_traces: bool = False,
    include_derived_agent_traces: bool = False,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only by the CLI
        raise RuntimeError("the CLI requires the 'datasets' package") from exc

    for repeat_index in range(spec.repeat):
        dataset = load_dataset(spec.dataset_id, split=split, streaming=True)
        emitted = 0
        for row_index, row in enumerate(dataset):
            try:
                if spec.kind == "stage3":
                    transformed = transform_stage3_row(
                        row,
                        source_dataset=spec.dataset_id,
                        reasoning_subdatasets=reasoning_subdatasets,
                        analysis_tag_pairs=analysis_tag_pairs,
                        final_answer_markers=final_answer_markers,
                        fallback_tokenizer=fallback_tokenizer,
                        fallback_answer_tokens=fallback_answer_tokens,
                        compress_reasoning=reasoning_row_is_compressed(
                            f"{spec.dataset_id}:{repeat_index}:{row_index}",
                            reasoning_compression_fraction,
                        ),
                    )
                else:
                    if not include_failed_agent_traces and not agent_row_is_successful(row):
                        continue
                    if (
                        not include_derived_agent_traces
                        and not agent_row_is_primary_trace(row)
                    ):
                        continue
                    transformed = transform_agent_row(row, source_dataset=spec.dataset_id)
                yield transformed
                emitted += 1
                if spec.max_rows is not None and emitted >= spec.max_rows:
                    break
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{spec.dataset_id} repeat {repeat_index}, row {row_index}: {exc}"
                ) from exc


def write_jsonl(
    rows: Iterable[Mapping[str, Any]], output: str | os.PathLike[str], *, overwrite: bool
) -> int:
    """Atomically write a stream as JSONL (optionally gzip-compressed)."""

    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        opener = gzip.open if output_path.suffix == ".gz" else open
        with opener(temporary_path, "wt", encoding="utf-8") as handle:
            for row in rows:
                if tuple(row.keys()) != SCHEMA_KEYS:
                    raise ValueError("row does not match the canonical output schema/order")
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count


def _weights(values: Sequence[float] | None, count: int, option: str) -> list[float]:
    if values is None:
        return [1.0] * count
    if len(values) != count:
        raise ValueError(f"{option} must be supplied once per corresponding dataset")
    if any(value <= 0 for value in values):
        raise ValueError(f"{option} values must be positive")
    return list(values)


def build_source_specs(args: argparse.Namespace) -> list[SourceSpec]:
    """Resolve CLI source defaults without introducing duplicate datasets."""

    stage3_ids = args.stage3_dataset or [DEFAULT_STAGE3_DATASET]
    agent_ids = args.agent_dataset or [DEFAULT_AGENT_DATASET]
    stage3_weights = _weights(args.stage3_weight, len(stage3_ids), "--stage3-weight")
    agent_weights = _weights(args.agent_weight, len(agent_ids), "--agent-weight")
    for option, value in (
        ("--stage3-max-rows", args.stage3_max_rows),
        ("--agent-max-rows", args.agent_max_rows),
        ("--coldstart-max-rows", args.coldstart_max_rows),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{option} must be positive")
    if args.include_coldstart and args.coldstart_weight <= 0:
        raise ValueError("--coldstart-weight must be positive")
    for option, value in (
        ("--stage3-repeat", args.stage3_repeat),
        ("--agent-repeat", args.agent_repeat),
        ("--coldstart-repeat", args.coldstart_repeat),
    ):
        if value <= 0:
            raise ValueError(f"{option} must be positive")

    specs = [
        SourceSpec(
            dataset_id,
            "stage3",
            weight,
            args.stage3_max_rows,
            args.stage3_repeat,
        )
        for dataset_id, weight in zip(stage3_ids, stage3_weights)
    ]
    specs.extend(
        SourceSpec(
            dataset_id,
            "agent",
            weight,
            args.agent_max_rows,
            args.agent_repeat,
        )
        for dataset_id, weight in zip(agent_ids, agent_weights)
    )
    if args.include_coldstart:
        specs.append(
            SourceSpec(
                COLDSTART_AGENT_DATASET,
                "agent",
                args.coldstart_weight,
                args.coldstart_max_rows,
                args.coldstart_repeat,
            )
        )

    duplicates = {
        spec.dataset_id
        for spec in specs
        if sum(other.dataset_id == spec.dataset_id for other in specs) > 1
    }
    if duplicates:
        raise ValueError(f"dataset IDs must be unique; duplicates: {sorted(duplicates)!r}")
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage3-dataset",
        action="append",
        help=f"stage-3 dataset ID (default: {DEFAULT_STAGE3_DATASET})",
    )
    parser.add_argument(
        "--stage3-weight",
        action="append",
        type=float,
        help="weight for each --stage3-dataset",
    )
    parser.add_argument("--stage3-max-rows", type=int, help="row cap per stage-3 source pass")
    parser.add_argument("--stage3-repeat", type=int, default=1, help="number of passes over each stage-3 source")
    parser.add_argument(
        "--agent-dataset",
        action="append",
        help=f"agent dataset ID (default: {DEFAULT_AGENT_DATASET})",
    )
    parser.add_argument(
        "--agent-weight",
        action="append",
        type=float,
        help="weight for each --agent-dataset",
    )
    parser.add_argument("--agent-max-rows", type=int, help="successful-row cap per agent source pass")
    parser.add_argument("--agent-repeat", type=int, default=1, help="number of passes over each agent source (explicit upsampling)")
    parser.add_argument(
        "--include-coldstart",
        action="store_true",
        help=f"also include {COLDSTART_AGENT_DATASET}",
    )
    parser.add_argument(
        "--include-failed-agent-traces",
        action="store_true",
        help="include rows whose OpenThoughts result field records an error",
    )
    parser.add_argument(
        "--include-derived-agent-traces",
        action="store_true",
        help="include Agent2 summary/answer variants in addition to primary traces",
    )
    parser.add_argument("--coldstart-weight", type=float, default=1.0)
    parser.add_argument("--coldstart-max-rows", type=int)
    parser.add_argument("--coldstart-repeat", type=int, default=1)
    parser.add_argument("--split", default="train", help="input dataset split (default: train)")
    parser.add_argument(
        "--reasoning-subdataset",
        action="append",
        help="selected sub_dataset; repeat to replace conservative defaults",
    )
    parser.add_argument(
        "--analysis-tag-pair",
        action="append",
        nargs=2,
        metavar=("START", "END"),
        help="explicit analysis delimiters; repeat to replace defaults",
    )
    parser.add_argument(
        "--final-answer-marker",
        action="append",
        help="explicit answer-boundary marker; repeat to replace defaults",
    )
    parser.add_argument(
        "--decoder-tokenizer",
        default=DEFAULT_DECODER_TOKENIZER,
        help="fast tokenizer used for the last-N-token reasoning fallback",
    )
    parser.add_argument(
        "--fallback-answer-tokens",
        type=int,
        default=DEFAULT_FALLBACK_ANSWER_TOKENS,
        help="trainable suffix size when no explicit answer boundary is found",
    )
    parser.add_argument(
        "--reasoning-compression-fraction",
        type=float,
        default=DEFAULT_REASONING_COMPRESSION_FRACTION,
        help="deterministic fraction of reasoning rows whose CoT is compressed",
    )
    parser.add_argument("--max-rows", type=int, help="total output row cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="output .jsonl or .jsonl.gz path")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = build_source_specs(args)
        reasoning_subdatasets = set(args.reasoning_subdataset or DEFAULT_REASONING_SUBDATASETS)
        analysis_tag_pairs = tuple(
            tuple(pair) for pair in (args.analysis_tag_pair or DEFAULT_ANALYSIS_TAG_PAIRS)
        )
        final_answer_markers = tuple(args.final_answer_marker or DEFAULT_FINAL_ANSWER_MARKERS)
        if args.fallback_answer_tokens <= 0:
            raise ValueError("--fallback-answer-tokens must be positive")
        if not 0.0 <= args.reasoning_compression_fraction <= 1.0:
            raise ValueError("--reasoning-compression-fraction must be between 0 and 1")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "the CLI requires transformers for the token fallback"
            ) from exc
        fallback_tokenizer = AutoTokenizer.from_pretrained(
            args.decoder_tokenizer,
            use_fast=True,
        )
        streams = [
            _iter_source(
                spec,
                split=args.split,
                reasoning_subdatasets=reasoning_subdatasets,
                analysis_tag_pairs=analysis_tag_pairs,
                final_answer_markers=final_answer_markers,
                fallback_tokenizer=fallback_tokenizer,
                fallback_answer_tokens=args.fallback_answer_tokens,
                reasoning_compression_fraction=args.reasoning_compression_fraction,
                include_failed_agent_traces=args.include_failed_agent_traces,
                include_derived_agent_traces=args.include_derived_agent_traces,
            )
            for spec in specs
        ]
        rows = weighted_interleave(
            streams,
            [spec.weight for spec in specs],
            seed=args.seed,
            max_rows=args.max_rows,
        )
        count = write_jsonl(rows, args.output, overwrite=args.overwrite)
    except (FileExistsError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"wrote {count:,} rows to {Path(args.output).resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

