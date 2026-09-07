"""Build selective-expansion tasks from real document-grounded QA sources."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from data.synthetic_expansion_agent import (
    GENERATOR_VERSION,
    MEMORY_END,
    MEMORY_MARKERS,
    MEMORY_START,
    format_rollout_user_prompt,
    format_training_user_prompt,
    verify_trace,
)


REAL_GENERATOR_VERSION = "real-expansion-v1"
MIN_SEGMENT_WORDS = 512
TARGET_SEGMENT_WORDS = 768
MAX_SEGMENT_WORDS = 896


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for marker in MEMORY_MARKERS:
        text = text.replace(marker, "")
    return text


def _focus_window(text: str, focus: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    normalized_focus = _clean_text(focus).casefold()
    character = text.casefold().find(normalized_focus) if normalized_focus else -1
    if character < 0:
        return " ".join(words[:max_words])
    word_index = len(text[:character].split())
    start = max(0, min(word_index - max_words // 3, len(words) - max_words))
    return " ".join(words[start : start + max_words])


def build_real_segment(
    primary: Mapping[str, str],
    companions: Sequence[Mapping[str, str]],
    *,
    focus: str = "",
    min_words: int = MIN_SEGMENT_WORDS,
    max_words: int = MAX_SEGMENT_WORDS,
) -> str:
    """Create one long segment using source text only, never synthetic padding."""

    primary_text = _clean_text(primary.get("text"))
    if not primary_text:
        raise ValueError("primary source document is empty")
    primary_window = _focus_window(primary_text, focus, max_words)
    pieces = [
        f"SOURCE DOCUMENT {primary.get('document_id', 'unknown')}:\n{primary_window}"
    ]
    word_count = len(" ".join(pieces).split())
    for companion in companions:
        if word_count >= min_words:
            break
        companion_text = _clean_text(companion.get("text"))
        if not companion_text:
            continue
        remaining = max_words - word_count
        if remaining <= 16:
            break
        companion_window = " ".join(companion_text.split()[:remaining])
        pieces.append(
            f"RELATED SOURCE DOCUMENT {companion.get('document_id', 'unknown')}:\n"
            f"{companion_window}"
        )
        word_count = len(" ".join(pieces).split())
    result = "\n\n".join(pieces)
    words = result.split()
    if len(words) < min_words:
        raise ValueError(
            f"not enough real source text for a segment: {len(words)} < {min_words} words"
        )
    if len(words) > max_words:
        result = " ".join(words[:max_words])
    return result


def _stable_task_id(source_dataset: str, source_row_id: str) -> str:
    digest = hashlib.sha256(
        f"{REAL_GENERATOR_VERSION}:{source_dataset}:{source_row_id}".encode()
    ).hexdigest()[:24]
    return f"rea-{digest}"


def make_real_expansion_task(
    *,
    source_dataset: str,
    source_row_id: str,
    question: str,
    gold_answer: str,
    support_document: Mapping[str, str],
    distractor_documents: Sequence[Mapping[str, str]],
    companion_documents: Sequence[Mapping[str, str]],
    source_category: str,
    seed: int,
    support_focus: str = "",
) -> dict[str, Any]:
    """Convert a gold real-source QA row into the common expansion task schema."""

    question = _clean_text(question)
    gold_answer = _clean_text(gold_answer)
    if not question or not gold_answer:
        raise ValueError("question and gold answer must be nonempty")
    if not distractor_documents:
        raise ValueError("at least one distractor document is required")

    rng = random.Random(
        f"{REAL_GENERATOR_VERSION}:{seed}:{source_dataset}:{source_row_id}"
    )
    requested = [support_document, *distractor_documents]
    requested_ids = {str(document["document_id"]) for document in requested}
    raw_segments: list[dict[str, str]] = []
    for index, document in enumerate(requested):
        companions = [
            companion
            for companion in companion_documents
            if str(companion.get("document_id")) not in requested_ids
        ]
        rng.shuffle(companions)
        is_support = index == 0
        focus = (support_focus or gold_answer) if is_support else ""
        text = build_real_segment(document, companions, focus=focus)
        if is_support:
            summary = (
                f"A potentially relevant {source_category} source for the question: "
                f"{question} Exact answer-bearing details are omitted."
            )
        else:
            title = _clean_text(document.get("title") or document.get("document_id"))
            summary = (
                f"Another {source_category} source titled {title}; exact contents omitted."
            )
        raw_segments.append(
            {
                "record_id": str(document["document_id"]),
                "summary": summary,
                "text": text,
                "is_support": is_support,
            }
        )
    rng.shuffle(raw_segments)

    segments: list[dict[str, str]] = []
    support_segment_ids: list[str] = []
    for position, raw in enumerate(raw_segments, start=1):
        segment_id = f"seg_{position}"
        segment = {
            "segment_id": segment_id,
            "record_id": raw["record_id"],
            "summary": raw["summary"],
            "text": raw["text"],
        }
        segments.append(segment)
        if raw["is_support"]:
            support_segment_ids.append(segment_id)

    rendered_question = (
        f"{question}\nReturn exactly `FINAL: ANSWER`, replacing ANSWER with your answer."
    )
    expected_final = f"FINAL: {gold_answer}"
    task = {
        "schema_version": 2,
        "generator_version": REAL_GENERATOR_VERSION,
        "task_id": _stable_task_id(source_dataset, source_row_id),
        "family": source_category,
        "source_dataset": source_dataset,
        "source_row_id": source_row_id,
        "question": rendered_question,
        "raw_question": question,
        "gold_answer": gold_answer,
        "expected_final": expected_final,
        "support_segment_ids": support_segment_ids,
        "segments": segments,
        "seed": seed,
    }
    task["training_user_prompt"] = format_training_user_prompt(
        segments, rendered_question
    )
    task["user_prompt"] = task["training_user_prompt"]
    task["rollout_user_prompt"] = format_rollout_user_prompt(
        segments, rendered_question
    )
    return task


def validate_real_task(task: Mapping[str, Any]) -> None:
    segments = task.get("segments")
    if not isinstance(segments, Sequence) or not segments:
        raise ValueError("task has no segments")
    segment_ids = set()
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("segment is not a mapping")
        segment_id = segment.get("segment_id")
        text = segment.get("text")
        if not isinstance(segment_id, str) or not re.fullmatch(r"seg_[1-9][0-9]*", segment_id):
            raise ValueError("invalid segment ID")
        if segment_id in segment_ids:
            raise ValueError("duplicate segment ID")
        segment_ids.add(segment_id)
        if not isinstance(text, str) or len(text.split()) < MIN_SEGMENT_WORDS:
            raise ValueError("segment is below the real-text word floor")
        if any(marker in text for marker in MEMORY_MARKERS):
            raise ValueError("raw segment text contains memory markers")
    support = set(task.get("support_segment_ids", []))
    if not support or not support <= segment_ids:
        raise ValueError("support segments are missing or invalid")
    prompt = task.get("training_user_prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("training prompt is not text")
    if prompt.count(MEMORY_START) != len(segments) or prompt.count(MEMORY_END) != len(segments):
        raise ValueError("training prompt memory-block count is inconsistent")


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _final_payload(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        matches = re.findall(r"(?im)^FINAL:\s*(.+?)\s*$", message["content"])
        if matches:
            return matches[-1].strip()
    return ""


def _numeric_value(text: str) -> tuple[Decimal, bool] | None:
    match = re.fullmatch(
        r"\s*[$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%)?\s*",
        text,
    )
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", "")), bool(match.group(2))
    except InvalidOperation:
        return None


def _token_f1(candidate: str, gold: str) -> float:
    candidate_tokens = _normalized_tokens(candidate)
    gold_tokens = _normalized_tokens(gold)
    if not candidate_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(candidate_tokens) & Counter(gold_tokens)).values())
    return 2 * overlap / (len(candidate_tokens) + len(gold_tokens))


def verify_real_trace(
    task: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify tool structure first, then apply the source's native answer metric."""

    structural = verify_trace(task, messages)
    result = structural.as_dict()
    if structural.reason not in {"accepted", "wrong_final_answer"}:
        return result
    if not structural.expanded_support:
        result.update(accepted=False, reason="no_tool_call")
        return result
    if structural.missing_support:
        result.update(accepted=False, reason="missing_support")
        return result
    payload = _final_payload(messages)
    result["parsed_final"] = payload or None
    if not payload:
        result.update(accepted=False, reason="missing_final_answer")
        return result

    source = str(task.get("source_dataset", ""))
    gold = str(task.get("gold_answer", "")).strip()
    payload_tokens = _normalized_tokens(payload)
    gold_tokens = _normalized_tokens(gold)
    metric = "normalized_exact"
    accepted = payload_tokens == gold_tokens

    if source == "stanfordnlp/contract-nli":
        normalized = " ".join(payload_tokens)
        if normalized.startswith("entail"):
            predicted = "entailment"
        elif normalized.startswith("contradict"):
            predicted = "contradiction"
        elif normalized.startswith("not mention") or "neither entailed nor contradicted" in normalized:
            predicted = "notmentioned"
        else:
            predicted = normalized
        accepted = predicted == "".join(gold_tokens)
        metric = "contract_nli_label"
    elif source == "qiaojin/PubMedQA:pqa_labeled":
        predicted = payload_tokens[0] if payload_tokens else ""
        accepted = predicted == (gold_tokens[0] if gold_tokens else "")
        metric = "pubmedqa_decision"
    elif source == "bevaya/FinQA":
        candidate_number = _numeric_value(payload)
        gold_number = _numeric_value(gold)
        accepted = candidate_number is not None and candidate_number == gold_number
        metric = "finqa_numeric"
    elif source == "PrimeQA/clapnq":
        normalized_payload = " ".join(payload_tokens)
        normalized_gold = " ".join(gold_tokens)
        containment = (
            len(payload_tokens) >= 2
            and (
                normalized_payload in normalized_gold
                or normalized_gold in normalized_payload
            )
        )
        score = _token_f1(payload, gold)
        accepted = containment or score >= 0.60
        result["answer_score"] = score
        metric = "clapnq_containment_or_token_f1"

    result["answer_metric"] = metric
    result["accepted"] = accepted
    result["reason"] = f"accepted:{metric}" if accepted else f"wrong_answer:{metric}"
    return result


def json_record(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = [
    "GENERATOR_VERSION",
    "MAX_SEGMENT_WORDS",
    "MIN_SEGMENT_WORDS",
    "REAL_GENERATOR_VERSION",
    "build_real_segment",
    "make_real_expansion_task",
    "validate_real_task",
    "verify_real_trace",
]
