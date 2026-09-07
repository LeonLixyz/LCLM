"""Deterministic synthetic tasks for native selective-memory expansion.

Each task presents positional ``seg_i`` context blocks whose visible bodies are
compressed summaries wrapped in LCLM memory markers.  Exact source text stays
outside the model context until the assistant calls the native ``expand`` tool
for a segment.  Accepted traces must expand every programmatically required
segment and emit the exact final answer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"
MEMORY_MARKERS = (MEMORY_START, MEMORY_END, "<|memory|>")

SCHEMA_VERSION = 2
GENERATOR_VERSION = "synthetic-expansion-v2"
DEFAULT_DISTRACTORS = 32
DEFAULT_MAX_TOOL_CALLS = 8
MIN_SEGMENT_WORDS = 512

TASK_FAMILIES = (
    "latest_state",
    "two_hop_join",
    "multi_key_lookup",
    "numeric_comparison",
    "set_intersection",
)

EXPAND_TOOL = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": (
            "Return the original uncompressed text of one compressed context "
            "segment. Use this when exact details are missing from a segment summary."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "pattern": "^seg_[1-9][0-9]*$",
                    "description": "Segment identifier to expand, such as seg_3.",
                }
            },
            "required": ["segment_id"],
            "additionalProperties": False,
        },
    },
}

TEACHER_SYSTEM_PROMPT = """Generate a correct native-tool agent trajectory for the task.
The user supplies context as numbered compressed segments named seg_i. You see
lossy routing summaries rather than the model's actual memory representations.
When exact evidence is needed, call the expand tool with the relevant
segment_id. After each call, the tool returns that segment's original text.

Use native tool calls, not textual descriptions of calls. Expand every segment
needed to ground the answer, including multiple segments when the question
requires combining evidence. Never guess details omitted from the summaries.
Do not reveal hidden reasoning. When sufficient evidence has been expanded,
finish with exactly the requested FINAL line and no further tool call.

This is a teacher-only generation instruction. Do not repeat or refer to it in
the trajectory.
"""

_NAMESPACE = uuid.UUID("5c8f153b-180d-55ad-8ba8-0ef20c37f03a")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: str
    parsed_final: str | None
    expanded_support: tuple[str, ...]
    missing_support: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "parsed_final": self.parsed_final,
            "expanded_support": list(self.expanded_support),
            "missing_support": list(self.missing_support),
        }


def _normalized_answer(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def _stable_task_id(seed: int, index: int, family: str) -> str:
    name = f"{GENERATOR_VERSION}:{seed}:{index}:{family}"
    return f"sea-{uuid.uuid5(_NAMESPACE, name)}"


def _code(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(100, 1000)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"


def _person(rng: random.Random) -> str:
    first = rng.choice(
        ("Ari", "Bela", "Cato", "Dara", "Evin", "Fara", "Galen", "Hana", "Ivo", "Jora")
    )
    last = rng.choice(
        ("Morrow", "Vale", "Sato", "Kern", "Ilyan", "Pike", "Renn", "Sol", "Tarin", "Wren")
    )
    return f"{first} {last}"


def _source(record_id: str, summary: str, text: str) -> dict[str, str]:
    if any(marker in summary or marker in text for marker in MEMORY_MARKERS):
        raise ValueError("source text may not contain LCLM memory markers")
    return {"record_id": record_id, "summary": summary, "text": text}


def _long_document(rng: random.Random, record_id: str, core_text: str) -> str:
    """Bury a programmatic fact in a deterministic 512+ token record.

    The lower bound is enforced in whitespace-delimited words. Since every word
    produces at least one decoder token, this guarantees at least 512 tokens
    without importing a tokenizer into the pure task generator. The pinned Qwen
    tokenizer is checked separately during pilot validation.
    """

    desks = (
        "archive control",
        "quality assurance",
        "operations review",
        "records compliance",
        "logistics audit",
        "inventory governance",
    )
    retention = ("standard", "extended", "priority", "regulated")
    conditions = (
        "normal operating conditions",
        "routine verification conditions",
        "the scheduled review window",
        "a controlled reconciliation period",
    )
    notes: list[str] = []
    entry = 1
    while len(" ".join(notes).split()) < MIN_SEGMENT_WORDS - len(core_text.split()):
        batch = _code(rng, "BAT")
        ticket = _code(rng, "TKT")
        reviewer = rng.randrange(100, 999)
        cycle = rng.randrange(10, 100)
        day = rng.randrange(1, 29)
        notes.append(
            f"Administrative entry {entry} for {record_id} states that {rng.choice(desks)} "
            f"reviewed batch {batch} under {rng.choice(conditions)} on day {day} of cycle "
            f"{cycle}. Reviewer {reviewer} cross-checked ticket {ticket}, confirmed the "
            f"{rng.choice(retention)} retention class, and recorded no exception that changes "
            "the authoritative entity facts elsewhere in this document."
        )
        entry += 1

    insertion = rng.randrange(1, len(notes))
    notes.insert(insertion, core_text)
    text = "\n\n".join(notes)
    if len(text.split()) < MIN_SEGMENT_WORDS:
        raise AssertionError("long-document construction missed its word floor")
    return text


def _long_source(
    rng: random.Random,
    record_id: str,
    summary: str,
    core_text: str,
) -> dict[str, str]:
    return _source(record_id, summary, _long_document(rng, record_id, core_text))


def _distractor_sources(rng: random.Random, count: int) -> list[dict[str, str]]:
    statuses = ("queued", "approved", "delayed", "packed", "cancelled", "released")
    regions = ("Northern", "Southern", "Eastern", "Western", "Central")
    sources: list[dict[str, str]] = []
    for index in range(count):
        kind = index % 5
        if kind == 0:
            key = _code(rng, "AST")
            text = (
                f"The movement ledger for asset {key} records an assignment to Bay "
                f"{rng.randrange(1, 90)} at {8 + index % 10:02d}:15."
            )
            summary = f"Movement ledger for asset {key}; exact bay and timing details omitted."
        elif kind == 1:
            key = _code(rng, "ORD")
            text = f"Order {key} currently has status {rng.choice(statuses)}."
            summary = f"Status record for order {key}; exact status omitted."
        elif kind == 2:
            key = _code(rng, "VND")
            text = (
                f"Vendor {key} is registered in the {rng.choice(regions)} region. "
                f"The registry was renewed in cycle {rng.randrange(20, 90)}."
            )
            summary = f"Registry record for vendor {key}; exact region and registry details omitted."
        elif kind == 3:
            key = _code(rng, "SNS")
            text = (
                f"Sensor {key} reported {rng.randrange(100, 900)} calibrated units. "
                f"The reading passed calibration batch {rng.randrange(1000, 9999)}."
            )
            summary = f"Calibrated reading for sensor {key}; exact numeric value omitted."
        else:
            key = _code(rng, "PRJ")
            members = ", ".join(_person(rng) for _ in range(4))
            text = f"Project {key} roster: {members}."
            summary = f"Roster record for project {key}; member names omitted."
        record_id = f"d-{index:04d}-{key}"
        sources.append(_long_source(rng, record_id, summary, text))
    return sources


def _latest_state_task(rng: random.Random) -> tuple[str, str, list[dict[str, str]], list[str]]:
    asset = _code(rng, "AST")
    hours = (8, 11, 15)
    bays = rng.sample(range(1, 90), len(hours))
    sources: list[dict[str, str]] = []
    support: list[str] = []
    for hour, bay in zip(hours, bays):
        record_id = f"state-{asset}-{hour:02d}00"
        core = (
            f"The authoritative movement event at {hour:02d}:00 states that asset "
            f"{asset} was reassigned to Bay {bay}."
        )
        sources.append(
            _long_source(
                rng,
                record_id,
                f"Movement record for asset {asset} at {hour:02d}:00; exact bay omitted.",
                core,
            )
        )
        support.append(record_id)
    question = (
        f"According to all three movement records, list the bays assigned to asset {asset} "
        "at 08:00, 11:00, and 15:00 in chronological order, then repeat the latest bay. "
        "Return exactly `FINAL: Bay A | Bay B | Bay C | LATEST Bay C`."
    )
    answer = f"Bay {bays[0]} | Bay {bays[1]} | Bay {bays[2]} | LATEST Bay {bays[2]}"
    return question, answer, sources, support


def _two_hop_join_task(rng: random.Random) -> tuple[str, str, list[dict[str, str]], list[str]]:
    shipment = _code(rng, "SHP")
    vendor = _code(rng, "VND")
    facility = _code(rng, "FAC")
    region = rng.choice(("Northern", "Southern", "Eastern", "Western", "Central"))
    shipment_id = f"join-shipment-{shipment}"
    vendor_id = f"join-vendor-{vendor}"
    facility_id = f"join-facility-{facility}"
    sources = [
        _long_source(
            rng,
            shipment_id,
            f"Manifest for shipment {shipment}; supplier identity omitted.",
            f"Shipment {shipment} was supplied by vendor {vendor}.",
        ),
        _long_source(
            rng,
            vendor_id,
            f"Assignment record for vendor {vendor}; facility identity omitted.",
            f"Vendor {vendor} is assigned to compliance facility {facility}.",
        ),
        _long_source(
            rng,
            facility_id,
            f"Registry record for facility {facility}; exact region omitted.",
            f"Compliance facility {facility} is registered in the {region} region.",
        ),
    ]
    question = (
        f"In which region is the compliance facility assigned to the supplier of shipment "
        f"{shipment} registered? Follow the shipment, vendor, and facility records. "
        "Return exactly `FINAL: REGION`."
    )
    return question, region, sources, [shipment_id, vendor_id, facility_id]


def _multi_key_task(rng: random.Random) -> tuple[str, str, list[dict[str, str]], list[str]]:
    statuses = ("queued", "approved", "delayed", "packed", "cancelled", "released")
    orders = [_code(rng, "ORD") for _ in range(5)]
    selected = [rng.choice(statuses) for _ in orders]
    sources: list[dict[str, str]] = []
    support: list[str] = []
    for order, status in zip(orders, selected):
        record_id = f"order-{order}"
        sources.append(
            _long_source(
                rng,
                record_id,
                f"Status record for order {order}; exact status omitted.",
                f"Order {order} currently has status {status}.",
            )
        )
        support.append(record_id)
    question = (
        f"Report the statuses of orders {', '.join(orders)} in that order. "
        "Return exactly `FINAL: STATUS1 | STATUS2 | STATUS3 | STATUS4 | STATUS5`."
    )
    return question, " | ".join(selected), sources, support


def _numeric_comparison_task(
    rng: random.Random,
) -> tuple[str, str, list[dict[str, str]], list[str]]:
    sensors = [_code(rng, "SNS") for _ in range(4)]
    values = rng.sample(range(120, 900), len(sensors))
    winner_index = max(range(len(values)), key=values.__getitem__)
    difference = max(values) - min(values)
    sources: list[dict[str, str]] = []
    support: list[str] = []
    for sensor, value in zip(sensors, values):
        record_id = f"sensor-{sensor}"
        sources.append(
            _long_source(
                rng,
                record_id,
                f"Calibrated reading for sensor {sensor}; exact numeric value omitted.",
                f"Sensor {sensor} reported {value} calibrated units.",
            )
        )
        support.append(record_id)
    question = (
        f"Among sensors {', '.join(sensors)}, which reported the largest value, and what is "
        "the difference between the largest and smallest readings? "
        "Return exactly `FINAL: SENSOR_ID | MAX_MINUS_MIN`."
    )
    return question, f"{sensors[winner_index]} | {difference}", sources, support


def _intersection_task(rng: random.Random) -> tuple[str, str, list[dict[str, str]], list[str]]:
    projects = [_code(rng, "PRJ") for _ in range(3)]
    shared = _person(rng)

    def unique_people(excluded: set[str]) -> list[str]:
        people: list[str] = []
        while len(people) < 3:
            candidate = _person(rng)
            if candidate not in excluded and candidate not in people:
                people.append(candidate)
        return people

    used = {shared}
    rosters: list[list[str]] = []
    for _ in projects:
        project_only = unique_people(used)
        used.update(project_only)
        members = project_only + [shared]
        rng.shuffle(members)
        rosters.append(members)
    sources: list[dict[str, str]] = []
    support: list[str] = []
    for project, members in zip(projects, rosters):
        record_id = f"roster-{project}"
        sources.append(
            _long_source(
                rng,
                record_id,
                f"Roster record for project {project}; member names omitted.",
                f"Project {project} roster: {', '.join(members)}.",
            )
        )
        support.append(record_id)
    question = (
        f"Who is the only person appearing in all three project rosters: "
        f"{projects[0]}, {projects[1]}, and {projects[2]}? "
        "Return exactly `FINAL: PERSON NAME`."
    )
    return question, shared, sources, support


_FAMILY_BUILDERS = {
    "latest_state": _latest_state_task,
    "two_hop_join": _two_hop_join_task,
    "multi_key_lookup": _multi_key_task,
    "numeric_comparison": _numeric_comparison_task,
    "set_intersection": _intersection_task,
}


def format_training_user_prompt(
    segments: Sequence[Mapping[str, str]], question: str
) -> str:
    blocks = [
        f"{segment['segment_id']}\n{MEMORY_START}{segment['text']}{MEMORY_END}"
        for segment in segments
    ]
    return "Context segments:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion:\n{question}"


def format_rollout_user_prompt(
    segments: Sequence[Mapping[str, str]], question: str
) -> str:
    """Present the teacher with routing clues, never raw source documents."""

    blocks = [
        f"{segment['segment_id']}\n[compressed segment]\n{segment['summary']}"
        for segment in segments
    ]
    return "Context segments:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion:\n{question}"


def generate_task(
    index: int,
    *,
    seed: int = 20260820,
    distractors: int = DEFAULT_DISTRACTORS,
    family: str | None = None,
) -> dict[str, Any]:
    if index < 0:
        raise ValueError("index must be non-negative")
    if distractors < 0:
        raise ValueError("distractors must be non-negative")
    selected_family = family or TASK_FAMILIES[index % len(TASK_FAMILIES)]
    if selected_family not in _FAMILY_BUILDERS:
        raise ValueError(f"unknown task family: {selected_family}")

    rng = random.Random(f"{GENERATOR_VERSION}:{seed}:{index}:{selected_family}")
    question, gold_answer, support_sources, support_record_ids = _FAMILY_BUILDERS[
        selected_family
    ](rng)
    sources = support_sources + _distractor_sources(rng, distractors)
    rng.shuffle(sources)
    segments: list[dict[str, str]] = []
    record_to_segment: dict[str, str] = {}
    for position, source in enumerate(sources, start=1):
        segment_id = f"seg_{position}"
        segment = {"segment_id": segment_id, **source}
        segments.append(segment)
        record_to_segment[source["record_id"]] = segment_id
    support_segment_ids = [record_to_segment[record_id] for record_id in support_record_ids]
    task_id = _stable_task_id(seed, index, selected_family)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "task_id": task_id,
        "family": selected_family,
        "question": question,
        "user_prompt": format_training_user_prompt(segments, question),
        "training_user_prompt": format_training_user_prompt(segments, question),
        "rollout_user_prompt": format_rollout_user_prompt(segments, question),
        "gold_answer": gold_answer,
        "expected_final": f"FINAL: {gold_answer}",
        "support_segment_ids": support_segment_ids,
        "segments": segments,
        "seed": seed,
        "index": index,
    }


def generate_tasks(
    count: int,
    *,
    seed: int = 20260820,
    distractors: int = DEFAULT_DISTRACTORS,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [generate_task(index, seed=seed, distractors=distractors) for index in range(count)]


def segment_map(task: Mapping[str, Any]) -> dict[str, str]:
    return {
        segment["segment_id"]: segment["text"]
        for segment in task.get("segments", [])
        if isinstance(segment, Mapping)
        and isinstance(segment.get("segment_id"), str)
        and isinstance(segment.get("text"), str)
    }


def expand(task: Mapping[str, Any], segment_id: str) -> str:
    if not isinstance(segment_id, str) or not re.fullmatch(r"seg_[1-9][0-9]*", segment_id):
        raise ValueError("segment_id must match seg_i")
    originals = segment_map(task)
    if segment_id not in originals:
        raise KeyError(f"unknown segment_id: {segment_id}")
    return originals[segment_id]


def _parse_tool_arguments(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("tool arguments must be a mapping")
    arguments = dict(raw)
    unexpected = set(arguments) - {"segment_id"}
    if unexpected:
        raise ValueError(f"unexpected expand arguments: {sorted(unexpected)}")
    segment_id = arguments.get("segment_id")
    if not isinstance(segment_id, str) or not re.fullmatch(r"seg_[1-9][0-9]*", segment_id):
        raise ValueError("expand segment_id must match seg_i")
    return {"segment_id": segment_id}


def canonicalize_assistant_response(response: Mapping[str, Any]) -> dict[str, Any]:
    content = response.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("assistant content must be a string or null")
    raw_calls = response.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValueError("assistant tool_calls must be a list")
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            raise ValueError(f"tool_calls[{index}] must be a mapping")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"tool_calls[{index}] has no call ID")
        if not isinstance(function, Mapping) or function.get("name") != "expand":
            raise ValueError(f"tool_calls[{index}] is not an expand call")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "expand",
                    "arguments": _parse_tool_arguments(function.get("arguments")),
                },
            }
        )
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        message["tool_calls"] = calls
    return message


def messages_for_openai_api(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = copy.deepcopy(list(messages))
    for message in output:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("assistant tool call has no function mapping")
            arguments = function.get("arguments")
            if isinstance(arguments, Mapping):
                function["arguments"] = json.dumps(
                    dict(arguments), ensure_ascii=False, sort_keys=True
                )
            elif not isinstance(arguments, str):
                raise ValueError("assistant tool arguments must be a mapping or JSON string")
    return output


def verify_trace(task: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> VerificationResult:
    originals = segment_map(task)
    call_segments: dict[str, str] = {}
    expanded: set[str] = set()
    tool_call_count = 0
    final_content: str | None = None

    for message in messages:
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            tool_call_count += len(calls)
            for call in calls:
                try:
                    canonical = canonicalize_assistant_response(
                        {"content": "", "tool_calls": [call]}
                    )["tool_calls"][0]
                except (TypeError, ValueError):
                    return VerificationResult(False, "invalid_tool_call", None, (), ())
                call_id = canonical["id"]
                segment_id = canonical["function"]["arguments"]["segment_id"]
                if call_id in call_segments:
                    return VerificationResult(False, "duplicate_tool_call_id", None, (), ())
                if segment_id not in originals:
                    return VerificationResult(False, "unknown_segment", None, (), ())
                call_segments[call_id] = segment_id
            if not calls and isinstance(message.get("content"), str):
                final_content = message["content"]
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in call_segments:
                return VerificationResult(False, "orphan_tool_result", None, (), ())
            segment_id = call_segments[call_id]
            if message.get("name") not in {None, "expand"}:
                return VerificationResult(False, "wrong_tool_result_name", None, (), ())
            if message.get("content") != originals[segment_id]:
                return VerificationResult(False, "incorrect_expansion", None, (), ())
            expanded.add(segment_id)

    support = set(task["support_segment_ids"])
    missing = support - expanded
    expanded_support = tuple(sorted(support & expanded))
    missing_support = tuple(sorted(missing))
    if tool_call_count == 0:
        return VerificationResult(False, "no_tool_call", None, expanded_support, missing_support)
    if missing:
        return VerificationResult(False, "missing_support", None, expanded_support, missing_support)
    if not final_content:
        return VerificationResult(False, "missing_final_answer", None, expanded_support, missing_support)
    if _normalized_answer(final_content) != _normalized_answer(task["expected_final"]):
        return VerificationResult(
            False, "wrong_final_answer", final_content, expanded_support, missing_support
        )
    return VerificationResult(True, "accepted", final_content, expanded_support, missing_support)


CompletionFunction = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Mapping[str, Any]
]


def run_agent_rollout(
    task: Mapping[str, Any],
    complete: CompletionFunction,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> dict[str, Any]:
    if max_tool_calls <= 0:
        raise ValueError("max_tool_calls must be positive")
    messages: list[dict[str, Any]] = []
    training_system_prompt = task.get("training_system_prompt")
    if training_system_prompt:
        messages.append({"role": "system", "content": training_system_prompt})
    messages.append({"role": "user", "content": task["training_user_prompt"]})
    rollout_messages: list[dict[str, Any]] = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": task["rollout_user_prompt"]},
    ]
    tool_call_count = 0
    failure_reason: str | None = None

    while True:
        try:
            assistant = canonicalize_assistant_response(
                complete(rollout_messages, [EXPAND_TOOL] if tool_call_count < max_tool_calls else [])
            )
        except (TypeError, ValueError) as exc:
            failure_reason = f"invalid_assistant_response:{exc}"
            break
        calls = assistant.get("tool_calls") or []
        if tool_call_count + len(calls) > max_tool_calls:
            failure_reason = "tool_call_limit"
            break
        messages.append(copy.deepcopy(assistant))
        rollout_messages.append(copy.deepcopy(assistant))
        if not calls:
            break
        for call in calls:
            segment_id = call["function"]["arguments"]["segment_id"]
            try:
                original = expand(task, segment_id)
            except (KeyError, ValueError):
                failure_reason = "unknown_segment"
                break
            tool_message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": "expand",
                "content": original,
            }
            messages.append(copy.deepcopy(tool_message))
            rollout_messages.append(copy.deepcopy(tool_message))
            tool_call_count += 1
        if failure_reason is not None:
            break

    verification = verify_trace(task, messages)
    if failure_reason is not None:
        verification = VerificationResult(
            False,
            failure_reason,
            verification.parsed_final,
            verification.expanded_support,
            verification.missing_support,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "data_type": "synthetic_expansion_agent",
        "source_dataset": GENERATOR_VERSION,
        "sub_dataset": task["family"],
        "task_id": task["task_id"],
        "compression_scope": "input_segments",
        "compression_arm": "compressed",
        "messages": copy.deepcopy(messages),
        "tools": [copy.deepcopy(EXPAND_TOOL)],
        "task": task["question"],
        "model": None,
        "model_revision": None,
        "gold_answer": task["gold_answer"],
        "support_segment_ids": list(task["support_segment_ids"]),
        "segment_count": len(task["segments"]),
        "tool_call_count": tool_call_count,
        "verification": verification.as_dict(),
        "rollout_failure_reason": failure_reason,
    }
