"""Inert, gold-independent LexGLUE prompt correction for a new task version.

No source fetching, generation, verifier changes, or writes occur here. Call this
explicitly when freezing a new task snapshot. Existing source normalization can
run before or after this transform: both prompt builders use the same question.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

VERSION = "lexglue-upstream-ontology-20260909-v1"
DEFINITIONS_SHA256 = "9fd5814221ddc80c087a6b6a07f78a0397bad0e662a2ef8703f11034d66f6c3e"
DEFINITIONS_PATH = Path(__file__).with_name("lexglue_task_definitions.v1.json")
CONFIGS = ("case_hold", "ecthr_a", "ecthr_b", "eurlex", "ledgar", "scotus", "unfair_tos")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_definitions(path: str | Path = DEFINITIONS_PATH) -> dict[str, Any]:
    """Load only the exact reviewed ontology bytes; no implicit latest version."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEFINITIONS_SHA256:
        raise ValueError("LexGLUE definition hash mismatch")
    data = json.loads(raw)
    if data["version"] != VERSION or set(data["configs"]) != set(CONFIGS):
        raise ValueError("LexGLUE definition version/config mismatch")
    return data


def config_from_source_row_id(source_row_id: str) -> str:
    if not isinstance(source_row_id, str):
        raise ValueError("Missing LexGLUE source_row_id")
    match = re.fullmatch(r"([a-z_]+)-(0|[1-9][0-9]*)", source_row_id)
    if not match or match[1] not in CONFIGS:
        raise ValueError("Unrecognized LexGLUE source_row_id")
    return match[1]


def corrected_question(config: str, question: str, *, definitions_path: str | Path = DEFINITIONS_PATH) -> str:
    """Change task-level instructions using only config and the original question.

    The frozen old label list must match in full. This refuses to silently fix
    previously edited prompts or alternative ontologies. Gold/evidence is never
    an argument to this function.
    """
    definitions = load_definitions(definitions_path)
    if config not in CONFIGS or not isinstance(question, str):
        raise ValueError("Unknown config or malformed question")
    if config == "case_hold":
        return question
    spec = definitions["configs"][config]
    labels = [item["label"] for item in spec["labels"]]
    old = (f"Classify this document for {config}. Choose labels from: "
           + ", ".join(labels)
           + ". Return labels in the listed order, separated by |; use none when no label applies.")
    if question.count(old) != 1:
        raise ValueError("Original LexGLUE question/label ontology mismatch")
    new = spec["objective"]
    if spec.get("criterion"):
        new += "\n" + spec["criterion"]
    new += "\nAllowed output labels and meanings, in output order:\n"
    new += "\n".join(f"{item['label']}: {item['meaning']}" for item in spec["labels"])
    if spec["cardinality"] == "single":
        new += "\nReturn exactly one label from the list. Do not return multiple labels or none."
    else:
        new += "\nReturn only the applicable labels in the listed order, separated by |; use none when no label applies."
    new += " Return label codes/names exactly as listed, without the explanatory meanings."
    return question.replace(old, new, 1)


def amend_lexglue_task(task: Mapping[str, Any], *, definitions_path: str | Path = DEFINITIONS_PATH) -> dict[str, Any]:
    """Return a copy with matching teacher/training prompts and explicit provenance.

    All gold fields, evidence, tools, IDs and attempt provenance are copied
    unchanged and never used to derive the correction. CaseHOLD is unchanged.
    Reapplying this version is idempotent and checks stored prompt consistency.
    """
    from data.synthetic_expansion_agent import format_rollout_user_prompt, format_training_user_prompt

    if task.get("family") != "lex_glue" or task.get("source_dataset") != "coastalcph/lex_glue":
        raise ValueError("Not a recognized LexGLUE task")
    config = config_from_source_row_id(task.get("source_row_id"))
    load_definitions(definitions_path)
    if config == "case_hold":
        return copy.deepcopy(dict(task))
    previous = task.get("task_definition_correction")
    if previous:
        if previous.get("version") != VERSION or previous.get("definitions_sha256") != DEFINITIONS_SHA256:
            raise ValueError("Different task-definition correction already applied")
        original = previous["original_question"]
        if previous.get("original_question_sha256") != _sha(original):
            raise ValueError("Original question provenance mismatch")
        question = corrected_question(config, original, definitions_path=definitions_path)
        if task["question"] != question or task.get("raw_question") != question:
            raise ValueError("Corrected question changed")
    else:
        original = task["question"]
        if task.get("raw_question", original) != original:
            raise ValueError("Original raw_question differs from question")
        question = corrected_question(config, original, definitions_path=definitions_path)
    training = format_training_user_prompt(task["segments"], question)
    rollout = format_rollout_user_prompt(task["segments"], question)
    if previous:
        for key, expected in (("training_user_prompt", training), ("user_prompt", training), ("rollout_user_prompt", rollout)):
            if task.get(key) != expected:
                raise ValueError("Corrected stored prompt changed")
    out = copy.deepcopy(dict(task))
    out.update(question=question, raw_question=question, training_user_prompt=training,
               user_prompt=training, rollout_user_prompt=rollout)
    out["task_definition_correction"] = {
        "version": VERSION, "config": config, "definitions_sha256": DEFINITIONS_SHA256,
        "original_question": original, "original_question_sha256": _sha(original),
        "corrected_question_sha256": _sha(question), "gold_fields_used": False,
        "task_identity_preserved": True, "requires_new_frozen_task_snapshot": True,
    }
    return out
