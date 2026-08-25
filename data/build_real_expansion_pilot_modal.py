#!/usr/bin/env python3
"""Build long ``seg_i`` expansion tasks from pinned real-source train data."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lclm-build-real-expansion-pilot"
PROJECT_ROOT = Path("/opt/lclm")
SOURCE_ROOT = Path("/data/stage3-agent/real-expansion/sources")
OUTPUT_ROOT = Path("/data/stage3-agent/real-expansion/pilots")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets==3.6.0", "pyarrow>=18,<22")
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "_modal_run"],
    )
    .env({"PYTHONPATH": str(PROJECT_ROOT)})
)
data_volume = modal.Volume.from_name("lclm-stage3-data")
app = modal.App(APP_NAME)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _table_text(table: Any) -> str:
    if not isinstance(table, list):
        return ""
    return "\n".join(
        " | ".join(str(cell) for cell in row) if isinstance(row, list) else str(row)
        for row in table
    )


def _load(path: str):
    from datasets import load_from_disk

    return load_from_disk(str(SOURCE_ROOT / path))


def _maud() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dataset = _load("maud/materialized/default/train")
    documents = []
    candidates = []
    for index, row in enumerate(dataset):
        text = str(row.get("text") or "").strip()
        answer = str(row.get("answer") or "").strip()
        question = str(row.get("question") or "").strip()
        doc_id = f"maud-{index}-{row.get('contract_name', 'contract')}"
        if text:
            documents.append(
                {"document_id": doc_id, "title": str(row.get("contract_name", doc_id)), "text": text}
            )
        if text and question and answer and len(answer.split()) <= 16:
            candidates.append(
                {
                    "document_id": doc_id,
                    "source_row_id": str(row.get("id") or index) + f"-{index}",
                    "question": question,
                    "answer": answer,
                    "focus": text[:500],
                }
            )
    return documents, candidates


def _finqa() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dataset = _load("finqa/materialized/default/train")
    documents = []
    candidates = []
    for index, row in enumerate(dataset):
        parts = [
            "\n".join(str(x) for x in row.get("pre_text", [])),
            "FINANCIAL TABLE:\n" + _table_text(row.get("table_ori") or row.get("table")),
            "\n".join(str(x) for x in row.get("post_text", [])),
        ]
        text = "\n\n".join(part for part in parts if part.strip())
        doc_id = f"finqa-{row.get('id') or index}"
        documents.append(
            {"document_id": doc_id, "title": str(row.get("filename", doc_id)), "text": text}
        )
        answer = str(row.get("answer") or "").strip()
        if answer and len(answer.split()) <= 12:
            evidence = ""
            try:
                gold = json.loads(row.get("gold_inds") or "{}")
                evidence = " ".join(str(value) for value in gold.values())
            except (TypeError, json.JSONDecodeError):
                pass
            candidates.append(
                {
                    "document_id": doc_id,
                    "source_row_id": str(row.get("id") or index),
                    "question": str(row.get("question") or ""),
                    "answer": answer,
                    "focus": evidence,
                }
            )
    return documents, candidates


def _pubmedqa() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dataset = _load("pubmedqa_labeled/materialized/pqa_labeled/train")
    documents = []
    candidates = []
    for index, row in enumerate(dataset):
        context = row.get("context") or {}
        contexts = context.get("contexts", []) if isinstance(context, dict) else []
        text = "\n\n".join(str(value) for value in contexts)
        doc_id = f"pubmed-{row.get('pubid') or index}"
        documents.append({"document_id": doc_id, "title": doc_id, "text": text})
        answer = str(row.get("final_decision") or "").strip()
        if answer in {"yes", "no", "maybe"}:
            candidates.append(
                {
                    "document_id": doc_id,
                    "source_row_id": str(row.get("pubid") or index),
                    "question": str(row.get("question") or ""),
                    "answer": answer,
                    "focus": str(row.get("long_answer") or ""),
                }
            )
    return documents, candidates


def _clapnq() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dataset = _load("clapnq/materialized/default/train")
    documents = []
    candidates = []
    for index, row in enumerate(dataset):
        passages = row.get("passages") or []
        text = "\n\n".join(
            f"{passage.get('title', '')}\n{passage.get('text', '')}"
            for passage in passages
            if isinstance(passage, dict)
        )
        doc_id = f"clapnq-{row.get('id') or index}"
        title = str(passages[0].get("title", doc_id)) if passages else doc_id
        documents.append({"document_id": doc_id, "title": title, "text": text})
        outputs = row.get("output") or []
        if not outputs or not isinstance(outputs[0], dict):
            continue
        answer = str(outputs[0].get("answer") or "").strip()
        if not answer or len(answer.split()) > 48:
            continue
        selected = outputs[0].get("selected_sentences") or []
        candidates.append(
            {
                "document_id": doc_id,
                "source_row_id": str(row.get("id") or index),
                "question": str(row.get("input") or ""),
                "answer": answer,
                "focus": str(selected[0]) if selected else answer,
            }
        )
    return documents, candidates


def _contract_nli() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    documents_ds = _load("contract_nli/materialized_raw/documents/train")
    tasks_ds = _load("contract_nli/materialized_raw/tasks/train")
    documents = [
        {
            "document_id": f"contract-nli-{row['document_id']}",
            "title": str(row.get("file_name") or row["document_id"]),
            "text": str(row["text"]),
        }
        for row in documents_ds
    ]
    candidates = []
    for row in tasks_ds:
        evidence = ""
        try:
            evidence_values = json.loads(row.get("evidence_text_json") or "[]")
            evidence = " ".join(str(value) for value in evidence_values)
        except (TypeError, json.JSONDecodeError):
            pass
        candidates.append(
            {
                "document_id": f"contract-nli-{row['document_id']}",
                "source_row_id": str(row["task_id"]),
                "question": (
                    "Does the contract entail, contradict, or not mention this statement: "
                    + str(row["hypothesis"])
                ),
                "answer": str(row["choice"]),
                "focus": evidence,
            }
        )
    return documents, candidates


ADAPTERS = {
    "theatticusproject/maud": ("legal", _maud),
    "bevaya/FinQA": ("finance", _finqa),
    "qiaojin/PubMedQA:pqa_labeled": ("health", _pubmedqa),
    "PrimeQA/clapnq": ("grounded_qa", _clapnq),
    "stanfordnlp/contract-nli": ("legal", _contract_nli),
}


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume},
)
def build_pilot(
    *,
    run_name: str,
    per_source: int,
    segment_count: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    from collections import Counter

    from data.real_expansion_agent import make_real_expansion_task, validate_real_task

    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a simple directory name")
    if per_source <= 0 or segment_count < 2:
        raise ValueError("per_source must be positive and segment_count must be at least two")

    output_dir = OUTPUT_ROOT / run_name
    tasks_path = output_dir / "tasks.jsonl"
    manifest_path = output_dir / "manifest.json"
    if tasks_path.exists() and not force:
        raise FileExistsError(f"pilot already exists: {output_dir}")

    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    source_revisions: dict[str, str | None] = {}
    skipped = Counter()
    for source_dataset, (category, adapter) in ADAPTERS.items():
        documents, candidates = adapter()
        by_id = {document["document_id"]: document for document in documents}
        eligible = [candidate for candidate in candidates if candidate["document_id"] in by_id]
        rng.shuffle(eligible)
        created = 0
        for candidate in eligible:
            if created >= per_source:
                break
            other_documents = [
                document
                for document in documents
                if document["document_id"] != candidate["document_id"]
            ]
            if len(other_documents) < segment_count - 1:
                skipped[f"{source_dataset}:small_pool"] += 1
                break
            distractors = rng.sample(other_documents, segment_count - 1)
            try:
                task = make_real_expansion_task(
                    source_dataset=source_dataset,
                    source_row_id=candidate["source_row_id"],
                    question=candidate["question"],
                    gold_answer=candidate["answer"],
                    support_document=by_id[candidate["document_id"]],
                    distractor_documents=distractors,
                    companion_documents=other_documents,
                    source_category=category,
                    seed=seed,
                    support_focus=candidate["focus"],
                )
                validate_real_task(task)
            except (TypeError, ValueError) as exc:
                skipped[f"{source_dataset}:{type(exc).__name__}"] += 1
                continue
            tasks.append(task)
            created += 1
        if created < per_source:
            raise RuntimeError(f"only built {created}/{per_source} tasks for {source_dataset}")

        source_key = {
            "theatticusproject/maud": "maud",
            "bevaya/FinQA": "finqa",
            "qiaojin/PubMedQA:pqa_labeled": "pubmedqa_labeled",
            "PrimeQA/clapnq": "clapnq",
            "stanfordnlp/contract-nli": "contract_nli",
        }[source_dataset]
        snapshot_manifest = SOURCE_ROOT / source_key / "snapshot-manifest.json"
        revision = None
        if snapshot_manifest.exists():
            with snapshot_manifest.open() as handle:
                revision = json.load(handle).get("revision")
        source_revisions[source_dataset] = revision

    rng.shuffle(tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(tasks_path, tasks)
    word_counts = [len(segment["text"].split()) for task in tasks for segment in task["segments"]]
    manifest = {
        "status": "complete",
        "run_name": run_name,
        "tasks": len(tasks),
        "per_source": per_source,
        "segment_count": segment_count,
        "segments": sum(len(task["segments"]) for task in tasks),
        "minimum_segment_words": min(word_counts),
        "maximum_segment_words": max(word_counts),
        "total_source_words": sum(word_counts),
        "seed": seed,
        "source_revisions": source_revisions,
        "tasks_by_source": dict(Counter(task["source_dataset"] for task in tasks)),
        "skipped": dict(skipped),
        "output_dir": str(output_dir),
    }
    _write_json(manifest_path, manifest)
    data_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    run_name: str = "real-pilot-v1",
    per_source: int = 4,
    segment_count: int = 8,
    seed: int = 20260825,
    force: bool = False,
) -> None:
    result = build_pilot.remote(
        run_name=run_name,
        per_source=per_source,
        segment_count=segment_count,
        seed=seed,
        force=force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
