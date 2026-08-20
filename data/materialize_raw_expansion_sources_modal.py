#!/usr/bin/env python3
"""Materialize source repositories whose upstream dataset loaders are broken.

The raw snapshot remains the source of truth.  This job produces stable Arrow
tables with JSON-encoded variable structures, and deliberately reads only
training annotations.  Long documents are stored once and joined to task rows
by document ID so conversion does not duplicate the source corpus.
"""

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Iterable

import modal


APP_NAME = "lclm-real-expansion-raw-materialize"
PROJECT_ROOT = Path("/opt/lclm")
DATA_ROOT = Path("/data/stage3-agent/real-expansion/sources")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets==3.6.0", "pyarrow>=18,<22")
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "_modal_run"],
    )
)
data_volume = modal.Volume.from_name("lclm-stage3-data")
app = modal.App(APP_NAME)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _save(rows: Iterable[dict[str, Any]], path: Path) -> dict[str, Any]:
    from datasets import Dataset

    values = list(rows)
    if not values:
        raise ValueError(f"refusing to save empty materialization at {path}")
    dataset = Dataset.from_list(values)
    dataset.save_to_disk(path)
    return {"path": str(path), "rows": len(dataset), "columns": dataset.column_names}


def _tatqa(source: Path, output: Path) -> list[dict[str, Any]]:
    with (source / "tatqa_train.json").open() as handle:
        records = json.load(handle)

    documents: dict[str, dict[str, Any]] = {}
    tasks = []
    for index, record in enumerate(records):
        doc_id = str(record.get("id") or f"tatqa-{index}")
        documents.setdefault(
            doc_id,
            {
                "document_id": doc_id,
                "pre_text_json": _json(record.get("pre_text", [])),
                "post_text_json": _json(record.get("post_text", [])),
                "table_json": _json(record.get("table", [])),
                "table_original_json": _json(record.get("table_ori", {})),
            },
        )
        qa = record.get("qa") or {}
        tasks.append(
            {
                "task_id": str(qa.get("uid") or f"{doc_id}-{index}"),
                "document_id": doc_id,
                "question": str(qa.get("question", "")),
                "answer_json": _json(qa.get("answer", qa.get("exe_ans", []))),
                "answer_type": str(qa.get("answer_type", "")),
                "derivation": str(qa.get("derivation", "")),
                "scale": str(qa.get("scale", "")),
                "program": str(qa.get("program", "")),
                "gold_evidence_json": _json(qa.get("gold_inds", qa.get("rel_paragraphs", []))),
            }
        )
    return [
        _save(documents.values(), output / "documents" / "train"),
        _save(tasks, output / "tasks" / "train"),
    ]


def _cuad(source: Path, output: Path) -> list[dict[str, Any]]:
    with (source / "CUAD_v1" / "CUAD_v1.json").open() as handle:
        payload = json.load(handle)

    documents = []
    tasks = []
    for contract_index, contract in enumerate(payload["data"]):
        title = str(contract.get("title") or f"contract-{contract_index}")
        for paragraph_index, paragraph in enumerate(contract.get("paragraphs", [])):
            doc_id = f"{title}::{paragraph_index}"
            documents.append(
                {"document_id": doc_id, "title": title, "text": paragraph["context"]}
            )
            for qa in paragraph.get("qas", []):
                tasks.append(
                    {
                        "task_id": str(qa["id"]),
                        "document_id": doc_id,
                        "question": str(qa["question"]),
                        "answers_json": _json(qa.get("answers", [])),
                        "is_impossible": bool(qa.get("is_impossible", False)),
                    }
                )
    return [
        _save(documents, output / "documents" / "train"),
        _save(tasks, output / "tasks" / "train"),
    ]


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe zip member: {member.filename}")
        handle.extractall(destination)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _acord(source: Path, output: Path) -> list[dict[str, Any]]:
    extracted = output.parent / "extracted"
    _safe_extract(source / "ACORD Dataset & ReadMe.zip", extracted)
    roots = list(extracted.rglob("corpus.jsonl"))
    if len(roots) != 1:
        raise ValueError(f"expected one ACORD corpus, found {len(roots)}")
    root = roots[0].parent
    corpus = _jsonl(root / "corpus.jsonl")
    queries = {
        str(row["_id"]): row for row in _jsonl(root / "queries.jsonl")
    }
    tasks = []
    with (root / "qrels" / "train.tsv").open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            query_id = str(row["query-id"])
            query = queries.get(query_id, {})
            tasks.append(
                {
                    "task_id": f"acord-train-{index}",
                    "query_id": query_id,
                    "query": str(query.get("text", query_id)),
                    "query_metadata_json": _json(query.get("metadata", {})),
                    "document_id": str(row["corpus-id"]),
                    "relevance": int(row["score"]),
                }
            )
    documents = [
        {
            "document_id": str(row["_id"]),
            "text": str(row.get("text", "")),
            "metadata_json": _json(row.get("metadata", {})),
        }
        for row in corpus
    ]
    return [
        _save(documents, output / "documents" / "train"),
        _save(tasks, output / "tasks" / "train"),
    ]


def _contract_nli(source: Path, output: Path) -> list[dict[str, Any]]:
    extracted = output.parent / "extracted"
    _safe_extract(source / "resources" / "contract-nli.zip", extracted)
    train_files = list(extracted.rglob("train.json"))
    if len(train_files) != 1:
        raise ValueError(f"expected one ContractNLI train file, found {len(train_files)}")
    with train_files[0].open() as handle:
        payload = json.load(handle)
    labels = payload["labels"]
    documents = []
    tasks = []
    for document in payload["documents"]:
        doc_id = str(document["id"])
        text = str(document["text"])
        spans = document["spans"]
        documents.append(
            {
                "document_id": doc_id,
                "file_name": str(document.get("file_name", "")),
                "text": text,
                "document_type": str(document.get("document_type", "")),
                "source_url": str(document.get("url", "")),
                "spans_json": _json(spans),
            }
        )
        annotations = document["annotation_sets"][0]["annotations"]
        for label_id, annotation in annotations.items():
            evidence_indices = annotation.get("spans", [])
            evidence = [text[spans[index][0] : spans[index][1]] for index in evidence_indices]
            label = labels[label_id]
            tasks.append(
                {
                    "task_id": f"{doc_id}::{label_id}",
                    "document_id": doc_id,
                    "label_id": label_id,
                    "hypothesis": str(label["hypothesis"]),
                    "short_description": str(label.get("short_description", "")),
                    "choice": str(annotation["choice"]),
                    "evidence_span_indices_json": _json(evidence_indices),
                    "evidence_text_json": _json(evidence),
                }
            )
    return [
        _save(documents, output / "documents" / "train"),
        _save(tasks, output / "tasks" / "train"),
    ]


def _convfinqa(source: Path, output: Path) -> list[dict[str, Any]]:
    extracted = output.parent / "extracted"
    _safe_extract(source / "data.zip", extracted)
    train_files = list(extracted.rglob("train.json"))
    if len(train_files) != 1:
        raise ValueError(f"expected one ConvFinQA train file, found {len(train_files)}")
    with train_files[0].open() as handle:
        records = json.load(handle)
    documents = []
    tasks = []
    for index, record in enumerate(records):
        doc_id = str(record.get("id") or f"convfinqa-{index}")
        documents.append(
            {
                "document_id": doc_id,
                "filename": str(record.get("filename", "")),
                "pre_text_json": _json(record.get("pre_text", [])),
                "post_text_json": _json(record.get("post_text", [])),
                "table_json": _json(record.get("table", [])),
                "table_original_json": _json(record.get("table_ori", [])),
            }
        )
        annotation = record.get("annotation") or {}
        qa = record.get("qa") or {}
        tasks.append(
            {
                "task_id": doc_id,
                "document_id": doc_id,
                "question": str(qa.get("question", "")),
                "answer": str(qa.get("answer", "")),
                "dialogue_json": _json(annotation.get("dialogue_break", [])),
                "answers_json": _json(annotation.get("answer_list", [])),
                "turn_programs_json": _json(annotation.get("turn_program", [])),
                "gold_evidence_json": _json(qa.get("gold_inds", {})),
            }
        )
    return [
        _save(documents, output / "documents" / "train"),
        _save(tasks, output / "tasks" / "train"),
    ]


MATERIALIZERS = {
    "tatqa": _tatqa,
    "cuad": _cuad,
    "acord": _acord,
    "contract_nli": _contract_nli,
    "convfinqa": _convfinqa,
}


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3 * 60 * 60,
    max_containers=3,
    volumes={"/data": data_volume},
)
def materialize_source(key: str, force: bool = False) -> dict[str, Any]:
    import shutil
    import time

    started = time.time()
    root = DATA_ROOT / key
    source = root / "repository"
    output = root / "materialized_raw"
    manifest_path = root / "raw-materialization-manifest.json"
    if manifest_path.is_file() and not force:
        with manifest_path.open() as handle:
            previous = json.load(handle)
        if previous.get("status") == "complete":
            return {"key": key, "status": "skipped_complete"}
    if force and output.exists():
        shutil.rmtree(output)
    result: dict[str, Any] = {"key": key, "status": "running", "tables": []}
    try:
        result["tables"] = MATERIALIZERS[key](source, output)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    result["elapsed_seconds"] = time.time() - started
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    data_volume.commit()
    return result


@app.local_entrypoint()
def main(source: str = "all", force: bool = False) -> None:
    keys = list(MATERIALIZERS) if source == "all" else [x.strip() for x in source.split(",")]
    unknown = set(keys) - set(MATERIALIZERS)
    if unknown:
        raise ValueError(f"unknown raw materializers: {sorted(unknown)}")
    results = list(materialize_source.map(keys, kwargs={"force": force}, order_outputs=False))
    print(json.dumps(results, ensure_ascii=False, indent=2))
