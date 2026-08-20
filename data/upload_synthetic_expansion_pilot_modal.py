#!/usr/bin/env python3
"""Publish the verified synthetic expansion-agent pilot to Hugging Face."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import modal


APP_NAME = "lclm-upload-synthetic-expansion-pilot"
VOLUME_NAME = "lclm-stage3-data"
SOURCE_DIR = Path(
    "/data/stage3-agent/synthetic-expansion/pilot-long-seg-20260821-v1"
)
HARVEST_DIR = Path(
    "/data/stage3-agent/synthetic-expansion/harvest-pilot-long-seg-20260821-v1"
)
REPORT_PATH = Path(
    "/data/stage3-agent/synthetic-expansion/hf-upload-pilot-long-seg-20260821-v1.json"
)
REPO_ID = "leonli66/stage3-synthetic-expansion-agent"

EXPECTED_ACCEPTED = 3
EXPECTED_REJECTED = 2
EXPECTED_TOOL_CALLS = 11

PROJECT_ROOT = Path("/opt/lclm")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=0.36,<2")
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "_modal_run"],
    )
    .env({"PYTHONPATH": str(PROJECT_ROOT)})
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


DATASET_CARD = """---
pretty_name: Stage 3 Synthetic Expansion Agents (Pilot)
license: apache-2.0
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Stage 3 Synthetic Expansion Agents — Verified Pilot

This is a small inspection pilot for native selective-expansion training. It is
not the final-scale mixture.

Each initial user context contains 27–29 positional segments named `seg_1`,
`seg_2`, and so on. Every segment contains its full source document wrapped in
`<|memory_start|>...<|memory_end|>`. The assistant receives a native Qwen tool
schema and may call `expand({"segment_id": "seg_i"})`; the matching tool result
then places that same full document into the conversation. The Qwen teacher saw
only short routing descriptions while generating calls; those descriptions are
not stored as the train-time memory bodies.

## Pilot results

- Requested tasks: 5, one per task family
- Verified accepted traces: 3
- Native expansion calls: 11 (3–5 per trace)
- Rejected traces: 2 (wrong exact final answers)
- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507`
- Compression scope: initial input segments only
- Per-segment floor: 512 whitespace-delimited words (882–972 Qwen tokens here)
- Raw memory context: 25,627–27,328 Qwen tokens per accepted trace

The accepted split contains one latest-state task, one multi-key lookup, and one
multi-hop join. The audit directory retains all five programmatic tasks and both
rejected traces.

## Training semantics

The `tools` field is passed to Qwen's native chat template, which renders the
tool schema in its system tool block. Training loss applies to every assistant
tool-call turn and the final assistant answer. System, user, and tool-result
tokens are loss-masked. Memory bodies are extracted for the encoder/adapter path.

## Verification

An accepted trace must use only native `expand` calls with a valid `seg_i`,
receive the exact original text mapped to each call, expand every supporting
segment, preserve call/result IDs, contain memory markers only in message text,
and emit the exact requested `FINAL:` response.
"""


def _jsonl_count(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def upload() -> dict[str, object]:
    from huggingface_hub import HfApi

    from data.harvest_synthetic_expansion import harvest_run_directories

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token is missing from the Modal secret")

    accepted_path = SOURCE_DIR / "accepted.jsonl"
    rejected_path = SOURCE_DIR / "rejected.jsonl"
    if _jsonl_count(accepted_path) != EXPECTED_ACCEPTED:
        raise RuntimeError("pilot accepted-row count changed")
    if _jsonl_count(rejected_path) != EXPECTED_REJECTED:
        raise RuntimeError("pilot rejected-row count changed")

    harvested, harvest_report = harvest_run_directories([SOURCE_DIR])
    if len(harvested) != EXPECTED_ACCEPTED:
        raise RuntimeError("audited harvest count changed")
    if harvest_report.get("tool_calls") != EXPECTED_TOOL_CALLS:
        raise RuntimeError("audited tool-call count changed")

    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    harvested_path = HARVEST_DIR / "accepted.jsonl"
    with harvested_path.open("w") as handle:
        for row in harvested:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    harvest_report_path = HARVEST_DIR / "report.json"
    with harvest_report_path.open("w") as handle:
        json.dump(harvest_report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    with tempfile.TemporaryDirectory(prefix="synthetic-expansion-hf-") as temporary:
        upload_root = Path(temporary)
        (upload_root / "data").mkdir(parents=True)
        shutil.copy2(harvested_path, upload_root / "data" / "train.jsonl")
        (upload_root / "metadata").mkdir(parents=True)
        shutil.copy2(
            harvest_report_path,
            upload_root / "metadata" / "harvest-report.json",
        )
        audit = upload_root / "audit" / SOURCE_DIR.name
        audit.mkdir(parents=True)
        for name in (
            "accepted.jsonl",
            "rejected.jsonl",
            "tasks.jsonl",
            "manifest.json",
            "report.json",
        ):
            shutil.copy2(SOURCE_DIR / name, audit / name)
        (upload_root / "README.md").write_text(DATASET_CARD)

        forbidden_local = [
            path.relative_to(upload_root).as_posix()
            for path in upload_root.rglob("state.json")
        ]
        if forbidden_local:
            raise RuntimeError(f"refusing to upload state.json files: {forbidden_local}")

        api = HfApi(token=token)
        api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=upload_root,
            ignore_patterns=["state.json", "**/state.json", "**/.cache/**"],
            delete_patterns=["audit/*"],
            commit_message="Replace pilot with long-document expansion traces",
        )

    expected_files = {
        "README.md",
        "data/train.jsonl",
        "metadata/harvest-report.json",
        *{
            f"audit/{SOURCE_DIR.name}/{name}"
            for name in (
                "accepted.jsonl",
                "rejected.jsonl",
                "tasks.jsonl",
                "manifest.json",
                "report.json",
            )
        },
    }
    remote_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    missing_remote = sorted(expected_files - remote_files)
    forbidden_remote = sorted(
        name for name in remote_files if name == "state.json" or name.endswith("/state.json")
    )
    if missing_remote:
        raise RuntimeError(f"remote dataset is missing files: {missing_remote}")
    if forbidden_remote:
        raise RuntimeError(f"forbidden state.json files were uploaded: {forbidden_remote}")

    dataset_info = api.dataset_info(REPO_ID)
    report = {
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": dataset_info.sha,
        "harvested_rows": EXPECTED_ACCEPTED,
        "tool_calls": EXPECTED_TOOL_CALLS,
        "remote_files": len(remote_files),
        "forbidden_files": forbidden_remote,
        "status": "complete",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = REPORT_PATH.with_suffix(".json.tmp")
    with temporary_report.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_report, REPORT_PATH)
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(upload.remote(), indent=2, sort_keys=True))
