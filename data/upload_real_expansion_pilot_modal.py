#!/usr/bin/env python3
"""Publish the audited real-source expansion-agent pilot to Hugging Face."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import modal


APP_NAME = "lclm-upload-real-expansion-pilot"
VOLUME_NAME = "lclm-stage3-data"
RUN_DIR = Path("/data/stage3-agent/real-expansion/pilots/real-pilot-20260825-v1")
HARVEST_DIR = RUN_DIR / "harvest-v1"
REPO_ID = "leonli66/stage3-real-expansion-agent"
EXPECTED_ACCEPTED = 10
EXPECTED_REJECTED = 10

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub>=0.36,<2"
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


DATASET_CARD = """---
pretty_name: Stage 3 Real-Source Expansion Agents (Pilot)
task_categories:
- question-answering
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Stage 3 Real-Source Expansion Agents — Pilot

This inspection pilot converts pinned training examples from real legal,
financial, biomedical, and grounded-QA corpora into native selective-expansion
traces. It is not the final-scale mixture.

Each row contains eight positional `seg_i` blocks. Every initial segment holds
512–896 words of real source material wrapped in
`<|memory_start|>...<|memory_end|>`. Qwen3-235B-A22B-Instruct-2507 receives a
native `expand({"segment_id": "seg_i"})` tool, expands a source segment, and
answers using the returned plaintext.

## Results

- Source tasks: 20 (four from each source)
- Accepted traces: 10
- Rejected traces retained in `audit/rejected.jsonl`: 10
- Segments in the source task set: 160
- Total real-source words: 116,768
- Accepted sources: ContractNLI 4, FinQA 2, PubMedQA 2, CLAPNQ 2
- Native expansion calls per accepted trace: 1
- Model revision: `ac9c66cc9b46af7306746a9250f23d47083d689e`

## Verification

All accepted traces must use a valid native tool call, receive the exact source
text for that `seg_i`, and expand every annotated support segment. Answers use
the source's native metric: ContractNLI canonical labels, FinQA numeric
equivalence, PubMedQA decision labels, and CLAPNQ grounded containment/token-F1.
Strictly wrong, missing-final, and tool-limit traces are excluded from train.

## Provenance

`metadata/source-manifest.json` pins all upstream dataset revisions. Licensing
continues to follow each upstream source; users should review those terms before
redistribution or commercial use.
"""


def _count(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def upload() -> dict[str, object]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token is missing")
    accepted = HARVEST_DIR / "accepted.jsonl"
    rejected = HARVEST_DIR / "rejected.jsonl"
    if _count(accepted) != EXPECTED_ACCEPTED or _count(rejected) != EXPECTED_REJECTED:
        raise RuntimeError("audited pilot row count changed")
    with tempfile.TemporaryDirectory(prefix="real-expansion-hf-") as temporary:
        root = Path(temporary)
        (root / "data").mkdir()
        (root / "audit").mkdir()
        (root / "metadata").mkdir()
        shutil.copy2(accepted, root / "data" / "train.jsonl")
        shutil.copy2(rejected, root / "audit" / "rejected.jsonl")
        shutil.copy2(HARVEST_DIR / "report.json", root / "metadata" / "harvest-report.json")
        shutil.copy2(RUN_DIR / "manifest.json", root / "metadata" / "source-manifest.json")
        shutil.copy2(
            RUN_DIR / "generation-manifest.json",
            root / "metadata" / "generation-manifest.json",
        )
        shutil.copy2(RUN_DIR / "tasks.jsonl", root / "audit" / "source-tasks.jsonl")
        (root / "README.md").write_text(DATASET_CARD)
        if any(root.rglob("state.json")):
            raise RuntimeError("refusing to upload state.json")
        api = HfApi(token=token)
        api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=root,
            ignore_patterns=["state.json"],
            commit_message="Publish audited real-source expansion-agent pilot",
        )
    return {
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/datasets/{REPO_ID}",
        "accepted": EXPECTED_ACCEPTED,
        "rejected": EXPECTED_REJECTED,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(upload.remote(), ensure_ascii=False, indent=2, sort_keys=True))
