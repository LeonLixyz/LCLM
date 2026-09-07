#!/usr/bin/env python3
"""Publish the validated Stage-3 50/50 CoT mixture from a Modal Volume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "lclm-upload-stage3-cot50"
VOLUME_NAME = "lclm-stage3-data"
DATASET_DIR = Path("/data/stage3-final-mixture-cot50-v1")
REPORT_PATH = Path("/data/stage3-cleaning/cot50-rewrite-v1/upload-report.json")
REPO_ID = "leonli66/stage3-final-mixture-cot50"
EXPECTED_SHARDS = 2033

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub>=0.36,<2"
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


DATASET_CARD = """---
pretty_name: Stage 3 Final Mixture — 50% CoT Compression
task_categories:
- text-generation
---

# Stage 3 Final Mixture — 50% CoT Compression

This is a deterministic capability-preserving rewrite of
`leonli66/stage3-final-mixture` for LCLM Stage-3 post-training.

This transformation does not relicense the original mixture. Its constituent
sources retain their own terms; no blanket Apache-2.0 data license is asserted.

Only the `reasoning_data` and `dolci_think` subsets change. Their
`compression_prompt` is the ordinary `prompt`. A deterministic 50% arm keeps
the complete assistant target as ordinary SFT; the other arm wraps the inferred
reasoning prefix in `<|memory_start|>...<|memory_end|>` while keeping the final
answer trainable. All non-reasoning rows are unchanged.

## Counts

- Total rows: 20,326,114
- Reasoning rows: 5,407,421
- Assigned CoT compression: 2,702,448 (49.98%)
- Assigned ordinary uncompressed SFT: 2,704,973 (50.02%)
- Effective CoT-compressed rows: 2,698,545
- Compression-assigned rows too short to split: 3,903

## Boundary rules

Priority is: explicit analysis tags; final-answer markers; a final boxed answer;
an answer line; the final balanced fenced block; a GSM-style final `####` line;
then a fallback that keeps the final 128
`Qwen/Qwen3-4B-Instruct-2507` tokens uncompressed.

The transformation is text-preserving: deleting the two LCLM memory tags from
every rewritten target reconstructs the original target exactly. All 2,033
Parquet shards were validated against the source.
"""


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def upload() -> dict[str, object]:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token is missing from the Modal secret")
    api = HfApi(token=token)
    local_shards = sorted(DATASET_DIR.glob("*.parquet"))
    if len(local_shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"expected {EXPECTED_SHARDS} Parquet shards, found {len(local_shards)}"
        )

    api.create_repo(
        REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    expected_names = {path.name for path in local_shards}
    remote_files_before = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    unexpected_data = {
        name
        for name in remote_files_before
        if name.endswith(".parquet") and name not in expected_names
    }
    if unexpected_data:
        raise RuntimeError(
            f"refusing to mix with unexpected existing Parquet files: "
            f"{sorted(unexpected_data)[:10]}"
        )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=DATASET_DIR,
        allow_patterns="*.parquet",
        ignore_patterns=["state.json", "*.stats.json", "**/.cache/**"],
        num_workers=16,
        print_report=True,
        print_report_every=60,
    )
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="README.md",
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        commit_message="Document the validated 50/50 CoT mixture",
    )

    remote_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    remote_shards = {name for name in remote_files if name.endswith(".parquet")}
    if remote_shards != expected_names:
        raise RuntimeError(
            f"remote shard mismatch: expected {len(expected_names)}, "
            f"found {len(remote_shards)}"
        )
    forbidden = {
        name
        for name in remote_files
        if name.endswith("state.json") or name.endswith(".stats.json")
    }
    if forbidden:
        raise RuntimeError(f"forbidden internal files were uploaded: {sorted(forbidden)}")

    report = {
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/datasets/{REPO_ID}",
        "parquet_shards": len(remote_shards),
        "remote_files": len(remote_files),
        "forbidden_files": [],
        "status": "complete",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, REPORT_PATH)
    volume.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(upload.remote(), indent=2, sort_keys=True))

