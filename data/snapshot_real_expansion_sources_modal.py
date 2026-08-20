#!/usr/bin/env python3
"""Snapshot registered real-document sources onto the shared Modal volume.

Raw repositories are retained for provenance.  Eligible Hugging Face train
partitions are also materialized with ``datasets.save_to_disk`` so conversion
jobs never depend on mutable upstream files or dataset-server availability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lclm-real-expansion-source-snapshot"
PROJECT_ROOT = Path("/opt/lclm")
DATA_ROOT = Path("/data/stage3-agent/real-expansion/sources")
HF_CACHE_ROOT = Path("/hf-cache/real-expansion-sources")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "datasets==3.6.0",
        "huggingface-hub>=0.36,<1",
        "pyarrow>=18,<22",
    )
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "_modal_run"],
    )
    .env({"PYTHONPATH": str(PROJECT_ROOT)})
)

data_volume = modal.Volume.from_name("lclm-stage3-data")
cache_volume = modal.Volume.from_name("lclm-hf-cache")
app = modal.App(APP_NAME)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=6 * 60 * 60,
    max_containers=4,
    volumes={"/data": data_volume, "/hf-cache": cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def snapshot_source(request: dict[str, Any]) -> dict[str, Any]:
    import shutil
    import subprocess
    import time

    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    from data.real_expansion_sources import SOURCE_SPECS

    source_by_key = {spec.key: spec for spec in SOURCE_SPECS}
    key = request["key"]
    spec = source_by_key[key]
    force = bool(request.get("force"))
    materialize = bool(request.get("materialize")) and spec.train_eligible
    destination = DATA_ROOT / key
    manifest_path = destination / "snapshot-manifest.json"
    if manifest_path.is_file() and not force:
        with manifest_path.open() as handle:
            previous = json.load(handle)
        if previous.get("status") == "complete":
            return {"key": key, "status": "skipped_complete"}

    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result: dict[str, Any] = {
        "key": key,
        "source": spec.as_manifest_record(),
        "status": "running",
        "materialized": [],
    }
    _atomic_json(manifest_path, result)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        if spec.acquisition == "huggingface":
            api = HfApi(token=token)
            info = api.dataset_info(spec.source_id)
            repo_dir = destination / "repository"
            snapshot_download(
                repo_id=spec.source_id,
                repo_type="dataset",
                revision=info.sha,
                local_dir=repo_dir,
                token=token,
            )
            result["revision"] = info.sha
            result["repository_files"] = sum(1 for path in repo_dir.rglob("*") if path.is_file())

            if materialize:
                for partition in spec.partitions:
                    dataset = load_dataset(
                        spec.source_id,
                        partition.config,
                        split=partition.split,
                        cache_dir=str(HF_CACHE_ROOT),
                        token=token,
                        trust_remote_code=True,
                    )
                    partition_dir = (
                        destination / "materialized" / partition.config / partition.split
                    )
                    dataset.save_to_disk(partition_dir)
                    result["materialized"].append(
                        {
                            "config": partition.config,
                            "split": partition.split,
                            "rows": len(dataset),
                            "columns": list(dataset.column_names),
                            "path": str(partition_dir),
                        }
                    )
                    data_volume.commit()
                    cache_volume.commit()
        elif spec.acquisition == "github":
            repo_dir = destination / "repository"
            completed = subprocess.run(
                ["git", "clone", "--depth", "1", spec.source_id, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result["revision"] = revision
            result["clone_output"] = completed.stderr[-2000:]
            result["repository_files"] = sum(1 for path in repo_dir.rglob("*") if path.is_file())
        else:
            result["status"] = "manual_acquisition_required"
            result["elapsed_seconds"] = time.time() - started
            _atomic_json(manifest_path, result)
            data_volume.commit()
            return result

        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["elapsed_seconds"] = time.time() - started
    _atomic_json(manifest_path, result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def main(
    source: str = "all",
    materialize: bool = True,
    include_review: bool = False,
    include_large: bool = False,
    force: bool = False,
) -> None:
    from data.real_expansion_sources import SOURCE_SPECS

    selected = []
    requested = {item.strip() for item in source.split(",") if item.strip()}
    for spec in SOURCE_SPECS:
        if source != "all" and spec.key not in requested:
            continue
        if not include_review and spec.status in {
            "license_review",
            "derived_duplicate",
            "identifier_needed",
        }:
            continue
        if not include_large and spec.status == "large_source":
            continue
        selected.append(
            {"key": spec.key, "force": force, "materialize": materialize}
        )
    if source != "all":
        missing = requested - {request["key"] for request in selected}
        if missing:
            raise ValueError(f"unknown or excluded source keys: {sorted(missing)}")
    results = list(
        snapshot_source.map(selected, order_outputs=False, return_exceptions=True)
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
