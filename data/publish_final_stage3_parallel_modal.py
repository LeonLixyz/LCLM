"""Resume the authorized final HF release with disjoint CPU upload workers."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import modal
from data.publish_final_stage3_modal import (
    image, volume, agent_volume, REPOS, PUBLICATION, IGNORE,
    manifests, load_staged_release, finalize,
)

app = modal.App("lclm-stage3-final-hf-throttled-20260911")
OPTIONS = dict(image=image, volumes={"/data": volume, "/runs": agent_volume},
               secrets=[modal.Secret.from_name("huggingface")], timeout=86400)
PARTS = {"raw": 4, "packed": 8}


@app.function(**OPTIONS, cpu=4, memory=8192)
def upload_part(kind: str, index: int, count: int):
    from data.hf_upload_client import install
    install()
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    started = time.time()
    stage, entries = load_staged_release(kind)
    assigned = [name for name in sorted(entries) if name.endswith(".parquet")][index::count]
    api = HfApi()
    info = api.repo_info(REPOS[kind], repo_type="dataset")
    if not info.private:
        raise ValueError("Destination must stay private")
    remote = {f.path: f for f in api.list_repo_tree(REPOS[kind], repo_type="dataset",
              recursive=True, revision=info.sha) if isinstance(f, RepoFile)}
    pending = []
    for name in assigned:
        if name in remote:
            f, expected = remote[name], entries[name]
            if f.size != expected["bytes"] or f.lfs is None or f.lfs.sha256 != expected["sha256"]:
                raise ValueError(f"Existing uploaded data differs: {name}")
        else:
            pending.append(name)
    local = Path("/tmp")/f"hf-{kind}-{index}"
    local.mkdir(parents=True, exist_ok=True)
    cache = stage/".cache/huggingface/upload"
    def prepare(name):
        target = local/name
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(entries[name]["source"])
        digest = hashlib.sha256()
        with source.open("rb") as reader, target.open("wb") as writer:
            while chunk := reader.read(8*1024*1024):
                digest.update(chunk)
                writer.write(chunk)
        if target.stat().st_size != entries[name]["bytes"] or digest.hexdigest() != entries[name]["sha256"]:
            raise ValueError(f"Source changed while staging: {name}")
        shutil.copystat(source, target)
        cached = cache/f"{name}.metadata"
        if cached.exists():
            dest = local/".cache/huggingface/upload"/f"{name}.metadata"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare, pending))
    print(json.dumps({"event": "part_start", "kind": kind, "index": index,
                      "assigned": len(assigned), "already_verified": len(assigned)-len(pending),
                      "pending_files": len(pending), "pending_bytes": sum(entries[n]["bytes"] for n in pending)}), flush=True)
    if pending:
        args = Path("/tmp/upload-args.json")
        args.write_text(json.dumps(dict(repo_id=REPOS[kind], repo_type="dataset", private=True,
            folder_path=str(local), allow_patterns=pending, ignore_patterns=IGNORE,
            num_workers=8, print_report=True, print_report_every=60)))
        subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("hf_upload_client.py")),
                        str(args)], check=True)
    # Verify this disjoint partition against the build certificate at its commit.
    info = api.repo_info(REPOS[kind], repo_type="dataset")
    remote = {f.path: f for f in api.list_repo_tree(REPOS[kind], repo_type="dataset",
              recursive=True, revision=info.sha) if isinstance(f, RepoFile)}
    for name in assigned:
        f, expected = remote[name], entries[name]
        if f.size != expected["bytes"] or f.lfs is None or f.lfs.sha256 != expected["sha256"]:
            raise ValueError(f"Upload differs from build: {name}")
    result = dict(status="complete", kind=kind, index=index, files=len(assigned),
                  bytes=sum(entries[n]["bytes"] for n in assigned),
                  elapsed_seconds=round(time.time()-started, 1), revision=info.sha)
    path = PUBLICATION/"parallel-parts"/f"{kind}-{index:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    volume.commit()
    print(json.dumps(result), flush=True)
    return result


@app.function(**OPTIONS, cpu=2, memory=4096)
def coordinate():
    from data.hf_upload_client import install
    install()
    from huggingface_hub import HfApi, hf_hub_download
    started = time.time()
    manifests()
    api = HfApi()
    if api.whoami()["name"] != "leonli66":
        raise ValueError("Unexpected HF account")
    staged = {}
    for kind, repo in REPOS.items():
        stage, entries = staged[kind] = load_staged_release(kind)
        info = api.repo_info(repo, repo_type="dataset")
        if not info.private:
            raise ValueError("Destination must stay private")
        marker = Path(hf_hub_download(repo, "release.json", repo_type="dataset", revision=info.sha))
        if marker.read_bytes() != (stage/"release.json").read_bytes():
            raise ValueError("Destination belongs to another release")
        meta = Path("/tmp")/f"metadata-{kind}"
        for name, record in entries.items():
            if name.endswith(".parquet"):
                continue
            payload = (stage/name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError(f"Changed release metadata: {name}")
            path = meta/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=meta,
                          ignore_patterns=IGNORE, commit_message="Preserve final build provenance and usage")
    calls = {(kind, index): upload_part.spawn(kind, index, count)
             for kind, count in PARTS.items() for index in range(count)}
    print(json.dumps({f"{k[0]}-{k[1]}": c.object_id for k,c in calls.items()}), flush=True)
    results, errors = {}, {}
    for (kind, index), call in calls.items():
        try:
            results[f"{kind}-{index}"] = call.get()
        except Exception as exc:
            errors[f"{kind}-{index}"] = str(exc)
    if errors:
        raise RuntimeError(json.dumps(errors))
    # All partition writers are finished before the final repository-wide check.
    completed = {kind: finalize(api, kind, entries, stage, started)
                 for kind, (stage, entries) in staged.items()}
    (PUBLICATION/"completion.json").write_text(json.dumps(completed, indent=2)+"\n")
    volume.commit()
    return completed


@app.local_entrypoint()
def main():
    print(json.dumps(coordinate.remote(), indent=2))
