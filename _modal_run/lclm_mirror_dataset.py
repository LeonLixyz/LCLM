"""Mirror tonychenxyz/ruler-full -> latent-context/ruler-full.

One-shot Modal job. Downloads the dataset snapshot and re-uploads under the
latent-context org so the eval scripts no longer reference tonychenxyz.
"""
import modal

app = modal.App("lclm-mirror-ruler")

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub>=0.20")
)


@app.function(image=img, cpu=4, timeout=30*60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def mirror():
    import os
    from huggingface_hub import HfApi, snapshot_download, create_repo

    src = "tonychenxyz/ruler-full"
    dst = "latent-context/ruler-full"

    print(f">>> Downloading {src} ...", flush=True)
    local = snapshot_download(repo_id=src, repo_type="dataset")
    print(f"  local: {local}", flush=True)
    print(f"  contents: {sorted(os.listdir(local))[:20]}", flush=True)

    print(f">>> Creating + uploading to {dst} ...", flush=True)
    create_repo(dst, repo_type="dataset", exist_ok=True, private=False)
    HfApi().upload_folder(
        folder_path=local,
        repo_id=dst,
        repo_type="dataset",
        ignore_patterns=["state.json", ".cache/*"],
        commit_message=f"Mirror of {src}",
    )
    print(f"  ✓ done: https://huggingface.co/datasets/{dst}", flush=True)


@app.local_entrypoint()
def main():
    mirror.remote()
