"""Patch model_config.json on the 3 already-uploaded latent-context repos.

Drops: model_type, schema_version, encoder_name, decoder_name, num_adapter_layers.
Leaves the safetensors weights untouched.

Run:
    modal run --detach _modal_run/lclm_patch_configs.py
"""
import modal

app = modal.App("lclm-patch-configs")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub>=0.20")
)

REPOS = [
    "latent-context/0.6b-4b-LCLM-4x",
    "latent-context/0.6b-4b-LCLM-8x",
    "latent-context/0.6b-4b-LCLM-16x",
]


@app.function(image=upload_image, cpu=2, timeout=20*60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def patch_one(repo_id: str):
    import json, tempfile, os
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    print(f">>> {repo_id}", flush=True)
    local = hf_hub_download(repo_id=repo_id, filename="model_config.json", repo_type="model")
    with open(local) as f:
        old = json.load(f)
    print(f"  old: {old}", flush=True)

    KEEP = {"compression_ratio", "encoder_window_size", "pooling",
            "encoder_mask_type", "boundary_overlap", "adapter_type"}
    new = {k: old[k] for k in KEEP if k in old}
    # Migrate legacy chunk_size if present
    if "compression_ratio" not in new and "chunk_size" in old:
        new["compression_ratio"] = old["chunk_size"]
    # Only keep num_adapter_layers if adapter_type != mlp
    if new.get("adapter_type") and new["adapter_type"] != "mlp" and "num_adapter_layers" in old:
        new["num_adapter_layers"] = old["num_adapter_layers"]
    print(f"  new: {new}", flush=True)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(new, tmp, indent=2)
    tmp.close()
    api.upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="model_config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Minimize model_config.json (only loader-required fields)",
    )
    os.unlink(tmp.name)
    print(f"  ✓ patched {repo_id}", flush=True)
    return {"repo_id": repo_id, "new_config": new}


@app.function(image=upload_image, cpu=2, timeout=30*60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def orchestrate():
    results = list(patch_one.map(REPOS))
    print("\n========== FINAL ==========", flush=True)
    for r in results:
        print(f"  ✓ {r['repo_id']}: {r['new_config']}", flush=True)
    return results


@app.local_entrypoint()
def main():
    orchestrate.remote()
