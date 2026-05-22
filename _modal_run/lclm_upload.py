"""Upload the 3 migrated LCLM checkpoints to huggingface.co/latent-context.

- cs4 -> latent-context/0.6b-4b-LCLM-4x
- cs8 -> latent-context/0.6b-4b-LCLM-8x
- cs16 -> latent-context/0.6b-4b-LCLM-16x

Fixes model_config.json on the way out: chunk_size -> compression_ratio
(the recent code rename).

Run:
    modal run --detach _modal_run/lclm_upload.py
"""
from __future__ import annotations
import modal, pathlib

app = modal.App("lclm-upload")
lclm_vol = modal.Volume.from_name("lclm-test")

upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub>=0.20", "safetensors")
)

CASES = [
    ("cs4-causal-mlp-3e5",  "latent-context/0.6b-4b-LCLM-4x",  4),
    ("cs8-causal-mlp-3e5",  "latent-context/0.6b-4b-LCLM-8x",  8),
    ("cs16-causal-mlp-3e5", "latent-context/0.6b-4b-LCLM-16x", 16),
]

CARD = """\
---
license: apache-2.0
library_name: lclm
base_model:
  - Qwen/Qwen3-Embedding-0.6B
  - Qwen/Qwen3-4B-Instruct-2507
tags:
  - latent-context-language-model
  - context-compression
  - long-context
---

# 0.6b-4b LCLM, {ratio}× compression

Latent Context Language Model: encoder–decoder soft-token compressor described
in *End-to-End Context Compression at Scale*. A Qwen3-Embedding-0.6B encoder
compresses input tokens at **{ratio}×** into latent vectors that a
Qwen3-4B-Instruct-2507 decoder consumes in place of the raw tokens.

## Quick load

```python
from latent_context import LCLM

model = LCLM.from_pretrained("{repo}")

prompt = (
    "<|memory_start|>"
    "<long document, code, or text to compress>"
    "<|memory_end|> "
    "Summarize the document above."
)
# model.generate(...) — see latent_context/inference/hf.py
```

## vLLM serving

```python
from inference.vllm import LCLMVLLMDecoder

runner = LCLMVLLMDecoder("{repo}", tensor_parallel_size=2)
outputs = runner.generate(
    prompts=[[{{"role": "user", "content": "<|memory_start|>...<|memory_end|> Summarize."}}]],
    max_tokens=512, temperature=0.0,
)
```

## Configuration

| field | value |
|---|---|
| encoder | Qwen/Qwen3-Embedding-0.6B |
| decoder | Qwen/Qwen3-4B-Instruct-2507 |
| compression_ratio | {ratio} |
| encoder_window_size | 1024 |
| pooling | mean |
| encoder_mask_type | causal |
| boundary_overlap | 0 |
| adapter_type | mlp |
| num_adapter_layers | 1 |

## Layout

```
decoder/  # Qwen3-4B-Instruct-2507 weights (post-train, finetuned)
encoder/  # Qwen3-Embedding-0.6B weights (post-train, finetuned)
adapter/  # MLP that projects encoder dim -> decoder dim
model_config.json
processor_config.json
```

Code: https://github.com/LeonLixyz/Code-LLaVA
"""


@app.function(image=upload_image, cpu=4, timeout=60*60, volumes={"/lclm": lclm_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def upload_one(src_name: str, repo_id: str, ratio: int):
    import os, json, shutil, tempfile
    from huggingface_hub import HfApi, create_repo

    src = f"/lclm/migrated/{src_name}"
    assert os.path.isdir(src), f"{src} missing"
    print(f">>> {src_name} -> {repo_id} ({ratio}x)", flush=True)

    # Patch the config in a temp staging dir to avoid mutating the source on volume.
    stage = tempfile.mkdtemp(prefix="lclm_upload_")
    print(f"  staging in {stage}", flush=True)
    for entry in os.listdir(src):
        s, d = os.path.join(src, entry), os.path.join(stage, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

    # Rewrite model_config.json: chunk_size -> compression_ratio
    cfg_path = os.path.join(stage, "model_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "chunk_size" in cfg and "compression_ratio" not in cfg:
        cfg["compression_ratio"] = cfg.pop("chunk_size")
        print(f"  migrated chunk_size -> compression_ratio in model_config.json", flush=True)
    print(f"  cfg: {cfg}", flush=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    # Same for processor_config.json if it has chunk_size
    pc = os.path.join(stage, "processor_config.json")
    if os.path.isfile(pc):
        with open(pc) as f:
            pcfg = json.load(f)
        if "chunk_size" in pcfg and "compression_ratio" not in pcfg:
            pcfg["compression_ratio"] = pcfg.pop("chunk_size")
        with open(pc, "w") as f:
            json.dump(pcfg, f, indent=2)

    # Write README.md (model card)
    with open(os.path.join(stage, "README.md"), "w") as f:
        f.write(CARD.format(repo=repo_id, ratio=ratio))

    # Upload
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"  uploading {stage} -> {repo_id} ...", flush=True)
    api.upload_folder(
        folder_path=stage,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["state.json"],
        commit_message=f"Upload 0.6b-4b LCLM {ratio}× compression checkpoint",
    )
    print(f"  ✓ done: https://huggingface.co/{repo_id}", flush=True)
    return {"name": src_name, "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}


@app.function(image=upload_image, cpu=2, timeout=2*60*60, volumes={"/lclm": lclm_vol},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def orchestrate():
    print(f"Uploading {len(CASES)} checkpoints in parallel...", flush=True)
    results = list(upload_one.starmap(CASES))
    print("\n========== FINAL ==========", flush=True)
    for r in results:
        print(f"  ✓ {r['name']} -> {r['url']}", flush=True)
    return results


@app.local_entrypoint()
def main():
    print("Dispatching orchestrate()...", flush=True)
    orchestrate.remote()
