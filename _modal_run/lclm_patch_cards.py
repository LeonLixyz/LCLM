"""Patch the README.md (model card) on the 3 latent-context HF repos.

Applies the agreed wording:
  - "trained an encoder initialized from … and a decoder initialized from …"
  - "in place of the original context" (was "raw tokens")
  - "latent tokens" (was "latent vectors")
  - Adds memory_start/memory_end paragraph explaining the wrapping
  - Drops the Layout section
  - Drops num_adapter_layers from the config table
  - Adds "(finetuned)" tag on the base-model rows

Run:
    modal run --detach _modal_run/lclm_patch_cards.py
"""
import modal

app = modal.App("lclm-patch-cards")

img = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub>=0.20")
)

REPOS = [
    ("latent-context/0.6b-4b-LCLM-4x",  4),
    ("latent-context/0.6b-4b-LCLM-8x",  8),
    ("latent-context/0.6b-4b-LCLM-16x", 16),
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

Latent Context Language Model: an encoder–decoder compressor described in
*End-to-End Context Compression at Scale*. We trained an encoder initialized
from Qwen3-Embedding-0.6B that compresses input tokens at **{ratio}×** into
latent tokens, and a decoder initialized from Qwen3-4B-Instruct-2507 that
consumes those latents in place of the original context.

Text the model should compress goes between `<|memory_start|>` and
`<|memory_end|>`; the encoder turns the wrapped span into latent tokens that
the decoder treats just like normal input embeddings.

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
| encoder | Qwen/Qwen3-Embedding-0.6B (finetuned) |
| decoder | Qwen/Qwen3-4B-Instruct-2507 (finetuned) |
| compression_ratio | {ratio} |
| encoder_window_size | 1024 |
| pooling | mean |
| encoder_mask_type | causal |
| boundary_overlap | 0 |
| adapter_type | mlp |

Code: https://github.com/LeonLixyz/LCLM
"""


@app.function(image=img, cpu=2, timeout=20*60,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def patch_one(repo_id: str, ratio: int):
    import os, tempfile
    from huggingface_hub import HfApi
    text = CARD.format(repo=repo_id, ratio=ratio)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    tmp.write(text); tmp.close()
    HfApi().upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card wording",
    )
    os.unlink(tmp.name)
    print(f"  ✓ {repo_id}", flush=True)
    return repo_id


@app.local_entrypoint()
def main():
    for r, k in REPOS:
        patch_one.remote(r, k)
