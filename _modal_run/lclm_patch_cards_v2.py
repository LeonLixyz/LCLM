"""Patch HF model cards (latent-context/0.6b-4b-LCLM-{4,8,16}x):

- Shorten the memory-wrapping paragraph.
- Drop the "We trained an encoder initialized from ..." sentence.
- Drop the "(finetuned)" suffix on encoder/decoder rows in the config table.
- Add an explicit note that running these checkpoints requires the
  LCLM codebase (https://github.com/LeonLixyz/LCLM).

Run locally:
    python _modal_run/lclm_patch_cards_v2.py
"""
import os
from huggingface_hub import HfApi


CARD = """---
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

# 0.6b-4b LCLM, {RATIO}× compression

Latent Context Language Model: an encoder–decoder compressor described in
*End-to-End Context Compression at Scale*.

The text to compress should be wrapped between `<|memory_start|>` and
`<|memory_end|>`.

> Running these checkpoints requires the LCLM codebase:
> <https://github.com/LeonLixyz/LCLM>. Standard `transformers.AutoModel` /
> `vllm.LLM` will not load this format on its own.

## Quick load

```python
from latent_context import LCLM

model = LCLM.from_pretrained("latent-context/0.6b-4b-LCLM-{RATIO}x")

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

runner = LCLMVLLMDecoder(
    "latent-context/0.6b-4b-LCLM-{RATIO}x", tensor_parallel_size=2,
)
outputs = runner.generate(
    prompts=[
        [{"role": "user",
          "content": "<|memory_start|>...<|memory_end|> Summarize."}],
    ],
    max_tokens=512, temperature=0.0,
)
```

Encode-once / decode-many via the two-stage CLI (`inference/encode.py` →
`inference/decode.py`) is also supported — see
`inference/examples/README.md` in the codebase.

## Configuration

| field | value |
|---|---|
| encoder | Qwen/Qwen3-Embedding-0.6B |
| decoder | Qwen/Qwen3-4B-Instruct-2507 |
| compression_ratio | {RATIO} |
| encoder_window_size | 1024 |
| pooling | mean |
| encoder_mask_type | causal |
| boundary_overlap | 0 |
| adapter_type | mlp |

Code: <https://github.com/LeonLixyz/LCLM>
"""


def main():
    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for ratio in [4, 8, 16]:
        repo = f"latent-context/0.6b-4b-LCLM-{ratio}x"
        body = CARD.replace("{RATIO}", str(ratio))
        path = f"/tmp/_card_{ratio}x.md"
        with open(path, "w") as f:
            f.write(body)
        api.upload_file(
            path_or_fileobj=path, path_in_repo="README.md",
            repo_id=repo, repo_type="model", token=token,
            commit_message="card: shorten wording, drop init-from line, clean encoder/decoder rows",
        )
        print(f"✓ {repo}")


if __name__ == "__main__":
    main()
