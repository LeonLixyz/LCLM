# LCLM Inference & Evaluation Examples

Runnable examples for the published checkpoints
`latent-context/0.6b-4b-LCLM-{4,8,16}x`. **All of these require this
repo (`LCLM`) on your PYTHONPATH** — the published checkpoints are not
loadable with vanilla `transformers.AutoModel` or `vllm.LLM` alone.

```bash
git clone https://github.com/LeonLixyz/LCLM && cd LCLM
pip install -e .
```

## 1. One-shot HF generation

Generate a single completion through the HuggingFace path
(`inference/hf.py`). Fine for debugging or single-prompt smoke tests.

```bash
python -m inference.examples.example_hf \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompt "<|memory_start|>$(cat my_doc.txt)<|memory_end|> Summarize."
```

## 2. Two-stage encode → decode (vLLM)

The vLLM path runs the encoder and the decoder in **separate processes**
that hand off via a `.pt` file on disk. Running both in one process
OOMs — vLLM grabs all available GPU memory at init, leaving none for
the HF encoder.

```bash
# Step 1: HF encoder over a jsonl of prompts → embeds.pt
python -m inference.vllm_inference.encode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-jsonl prompts.jsonl \
    --out embeds.pt \
    --max-encode-batch-size 128

# Step 2: vLLM decoder reads embeds.pt → completions.jsonl
python -m inference.vllm_inference.decode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --embeds-pt embeds.pt \
    --out completions.jsonl \
    --max-tokens 128 \
    --temperature 0.0
```

`prompts.jsonl` format: one JSON object per line. The `prompt` field is
either a chat-template string or a list of chat messages; any other
fields are passed through as metadata.

```jsonl
{"prompt": "<|im_start|>user\n<|memory_start|>...<|memory_end|> Question?<|im_end|>\n<|im_start|>assistant\n", "answers": ["expected answer"]}
{"prompt": [{"role": "user", "content": "<|memory_start|>...<|memory_end|> Question?"}], "answers": ["expected answer"]}
```

## 3. RULER NIAH evaluation

End-to-end RULER needle-in-a-haystack eval over the published
`latent-context/lclm-eval` dataset.

```bash
# Prepare per-task prompts.jsonl from latent-context/lclm-eval at a chosen ctx length
python -m inference.examples.prepare_ruler_niah --ctx 4096 --out-dir _ruler_prompts

# Run encode + decode + scoring; writes summary.json
python -m inference.examples.eval_ruler_niah \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-dir _ruler_prompts \
    --out-dir _ruler_results
```

Scoring uses the official RULER scorer (case-insensitive substring
match per needle, mean across needles per sample) — same as the
upstream NVIDIA RULER reference implementation.

## Eval datasets

All eval data lives in a single HF dataset with one config per
benchmark:

```python
from datasets import load_dataset

# configs: ruler / gsm8k / longhealth5 / longbench
ds = load_dataset("latent-context/lclm-eval", "ruler", split="test")
print(ds[0]["prompt"][:200])
print(ds[0]["category"])
print(ds[0]["extra_info"]["ground_truth"]["answers"])
```

Schema for every row: `prompt: str`, `category: str`, `extra_info: dict`.

