# LCLM — Latent Context Language Models

Code for *End-to-End Context Compression at Scale*. An LCLM is an
encoder–decoder soft-token compressor: an encoder maps a long input to
a short sequence of latent tokens, and a decoder consumes those latents
in place of the original tokens. With a 0.6B encoder and a 4B decoder
trained end-to-end on ~350B tokens, LCLMs achieve a new Pareto frontier
between long-context accuracy, time-to-first-token, and peak GPU memory.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
# If flash-attn fails to build:
uv run pip install flash-attn --no-build-isolation
```

If you need `nvcc`: `conda install -c nvidia cuda-nvcc`.

## Quick start

### Load an LCLM (HuggingFace Transformers)

```python
from latent_context import LCLM

# Auto-detects both layouts:
#   new:     model/{decoder, encoder, adapter}/
#   legacy:  model/{llm, embedder, projectors}/
model = LCLM.from_pretrained("latent-context/0.6b-4b-LCLM-16x")

prompt = (
    "<|memory_start|>"
    "<long document, code, or text to compress>"
    "<|memory_end|> "
    "Summarize the document above."
)
```

Or via the function form:

```python
from latent_context import from_pretrained

model, tokenizer, processor = from_pretrained("latent-context/0.6b-4b-LCLM-16x")
```

### Production inference (vLLM)

The decoder runs in vLLM (paged attention, continuous batching); the
encoder stays in HuggingFace Transformers.

```python
from inference.vllm import LCLMVLLMDecoder

runner = LCLMVLLMDecoder("latent-context/0.6b-4b-LCLM-16x", tensor_parallel_size=2)
outputs = runner.generate(
    prompts=[[{"role": "user", "content": "<|memory_start|>...<|memory_end|> Summarize."}]],
    max_tokens=512, temperature=0.0,
)
print(outputs[0])
```

Runnable end-to-end demos:

```bash
python -m inference.examples.example_hf   --checkpoint latent-context/0.6b-4b-LCLM-16x --prompt "..."
python -m inference.examples.example_vllm --checkpoint latent-context/0.6b-4b-LCLM-16x --prompt "..."
```

## Repo layout

| Path              | What it is |
| ----------------- | ---------- |
| `latent_context/` | Model package — `LCLM`, `LatentEncoder`, `Adapter`, `LCLMProcessor`, `from_pretrained`. Legacy class names (`LCLM`, `Encoder`, `Adapter`) are also exported for backward compat with already-published checkpoints. |
| `train/`          | `launch_train.py` (CLI) + `trainer.py` (training loop, checkpointing, auto-resume). |
| `scripts/`        | Launch wrappers (`train.sh`, `run_pipeline.sh`, `convert_checkpoint.sh`) + YAML configs (`experiment_config/`, `pretrain_config/`, `distributed_configs/`). |
| `inference/`      | `hf.py` (reference) + `vllm.py` (production) + `examples/`. |
| `agent/`          | Agent app — `EXPAND(i)` tool over compressed segments. Per-benchmark runners + Modal launcher. |
| `data/`           | Training-time datasets, collators, dynamic packing utilities. |
| `utils/`          | Generic helpers (env, seed, scheduler, NaN checks) + `utils/checkpoints/` shell scripts for converting FSDP/DeepSpeed checkpoints to the HF layout. |

## Training

End-to-end pipeline (adapter → encoder → decoder → SFT) via accelerate/DeepSpeed:

```bash
OUTPUT_DIR=./checkpoints bash scripts/run_pipeline.sh \
  scripts/experiment_config/0.6b-4b-cs16-mean-w1024-bidirectional-mlp-O0.yaml
```

For FSDP, swap to a `*-fsdp.yaml` distributed config (or set
`DISTRIBUTED_TYPE=fsdp`). Distributed configs live in
`scripts/distributed_configs/`.

Key env vars:

| Var | Default | What it does |
|-----|---------|--------------|
| `OUTPUT_DIR` | (required) | Where checkpoints get written. |
| `AUTO_RESUME` | `true` | Resume from latest matching checkpoint each `SAVE_STEPS`. |
| `RESUME_FROM_CHECKPOINT` | `""` | Resume from a specific HF checkpoint (lower priority than `AUTO_RESUME`). |
| `DISTRIBUTED_TYPE` | `deepspeed` | `deepspeed` or `fsdp`. |
| `DIST_TRAIN_CONFIG` | `scripts/distributed_configs/deepspeed_zero1_multi_node.yaml` | Accelerate config path. |

See `train/trainer.py` for the checkpoint / resume logic and
`utils/checkpoints/` for converting raw distributed checkpoints to
HF-style ones the loader can read.

## Citation

```bibtex
@article{lclm2026,
  title={End-to-End Context Compression at Scale},
  author={...},
  year={2026},
}
```
