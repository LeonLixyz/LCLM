# LCLM — Latent Context Language Models

Code for *End-to-End Context Compression at Scale*. An LCLM is an
encoder–decoder soft-token compressor: an encoder maps a long input to
a short sequence of latent tokens, and a decoder consumes those latents
in place of the original tokens.

Trained checkpoints: [`latent-context/0.6b-4b-LCLM-{4,8,16}x`](https://huggingface.co/latent-context).
Eval datasets: [`latent-context/lclm-eval`](https://huggingface.co/datasets/latent-context/lclm-eval).

## Install

```bash
git clone https://github.com/LeonLixyz/LCLM && cd LCLM
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
# If flash-attn fails to build:
uv run pip install flash-attn --no-build-isolation
```

If you need `nvcc`: `conda install -c nvidia cuda-nvcc`.

## Repo layout

```
LCLM/
├── latent_context/        # Model package: LCLM, LatentEncoder, Adapter,
│                          # LCLMProcessor, from_pretrained.
├── inference/             # Inference entry points.
│   ├── hf.py              #   reference HF path
│   ├── vllm.py            #   vLLM-decoder path (LCLMVLLMDecoder)
│   ├── encode.py          #   two-stage CLI: HF encoder → embeds.pt
│   ├── decode.py          #   two-stage CLI: vLLM decoder reads embeds.pt
│   └── examples/          #   runnable demos + eval drivers (see README)
├── train/                 # Training entry points.
│   ├── launch_train.py    #   CLI
│   └── trainer.py         #   training loop, checkpointing, auto-resume
├── scripts/               # Launch wrappers + YAML configs.
│   ├── run_pipeline.sh    #   end-to-end pipeline (adapter→encoder→decoder→SFT)
│   ├── convert_checkpoint.sh
│   ├── experiment_config/ #   per-experiment YAMLs
│   ├── pretrain_config/   #   pretrain-stage YAMLs
│   └── distributed_configs/  # accelerate / deepspeed / fsdp
├── agent/                 # Agent app — EXPAND(i) tool over compressed segments.
├── data/                  # Training datasets, collators, dynamic packing.
└── utils/                 # Helpers + checkpoint-conversion shell scripts.
```

## Inference

All four entry points are documented in
[`inference/examples/README.md`](inference/examples/README.md). Quick
pointers:

```python
# 1. One-shot via HuggingFace Transformers (single process, single GPU)
from latent_context import LCLM
model = LCLM.from_pretrained("latent-context/0.6b-4b-LCLM-16x")
# see inference/hf.py for generate_text
```

```bash
# 2. Two-stage CLI via vLLM (HF encoder + vLLM decoder in separate
#    processes — this is the path for batched eval / serving). Running
#    both in one process OOMs: vLLM grabs all GPU memory at init.
python -m inference.encode --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-jsonl prompts.jsonl --out embeds.pt
python -m inference.decode --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --embeds-pt embeds.pt --out completions.jsonl

# 3. End-to-end RULER NIAH eval (wraps the two-stage CLI + scoring)
python -m inference.examples.prepare_ruler_niah --ctx 4096 --out-dir _ruler_prompts
python -m inference.examples.eval_ruler_niah \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-dir _ruler_prompts --out-dir _ruler_results
```

Text to compress should be wrapped between `<|memory_start|>` and
`<|memory_end|>` in the prompt. Scoring uses the official RULER scorer
(case-insensitive substring match per needle).

## Training

End-to-end pipeline (adapter → encoder → decoder → SFT) via
accelerate / DeepSpeed:

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
| `RESUME_FROM_CHECKPOINT` | `""` | Resume from a specific HF checkpoint. |
| `DISTRIBUTED_TYPE` | `deepspeed` | `deepspeed` or `fsdp`. |
| `DIST_TRAIN_CONFIG` | `scripts/distributed_configs/deepspeed_zero1_multi_node.yaml` | Accelerate config path. |

See `train/trainer.py` for the checkpoint / resume logic and
`utils/checkpoints/` for converting raw distributed checkpoints to the
HF-style layout the loader expects.

## Citation

```bibtex
@article{lclm2026,
  title={End-to-End Context Compression at Scale},
  author={...},
  year={2026},
}
```
