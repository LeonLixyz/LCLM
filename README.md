# LCLM — Latent Context Language Models

Code for *End-to-End Context Compression at Scale*. An LCLM is an
encoder–decoder soft-token compressor: an encoder maps a long input to
a short sequence of latent tokens, and a decoder consumes those latents
in place of the original tokens.

🤗 [Checkpoints](https://huggingface.co/latent-context) · [Eval datasets](https://huggingface.co/datasets/latent-context/lclm-eval)

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
│   ├── hf.py              #   reference HF path (single process, single GPU)
│   ├── vllm_inference/    #   two-stage vLLM CLI
│   │   ├── encode.py      #     HF encoder → embeds.pt
│   │   └── decode.py      #     vLLM decoder reads embeds.pt
│   └── examples/          #   runnable demos + eval drivers (see README)
├── train/                 # Training entry points.
│   ├── launch_train.py    #   CLI
│   └── trainer.py         #   training loop, checkpointing, auto-resume
├── scripts/               # Launch wrappers + YAML configs.
│   ├── run_pipeline.sh    #   end-to-end (adapter → enc/dec continual pretrain → SFT)
│   ├── convert_checkpoint.sh
│   ├── experiment_config/ #   per-experiment YAMLs
│   ├── pretrain_config/   #   pretrain-stage YAMLs
│   └── distributed_configs/  # accelerate / deepspeed / fsdp
├── agent/                 # Agent app — EXPAND(i) tool over compressed segments.
├── data/                  # Training datasets, collators, dynamic packing.
└── utils/                 # Helpers + checkpoint-conversion shell scripts.
```

## Inference

Text to compress should be wrapped between `<|memory_start|>` and
`<|memory_end|>` in the prompt. See
[`inference/examples/README.md`](inference/examples/README.md) for
runnable demos and the RULER NIAH eval driver.

#### HF inference

```python
from latent_context import LCLM
model = LCLM.from_pretrained("latent-context/0.6b-4b-LCLM-16x")
# see inference/hf.py for generate_text
```

#### vLLM inference

Two-stage CLI — HF encoder and vLLM decoder run in **separate
processes** that hand off via a `.pt` file. Running both in one process
OOMs (vLLM grabs all GPU memory at init).

```bash
python -m inference.vllm_inference.encode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-jsonl prompts.jsonl --out embeds.pt
python -m inference.vllm_inference.decode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --embeds-pt embeds.pt --out completions.jsonl
```

## Training

Driven by a single experiment YAML that defines four stages: **adapter
warm-up → encoder continual pretrain → decoder continual pretrain → SFT.** Each stage runs
under accelerate (DeepSpeed by default) and the pipeline converts the
distributed checkpoint to the HF layout between stages.

### One-line full pipeline

```bash
OUTPUT_DIR=./checkpoints bash scripts/run_pipeline.sh \
    scripts/experiment_config/0.6b-4b-cs4-mean-w1024-causal-mlp-O0.yaml
```

`OUTPUT_DIR` is required; everything else lives in the YAML.

### Configs

| Path | What's in it |
|---|---|
| `scripts/experiment_config/` | Full end-to-end runs. Naming: `{enc}-{dec}-cs{N}-{pooling}-w{W}-{mask}-{adapter}-O{O}.yaml` — e.g. `0.6b-4b-cs16-mean-w1024-bidirectional-mlp-O0.yaml`. |
| `scripts/pretrain_config/` | Pretrain-only sweeps over adapter / encoder layouts. Naming: `{pooling}-w{W}-{mask}-{adapter}-O{O}.yaml`. |
| `scripts/distributed_configs/` | Accelerate launcher configs: `deepspeed_zero{1,2,3}*.yaml`, `fsdp_*.yaml`, `ddp_multi_node.yaml`. |

To match the released checkpoints, the relevant axes are
`pooling=mean`, `mask=causal`, `adapter=mlp`, `boundary_overlap=0`,
`encoder_window_size=1024`. Pick the `csN` matching the compression
ratio you want (4 / 8 / 16).

### Single stage

```bash
# launch_train.py is the CLI; trainer.py owns the loop.
accelerate launch \
    --config_file scripts/distributed_configs/deepspeed_zero1.yaml \
    -m train.launch_train \
    --config scripts/experiment_config/0.6b-4b-cs4-mean-w1024-causal-mlp-O0.yaml \
    --stage 1 \
    --output_dir ./checkpoints
```

### FSDP

Swap the accelerate config:

```bash
DIST_TRAIN_CONFIG=scripts/distributed_configs/fsdp_hybrid_shard.yaml \
DISTRIBUTED_TYPE=fsdp \
OUTPUT_DIR=./checkpoints bash scripts/run_pipeline.sh \
    scripts/experiment_config/0.6b-4b-cs4-mean-w1024-causal-mlp-O0.yaml
```

### Env vars

| Var | Default | What it does |
|-----|---------|--------------|
| `OUTPUT_DIR` | (required) | Where checkpoints get written. |
| `AUTO_RESUME` | `true` | Resume from latest matching checkpoint each `SAVE_STEPS`. |
| `RESUME_FROM_CHECKPOINT` | `""` | Resume from a specific HF checkpoint. |
| `DISTRIBUTED_TYPE` | `deepspeed` | `deepspeed` or `fsdp`. |
| `DIST_TRAIN_CONFIG` | `scripts/distributed_configs/deepspeed_zero1_multi_node.yaml` | Accelerate config path. |
| `DS_HOSTFILE` | unset | DeepSpeed hostfile for multi-node. |

### Checkpoint conversion

`scripts/convert_checkpoint.sh` converts a raw FSDP / DeepSpeed
checkpoint to the HF-style `{decoder, encoder, adapter}/` layout the
LCLM loader (and the published checkpoints) use. The pipeline calls it
between stages automatically. See `utils/checkpoints/` for the inner
scripts and `train/trainer.py` for the checkpoint / resume logic.

## Citation

```bibtex
@article{lclm2026,
  title={End-to-End Context Compression at Scale},
  author={...},
  year={2026},
}
```
