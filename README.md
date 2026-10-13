# LCLM — Latent Context Language Models

Code for *End-to-End Context Compression at Scale*. An LCLM is an
encoder–decoder soft-token compressor: an encoder maps a long input to
a short sequence of latent tokens, and a decoder consumes those latents
in place of the original tokens.

📄 [arXiv](https://arxiv.org/abs/2606.09659) · 🤗 [Checkpoints](https://huggingface.co/latent-context) · [Eval datasets](https://huggingface.co/datasets/latent-context/lclm-eval)

| What | Where | Paper |
|---|---|---|
| Model, HF and vLLM inference | `latent_context/`, `inference/` | §3 |
| Optimized encoder and timing benchmarks | `inference/optimized_encoder.py`, [`benchmarks/timing/`](benchmarks/timing/README.md) | §6, Figures 1, 4, 5 |
| Agent with latent context (`expand` tool) | [`agent/`](agent/README.md) | §7.1, Figure 6 |
| Online latent compaction | [`compaction/`](compaction/README.md) | §7.2, Figure 7 |
| Training pipeline and data | `train/`, `data/`, [`docs/training.md`](docs/training.md), [`docs/data.md`](docs/data.md) | §4 |

The checkpoints `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` are at **v1.1**,
the models evaluated in the current version of the paper: v1.1 improves the
agent with latent context and online latent compaction. The first release
(May 2026) remains available on each model repo as revision `v1.0`.

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
│   ├── optimized_encoder.py  # FlashAttention + torch.compile + CUDA-graph encoder (inference only)
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
│   ├── experiment_config/ #   per-experiment YAMLs (sft/: stage-3 configs of the released models)
│   ├── pretrain_config/   #   pretrain-stage YAMLs
│   └── distributed_configs/  # accelerate / deepspeed / fsdp
├── agent/                 # Agent with latent context: native `expand` tool over compressed segments.
├── compaction/            # Online latent compaction of the model's own generation + MATH eval.
├── benchmarks/timing/     # TTFT / peak-memory benchmarks (LCLM, KV-cache baselines, vLLM).
├── data/                  # Runtime datasets, collators and packing utilities.
├── docs/                  # Training, data, agent and timing guides.
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

Two stages: first the HF encoder compresses every prompt into latent
tokens written to a `.pt` file, then vLLM reads that file and decodes
generations from the latents.

```bash
python -m inference.vllm_inference.encode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --prompts-jsonl prompts.jsonl --out embeds.pt
python -m inference.vllm_inference.decode \
    --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --embeds-pt embeds.pt --out completions.jsonl
```

#### Optimized encoder

For faster compression, `enable_encoder_optimization` runs the encoder with
FlashAttention, `torch.compile` and CUDA graphs, 32 windows (32,768 tokens) per
forward pass. It is inference only, and the first call per batch shape compiles.
See [`docs/timing.md`](docs/timing.md).

```python
from inference.hf import load_model
from inference.optimized_encoder import enable_encoder_optimization

model, tokenizer, processor = load_model("latent-context/0.6b-4b-LCLM-16x", device="cuda:0")
model.eval()
enable_encoder_optimization(model)
```

## Paper experiments

Each folder has its own README with smoke tests, full commands, Modal
launchers and the expected numbers.

**Timing (Figures 1, 4, 5).** [`benchmarks/timing/`](benchmarks/timing/README.md)
measures TTFT and peak memory for LCLMs (optimized encoder + Hugging Face
decoder), the KV-cache compression baselines (KVPress, Attention Matching) and
uncompressed models served with vLLM.

```bash
python -m benchmarks.timing.lclm_hf_scaling --checkpoint latent-context/0.6b-4b-LCLM-16x
python -m benchmarks.timing.vllm_ttft --model lclm-16x
```

**Agent with latent context (Figure 6).** [`agent/`](agent/README.md) splits
the context into 1,024-token segments, compresses each one, and gives the model
a native `expand` tool that returns a segment's original text.

```bash
for n in 4096 8192 16384; do
  python -m agent.run_niah_agent --context-length $n --gpus 8 --output-dir runs/niah_agent/$n
done
python -m agent.aggregate_niah runs/niah_agent --per-task
```

**Online latent compaction (Figure 7).** [`compaction/`](compaction/README.md)
encodes the model's own generated tokens into latents every 512 tokens while
keeping the latest 256 raw, and evaluates on the MATH test set.

```bash
pip install math-verify==0.9.0
python -m compaction.eval_math --checkpoint latent-context/0.6b-4b-LCLM-16x --out runs/math_full
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
| `scripts/distributed_configs/` | Accelerate launcher configs: `deepspeed_zero{1,2}*.yaml`, `fsdp_*.yaml`, `ddp_multi_node.yaml`. |

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

### Stage-3 SFT

The released checkpoints were fine-tuned with the configs in
`scripts/experiment_config/sft/` on the public 16k packed dataset described in
[docs/data.md](docs/data.md). It includes native agents, expanded agents,
reasoning and other data, with category-specific packing followed by global
sequence shuffling. The [training guide](docs/training.md) covers
configuration, launch commands and DeepSpeed ZeRO-1/2.

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
@misc{li2026endtoendcontextcompressionscale,
      title={End-to-End Context Compression at Scale},
      author={Ang Li and Sean McLeish and Haozhe Chen and Nimit Kalra and Zaiqian Chen and Artem Gazizov and Venkata Anoop Suhas Kumar Morisetty and Bhavya Kailkhura and Harshitha Menon and Zhuang Liu and Brian R. Bartoldson and Tom Goldstein and Sanae Lotfi and Micah Goldblum and Pavel Izmailov},
      year={2026},
      eprint={2606.09659},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.09659},
}
```
