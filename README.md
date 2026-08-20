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

### Stage-3 + agent data

The mixture builder keeps ordinary stage-3 rows unchanged, but for the
reasoning-heavy `reasoning_data` and `dolci_think` subsets it restores the
original (uncompressed) prompt and compresses only a safely identified target
analysis span. The final answer/code suffix remains supervised. Ambiguous CoT
splits fail closed and stay uncompressed.

OpenThoughts trajectories remain full multi-turn conversations with
`compression_scope=none`; every assistant turn is supervised through the
Qwen3-4B-Instruct-2507 chat template. Rows whose `result` records an agent error
are excluded by default.

```bash
python data/build_stage3_agent_mixture.py \
    --agent-repeat 10 \
    --output ./data/stage3-agent.jsonl.gz

python data/preprocess_for_dynamic_packing.py \
    --input_path ./data/stage3-agent.jsonl.gz \
    --output_dir ./data/stage3-agent-packed \
    --llm_tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --embed_tokenizer Qwen/Qwen3-Embedding-0.6B \
    --reference_chunk_size 16 \
    --max_packed_length 32768
```

The default agent source is
`open-thoughts/OpenThoughts-Agent-SFT-100K`. Add the distinct pre-RL cold-start
set with `--include-coldstart`; failed traces require the explicit
`--include-failed-agent-traces` override, and derived summary/answer variants
require `--include-derived-agent-traces`. Source weights control streaming
interleave order, row caps control how many examples each source pass
contributes, and `--agent-repeat` is explicit trajectory upsampling (10 passes
is roughly a low-single-digit agent share against the full 20.3M-row stage-3
set).

Packed batches may freely mix compressed and uncompressed sequences. During
distributed training all ranks still enter encoder/adapter collectives; a
globally all-uncompressed optimizer step is a true no-op for those parameter
groups (including AdamW state and weight decay).

Synthetic selective-expansion traces use the same native agent path. Their
initial user context contains positional `seg_i` blocks whose bodies are wrapped
in `<|memory_start|>...<|memory_end|>`. Each memory body is the full source
document (at least 512 words), not a prewritten summary. During synthetic trace
generation only, the teacher sees short routing descriptions so it cannot answer
from the raw documents without calling `expand`; harvested training messages
contain the full source documents. The assistant calls the Qwen-native
`expand` tool with `{"segment_id": "seg_i"}`, receives that segment's original
text as a tool result, and may continue expanding before answering. Dynamic
preprocessing extracts memory bodies from agent message content, retains the
native tool schema and all tool calls, supervises every assistant turn, and
keeps system/user/tool-result tokens loss-masked.

Generate a five-family Qwen-235B pilot on Modal with:

```bash
modal run --detach data/generate_synthetic_expansion_modal.py \
    --count 5 \
    --distractors 24 \
    --run-name pilot-long-seg-v2
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
