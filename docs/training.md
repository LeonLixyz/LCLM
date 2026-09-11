# Training

Use the `train` branch and read [data.md](data.md) before configuring a run.
The supported DeepSpeed configurations are ZeRO-0, ZeRO-1 and ZeRO-2. Stage 3 of
the LCLM pipeline is the final SFT phase.

## Configure stage 3

Start from a model-matching YAML in `scripts/experiment_config/`, for example
`0.6b-4b-cs16-mean-w1024-bidirectional-mlp-O0.yaml`, and set:

- `stages[3].dataset`: the downloaded `packed-cs16-16384` or `packed-cs16-32768` root.
- `stages[3].max_packed_length`: 16384 or 32768, matching that root.
- `stages[3].distributed.type`: `deepspeed`, with
  `scripts/distributed_configs/deepspeed_zero2_multi_node.yaml` as its config.
- `training.compression_ratio`: 16; `data.packed_attention_backend`: `flash`.
- The intended model checkpoint, learning rates, schedule and output directory.

The parser otherwise infers stage 3's starting checkpoint as
`<output>/<experiment>/stage2-hf`. Verify that it exists or explicitly configure
`stages[3].resume_from_checkpoint`. Set it to `null` only when starting from the
configured base models is intended. Check `training.auto_resume` to avoid picking
up an unrelated output run.

The tested baseline uses BF16, mean pooling, a bidirectional 1024-token encoder
window, an MLP adapter, encoder batch cap 32, gradient accumulation 2 and gradient
checkpointing for both models. It sets `training.use_liger_kernel: false` and
`models.use_fused_ce: true`. Other experiment YAMLs may have different defaults.

## Launch

Run from the repository root on each training host. For one eight-GPU node:

```bash
OUTPUT_DIR=/path/to/checkpoints NUM_MACHINES=1 GPUS_PER_NODE=8 WORLD_SIZE=8 \
  MACHINE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  bash scripts/train.sh /path/to/prepared-training.yaml 3
```

For multi-node `nossh`, run this launcher once on **each node** with the same
configuration, dataset, model files and output storage. Set `NUM_MACHINES=N`,
`GPUS_PER_NODE=G`, `WORLD_SIZE=N*G`, a unique `MACHINE_RANK` from 0 to N-1, and
the same reachable rank-0 `MASTER_ADDR` and port. Use actual numeric values in
environment variables. Avoid launching once per GPU as well as once per node.
The launcher computes total `WORLD_SIZE` when omitted and rejects invalid counts,
ranks and loopback master addresses for multi-node runs.

ZeRO-2 partitions optimizer state and gradients across the data-parallel process
group, normally spanning all nodes. Validate the actual topology before a full
multi-node run: collectives, equal/disjoint data assignment, accumulation,
asymmetric memory, and full model/optimizer checkpoint save and restore.

## Validated behavior

The September 10, 2026 checks passed on a **single Modal H200:8 node**:

- 44 regressions covering data loading, memory labels, packed attention
  output/loss/gradient parity, document isolation and optimizer behavior.
- Eight-rank DDP/FSDP and BF16 ZeRO-1/2 checks, including absent/asymmetric memory,
  one-rank non-finite loss or gradients, rejected-window recovery and unchanged
  encoder/adapter optimizer state on memory-free updates.
- Exact accumulation scaling, ZeRO-2 CPU optimizer offload and FP16 loss-scale recovery.
- Real 16k/32k steps using pretrained Qwen3-4B-Instruct-2507 and
  Qwen3-Embedding-0.6B: 16 microbatches and 8 updates per rank, with exact
  **data-loader** resume, under FSDP and ZeRO-2.

Multi-node/RDMA and full model/optimizer checkpoint restore remain unverified.
The passing loader-resume check does not cover those checkpoint states.

The optimizer fix lives in `train/deepspeed_step.py`: backward and update are
separate, synchronized finite checks inspect ZeRO gradient buffers before any
update, and rejected windows skip both optimizer and scheduler. Memory-free
windows suppress encoder/adapter master gradients while allowing decoder updates.

Validation environment: Torch 2.8.0/CUDA 12.9, DeepSpeed 0.17.5, Accelerate 1.10.1,
Transformers 4.57.1, FlashAttention 2.8.3, TorchData 0.11.0, datasets 3.6.0,
PyArrow 21.0.0 and Liger 0.6.2. The Modal image is defined in
`scripts/stage3_deepspeed_modal.py`.

The focused regression in `tests/test_cot_compression.py` places memory inside
an assistant continuation. With packed FlashAttention, a real tiny Qwen3
encoder/decoder and an MLP adapter, it checks that changed CoT cannot affect
preceding predictions or another packed document, while subsequent loss reaches
both encoder and adapter. It also checks fused-CE parity and gradient checkpointing.
Run this GPU test on Modal alongside `tests/test_packed_flash_parity.py`. Both passed.
The source-to-packed audit made 120 comparisons across both lengths and checked
1,252 runtime example expansions, with no ID, label, memory-placement or length
mismatches. These are sample checks, separate from the full-file content-hash
verification. Details and the CoT boundary limitation are in
[data-verification.json](data-verification.json) and [data.md](data.md).

## Repeat bounded checks

These harnesses use the saved build and regression fixtures on Modal volume
`lclm-stage3-data`. Run from the repository root in the Modal conda environment:

```bash
export MODAL_ENVIRONMENT=leon-dev
# Prepare pinned models and regression fixtures if they are not already on the volume.
modal run -m scripts.stage3_readiness_modal --check prepare
# Current packed-loader check and DDP/FSDP regressions.
modal run -m scripts.stage3_readiness_modal --check loader
modal run -m scripts.stage3_readiness_modal --check gpu
# ZeRO-0/1/2, scaling, offload, FP16 and real-model matrix.
modal run --detach -m scripts.stage3_deepspeed_modal --run-name my-validation-run
```

Use a fresh result name when collecting new evidence. The original reports are
on `lclm-stage3-data` under `stage3-build-20260906/readiness-20260910-v1/` and
`stage3-build-20260906/deepspeed-ordering-20260910-v1/`. These are bounded tests,
not production training jobs.
