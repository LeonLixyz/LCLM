# Training

Read [data.md](data.md) before configuring a run. The supported DeepSpeed
configurations are ZeRO-0, ZeRO-1 and ZeRO-2. Stage 3 of the LCLM pipeline is
the final SFT phase, not ZeRO-3.

## Stage 3 (SFT) for the released checkpoints

The released `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` checkpoints were
fine-tuned from their stage-2 (continual pre-training) checkpoints with the
configs in `scripts/experiment_config/sft/`:

| Ratio | Config |
|---|---|
| 4x | `0.6b-4b-cs4-mean-w1024-causal-mlp-O0-sft.yaml` |
| 8x | `0.6b-4b-cs8-mean-w1024-causal-mlp-O0-sft.yaml` |
| 16x | `0.6b-4b-cs16-mean-w1024-causal-mlp-O0-sft.yaml` |

All three use the 16k packed mixture (`packed-cs16-16384`), a learning rate of
3e-5 for the encoder, adapter and decoder with cosine decay to 1e-6 and 500
warmup steps, mean pooling, a causal 1024-token encoder window, an MLP adapter
and no boundary overlap. Before launching, set:

- `stages[3].dataset`: the downloaded `packed-cs16-16384` root.
- `stages[3].resume_from_checkpoint`: your stage-2 HF checkpoint. If it is
  removed, the parser uses `<output>/<experiment>/stage2-hf`; set it to `null`
  only when starting from the configured base models is intended.
- `stages[3].distributed.config`: the accelerate config for your cluster. The
  released checkpoints were trained on 128 GPUs with two packed sequences per
  GPU and no gradient accumulation (256 sequences per optimizer update), using
  ZeRO-1 (`scripts/distributed_configs/deepspeed_zero1_multi_node.yaml` with
  `num_machines`/`num_processes` set to the cluster size).

Check `training.auto_resume` to avoid picking up an unrelated output run.

HF checkpoint resume validates `model_config.json` before loading weights and
requires the saved adapter. For legacy checkpoints, `summary_mean` maps to
`mean`, `chunk_size` to `compression_ratio`, and `batch_summary_tokens` to
`encoder_window_size`; the mask, overlap and adapter must also match.

For packed training, DeepSpeed takes its microbatch size from the actual loader.

## Launch

Run from the repository root on each training host. For one eight-GPU node:

```bash
OUTPUT_DIR=/path/to/checkpoints NUM_MACHINES=1 GPUS_PER_NODE=8 WORLD_SIZE=8 \
  MACHINE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  bash scripts/train.sh scripts/experiment_config/sft/0.6b-4b-cs16-mean-w1024-causal-mlp-O0-sft.yaml 3
```

For multi-node `nossh`, run this launcher once on **each node** with the same
configuration, dataset, model files and output storage. Set `NUM_MACHINES=N`,
`GPUS_PER_NODE=G`, `WORLD_SIZE=N*G`, a unique `MACHINE_RANK` from 0 to N-1, and
the same reachable rank-0 `MASTER_ADDR` and port. Do not also launch once per
GPU. The launcher computes the total `WORLD_SIZE` when omitted and rejects
invalid counts, ranks and loopback master addresses for multi-node runs.

ZeRO-2 partitions optimizer state and gradients across the data-parallel process
group, normally spanning all nodes. Validate collectives, data assignment and
checkpoint save/restore on your topology before a full multi-node run.

## Implementation notes

- `train/deepspeed_step.py` keeps the DeepSpeed backward separate from the
  optimizer update. All ranks run synchronized finite-loss and gradient checks on
  every microbatch; a window with a non-finite value skips both the optimizer and
  scheduler step and clears the accumulated gradients.
- On windows where no rank has memory to compress, encoder and adapter weights,
  momentum, weight decay and optimizer counters stay unchanged, while the decoder
  still updates. Ranks without memory run a dummy encoder/adapter pass so
  collectives stay matched.
- Packed sequences use FlashAttention with per-document isolation, and loss is
  computed on assistant tokens only.

Tested environment: Torch 2.8.0 / CUDA 12.9, DeepSpeed 0.17.5, Accelerate 1.10.1,
Transformers 4.57.1, FlashAttention 2.8.3, TorchData 0.11.0, datasets 3.6.0,
PyArrow 21.0.0 and Liger 0.6.2.

`tests/test_packed_flash_parity.py` and `tests/test_cot_compression.py` need a
GPU; the other tests in `tests/` run on CPU.
