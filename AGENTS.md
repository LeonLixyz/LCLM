# LCLM training instructions

## Scope and current readiness

- This file lives at the repository root on the `train` branch of
  `LeonLixyz/LCLM` (local checkout: `LCLM-stage3-build-20260906`). The neighboring
  `Code-LLaVA` checkout has a different trainer; these results do not validate it.
- Use **DeepSpeed ZeRO-2 at most**. Do not investigate or enable ZeRO-3 for this
  build. Pipeline `--stage 3` means the final training phase, not ZeRO stage 3.
- Read `STAGE3_PACKED_DATA_20260910.md`, `TRAINING_READINESS_20260910.md` and
  `DEEPSPEED_ORDERING_20260910.md` before changing the data path or training flow.
- Single-node H200:8 verification passed 44 regressions, distributed failure and
  recovery cases, and real 16k/32k training steps with a pretrained 4B decoder and
  0.6B encoder. FSDP and ZeRO-2 passed. Multi-node/RDMA and full model/optimizer
  checkpoint restore have not been validated by these smoke tests; the resume
  result recorded in those tests is exact **data-loader** resume.
- Existing passing tests establish their recorded scope, not universal training
  correctness. Do not start a full production run just to answer a status check.

## Final training data

On Modal volume `lclm-stage3-data`, mounted at `/data`, select one root:

- 16k: `/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-16384`
- 32k: `/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-32768`

Set `stages[3].dataset` to that version root and `stages[3].max_packed_length` to
16384 or 32768. Keep `training.compression_ratio: 16` and
`data.packed_attention_backend: flash`. `DynamicPackedDataset` expects the root
containing `data/mixed/`; retain file/row shuffling and equal work across ranks.
Do not concatenate the 16k and 32k variants: their underlying examples overlap.

Lengths include memory boundary tags after compression. Decoder IDs and labels
are preprocessed; memory strings are encoder-tokenized at runtime. Preserve
assistant-only supervision and mask memory vectors/boundary tags. Native agents
remain uncompressed; reasoning uses the reviewed CoT50 policy with prompts intact.
Sequences are category-pure, then globally shuffled across categories. Use the
manifest hashes and pinned tokenizer revisions in the data handoff; do not
silently replace tokenizers, re-pack, or change the compression ratio.

Both raw and packed datasets are uploaded and verified in public
repositories under `leonli66`. Read `HF_PUBLICATION_20260911.md` for repository
links, final pinned revisions, verification evidence and a download recipe.
Use the packed snapshot's selected version root exactly as the Modal root above.
Do not label older stage3 repositories as this final release or combine the
two packed lengths. Exclude `state.json` from uploads; use
`ignore_patterns=["state.json"]` for `HfApi.upload_folder()` and remove such
internal files from the upload staging directory.

## Configure and launch

- Entrypoint: `train/launch_train.py`; launcher: `scripts/train.sh`.
- Start from the model-matching YAML in `scripts/experiment_config/`. Select
  `stages[3].distributed.type: deepspeed` and a ZeRO-2 config. The existing
  `scripts/distributed_configs/deepspeed_zero2_multi_node.yaml` uses the `nossh`
  launcher and no CPU offload; its node/process numbers are template values
  overridden by `scripts/train.sh`.
- Set the dataset and packed length above. Use the requested model checkpoint,
  learning rates, schedule and output directory. The YAML parser otherwise
  infers stage 3's starting checkpoint as `<output>/<experiment>/stage2-hf`;
  verify it exists or explicitly set `stages[3].resume_from_checkpoint`.
  Set that field to `null` only when starting from the configured base models is
  intended. Check `training.auto_resume` to avoid loading a stale output run.
- The tested baseline uses BF16, packed FlashAttention, mean pooling, a
  bidirectional 1024-token encoder window, an MLP adapter, encoder batch cap 32,
  gradient accumulation 2 and checkpointing for both models. It sets
  `training.use_liger_kernel: false` and `models.use_fused_ce: true`. Other YAML defaults
  can differ; do not call those exact settings tested without checking them.
- Reproduce the validation environment: Torch 2.8.0/CUDA 12.9, DeepSpeed 0.17.5,
  Accelerate 1.10.1, Transformers 4.57.1, FlashAttention 2.8.3, TorchData 0.11.0,
  datasets 3.6.0 and PyArrow 21.0.0. See `scripts/stage3_deepspeed_modal.py` for
  the image recipe. Record the actual environment for new runs.

Run from this repository root on the training host, after preparing the chosen
config and output directory. For one eight-GPU node:

```bash
OUTPUT_DIR=/path/to/checkpoints NUM_MACHINES=1 GPUS_PER_NODE=8 WORLD_SIZE=8 \
  MACHINE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  bash scripts/train.sh /path/to/prepared-training.yaml 3
```

For multi-node `nossh`, run the launcher once on **each node** with the same
config/data/model files and output storage. Set `NUM_MACHINES=N`,
`GPUS_PER_NODE=G`, `WORLD_SIZE=N*G`, a unique `MACHINE_RANK` from 0 to N-1, and
the same reachable rank-0 `MASTER_ADDR`/`MASTER_PORT`. Supply actual numeric values
in environment variables. Avoid multiple launchers per node from SLURM task
fan-out. The script calculates total `WORLD_SIZE` if omitted, but specifying the
topology explicitly makes the intended launch reviewable.

ZeRO-2 partitions optimizer state and gradients across the data-parallel process
group, normally spanning all nodes. It is not automatically confined within each
node. Before claiming multi-node readiness, run a bounded test on the actual
topology that checks NCCL collectives, disjoint/equal-length data assignment,
accumulation scaling, absent/asymmetric memory, and checkpoint save/restore.
Do not suppress NCCL failures or substitute single-node results for that test.

## Preserve training correctness

- `train/deepspeed_step.py` owns separate DeepSpeed backward/update ordering and
  loss scaling. Never replace it with `accelerator.backward()` followed by a
  late gradient check: that wrapper can update weights during backward.
- Every rank enters the finite-loss/gradient reduction every microbatch. Invalid
  windows skip the optimizer and scheduler and clear accumulation buffers.
- Globally no-memory windows must not update encoder/adapter weights, momentum,
  weight decay or step counters. Inspect ZeRO master gradients, not only `.grad`
  on model parameters. Retain CPU-offload gradient-allocation restoration.
- Keep dummy encoder/adapter participation for empty-memory ranks, balanced
  dataset lengths, packed attention document isolation and memory-label masks.
- Run GPU verification/evals on Modal H200:8 per container, never locally. Use
  `conda activate modal` and `MODAL_ENVIRONMENT=leon-dev` for this build. Reuse
  passing evidence when behavior has not changed; do not launch duplicate jobs.
  Local syntax, configuration and launcher checks may use CPU-only stubs.
- The bounded ZeRO-2 verification entrypoint is
  `modal run --detach -m scripts.stage3_deepspeed_modal`; choose a fresh run name
  when collecting new evidence. This is separate from a production training run.
