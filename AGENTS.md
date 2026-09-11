# LCLM development and training

This repository's `train` branch contains the LCLM trainer. Its validation does
not cover the separate Code-LLaVA implementation.

## Start here

- [Training](docs/training.md): configuration, launch commands and validation scope.
- [Data](docs/data.md): public datasets, pinned revisions, packing rules and tokenizer pins.
- Entrypoint: `train/launch_train.py`; launcher: `scripts/train.sh`.
- Use DeepSpeed **ZeRO-2 at most**. Pipeline stage 3 means SFT, not ZeRO-3.

## Preserve correctness

- Keep DeepSpeed backward separate from the optimizer update in
  `train/deepspeed_step.py`. Accelerate's DeepSpeed backward wrapper can update
  weights before a later gradient check.
- All ranks must participate in finite-loss/gradient checks every microbatch.
  Invalid accumulation windows skip optimizer and scheduler updates and clear
  accumulated gradients. Do not swallow collective failures.
- On globally memory-free windows, encoder/adapter weights, momentum, weight
  decay and optimizer counters must remain unchanged. Inspect ZeRO master
  gradients and preserve CPU-offload gradient-buffer restoration.
- Keep dummy encoder/adapter participation on ranks without memory, equal
  loader lengths across ranks, packed-document attention isolation and memory
  label masks.
- Set DeepSpeed's microbatch size from the actual loader. Validate checkpoint
  compression/adapter metadata on resume; legacy `summary_mean` maps to windowed
  `mean`, with the saved window, ratio, mask and overlap preserved.
- Use one packed length per run. Preserve the pinned tokenizers, 16:1 compression,
  assistant-only supervision, category-specific packing and sequence shuffling.
- ZeRO-2 normally shards across the full data-parallel group, including multiple
  nodes. Single-node tests do not establish multi-node/RDMA or full checkpoint
  restore correctness.

## Verification and repository hygiene

- Run GPU tests and evals on Modal H200:8, never locally. Use `conda activate modal`
  and `MODAL_ENVIRONMENT=leon-dev`. CPU syntax and unit checks may run locally.
- Reuse passing GPU evidence when runtime behavior has not changed. Do not start
  production training to answer a status question.
- Keep current instructions in `docs/`. Store run logs, progress diaries and
  temporary recovery/upload scripts under ignored `_modal_run/`, not the source tree.
- Historical generation and release orchestration is preserved in Git at
  `c94333562bf8d7cc8162cc970b2519b20b1cf253`; do not restore it as a training dependency.
- Dataset uploads must exclude `state.json`: remove it from upload staging and
  use `ignore_patterns=["state.json"]` with `HfApi.upload_folder()`.
