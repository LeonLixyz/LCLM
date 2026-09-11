# DeepSpeed optimizer ordering correction — September 10, 2026

**ZeRO-2 is the intended training configuration and the maximum supported ZeRO
stage. Its optimizer correction and real 16k/32k training checks passed on eight
H200s. FSDP also passed its earlier verification.** The user explicitly limited
this work to ZeRO-2 at most; no ZeRO-3 investigation remains in scope.

Accelerate's DeepSpeed backward wrapper also updates the optimizer. The trainer
now uses separate engine backward and step calls so synchronized loss/gradient
checks happen before any update. It scans ZeRO accumulation/master buffers,
rejects the entire accumulation window when any rank has invalid loss or
gradients, clears those buffers, and skips that window's scheduler update. FP16
rejection also lowers the dynamic loss scale for recovery.

For a complete accumulation window without compressed memory on any rank, an
optimizer pre-step hook suppresses encoder/adapter master gradients. A post-step
hook restores the allocations used by CPU offload. Their weights, momentum,
weight decay and optimizer counters stay unchanged while the decoder can update.
Loss scaling applies exactly once to each accumulation microbatch, including
ZeRO-1 no-sync microbatches.

Changes are in `train/deepspeed_step.py`, `train/trainer.py` and
`utils/nan_checks.py` in this LCLM stage-3 checkout. A stage guard rejects ZeRO-3
before DeepSpeed preparation. The validation launcher defaults to stages 0–2
and rejects higher stages before allocating GPUs. Experimental ZeRO-3 encoder
and model changes, checkpoint wrapper and dedicated tests were removed; the
encoder/model files match their pre-investigation versions.

Passed verification:

- 44 existing regressions, including packed attention output/loss/gradient
  parity, document isolation, data loading and memory-label handling.
- Eight-rank BF16 ZeRO-1/2, ZeRO-2 CPU optimizer offload, and BF16 without ZeRO.
- Analytic SGD checks of exact accumulation scaling under ZeRO-1/2.
- Eight-rank FP16 ZeRO-2 rejection and recovery with dynamic loss scaling.
- Pretrained Qwen3-4B-Instruct-2507 decoder plus Qwen3-Embedding-0.6B encoder:
  16 microbatches and 8 optimizer updates per rank across real 16k/32k packs,
  with BF16, packed FlashAttention and gradient checkpointing under ZeRO-2.
  All eight ranks passed finite-loss/gradient, memory optimizer-state and exact
  data-loader resume checks. Peak allocated memory was 35.64–46.54 GiB per GPU.

The distributed tests include initially unused components, unused components
after momentum exists, memory on only one rank/early microbatch, NaN loss with
finite gradients, infinite gradients, and recovery. Master weights and optimizer
state were compared exactly on rejected or unused-component updates.

After removing the ZeRO-3 work, syntax/whitespace checks and stage-selection guards
were checked locally. Existing passing GPU evidence applies to the retained
ZeRO-2 behavior; no new GPU run was launched for this scope cleanup.

Validation pins Torch 2.8.0, Accelerate 1.10.1, DeepSpeed 0.17.5, Transformers
4.57.1 and FlashAttention 2.8.3. NVMe optimizer swapping and FP16 without ZeRO are
unsupported. Multi-node/RDMA is outside this single-node verification. The
separate Code-LLaVA checkout uses a different training implementation and is not
covered by these results.

Evidence is on Modal volume `lclm-stage3-data` under
`/data/stage3-build-20260906/deepspeed-ordering-20260910-v1/`, with local copies in
`_modal_run/deepspeed-ordering-final/`. V3 (app `ap-DAXJEy2SPQrFWhLzRvE8F0`)
contains the passing BF16/ZeRO-2 full-model results; V4 (app
`ap-l5ON9AMb8Nxi1eLmuMxw3u`) contains passing FP16/scaling results. Unnecessary
ZeRO-3 investigation logs and source snapshots remain archived as historical
records, including the cancelled V7 run; they are not a training prerequisite.

The data packs were not changed, and this correction did not launch a production
training run or upload datasets to Hugging Face.
