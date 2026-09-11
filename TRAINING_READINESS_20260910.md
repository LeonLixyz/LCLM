# Stage-3 shuffle and training readiness follow-up

Requested September 10: verify that agents are included, fully shuffle completed
sequences, and exercise training/NCCL. **Both global shuffles, source/rank audits,
final loader checks and GPU verification passed.** Verified at 19:36 UTC
(15:36 Eastern) on September 10, 2026.

The completed source packs contain 2,278,921 sequences at 16k and 1,277,010 at
32k, including native and expansion agents. Their original loader shuffles files
and rows within files. This permits expansion-only stretches because expansion
shards are separate. An exact global permutation has been materialized into a
new copy; each packed sequence and every serialized example remain intact.

Completed globally shuffled output:

`/data/stage3-build-20260906/global-shuffle-20260910-v2/packed-cs16-{16384,32768}`

Shuffle app: `ap-LSLXqRndywDPHuuTVrCGFj`. The v1 attempt hit Arrow's 2 GiB binary
offset limit and was stopped. V2 uses `large_binary` offsets and a fresh output
root. The scatter/gather permutation and payload-preservation test passed on
Modal. All positions, identities and serialized payloads are verified during
materialization. The final loader check (`ap-3fdbEOuYhR2AzC0zwfI4IM`) passed on
the new LargeBinary shards at both lengths, including compressed lengths,
supervised-token counts, memory/tag label masks and all four source kinds. The
independent source-certificate and 8-rank mixture audit
(`ap-YQRLv7vdVLPIYX9qRNvlId`) passed for both versions. All 4,493 original 16k
files and 2,535 original 32k files match their bound audits; all eight ranks
receive native agents, expanded agents, reasoning and other data in their first
4,096 sequences. Expanded-agent counts per sampled rank are 39–55 at 16k and
21–48 at 32k. The packs retain all 2,278,921 / 1,277,010 sequences. Default
eight-rank training drops only an epoch remainder of 1 / 2 sequences, respectively.

Every final shuffle position was checked against the exact permutation. The
source and output payload digests match, and every output file was reopened and
verified. Final paths, manifest hashes, counts and pinned tokenizers are in
[the data handoff](STAGE3_PACKED_DATA_20260910.md). Exact final manifests, completion
and independent audit are saved under `_modal_run/global-shuffle-final-20260910/`.

Trainer corrections in this working copy:

- All ranks enter the non-finite loss/gradient reduction on every microbatch.
  Previously, only ranks with non-finite local loss entered the reduction,
  which could mismatch NCCL collectives. Finite loss with non-finite gradients
  is now detected as well; collective errors are no longer swallowed.
- Training logs use the configured compression ratio rather than accessing
  `encoder` directly through a DDP wrapper.

Validation uses pinned Torch 2.8.0, Transformers 4.57.1, Accelerate 1.10.1,
FlashAttention 2.8.3, TorchData 0.11.0, and Liger 0.6.2 on Modal H200:8.
The bounded tests cover 8-rank DDP/FSDP, asymmetric and absent memory, injected
single-rank non-finite loss/gradients, packed attention parity and document
isolation. A separate production-trainer smoke uses the pinned pretrained
Qwen3-4B-Instruct-2507 decoder and Qwen3-Embedding-0.6B encoder, FSDP full sharding,
BF16, packed FlashAttention, gradient checkpointing, gradient accumulation=2,
and real 16k/32k packed sequences from all categories. No production checkpoint
is changed and no full training run or evaluation benchmark is launched.

Real-data/model preparation passed. The final full-model run
`ap-4lJDrlGnxcA7g6OpUG5T3x` exited successfully: 16 microbatches and 8 optimizer
steps per rank across both lengths, finite losses and gradients, exact loader
resume, compression optimizer state unchanged on all-uncompressed steps, and
advancing on compressed steps. Peak allocated memory ranged from 34.16 to
46.62 GiB per GPU. Earlier test-harness setup/cleanup errors were corrected.

The regression/NCCL app `ap-C5Q5QxaZ5J715fyo8ymECL` passed all three checks:
44 regression tests (including packed FlashAttention output/loss/gradient parity
and document isolation), 8-rank DDP, and 8-rank FSDP. Both distributed modes
passed asymmetric/absent memory patterns and injected single-rank non-finite
loss/gradient cases. Reports and all eight full-model rank records are saved
locally in `_modal_run/training-readiness-final/`.

The follow-up DeepSpeed correction separates backward from the optimizer update,
checks real ZeRO gradient buffers first, and prevents compression-group momentum
or weight decay from advancing on globally uncompressed windows. Rejected
windows also skip the scheduler and recover cleanly. Eight-rank BF16/FP16 and
CPU-offload tests passed, along with full-model ZeRO-2 at both lengths.
**ZeRO-2 is the intended setup and maximum supported ZeRO stage.** The user
confirmed this scope, so the experimental ZeRO-3 encoder/model changes and tests
were removed. Training and validation entry points reject higher ZeRO stages.
The earlier ZeRO-3 investigation is not a blocker for this build. See
[the DeepSpeed report](DEEPSPEED_ORDERING_20260910.md)
for the exact matrix and current result. The ordering issue originated in
[Accelerate's DeepSpeed wrapper](https://github.com/huggingface/accelerate/blob/v1.10.1/src/accelerate/utils/deepspeed.py#L242),
which steps the optimizer inside `backward` before the trainer's later cleanup.

The exact intended production checkpoint, configuration and GPU topology were
requested from the user; no answer has arrived yet. These tests validate the
specified single-node configurations. They do not establish multi-node/RDMA
correctness or guarantee that a long job cannot encounter hardware or numerical
failures.

The follow-up launcher review found and fixed a multi-node default: previously
`scripts/train.sh` defaulted `WORLD_SIZE` to the local GPU count even when
`NUM_MACHINES` exceeded one. It now defaults to nodes times GPUs per node and
rejects invalid counts, ranks, uneven process allocation and loopback master
addresses for multi-node launches. Eight CPU-only checks exercised its actual
argument construction and early errors with stubbed Python/Accelerate commands;
shell syntax and whitespace checks passed. This is launcher validation, not a
multi-node GPU/NCCL test. ZeRO-2 normally partitions across the full data-parallel
group, which spans nodes in this setup; node-local sharding is not implied by
choosing stage 2. A real multi-node run is still needed to verify transport,
distributed data assignment and full checkpoint save/restore on that topology.

Root `AGENTS.md` now records the intended ZeRO-2 scope, final data paths, compatible
settings, launch instructions, optimizer/data invariants and verification limits.
The fixes and instructions are present in the working tree; they have not been
committed or pushed as part of this follow-up.

The tested code and fixes are in this `LCLM-stage3-build-20260906` clone, on
`codex/stage3-full-20260906`. The separate `Code-LLaVA` checkout has a different
training implementation (including different model modules and data loader).
These validation results do not apply to that checkout, and no files there were
changed by this follow-up.

Relevant local launch logs and reports are under `_modal_run/readiness-*` and
`_modal_run/global-shuffle-v2-launch.log`. Cloud reports are under
`/data/stage3-build-20260906/readiness-20260910-v1/` on volume `lclm-stage3-data`.

Temporary shuffle cleanup completed (`ap-c4ox3UU8Ks5TGbJlYly8Ya`): 6,976
intermediate scatter files removed, freeing 352,312,600,031 bytes (352.3 GB).
Original packs, final shuffled packs, plans, source digests and audit reports
were preserved. The cleanup record is included in the final local evidence.
