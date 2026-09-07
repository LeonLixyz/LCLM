# Stage-3 rebuild — 2026-09-06

Work branch: `codex/stage3-full-20260906` on `LeonLixyz/LCLM`.
This is an in-progress build, **not a completed dataset release**.

## Latest source requirement

- `nvidia/Nemotron-SFT-Agentic-v2`: search, tool_calling, interactive_agent.
  Pinned revision `7c804833427f633ccd53b582dbf02525fd680f78`.
- `nvidia/Nemotron-Agentic-v1`: tool_calling, interactive_agent.
  Pinned revision `650d590978ca35c8f1ecea2faf136e5fac421b62`.
- `open-thoughts/OpenThoughts-Agent-SFT-100K`:
  pinned revision `45fb28fcc38d352133cb28a1c8a43a2f14fea97b`.

All three snapshots are downloaded to the `lclm-stage3-data` Modal volume,
under `/data/stage3-build-20260906/sources`.

Imported native agents have no compression. Assistant reasoning fields and
leading think/analysis blocks are removed. Tool arguments are decoded from
transport JSON without changing their values; structured tool results are
serialized losslessly. Missing result IDs are inferred only from an unambiguous
pending call. Conflicting schemas/ambiguous results are quarantined, not guessed.
OpenThoughts terminal JSON analysis/plan fields are discarded; commands become
native `terminal_step` calls. All assistant turns/calls are supervised; all
system/user/tool-result content is masked.

The v2 search harness's visible Thought/Search/Observation instructions are
removed, while retaining its task and final-answer contract.

## Verified

- Real tokenizer and packing-worker audit: first 50 rows of each of five
  Nemotron subsets. All 50 passed for v1 interactive and all three v2 subsets.
  v1 tool_calling: 38 passed; 4 conflicting schemas and 8 ambiguous result
  mappings quarantined. These are **sample**, not full-dataset, counts.
- Tokenizer: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Its chat template is identical to the pinned 235B-Instruct teacher's template:
  SHA256 `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`.
- Fixed real BPE prefix/newline alignment using full-render character offsets.
  Independent ChatML spans check the assistant-only loss mask.
- 118 tests passed on Modal, plus two-rank NCCL DDP and nested FSDP tests with
  real small Qwen encoder/decoder components and the LCLM adapter. Patterns:
  both ranks compressed, rank 0 only, neither, rank 1 only, neither, both.
  Tests include unequal encoder minibatch counts and unchanged encoder/adapter
  weights on globally uncompressed optimizer steps.
- These checks do not constitute a full-sized production training validation.

Audit artifacts: `/data/stage3-build-20260906/qwen-format-audit/` and
`/data/stage3-build-20260906/validation/` on the data volume.

## Active work

Commands use `/Users/leonli66/miniconda3/envs/modal/bin/modal` from this checkout.

- Native cleaning: `data/stage3_full_modal.py --action clean-agents`,
  app `ap-D9EQ445kdWpi65wyiyc9LG`; outputs `agents-qwen-v2` under the build root.
- Base packing: `data/stage3_pack_release_modal.py`,
  app `ap-vFCwnWaCX5lImxkgXdTUqF`; 64 partitions, at most 16 containers.
  Input `/data/stage3-final-mixture-cot50-v1`: 20,326,114 rows, including the
  previously completed 50/50 reasoning rewrite. Output
  `/data/stage3-build-20260906/packed-base-cs16-32768`.
- Expansion task build v3: `-m data.build_full_expansion_tasks_modal`,
  app `ap-UtHpSexNVfoX1VCij466Hs`.
- Qwen teacher: `data/generate_real_expansion_modal.py --full`,
  app `ap-1lg6UYUJIPBc62nK1hP0YB`. Teacher
  `Qwen/Qwen3-235B-A22B-Instruct-2507`, revision
  `ac9c66cc9b46af7306746a9250f23d47083d689e`, H200:8.
  Expansion task/trace root:
  `/data/stage3-agent/real-expansion/pilots/full-20260906-v3`.

Only v3 task builds are eligible for generation. Earlier v2 task artifacts are
unreleased diagnostics: short-document distractors and ambiguous document
identity were corrected in v3. CUAD is quarantined until its official training
split is resolved. FAv2's identifier is still requested from the user.
FinanceBench needs a non-commercial-license decision. TechQA, SWE corpora,
Natural Questions and unique LegalBench additions still need adapter/acquisition
work; do not claim that the full source registry has been generated.

## Remaining release gates

1. Inspect job outcomes and resumability. Never run duplicate expensive jobs.
2. Inspect real generated trajectories and source-level acceptance/rejections.
   Reference/evidence judging for free-form answers is an LLM heuristic, not
   a mathematical correctness guarantee. Teacher/judge prompts are not training
   messages. Verify source split and license provenance before inclusion.
3. Finish native-agent and expansion packing, with an explicit manifest of
   every raw row, skipped row/reason, packed sample, and packed batch.
4. Keep all_samples outputs, including partial packs and zero-memory samples.
   The new pack length is 32768; audit/report overlength exclusions explicitly.
   Dynamic packing pretokenizes the decoder, but intentionally stores raw
   memory strings for encoder tokenization at runtime.
5. Build/publish a new raw HF mixture and a new packed HF mixture with source
   configs/attributions, revision pins, counts, and consumer documentation.
   Arbitrary tool schemas need lossless JSON transport rather than Arrow
   schema inference. Preserve downloadable native JSON traces as well.
   Never upload `state.json` (`ignore_patterns=["state.json"]`).
6. Run final packed-data/model integration checks and push remaining code.
   No complete raw or packed release has been published by this build yet.
