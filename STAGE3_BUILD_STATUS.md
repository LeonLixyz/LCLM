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

## Latest checkpoint — 2026-09-07 01:27 EDT

- Base packing and recovery are **complete, all 64 partitions each**. Combined
  accounting passes: 20,326,114 input rows = **20,286,582 packed rows** + 27,634
  overlength + 11,897 under 18 tokens + 1 processing rejection. The packed count
  includes 334,260 recovered legacy BPE-prefix rows without duplication.
  Baseline has 1,071,885 packs; recovery has 45,546. Do not restart these jobs.
- Native agents remain complete: **1,244,170 cleaned raw / 1,230,344 packed
  trajectories**, with the requested Nemotron search/tool_calling subsets and
  OpenThoughts. All native rows are uncompressed. See subset counts below.
- V5 pilot completed **480 attempts / 247 accepted by its verifier**, across
  all 15 sources. A format audit of every accepted trace passed: exact expand
  bodies, explicit saved task system and tools, no teacher system in training
  messages, all assistant calls supervised, observations masked, and every
  segment at least **642 actual Qwen tokens**. There are 92 multi-expansion
  trajectories (distinct segments >1). This is NOT semantic release approval.
- Manual inspection of two accepted examples per source found suspicious
  free-form false positives: CLAPNQ `rea3-3927e0f41235145cef2c2067` asks for the
  father of the convention but answers its president; BillSum
  `rea3-a28561e3e68c11ed1bf74dcd` claims the bill establishes an authority whereas
  the reference describes amendments involving an existing authority.
  **Do not write review-passed.json or start full generation yet.**
- A diagnostic Qwen-235B semantic re-review is running in
  **`ap-mrHiNmdGgFMahqjOvLKFyy`**, H200:8, covering all 96 accepted free-form
  pilot answers plus 19 PubMedQA answers. It separately checks question/evidence
  without the reference, then with the reference, and requires both judgments.
  This does not alter trajectories or approve release. Expected report:
  `full-20260906-v5-audit/semantic-review-question-first-every-claim-v1.json`.
  Inspect the two suspect-example outcomes and other rejected/kept samples.
  If useful, integrate the calibrated filter with a new versioned generation
  manifest/output before scaling; it is currently diagnostic only.
- Latest focused tests: **64 passed**, plus eight pinned real-tokenizer
  boundary cases (app `ap-Ek8OEjfTkuTlKyNMbra5tT`). Earlier full 152-test,
  actual-packed-artifact, NCCL/DDP and FSDP results below remain valid for the
  packing/training code. Final expansion integration still must be run.
- No new complete raw/packed HF release has been uploaded. Full expansion
  generation has not started. Earlier source-license/split exclusions remain.

Pilot audit files are on `lclm-stage3-data` under
`/data/stage3-agent/real-expansion/pilots/full-20260906-v5-audit/`:
`format-audit.json`, `review-samples.json`, and `review-errors.json`.

## Historical checkpoint — 2026-09-07 00:43 EDT

### Follow-up checkpoint — 2026-09-07 00:57 EDT

- Base resume is still running, now **62/64 partitions complete**. Do not
  relaunch it. Once all 64 exist, the progress audit automatically verifies
  combined baseline/recovery row accounting.
- New Qwen pilot **`ap-IAptVFhDjDVFJAQGxKVdrw`** is running over all 15 sources,
  writing `full-20260906-v5-audit`. Model startup succeeded and source reports
  are appearing. Full v5 generation has not started.
- Implemented final-answer harvesting: native assistant tool calls remain
  unchanged, their free-text reasoning is removed, and the terminal answer is
  extracted only from one unambiguous FINAL line. Trailing text/control markers
  are rejected. Tool results, documents and task instructions are unchanged.
  Correctness is rechecked on these actual saved training messages. A concise
  document-task system prompt is now explicitly saved, separate from the
  teacher-only generation system prompt. Judge failures preserve diagnostic
  trajectories rather than dropping their messages.
- Fixed the runtime loader to discover the published `data/*/part-*/*.parquet`
  layout as well as legacy flat folders. It does not recursively ingest raw,
  in-progress or quarantine folders, and rejects ambiguous mixed layouts.
- Validation app **`ap-IjBVOE1vcVbDTce9J1oajn`** passed **152 tests**, the
  two-rank NCCL/DDP and FSDP smoke runs, and 18 actual packed-sample checks across
  base/native/recovery. Runtime expansion preserved labeled-token counts and
  masked memory spans; sampled lengths stayed <=32768; both loader ranks had
  equal lengths. Reports are in `validation/report.json` and
  `validation/packed-artifact-audit.json`. These are small-model and sample
  validations, not a production-scale or complete artifact scan. Re-run final
  integration including expansion packs once those exist.
- V5 early source outcomes remain selective: MAUD 13/32, FinQA 3/32,
  PubMedQA 19/32, ContractNLI 24/32, CLAPNQ 23/32 accepted. CLAPNQ still has six
  JSON failures; inspect preserved rejected traces/error fields before deciding
  whether further parser or judge prompting work is needed. Do not relax
  correctness rules to inflate acceptance.

The older checkpoint details below are historical where superseded above.

Native cleaning and lossless JSON transport export are complete: **1,244,170
trajectories**. All 64 native packing partitions completed successfully in app
`ap-TW2Yn7nLWNb2qnOWkWMRkp`: **1,230,344 packed trajectories in 147,706 packs**,
all uncompressed, with 1,273,215,572 labeled tokens. The other 13,826 raw
trajectories exceed the 32,768-token packing limit; no other processing exclusions.

| Native source/subset | Retained raw trajectories |
| --- | ---: |
| Nemotron v1 tool_calling | 199,207 |
| Nemotron v1 interactive_agent | 15,443 |
| Nemotron v2 tool_calling | 696,224 |
| Nemotron v2 search | 5,953 |
| Nemotron v2 interactive_agent | 278,592 |
| OpenThoughts main + terminal | 48,751 |

V2 retains 2,640,245 tool calls; v1 retains 382,594; OpenThoughts retains 525,121.
Native JSONL is in `agents-qwen-v2`, and Arrow-safe JSON-string transport is in
`agents-transport`, both under `/data/stage3-build-20260906`.

The original base packing app stopped after 49/64 partitions. Fifteen unfinished
partitions had partial outputs; two sampled files had missing Parquet footers.
All partial folders were inspected and **moved, not deleted**, to
`quarantine-base-partials/`; exact paths/metadata are in
`base-resume-inspection.json`. Completed outputs were untouched.
Baseline resume app **`ap-PTjkWElexk2FIJc34t3uC5`** is running only unfinished
partitions, launched with `LCLM_PACK_JOB=base-resume`.

An additional legacy SFT BPE/newline bug rejected valid reasoning responses.
The recovery pass is complete: all 20,326,114 input rows scanned, 527,286
whitespace-prefix candidates, **334,260 recovered/packed rows**, 977 overlength,
192,049 not recoverable by this fix. Outputs:
`packed-base-prefix-recovery/part-###/all_samples` and reports. The baseline retry
explicitly retains the old prefix eligibility rule; the separate recovery pass
includes only mismatches, so those rows cannot be duplicated. Publication now
requires all 64 recovery reports and combined per-partition accounting checks.

Latest focused validation: **48 tests passed** on Modal, plus eight real pinned
Qwen tokenizer boundary cases. Report `validation/prefix-recovery-tests.json`.
This supplements, not replaces, the earlier 118-test and NCCL/FSDP GPU evidence.

### Expansion generation remains gated

All 15 implemented source task builders completed: **375,305 tasks**, including
10,000 synthetic tasks. This is a prompt-task count, NOT accepted training traces.
Task input root remains `/data/stage3-agent/real-expansion/pilots/full-20260906-v3`.

The old full teacher app `ap-28zO9uYzlKe2iyeE8NvHPN` was STOPPED after roughly
3,000 MAUD rejects and zero accepted rows. MAUD lacked explicit label choices
(e.g. teacher answered cash while the label was All Cash). These are diagnostics.

Source-stratified pilot app `ap-ZiiXioSeQVo9F4zwGqdfmw` completed 32 tasks per
source, **480 attempted / 224 accepted by the then-current verifier**. Outputs
are in `full-20260906-v4-audit`; do not release these as final data. Pilot review
found issues requiring fixes before a new pilot:

- MAUD label ontology now appears identically in teacher and saved task inputs.
- ACORD qrels are 0–4, not 1–5 stars. Its v4 pilot had zero accepted rows. The
  normalization now fixes the question without changing the source gold label.
  Evidence: https://huggingface.co/datasets/theatticusproject/acord
- Judge JSON now accepts complete fenced JSON, but still rejects malformed or
  non-boolean votes. The v4 pilot had 26 JSON decode failures; not all necessarily
  come from code fences, so inspect v5 outcomes.
- Exact-answer normalization now preserves signs, decimal punctuation and
  percentage units. Previously negative and positive answers could compare equal.
- Financial strict comparisons reject arithmetic errors, rounding differences,
  and scale differences. Do not loosen them blindly to raise acceptance.
- V4 accepted teacher responses sometimes contain explanatory computation before
  FINAL. This is fixed by the tested v5 harvesting pass described above.
- Support currently means all chunks of the source document, not minimal
  evidence. This is conservative but may discard otherwise grounded trajectories.
- LLM reference/evidence judgment can accept noisy references. It is heuristic;
  manually inspect accepted free-form samples and record limitations.

Code targets the next pilot at **`full-20260906-v5-audit`** and full output at
**`full-20260906-v5`**, separate from v4 diagnostics. The v5 pilot is now running;
full generation has not been launched. Full generation requires the pilot report and an
explicit `review-passed.json` with `approved: true`, written only after review.
Do not restart full generation merely because a pilot job completed.

## Commands and continuation

Commands use `/Users/leonli66/miniconda3/envs/modal/bin/modal` from this checkout.

- Read bounded progress with `modal run -m data.audit_stage3_progress_modal`.
- Run focused + real-tokenizer tests with the same command plus `--tests`.
- Inspect selected pilot examples with `--samples --version v4 --sources finqa,clapnq`.
- Teacher model stays `Qwen/Qwen3-235B-A22B-Instruct-2507`, revision
  `ac9c66cc9b46af7306746a9250f23d47083d689e`, H200:8.

A same-task follow-up (`finish-stage-3-dataset-release`) is active every 30
minutes to continue this work. It must stop after release completion. Keep the
computer on and the app running for local continuation; Modal jobs run remotely.

Next pipeline commands, after their input completion gates are satisfied:

```sh
modal run --detach data/generate_real_expansion_modal.py --audit-full
# Only after reviewing the new pilot:
modal run --detach data/generate_real_expansion_modal.py --full
modal run --detach -m data.export_stage3_agent_transport_modal --kind expansion
LCLM_PACK_JOB=expansion modal run --detach data/stage3_pack_release_modal.py --kind expansion
modal run --detach -m data.publish_stage3_release_modal
```

The publisher intentionally requires all 64 partitions for each component,
including separate base recovery, and
a final `/data/stage3-build-20260906/release-review.json` with `approved: true`.
Only write that review record after inspecting source provenance, generated
trajectories, packing counts, skips, and integration results. It is an internal
verification gate, not a claim that the user reviewed the data. Proposed HF IDs:
`leonli66/stage3-final-mixture-cot50-native-agent-v2` and
`leonli66/stage3-final-mixture-cot50-native-agent-v2-packed-cs16-32k`.
They have not been created or uploaded yet.

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
