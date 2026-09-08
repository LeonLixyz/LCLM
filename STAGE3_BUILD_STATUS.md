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

## Latest checkpoint — 2026-09-08 04:33 EDT — V3 calibrated; full held-source review submitted

- Main **ap-NT9cm8Mc78qaGB9euDnK99** remains active and unchanged: ACORD
  **34,300 attempts / 29,372 accepted**. Corrected MultiDoc2Dial
  **ap-c9DCBmAQMHHS4aJt9MASov** remains active: **19,400 / 14,573** of 21,451.
  Eleven main sources are complete (100,833 attempted / 46,443 accepted,
  including held originals). Corrected MD replaces old MD, never additive.
- V3 diagnostic **ap-DzKVvk81JbxcitfinEMjn2** completed: **58 reviewed / 31
  kept / 3 JSON errors**, all four controls passed. Decisions SHA
  **b5156fed85ad65752eab0d513afcfb2c85e85126f2341eeef40576b4fb68c10a**.
  CPU sampler **ap-xBgIl7VhmKp4b92zwh4dXo** wrote ten full evidence/answer
  examples in V3 `manual-samples.json`, including newly kept answers versus V2.
- Reviewed controls and additional source/outcome samples. Newly kept apple
  abstention, We Are the World soloists and Waikato endpoints are grounded.
  Date-quotation errors remain excluded even where candidate answers are right;
  hamburger details from dialogue history are excluded under the source-only
  evidence policy. Do not describe all exclusions as false answers. Same-model
  judgment and limited manual samples do not establish a source-wide error rate.
- Hash-bound internal scale approval saved in
  `data/reviews/stage3-grounding-calibration-v3-review.json` and uploaded to V3
  `manual-review.json`. It permits reviewing **7,884 FaithDial + 622 CLAPNQ =
  8,506 candidates**, NOT filtering or release. Original source holds remain.
- Implemented fail-closed scale gate and review-only full job in
  `data/grounding_full_review_gate.py` and `data/full_grounding_review_modal.py`.
  **148 regression tests passed** on **ap-7sYVGuS9cYayG8x6G4UZ4X**.
  CPU preflight **ap-h6VnP0hzQFqrtq2XfWNs4m** passed all 8,506 rows and source
  hashes against format audits, binding calibration decisions, manual approval
  and protocol SHA **d633cf782271f5d4ff68705229570e6ba8ad2186ca27f878a8baa5c7dbb63563**.
- Deployed separate **lclm-held-sources-grounding-review-v3**, verified idle,
  then submitted exactly once: **ap-MRs5K5BukSZUvrSm1h2dlv**, call
  **fc-01M202D1QK2YP27ZN3AATPG608**. H200:8, max one container, 12-hour cap,
  concurrency 8, pinned Qwen235B-Instruct/no-thinking. Do not duplicate/redeploy
  while active. Outputs `/data/stage3-build-20260906/full-grounding-review-v3/`;
  decisions/progress checkpoint every 50 rows. Errors stay explicit, originals
  unchanged, no export/filter/publication side effect. Submission is confirmed;
  inference startup/completion still needs checking.
- Next: monitor this review and the two generators. Audit and manually review
  corrected MD once complete; continue other source manual reviews. After full
  held-source review, implement separate retained-row materialization with
  explicit exclusions and fresh audits/manual review before changing release
  selection. No expansion export/packing/HF release yet. Original base-mixture
  provenance/terms remain unresolved and block publication, not generation.

## Historical checkpoint — 2026-09-08 03:58 EDT — V2 controls passed; extra manual review and V3 guard fix

- Full generators still active and unchanged. Main **ap-NT9cm8Mc78qaGB9euDnK99**
  ACORD checkpoint **27,700 attempts / 23,721 accepted**; corrected
  **ap-c9DCBmAQMHHS4aJt9MASov** MultiDoc2Dial **15,500 / 11,597**. Eleven main
  sources remain complete. Never duplicate/redeploy either active generator.
- Diagnostic V2 **ap-xMrqsZfuSwBgQEdt15djGw** completed and is idle: **58 rows /
  29 kept / 4 JSON errors**. **All four calibration controls passed.** Read all
  four decisions; known unsupported answers still reject and both grounded
  positive controls now keep. No source-wide or release approval was inferred.
  V2 protocol SHA **d4761fef67921ca8bc01d2e44b0eb79699d682c34ee67d77a74f0e0f8cb01ceb**.
- Added and ran CPU sampler **ap-eW3giQAhr4Eq8E5aVKg7Uj** via
  `data.sample_grounding_calibration_modal`. It selects two non-control examples
  per source/outcome stratum (8 total), preserves all primary evidence and
  decisions, and records error rows. Artifact in V2 `manual-samples.json`;
  decisions SHA **46fc01e69decfafd7e86ddd85ffc06634722eb6a654ecee9cb62a7e82e88935f**.
  Sampler is explicitly pinned to V2 even as current diagnostic runner advances.
- Read all eight full evidence/answer/decision views. All four additional keeps
  grounded: Catalonia/Barcelona, Billa Hindi dubbing, sunset definition, braces
  applications. Two rejects appropriate: claimed freeway range excludes Hawaii's
  lower source values; Boxer answer adds skull/jaw/prey claims absent from source.
  Two are valid answers rejected conservatively: Phantom Menace dates are right
  but the judge miscopies a source year (1998 to 1988); apple-color abstention
  is accurate but `does not specify` was absent from the mechanical guard regex.
  Do NOT fuzzy-correct dates; invalid judge quotations remain rejected/diagnostic.
  These stratified samples are not a source-wide error-rate estimate.
- Manual review saved in `data/reviews/stage3-grounding-calibration-v2-review.json`
  and uploaded to V2 `manual-review.json`. Full-source/release approval remains
  false pending the narrow guard correction's revalidation.
- V3 changes only the missing abstention phrase plus version identifiers; judge
  prompts, evidence scope and quote semantics are unchanged. **131 tests passed**
  in **ap-VThLbktvzQ0DBhSrHmL9Aw**. Tests include appropriate `does not specify`,
  rejecting its mixed positive assertion variant, and refusing a changed date in
  a quote. The 58-row preflight and old-vote mechanical replay also passed;
  the apple sentence now clears its guard (whole-answer fit not yet confirmed).
- Submitted one separate bounded diagnostic V3 after verifying it was idle:
  app **ap-DzKVvk81JbxcitfinEMjn2**, deployment **lclm-grounding-claims-pilot-v3**,
  call **fc-01M200AP2WR3K3ME4PWW97CCQB**. Do not duplicate/redeploy while active.
  Same 58 rows, pinned Qwen235B/no-thinking, H200:8, max one container, two-hour
  cap; outputs `/data/stage3-build-20260906/grounding-calibration-claims-v3/`.
  Current runner targets V3; V1/V2 deployments and artifacts remain intact/idle.
- Next: inspect V3 controls and recovered apple abstention; if its results and
  additional manual sample checks support scaling, write explicit hash-bound
  approval for a conservative full re-review of the **8,506 previously accepted
  FaithDial+CLAPNQ candidates**, not release approval. Errors/invalid citations
  must remain counted quarantine reasons. Full-source re-review/materialization
  code still needs implementation and tests; preserve originals and reflect
  exclusions in selected-source accounting before any release gate can clear.
  Continue other source manual reviews and corrected MultiDoc2Dial completion/
  audit. No expansion export/packing/HF release; base provenance gates remain.

## Historical checkpoint — 2026-09-08 03:27 EDT — diagnostic v1 failed calibration; v2 submitted

- Main **ap-NT9cm8Mc78qaGB9euDnK99** still active: ACORD checkpoint **22,900
  attempts / 19,276 accepted**. Corrected full **ap-c9DCBmAQMHHS4aJt9MASov**
  still active: MultiDoc2Dial **12,600 / 9,357**. Eleven main sources complete.
  Do not duplicate/redeploy either generator.
- Diagnostic V1 **ap-D6B6a2sq9oR6r05ny7F7rz** completed and scaled to zero tasks:
  **58 reviewed / 25 kept / 1 JSON error**. **Calibration FAILED**: both known
  unsupported/ambiguous controls were correctly rejected, but both grounded
  controls were also rejected. Do not scale V1 or integrate its 25 keeps.
- Read all four saved control decisions. FaithDial hair-movement control passed
  every sentence (including both pure abstentions), then failed whole-answer fit.
  CLAPNQ Little Lion Man control had supported=true for each sentence, but model
  quotes detokenized source punctuation/contractions, failing strict whitespace-
  only matches. The negative legal control was rejected, yet some individual
  supporting quotes were unrelated states: exact quotation presence is not
  entailment. One separate row `rea3-147e05e976dc2d3250eb1477` had malformed JSON
  (Extra data), kept as an error. Manual failure review saved in
  `data/reviews/stage3-grounding-calibration-v1-review.json` and uploaded to V1
  `manual-review.json`. Its protocol SHA is
  **481ffed1317eb6bde6284e157991fcb09a2341f1e83094a3a0989bdb58b83f6d**.
- Protocol V2 conservatively normalizes source tokenization spacing around
  punctuation and contractions for quote matching; it does not delete semantic
  characters or merge ordinary words. Describe checks as detokenized quote
  presence, NOT byte-exact quoting. Whole-answer prompt clarifies current-turn
  justified abstention and returns a typed issue category; inconsistent booleans/
  categories fail closed. No references or manual labels enter judge prompts.
- **128 tests passed** in **ap-gfSE5Tk5euNUHp7hXodGBO** and again in
  **ap-saxoQDm1kjNpHEmdBYFGPW**. Second run replayed saved V1 sentence votes on
  all real candidates without model calls: five rows' quotation checks recover,
  including the known CLAPNQ positive; this is not a final-answer/model pass.
  Both runs preflighted the pinned 58-row input. Known V1 failures remain held.
- Deployed separate **lclm-grounding-claims-pilot-v2**, app
  **ap-xMrqsZfuSwBgQEdt15djGw**; verified idle then submitted once:
  **fc-01M1ZYJEVRMMTCA0T0CMZG15QJ**. Do not duplicate/redeploy while active.
  Same bounded 58 rows, H200:8/max one container/two-hour limit, pinned Qwen235B,
  no thinking and eight concurrent reviews. Output is separately versioned:
  `/data/stage3-build-20260906/grounding-calibration-claims-v2/`.
  V1 cloud deployment and outputs are untouched; Git history retains V1 code.
  Current `data.run_grounding_calibration_modal` addresses V2 only.
- V2 submission confirmed; completion still needs checking. Next inspect controls
  and additional non-control kept/rejected cases. Passing tuned controls alone is
  insufficient for scale. Neither source hold is lifted. No source-wide review,
  export, packing or HF publication launched. Continue remaining source manual
  reviews and full-generation monitoring, then corrected full-source audits.
  Base provenance and final packing/GPU/release gates remain unresolved.

## Historical checkpoint — 2026-09-08 02:58 EDT — claim-grounding diagnostic submitted

- Existing full jobs unchanged: main **ap-NT9cm8Mc78qaGB9euDnK99** ACORD latest
  inspected checkpoint **18,100 attempts / 15,073 accepted**; corrected full
  **ap-c9DCBmAQMHHS4aJt9MASov** MultiDoc2Dial **9,900 / 7,278**. Both active
  with one task each. Do not duplicate/redeploy them. Eleven main sources complete.
- Added pure `data/grounding_claim_review.py`: identity-scoped primary expanded
  evidence only, excluding RELATED SOURCE padding and distractor expansions;
  lossless sentence coverage, exact segment-specific evidence quotes for factual
  claims, typed fail-closed judge schema and explicit pure-abstention handling.
  Every sentence must pass before a fresh whole-answer/current-question fit vote.
  No reference answer, manual labels or control expectations are sent to judges;
  input training rows are never modified. Quotes can be mechanically checked;
  entailment/absence and sentence classification still remain model heuristics.
- **118 CPU tests passed** in **ap-nNhBgNXobrRusbN026ZRzq**, including exact quote
  scope, invented/distractor quotes, mixed-claim abstention bypass, legitimate
  missing-information lists, malformed votes, full sentence coverage and teacher/
  control separation. Preflight also parsed all **58 real calibration rows** and
  verified their pinned bytes, membership, controls and primary evidence.
  Input manifest SHA **a0cfe5bb46ed252905baa4a9bf2e6135ebf0635a68f5b0e3a3aec663b6105f6b**.
  Earlier iteration 117 tests passed in **ap-2BqBXuvkriTJcIW88Fdz09**.
- Deployed **lclm-grounding-claims-pilot-v1**, app
  **ap-D6B6a2sq9oR6r05ny7F7rz**, then submitted exactly once with
  **fc-01M1ZWX8PHPJJPG1VV4ZCDQWYQ**. New module
  `data/run_grounding_calibration_modal.py`; **do not duplicate/redeploy while active**.
  This is a separate **58-row diagnostic**, H200:8, max one container, two-hour
  function bound with server startup bounded to 90 minutes; server terminated
  in finally, scale-down 60 seconds. Same pinned Qwen235B-Instruct, no thinking,
  vLLM 0.21.0, temperature 0, 1,536 judge output tokens, concurrency eight.
- Output `/data/stage3-build-20260906/grounding-calibration-claims-v1/`:
  `manifest.json`, per-row fsynced/committed `decisions.jsonl`, final `report.json`.
  Resume is hash/version-bound and skips already judged IDs, including errors.
  Final control scoring occurs only after decisions, outside judge inputs.
  No full-source operation is exposed; results always retain
  `approved_for_full_source_review: false` and `approved_for_release: false`.
  Submission is confirmed; model/inference completion still needs checking.
- Next: inspect diagnostic report plus all four control decisions and sampled
  keep/reject cases against source passages. It must reject both known failures,
  retain both grounded controls and receive manual review before any broad source
  re-review. Failed calibration stays diagnostic; use a new version for protocol
  changes. FaithDial and CLAPNQ release holds remain. Continue other completed
  source manual reviews and existing full-generation monitoring. Corrected full
  MultiDoc2Dial still needs its own completed-source audit/review. No export,
  expansion packing, final GPU validation or HF release yet; base provenance gates
  also remain unresolved.

## Historical checkpoint — 2026-09-08 02:26 EDT — CLAPNQ held; 58-row grounding calibration prepared

- Both full generators remain active and unchanged, one task each. Main
  **ap-NT9cm8Mc78qaGB9euDnK99** ACORD checkpoint **13,200 attempts / 10,774 accepted**;
  corrected **ap-c9DCBmAQMHHS4aJt9MASov** MultiDoc2Dial **6,900 / 4,964**.
  Eleven main sources remain complete. Do not duplicate/redeploy either job.
- PubMedQA sample app **ap-75YNh0uET3KwsK7KrWwwSY** completed. Its source-wide
  audit has 233 calls for 233 accepted rows, no multi-expansion bucket. Reviewed
  the sole hash-selected single sample `rea3-8e26c2bdaf23db073c3afa14`: the
  Chingford abstract supports the concise yes answer; no invented clinical
  details. Recorded **sample pass**, not final release approval. Official
  train-only source-wide audit remains valid; base-mixture PubMed split status
  is a separate unresolved provenance issue.
- CLAPNQ sample app **ap-wgPgWlPvd7jzYhYOWcpUdt** completed. Multi sample
  `rea3-37018cffcd551f8a937b73c9` **fails**: a flattened state-law table loses
  Indiana column boundaries, yet answer asserts distinct adult/handheld/texting/
  hands-free rules. The supplied text does not unambiguously establish them;
  both automatic votes passed anyway. This review makes no claim about current
  Indiana law. Single `rea3-af40e67daccef4fc895e2a3a` (Little Lion Man meaning)
  is grounded and passes. Read all complete tool bodies and full task/answers.
  **CLAPNQ joins FaithDial on release hold**, enforced by the manual-review gate.
  Investigate original table structure if available, otherwise reject ambiguous
  table answers with explicit versioned exclusions; do not import current facts.
- Both reviews are in `data/reviews/stage3-full-{pubmedqa_labeled,clapnq}-review.json`
  and uploaded to the matching `{source}.review.json` on the manual sample volume
  path. All remain `approved_for_release: false`.
- Added CPU-only `data.prepare_grounding_calibration_modal`; ran successfully in
  **ap-4WQZ6jJBXAkq66PiSuFD4w**. Rehashed/scanned every **7,884 FaithDial + 622
  CLAPNQ accepted rows**, checked duplicates/accepted flags/review hashes. Saved
  **58 diagnostic candidates: 34 FaithDial + 24 CLAPNQ**, using up to 16 per
  single/multi source bucket plus all four manually checked positive/negative
  controls. CLAPNQ has only seven multi-expansion rows, all included.
  Output `/data/stage3-build-20260906/grounding-calibration-v1/`:
  `candidates.jsonl`, `manifest.json`; row file SHA
  **f1caf75d1e4716aaa8cb49c710b13cd1fa15726fdf6103405dbc4ae3dc0a2eac**.
  Original training rows retained byte-for-byte; original outputs not changed.
  Control expectations are separate metadata, forbidden in judge/training inputs.
- Calibration is **prepared, not judged or approved**; no extra GPU was started.
  Next implement a bounded pinned-Qwen235B/no-thinking per-claim evidence-quote
  diagnostic, then manually inspect control decisions and sampled keep/reject
  cases before scaling any source re-review. Existing BillSum claim checker is
  a starting point, but cannot be reused blindly: FaithDial includes legitimate
  abstention/missing-information answers, which lack positive evidence quotes.
  Handle abstention explicitly without admitting unsupported positive claims.
  Verbatim quote presence is mechanically checkable; entailment is still a
  same-model heuristic. Calibration must catch BOTH known failures before scale.
- No runtime training/packing code changed; previous 97 regression tests remain
  the latest suite. New preparer validated through its successful all-input CPU
  run. No HF release. Remaining completed-source manual reviews, corrected full
  source audit/review, source remediation, final packing/GPU and base provenance
  gates still need completion.

## Historical checkpoint — 2026-09-08 01:55 EDT — FaithDial semantic sample failed; release hold enforced

- Both full generators are still active, unchanged, one task each. Main
  **ap-NT9cm8Mc78qaGB9euDnK99** ACORD checkpoint **8,300 attempts / 6,521 accepted**;
  corrective **ap-c9DCBmAQMHHS4aJt9MASov** MultiDoc2Dial **4,300 / 3,162**.
  Do not redeploy/duplicate either. Eleven main sources remain complete.
- Prepared deterministic single/multi full-run samples for FaithDial in
  **ap-kyAX11fOEWMMERvuk2HGrX**, Watsonx in **ap-bfrxSe4CxXyfZ4YjopTClS**.
  Read task/reference/system/schema/calls, all complete tool observations and
  final answers. Evidence files are under the existing full manual sample root.
- **FaithDial is held from release.** Multi sample
  `rea3-485f8980a759552ce06d42ec` adds an unsupported age/background generalization
  about OCD, absent from all three expanded bodies, despite BOTH semantic votes
  accepting it. Required source 4171 contains only one risk-factor sentence;
  a distractor supplies the OCD definition. Single sample
  `rea3-5fdc65e84b8db045fbc2fdfe` correctly declines to invent a website's origin
  or reviews and passes. This is one failed example, not an all-row error rate.
  Do not discard the finding, silently edit the answer, or infer a source-wide
  pass from the existing all-row format audit.
- Both sampled FaithDial relevant passages are one sentence; their 512+ token
  segments are mostly unrelated RELATED SOURCE snippets. Record this as a
  short-evidence retrieval task limitation, not genuinely long source documents.
  Chronological sampled histories lack speaker labels but are not reversed.
  Follow-up: investigate unsupported extra claims, make a versioned correction/
  exclusion with exact accounting/provenance, then re-audit and re-review. Keep
  original generation files intact. Current selection deliberately cannot pass
  this source until that remediation is explicitly integrated and reviewed.
- Watsonx two samples **pass**: AI guardrails behavior and OpenScale definition
  are supported by the supplied pinned docs, including all verbose extra claims.
  FAQ trace expands five chunks although one contains the full answer. This
  remains inefficient expansion, not necessary five-hop reasoning. No assertion
  about today's product behavior is made.
- Reviews committed under `data/reviews/stage3-full-{faithdial,watsonx_docs_qa}-review.json`
  and uploaded to `/data/stage3-build-20260906/full-expansion-manual-review-samples/`
  as `{source}.review.json`. FaithDial status `sample_review_failed`, Watsonx
  `sample_review_passed`; both remain `approved_for_release: false`.
- Strengthened shared selection gate: each selected ORIGINAL source now requires
  a passing manual sample review, bound to the audit's accepted-file SHA and
  exactly the sampled task IDs/categories. Missing/failed/stale reviews block
  export and publication. Corrected MultiDoc2Dial retains its separate stricter
  full-source review gate. Reviews are included in the selection SHA, so changed
  reviews invalidate prior export/packing/final-review selection bindings.
- **97 CPU regression tests passed** on Modal **ap-a83S729gEHacBaLIMKbElG**,
  including six new passing/missing/failed/stale/wrong-ID review cases. No GPU
  validation, export, packing or HF publication launched this checkpoint.
  Next safe work: remaining completed-source manual reviews and a bounded
  FaithDial claim-grounding remediation plan/pilot; wait for existing corrected
  source full completion before its audit/review. Base source/license/split
  provenance and final packing/GPU/release gates still remain unresolved.

## Historical checkpoint — 2026-09-08 01:32 EDT — replacement-aware release code tested

- Both full generators remain active with one task each, unchanged:
  main **ap-NT9cm8Mc78qaGB9euDnK99**, corrective
  **ap-c9DCBmAQMHHS4aJt9MASov**. Do not duplicate or redeploy them.
  Inspected ACORD checkpoint **4,400 attempted / 3,490 accepted**; corrected
  MultiDoc2Dial **1,900 / 1,376**. Corrective inference is confirmed healthy.
  Main still has 11 completed sources; these are partial checkpoints, not
  release-approved totals. Old and corrected MultiDoc2Dial are never additive.
- Added `data/expansion_release_selection.py`, shared by export and publication.
  It requires the completed 15-source main run AND completed corrected full
  run, exact corrected input/parent/source/teacher provenance, its all-row audit,
  and a separate hash-bound **full-source** `release-review.json` with
  `approved_for_release: true`. Pilot approval cannot satisfy this gate.
  That review has **not** been granted or written.
- Selection takes 14 original sources and exactly one corrected MultiDoc2Dial,
  preserving old V6 files/reports as diagnostics. Selected counts are recomputed;
  per-source manifest/file hashes, correction review and original main counts
  remain in the release manifest. Export rejects duplicate task IDs; publication
  rehashes selected native JSONL before upload.
- Export, every expansion packing partition, and final release review now bind
  the same selection SHA. Stale existing exports/partitions fail closed. No export,
  packing, publication or new GPU generation was launched by this checkpoint.
- **91 regression tests passed** on Modal **ap-TmpQzg1Qft4frlE4qmfG9i**
  (earlier iteration 89 passed in **ap-LaWYJUTuQzlLPtKvFavD1L**). Includes real
  loader replacement accounting, pilot/missing/stale review rejection, input and
  parent hashes, per-source manifest mismatch, duplicate IDs, additive transport,
  changed upload bytes and release-entrypoint syntax. This is CPU release-logic
  validation, not final packed-data/GPU validation.
- Next: continue completed-source audits/manual sampling while generation runs;
  once the corrected full source finishes, run its own full audit and manual
  review under its versioned directory, then record a reviewed full-source
  `release-review.json` there. Once all main sources finish and selection gates
  pass, export/pack expansion, run final actual-expansion GPU validation, and
  record the selection SHA in the final build review. Original base-mixture
  source/terms/split provenance still blocks public release. No HF release yet.

## Historical checkpoint — 2026-09-08 00:58 EDT — 11 main sources complete; corrected source submitted

- Main generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, remains active. FaithDial completed:
  **18,357 attempts / 7,884 accepted**; Watsonx **45 / 31**. Main V6 has
  **11 completed sources / 100,833 attempts / 46,443 accepted**, including the
  old, release-held MultiDoc2Dial. ACORD is next, then BillSum, LexGLUE, synthetic.
- Full FaithDial + Watsonx format/label audits **passed** in
  **ap-iC0Wavg14heFJIGUZ9pxgg**: every **7,884 + 31 accepted rows**, zero
  failures, 8,145 + 73 calls, 100 + 12 multi-expansion rows, minima 723/705
  tokens. Reports in `/data/stage3-build-20260906/full-expansion-format-audit/`.
  All 11 completed V6 sources are format-audited (**46,443 rows**); that does
  not lift the old MultiDoc2Dial semantic/prompt release hold.
- Corrective MultiDoc2Dial pilot finished **32 attempts / 21 accepted**. Pilot
  GPU app **ap-BWrkXeukNZoYoArprKQaOe** has zero tasks and no active GPU.
  Audit **ap-yjB4XRhN52F6JEdmyXgc7V** passed all 21, 35 calls, 10 multi-expansion
  rows, minimum 606 Qwen tokens. Manifest SHA
  **962334d2a79e0b6c5e629ab1c0f983f4adac86c61ef8038cb17576746bb910c1**;
  accepted SHA **3791c9a1aa5bc57fcdcdfe3e09115012e5e9c004cc63a0674fe56d2fb90fe6b6**.
- Sample app **ap-FpsSpy1OmrVUtJyLCLdnjB** finished. Reviewed full evidence for
  one single and one multi-expansion corrected pilot trace. Current-request
  boundaries are clear and answer claims grounded in the expanded documents.
  Recorded internal **approval to scale only**, not release approval, in
  `data/reviews/stage3-md2d-corrective-pilot-review.json`, uploaded to the pilot's
  `review-passed.json`. Twenty-one pilot rows are not added to main-run counts.
- New corrected full app **ap-c9DCBmAQMHHS4aJt9MASov**
  (`lclm-md2d-corrective-full-v1`), call **fc-01M1ZP01R08PH9PAA822K4PWX9**,
  submitted after explicit hash-bound pilot/audit/manual-review checks.
  **Do not duplicate or redeploy either active full generator.**
  Output: `/data/stage3-agent/real-expansion/pilots/multidoc2dial-chronological-v1-full`.
  Same Qwen235B/no-thinking and verification settings, 24-hour bound, 21,451
  corrected tasks. Weight loading observed; inference progress still needs
  confirmation. These will replace old
  V6 MultiDoc2Dial at release time; never concatenate both versions.
- **37 regression tests passed** in **ap-b2w5pCuppN42jTAcBXh1jv**, including
  rejected missing/stale approvals, changed teacher, hashes and review IDs.
  Added shared corrective serving helper and gated full-generation module;
  active main deployment was not changed. Final source replacement/export/
  provenance integration remains to be implemented after full repair validation.

## Historical checkpoint — 2026-09-08 00:29 EDT — corrective pilot submitted; regressions passed

- Main generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, remains active and unchanged. Latest
  FaithDial checkpoint: **13,600 attempts / 5,889 accepted**. Combined main-run
  checkpoint: **96,031 attempts / 44,417 accepted**, not release-approved counts.
- Deployed isolated **32-task** corrective pilot app
  **ap-BWrkXeukNZoYoArprKQaOe** (`lclm-md2d-corrective-pilot-v1`) and spawned
  **fc-01M1ZM40P93D2WPDXNVD8BDPSZ**. Model weights are loading on a separate
  H200:8 container. **Do not duplicate/redeploy either active generator.**
  New module: `data/generate_multidoc2dial_repair_modal.py`; no full-run option
  is exposed. Same pinned Qwen235B, no thinking, 32-task reservoir, dual semantic
  checks, original documents and tools. Two-hour function bound, scale-to-zero.
  Pilot output: `/data/stage3-agent/real-expansion/pilots/multidoc2dial-chronological-v1-pilot`.
- Optional `input_provenance` now binds corrected task SHA/source/rendering
  version in its generation manifest; omission preserves exact original V6
  manifests. No redeploy of the main service. Audit and manual sampler CLIs now
  accept `--generation-dir` for separate corrective outputs, storing reports
  beneath that directory instead of overwriting baseline source audits.
- Regression app **ap-C0ZTZ7WEtmPRLfPJbrjegF**: **29 tests passed**. Initial
  run caught Modal's symlinked volume-root path comparison; corrected both sides
  before rerun. Report `/data/stage3-build-20260906/expansion-repair-regressions.json`.
- After pilot completion: audit with `data.audit_full_expansion_modal --sources
  multidoc2dial --generation-dir <pilot-output>`; sample via
  `data.sample_full_expansion_review_modal --source multidoc2dial --generation-dir
  <pilot-output>`, then manual review before full corrected-source regeneration.
  Corrective counts must remain separate from main V6 counts until reviewed
  replacement integration. Base provenance/terms and final release gates remain.
- FinQA manual sample app **ap-5vZQUn7LXPX9n29sXvTr1m** completed; both hash-
  selected single/multi samples reviewed against full expanded evidence and
  independently checked arithmetic. Recorded in `data/reviews/stage3-full-finqa-review.json`.
  Both sample answers pass; this is not all-row correctness/release approval.
  Noted conservative support coverage can require unnecessary extra expansion:
  multi-expansion counts must not be described as necessarily multi-hop tasks.

## Historical checkpoint — 2026-09-07 23:58 EDT — versioned dialogue repair inputs ready

- Main generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, still active. FaithDial latest inspected
  checkpoint: **9,300 attempts / 3,953 accepted**. Combined with nine completed
  sources: **91,731 attempts / 42,481 accepted**, before final source reviews.
- Verified the reversed-history encoding in the **pinned** IBM dataset builder
  at revision `1108a969d076f04c7367f0c2427d1c5d6d6bdaa0`, not just main.
- Added `data/multidoc2dial_dialogue.py`: reconstructs questions using raw
  dialogue IDs/turn IDs rather than splitting arbitrary utterance text on ||.
  It verifies exact upstream MRC encoding and next-agent reference before
  rendering chronological history and an explicit current-request section.
  Five tests passed on Modal (including literal delimiters, first/columnar turns
  and mismatched question/reference/turn IDs).
- Corrected-source CPU build **ap-HRWYfPYjEEJB22AjYPwgDM** completed via
  `data/prepare_multidoc2dial_repair_modal.py`. Target directory:
  `/data/stage3-agent/real-expansion/pilots/multidoc2dial-chronological-v1-inputs`.
  **All 21,451 tasks** roundtrip-validated against raw dialogues; new versioned
  IDs with parent IDs, original document bodies/segment IDs/support/answers
  unchanged. `multidoc2dial.build.json` saved with approved false, tasks SHA256
  **ed8c72b55cd790cdc009b60e59088174ba8dfd35fb9f4d76d594379e1b005649**,
  parent task SHA256 **3bcbf4e63cc096983319e958391e1745831ff7a56eb118db8fcc9084b9f82473**.
  Inspected all three `preview-questions.json` examples: no-history, two-history-
  turn and four-history-turn cases render chronological history and current
  request separately; expected next answers are not inserted in the prompt.
- This does **not** change the active V6 files or deployed generator. Next gate:
  separate corrected-prompt Qwen235B pilot, answer/format review, then versioned
  full-source regeneration and release integration. Old V6 MultiDoc2Dial remains
  release-held; no pilot/full repair generation has been launched yet.

## Historical checkpoint — 2026-09-07 23:28 EDT — ninth format audit passed; manual prompt issue found

- Generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, remains healthy. FaithDial checkpoint:
  **5,000 attempts / 1,922 accepted**. Nine completed sources plus checkpoint:
  **87,431 attempts / 40,450 accepted** (not final release-approved counts).
- MultiDoc2Dial full-format audit passed all **13,268 rows**, zero failures,
  **20,051 calls / 4,973 multi-expansion rows**, minimum segment **598 tokens**.
  App **ap-OyEpu1agGPUnA0nYWRdVLt** completed. Across nine completed sources:
  **38,528 rows format-audited**, 61,047 calls, 13,956 multi-expansion rows.
- New deterministic manual-QA sampler in `data/sample_full_expansion_review_modal.py`
  ran successfully in **ap-0obOuINRDqJJet7Q0rFeUK**. It binds the accepted-file
  hash, chooses lowest-hash single/multi examples, saves full rows and untruncated
  tool-evidence views. Volume: `full-expansion-manual-review-samples/multidoc2dial.*`.
- **Manual review found a MultiDoc2Dial prompt-rendering issue despite passing
  structure checks. Do not approve this source for final release yet.** Its
  upstream `question` field stores current utterance + [SEP] + reverse-ordered
  prior turns separated by ||; our adapter only replaces [SEP] with a newline.
  Both sampled questions therefore append unlabeled backwards history after
  the current question. Sample answers have supporting evidence, but the prompt
  chronology/boundaries should be corrected and revalidated separately.
- Recorded detailed non-approval in
  `data/reviews/stage3-full-multidoc2dial-review.json`. Source snapshot revision
  **1108a969d076f04c7367f0c2427d1c5d6d6bdaa0**. Upstream main builder confirms
  the encoding; verify pinned builder/raw dialogue before implementing repair.
  Next: versioned source-specific rendering correction + pilot/revalidation.
  Do not change active V6 inputs, manifest, generator deployment or saved traces.
  Other sources continue. Final release remains gated on this issue and base
  provenance/terms, complete generation, packing and final GPU validation.

## Historical checkpoint — 2026-09-07 22:52 EDT — ninth source complete, audit running

- Generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, remains active with one container.
  MultiDoc2Dial completed at 22:47 EDT: **21,451 attempts / 13,268 accepted**.
  Nine completed sources now total **82,431 attempts / 38,528 accepted**.
- FaithDial is next and running. Latest inspected checkpoint: **600 attempts /
  198 accepted**. Combined checkpoint total: **83,031 attempts / 38,726 accepted**.
  FaithDial has 18,357 tasks; no completion report yet. No new generation job
  or model switch was made.
- Launched the full MultiDoc2Dial accepted-row format/label audit in
  **ap-OyEpu1agGPUnA0nYWRdVLt** with
  `modal run --detach -m data.audit_full_expansion_modal --sources multidoc2dial`.
  No existing audit/report was present. Await
  `/data/stage3-build-20260906/full-expansion-format-audit/multidoc2dial.json`;
  do not treat the running audit as passed. Eight earlier source audits remain
  passed. Later-source audits and final semantic review are still pending.

## Historical checkpoint — 2026-09-07 22:23 EDT — generation progressing; base inventory saved

- Generator **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, remains active with one GPU container;
  normal inference logs observed. Do not redeploy or submit another generation
  call while active. Latest persisted MultiDoc2Dial checkpoint: 2,800 new
  attempts, **19,302 cumulative / 11,829 accepted**. Including eight completed
  sources: **80,282 attempts / 37,089 accepted**. MultiDoc2Dial still incomplete.
- Base inventory app **ap-3VhthOxDUtpOsCRVpJSbq4** stopped after saving
  `/data/stage3-build-20260906/base-source-provenance-inspection.json` with
  status `inspected`, approved false. All **2,033 shards / 20,326,114 rows**
  accounted for across **37 sub_dataset labels**. Every shard has a download
  receipt pinned to **b1b26053a7cd4ad669dd590c3cb6d14af91d946b**; none missing.
  HF metadata at that revision has no card, license or notice files. This
  completes runtime inspection, not attribution/terms clearance. Original
  upstream source mapping is still required; do not infer IDs from labels.
- The original base inventory includes **662 pubmedqa_labeled rows**. This is
  not the separate expansion source (which correctly uses 450 official training
  IDs). Base rows' official split membership has not been checked; count alone
  does not prove overlap because duplicates/transformations may exist. Include
  this question in the original-mixture provenance/split review before release.
- No new source has finished since the eight completed full-format audits.
  Remaining work: generation, later-source audits, final semantic/source review,
  expansion export/packing, final GPU validation and gated publication.

## Historical checkpoint — 2026-09-07 21:52 EDT — resumed generation healthy

- Deployed app **ap-NT9cm8Mc78qaGB9euDnK99**, call
  **fc-01M1ZB04K1DJ94V1PGWTGC9224**, is healthy and generating with one
  H200:8 container. Do not duplicate. Latest persisted MultiDoc2Dial checkpoint
  reports 200 new attempts, **16,702 cumulative attempts / 10,064 accepted**.
  Including eight completed sources: **77,682 attempts / 35,324 accepted**.
- User confirmed the choice to continue Qwen235B-Instruct without thinking.
  No model switch or task-manifest change. Server logs show normal inference
  after startup; new checkpoints demonstrate progress after the local submitter
  exited. This does not prove the cause of past cancellations.
- Base provenance inspector is being retried separately on CPU in
  **ap-3VhthOxDUtpOsCRVpJSbq4** after confirming no existing inspector app
  or persisted report. It does not modify base rows
  and cannot substitute for the missing original source/license mapping.

## Historical checkpoint — 2026-09-07 21:44 EDT — full generation submitted

- User explicitly requested generation of all tasks; the previous cancellation
  hold is lifted. No other LCLM app was listed before starting checkpoint audit
  `ap-QlOC44rwUt72Yqn0Y6Rq8i`. Its report passed at 01:44:03 UTC:
  **77,482 persisted attempts / 35,201 accepted**, including MultiDoc2Dial's
  **16,502 attempts / 9,941 accepted**. No duplicate IDs, partial JSONL lines,
  unknown task IDs or completed-report mismatches. Remaining: **297,273 tasks**.
- User suggested Qwen/Qwen3.8-2.4T-A95B and requested no thinking. Its official
  model card says thinking cannot be disabled; Modal lists a shared endpoint
  with the same restriction. Asked whether no thinking means generation itself
  or just the saved training traces. Default remains the approved, non-thinking
  Qwen3-235B-Instruct teacher; do not silently switch the V6 manifest/model.
- Qwen3.8 evidence: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B#api-usage
  and https://modal.com/library/qwen/qwen3-8-max . No Qwen3.8 endpoint created.
- Deployed existing 235B generator as app **ap-NT9cm8Mc78qaGB9euDnK99** and
  submitted **fc-01M1ZB04K1DJ94V1PGWTGC9224** using
  `python -m data.submit_expansion_generation`. The submitter exited normally;
  no local process is waiting on the remote function. Check this call/app before
  doing anything; **do not submit another job while this one is pending/running**.
  This removes the local waiting process from the job lifecycle; it does not
  establish the cause of the earlier cancellations. GPU startup not yet verified.
- Generation still uses the exact approved V6 manifest, Qwen235B revision and
  non-thinking requests. All saved attempted IDs are skipped. Same checkpoint
  paths, 24-hour function timeout and existing semantic/format gates apply.
  No public unauthenticated endpoint was created; this is a Modal class service.
- Launcher syntax-compiled locally; image deployment and job submission passed.
  Generator logic was not changed; recursive bytecode image exclusions added.
- Base provenance/terms and final release gates remain unresolved. No new HF
  release exists. Generation target remains 374,755 tasks, not accepted rows.

## Historical checkpoint — 2026-09-07 17:10 EDT — awaiting cancellation clarification

- **The resumed GPU app stopped again. The restart hold recorded here was
  lifted by the user's resume request above.** App
  `ap-UXSWSfrR3pbU6soJrhFl1B` received an input cancellation signal at
  **19:52:20 UTC / 15:52 EDT**, then stopped at 15:53:28 EDT. It was processing
  requests normally immediately beforehand. No inference-error cause appears
  in the inspected log tail; cancellation initiator is unknown. This is the
  second full-generation cancellation, despite using detached mode.
- Latest persisted MultiDoc2Dial progress report: 8,300 new attempts after
  resume, **16,415 cumulative attempts / 9,873 accepted**. Together with the
  eight completed sources this reports **77,395 attempts / 35,133 accepted**.
  These are checkpoint-reported counts; audit actual JSONL again before any
  authorized resume because additional rows may have flushed after the report.
- Base-mixture provenance gap: HF's original `leonli66/stage3-final-mixture`
  has no dataset card or license declaration. Local searches of LCLM and the
  sibling project code found the rewrite/audit scripts, but not its original
  37-subset construction manifest. Asked the user for the build script/source
  mapping. Source evidence: https://huggingface.co/datasets/leonli66/stage3-final-mixture
- Removed the unsupported blanket `license: apache-2.0` declaration from the
  legacy CoT-only uploader's card template; it now explicitly preserves upstream
  terms. No existing HF repository was changed by this edit.
- Added `modal run --detach -m data.stage3_provenance_modal --base` to inspect
  raw subset counts and cached download revisions. Attempt
  `ap-CkELuqTPjHgBWBdhO6LXw5` was also cancelled (20:08:44 UTC) and stopped;
  **no `base-source-provenance-inspection.json` exists**. Runtime validation of
  that inspector is pending. Do not claim the original base terms are cleared.
- Source notices, eight completed full-format audits, base/native/recovery
  packs and TechQA supplemental prompts remain saved. No HF release exists.
  Next: obtain cancellation clarification; then inspect/checkpoint-audit before
  restarting. Read-only provenance investigation can continue meanwhile.

## Historical checkpoint — 2026-09-07 15:21 EDT

- Full V6 teacher remains active in `ap-UXSWSfrR3pbU6soJrhFl1B`. Latest inspected
  MultiDoc2Dial checkpoint: **13,315 cumulative attempts / 7,910 accepted**
  (5,200 new attempts after resume). No duplicate job was launched.
- Collected **19 upstream notice/card files across all 17 native-agent and
  non-synthetic expansion sources** from their pinned downloaded checkouts.
  Bundle: `/data/stage3-build-20260906/source-notices-v1/`, with `index.json`
  recording source revisions, original paths, sizes and SHA256 values.
  Collection app: `ap-VquvZeacxxjqxlkqqWyRYC`; byte/coverage verification rerun:
  `ap-vj9iErJYKXpE2wQlkcG2xP`. This is notice preservation, **not legal clearance**.
- Inspected the native-agent dataset cards: Nemotron v1 declares CC-BY-4.0
  plus Glaive's Apache-2.0 notice; v2 declares CC-BY-4.0 plus Apache-2.0/MIT
  source terms; OpenThoughts declares Apache-2.0 and identifies its upstream
  task sources. Underlying source restrictions still require final review.
  Most HF snapshots contain only a dataset card, not standalone license text.
- Publisher now preserves this bundle in both HF repos and verifies all source
  IDs/revisions and notice bytes before upload. Final `release-review.json`
  must bind `source_notices_index_sha256` and set `base_mixture_terms_reviewed`
  only after actual review of the original base mixture. Do not preapprove
  those fields. The bundle does not cover a complete base-mixture terms audit.
- Focused validation: **141 tests passed** plus eight real-Qwen boundary cases,
  `ap-yYKKCjRUsKAuieoTAmYqoh`. Notice tests cover exact copies, missing sources,
  stale revisions, changed bytes, symlinks and unsafe relative paths.
- Generation/export/packing/final GPU validation remain incomplete; no new HF
  release exists. Continue the later-source audits and source/semantic review.

## Historical checkpoint — 2026-09-07 14:48 EDT

- Full V6 teacher still running in `ap-UXSWSfrR3pbU6soJrhFl1B`; do not duplicate.
  Latest inspected MultiDoc2Dial report has 2,500 new attempts after resume,
  **10,615 cumulative attempts / 6,196 accepted**. No later source has completed
  yet; the eight completed source audits remain valid.
- Supplemental TechQA prompt build completed in `ap-Vzy7dS7sSUN2yQqlLqTXtV`:
  **393 tasks**, 54 with multi-segment support, 3,151 total segments, minimum
  **705 Qwen tokens** per segment. Three of the 396 eligible questions were
  excluded because complete original source documents exceed 12 segments.
  Exact answer offsets are revalidated and every original support chunk is
  checked for preservation. Distractors/padding use only the 348-document
  retained training pool, with held-out answer documents excluded.
- Supplemental files are on the data volume under
  `/data/stage3-agent/real-expansion/pilots/techqa-supplement-20260907-v1/`:
  `techqa.tasks.jsonl`, `techqa.build.json`, and `preview-tasks.json`.
  Task-file SHA256: `8cde837935d354693289d82d19a61ea2289bb41e51584da1e6d268e3bd4947f4`.
  **These are prompt tasks, not generated traces, and are not included in the
  running 15-source build.** Next gate is a separate Qwen-235B pilot with
  source-appropriate answer verification and semantic review before inclusion.
- Focused tests: **133 passed** plus eight real-Qwen boundary cases,
  `ap-ofyiO3KScDkJaJzgnGiu1s`. Initial supplemental launch failed during local
  image assembly because concurrent imports modified a nested `.pyc`; added
  explicit recursive bytecode exclusions to the shared Modal image and retried
  successfully. No generation data was modified by that failed launch.
- Main release remains pending complete generation, audits for later sources,
  semantic/source-notice review, export, packing and final packed/GPU tests.

## Historical checkpoint — 2026-09-07 14:22 EDT

- Resumed teacher **healthy and generating** in `ap-UXSWSfrR3pbU6soJrhFl1B`.
  vLLM began serving at 18:15:23 UTC and active inference was observed. Latest
  inspected MultiDoc2Dial checkpoint: **8,415 cumulative attempts / 4,734
  accepted**, including 300 new attempts after resume. Its `processed` field
  counts new attempts this run; sum `reasons` for cumulative attempts.
- New full-format audit checks every accepted trace in each completed source,
  using pinned decoder/encoder tokenizers and bounded CPU worker pools on Modal.
  It verifies original expand bodies, tool schema/arguments, saved task system,
  teacher separation, harvested assistant content, all segment lengths >=512,
  source verifier/semantic-vote conditions, PubMed train-only document IDs and
  independently computed assistant-only labels for every tool call/answer.
- **All 25,260 accepted traces in the eight completed sources passed**, not just
  a sample. **8,983 multi-segment-expansion traces**, 40,996 native calls; minimum
  segment length **624 Qwen tokens**. PubMedQA + CLAPNQ audit:
  `ap-wCzE0z01UxdgM4WcH5rXNL`; other six completed sources:
  `ap-QcnoKOJflRERNtXqYvsqXb`. Both audit apps finished. Reports are under
  `/data/stage3-build-20260906/full-expansion-format-audit/<source>.json`.
  Free-form semantic judgments remain heuristics, not correctness guarantees.
- Export now requires complete source audit coverage, matching generation
  manifest/tokenizer revisions/counts, and the SHA256 of each actual accepted
  JSONL file to match its audit. Publication rechecks the exported audit receipts.
  This prevents pilot-only, stale or partial audit reports from passing release.
- Focused tests: **130 passed** plus eight real-Qwen boundary cases,
  `ap-274IWziLRspjOn0xZyVoZe`. New tests cover corrupt expand bodies, tool/schema
  problems, missing system, preambles, label mistakes and stale audit gates.
- Next: leave the resumed teacher running; audit subsequent sources only after
  their generation reports are complete. Run
  `modal run --detach -m data.audit_full_expansion_modal --sources <comma-separated-newly-complete-sources>`.
  Do not rerun the eight passed source audits unless files or audit rules change.
  Then finish source-stratified semantic review, export, packing, final packed
  artifact/GPU validation and source-notice review. **No HF release yet.**

## Historical checkpoint — 2026-09-07 14:13 EDT

- Original full V6 app `ap-AClcBdi6IpX7cwvr24zc0m` **stopped at 11:33 EDT**.
  Logs show an input cancellation signal at 15:32:53 UTC, followed by worker
  shutdown; they do not identify the initiator. Do not describe this as a
  model failure or a completed run.
- Read-only checkpoint audit **passed** in `ap-r2V9O6xkEOve7r87c3np6M`:
  **69,095 persisted attempts / 29,791 accepted**. Every output task ID belongs
  to its input shard, no duplicate IDs or incomplete JSONL lines were found,
  verdicts match accepted/rejected files, and all eight completed source reports
  agree with actual row counts. Per-file SHA256 values and counts are saved at
  `/data/stage3-build-20260906/expansion-resume-audit.json`.
- Completed sources (accepted / attempted): MAUD **11,851 / 23,016**, FinQA
  **1,270 / 6,191**, PubMedQA **233 / 450**, CLAPNQ **622 / 989**, ContractNLI
  **5,246 / 7,191**, TATQA **4,055 / 13,223**, ConvFinQA **325 / 2,090**,
  MultiHiertt **1,658 / 7,830**. MultiDoc2Dial is partial: **4,531 / 8,115**
  persisted, with 13,336 remaining tasks in that source.
- **Resume submitted in detached mode:** `ap-UXSWSfrR3pbU6soJrhFl1B`, using the
  same V6 manifest, task root, output root, model revision and verification
  rules. Model startup is not yet confirmed at this checkpoint. Do not launch
  another run; inspect this app and saved progress first. Completed task IDs
  are skipped, including the eight complete sources. All 374,755 prompt tasks
  remain in scope; no partial export or HF release has been made.
- New checkpoint-audit tests passed with the focused suite: **109 tests** plus
  eight actual-Qwen boundary cases, `ap-I8ZuOnxntxbLwDi1taNRCA`. The first audit
  attempt (`ap-aVshGELisZ6xaukc6KswaQ`) was canceled after a client disconnect;
  its detached retry above completed. Use detached mode for long future jobs.
- Next: confirm resumed teacher health and renewed MultiDoc2Dial progress;
  continue source review. Export/pack only after all source accounting passes,
  then final actual-expansion artifact/GPU validation and publication gates.

## Historical checkpoint — 2026-09-07 02:56 EDT

- Full V6 generation remains active in `ap-AClcBdi6IpX7cwvr24zc0m`; do not
  duplicate it. Latest inspected MAUD progress: **3,200 attempted / 802 accepted**,
  2,347 wrong answers, 39 missing support, 12 generation ValueErrors. This is a
  partial first-source count, not the full 374,755-task outcome.
- TechQA's training-only adapter is implemented and materialized on Modal in
  `ap-3bprag3yt5IGpx5DKLFOny`. Of 600 official training questions, **396 retained**,
  150 unanswerable excluded, 54 excluded for dev/validation answer-document
  overlap. All retained answers match exact original document offsets; the
  pool has **348 original documents**. Data, report, upstream README and license
  are under `sources/techqa/materialized_train_v1` within the real-expansion root.
  This is supplemental preparation, **not part of the running 15-source build**;
  it still needs long-segment task construction and its own reviewed pilot.
- Future release packing now pins decoder and encoder tokenizers. Encoder:
  `Qwen/Qwen3-Embedding-0.6B@97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
  Earlier base/native/recovery packs loaded main without explicit revisions.
  Do not retroactively claim pinned build provenance for them; final sampled
  validation uses pinned tokenizers and records that limitation in the manifest.
- Final GPU validation now requires actual expansion packs by default. An
  explicit intermediate mode writes a separate report. Publication additionally
  rejects an artifact audit missing any of base/native/recovery/expansion.
- Focused Modal tests: **103 passed**, plus all eight real-Qwen boundary cases,
  `ap-XHBFVLn6nIMpwnsGnhZLTM`. New tests cover optional tokenizer revision
  forwarding, final component gates and TechQA split/answer-span checks.
  The final expansion packed/GPU integration and HF release remain pending.
- Next: monitor full generation; resume only after confirming it stopped and
  inspecting persisted reports. Export/pack only after complete generation
  accounting passes, then run final integration and source-notice review.

## Historical checkpoint — 2026-09-07 02:28 EDT

- **Full V6 generation submitted:** `ap-AClcBdi6IpX7cwvr24zc0m`, Qwen3-235B
  Instruct on H200:8, output `full-20260906-v6`, input root
  `full-20260906-v3` (with corrected PubMed shard). It targets **374,755 tasks**
  across the 15 implemented sources. Do NOT start a duplicate. The function
  has a 24-hour timeout and task-ID-based resume; check stopped/running state
  and persistent source reports before any retry.
- V6 pilot finished **480 attempted / 222 accepted**. Accepted by source:
  MAUD 11, FinQA 5, PubMedQA 15, CLAPNQ 18, ContractNLI 27, TATQA 9,
  ConvFinQA 9, MultiHiertt 7, MultiDoc2Dial 20, FaithDial 11, Watsonx 22,
  ACORD 26, BillSum 3, LexGLUE 9, synthetic 30.
- All 222 accepted traces passed the format/label/expanded-body audit in
  `ap-ZNRkq6Cni4v4SeBh2foZPl`; shortest segment **645 Qwen tokens**. The audit
  additionally checks every PubMed document ID in every segment, including
  padding/distractors, against the official 450-example training allowlist.
- Reviewed two accepted examples per source and the BillSum sentence-level
  quotation evidence. Internal approval to scale is recorded in
  `data/reviews/stage3-v6-pilot-review.json` and on the volume at
  `full-20260906-v6-audit/review-passed.json`. This is **not** final release
  approval or user sign-off. The gate binds review/counts/source examples to
  generation manifest SHA256
  `de96a25ff748eae11cff76e60be5298f229b4d4f77899754dc507017dfe3de83`.
- Assembled `stage3-build-20260906/expansion-source-provenance.json`: all 15
  sources, exact upstream revisions, declared licenses, acquisition/conversion
  receipts, source task counts, official PubMed splits and registry exclusions.
  Export/publication now require all per-source attempted counts to match this
  manifest, complete source coverage, matching aggregate reasons, and accepted
  counts matching the actual JSONL export. Source attributions are added to
  the raw dataset card. Still review upstream notices before release.
- Latest focused validation: **90 passed** plus eight actual-tokenizer boundary
  cases, app `ap-YL6CaesvsO3Hl1f9KNs7Hg`. Final expansion packed/GPU integration
  remains pending; no new full HF release exists.
- TechQA archive inspection is complete (`techqa-archive-inspection.json`):
  600 training questions, 310 development questions, separate validation data,
  original technote documents and exact answer offsets. It is a regular tar
  archive, not WebDataset. Its embedded README declares **CDLA-Permissive-1.0**;
  corrected the registry's previously incorrect Apache-2.0 data-license entry.
  A source-specific training-only adapter and pilot are still needed. Do not
  silently add it to the currently running 15-source build. Other unresolved
  registry sources/FAv2/license choices remain as documented below.
- The progress audit now includes `full_v6` per-source progress/completion
  reports. Next: monitor the full job, inspect early acceptance/circuit-breaker
  behavior; work on final integration and unresolved source adapters meanwhile.
  Export and packing can start only after full generation count gates pass.

## Historical checkpoint — 2026-09-07 02:00 EDT

- **New V6 pilot is running:** `ap-9Nge1n4hfkSvPP1GtXSG2u`, 32 tasks per
  source / 480 attempts, Qwen3-235B-Instruct on H200:8. Output is
  `full-20260906-v6-audit`. Do not launch a duplicate. Full generation still has
  NOT started. Generator, exporter and publisher now target **V6**, not V5.
- V5 semantic review finished: 98/115 kept, 13 rejected, 4 malformed-response
  errors. This caught the CLAPNQ mismatch, but missed the BillSum establishment
  claim. Full source evidence confirms that claim is wrong: the bill refers to
  an authority established under a different act.
- Sentence-level BillSum recheck finished in `ap-hU75aCHm8WWjE7zjdREsUd`:
  1/16 kept, 11 rejected, 4 malformed/truncated JSON errors. The known bad
  establishment claim was rejected. Each sentence now needs a positive
  entailment vote and quotations that occur verbatim (whitespace-normalized)
  in the source. Explicit RELATED SOURCE padding is excluded from this check.
  This is deliberately conservative and can reject good summaries too.
  The same-model semantic judgments are still heuristics, not guarantees.
- V6 applies the dual question/evidence review to free-form and otherwise
  correct PubMedQA responses; it additionally applies the sentence check to
  BillSum. Structural failures and wrong biomedical decisions cannot be
  overridden by the judge. Saved training messages are unchanged by judging.
  Judge output allowance is now 2,048 tokens to reduce truncated quotations.
  New manifest fields prevent resuming with the old verification settings.
- **PubMedQA split fixed:** HF's single `train` contains all 1,000 labeled
  examples, including the official test set. Ran the official split script at
  `pubmedqa/pubmedqa@1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` in a fresh checkout.
  Only fold-0's **450 training examples** are now eligible; 50 dev + 500 test
  examples are excluded BEFORE building documents/distractors. The builder
  fails closed without the pinned, disjoint split manifest. Source evidence:
  https://github.com/pubmedqa/pubmedqa/blob/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/preprocess/split_dataset.py
- PubMed repair completed in `ap-FnXDjtVsAywD5FEamLlhiM`. Original task/build/
  aggregate-manifest files were **moved, not deleted**, to
  `full-20260906-v3/quarantine-pubmed-original1000/`. Current task root remains
  `full-20260906-v3`, with the repaired PubMed shard. Total eligible prompt
  tasks are now **374,755**, not 375,305. Old PubMed pilots remain diagnostics.
  Report: `sources/pubmedqa_labeled/official-splits/task-repair-report.json`.
- Latest focused validation: **78 passed**, plus all eight pinned-tokenizer
  boundary cases, app `ap-OQeuMxe9UTKLdQhh3i3v3q`. Final GPU integration must
  still include the eventual expansion packs.
- Next: inspect V6 source outcomes and accepted samples; run
  `modal run -m data.review_expansion_pilot_modal` (now targets V6). Only write
  V6 `review-passed.json` after actual review. Then full V6 generation may start.
  Do not re-run V5 semantic diagnostics; their reports are complete.
- No HF release yet. Base/native/recovery completion below remains current.

## Historical checkpoint — 2026-09-07 01:27 EDT

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
