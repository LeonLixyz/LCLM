# Category-isolated 16k/32k Stage-3 packing — 2026-09-09

**Final checkpoint, Sep10 18:32 UTC:** the reviewed build, including expansion, is complete. Both16k/32k versions are verified and trainable. Final counts and paths: [STAGE3_PACKED_DATA_20260910.md](STAGE3_PACKED_DATA_20260910.md). Earlier checkpoints below are historical.


**Verified complete, 2026-09-09:** Both final manifests exist and report
`status=complete`, with every output byte digest verified and both actual-loader
variants passed.16k:2,255,002 sequences /21,311,188 examples.32k:1,265,307
sequences /21,516,926 examples. Agent/reasoning/other categories remain pure per
sequence; completed sequences are shuffled. Expansion append is still pending.

Final manifest hashes:
-16k:`82f8e84b38e331b2f267f4654a09b7c37aec110ce20777504385be5f3372bd3a`.
-32k:`1d6e63a20b9eeee82d02a91f3cce8f3bb2226d766828f994170af88ebbc0e314`.

Local exact copies and machine-readable exclusion accounting are in
`_modal_run/grouped-packing-final/`. The259,096/53,358 excluded totals include
short rows and one processing rejection; they are not all overlength. Earlier
execution checkpoints below are historical.

User requested independent agent/reasoning/other packed sequences and both
16,384- and 32,768-token versions. Whole completed sequences are shuffled
together afterward. Synthetic expansion remains excluded until its generation,
grounding review, and approved export complete.

## Implementation and execution

- Unique CPU app: `lclm-stage3-grouped-packing-20260909-v3`; deploy once and invoke
  its deployed functions. Generation apps are separate and untouched.
- Code: `data/grouped_stage3_packing.py`, `data/grouped_stage3_packing_modal.py`.
- Tests: `tests/test_grouped_stage3_packing.py` plus existing worker/loader suites.
- Status: **all 64 available-data partitions complete; final integrity audit running**.
  Original app `ap-6wcNzANRV21Rq14PbU8m5H`,
  call `fc-01M22CD5A03YKF3RB02Q6WRJ3K`.
  V3 deployed from immutable `/tmp/lclm-grouped-packing-v3-frozen`;
  preflight `fc-01M22C88DK0YQDDSBHQ5SMQP9C` passed 35 tests and representative
  actual-tokenizer/runtime checks in 23.8 seconds. Superseded v1/v2
  preflights were cancelled after passing 32/35 CPU tests. Neither ran packing.
  V3 preserves the independently verified math-reconstruction task semantics and
  bounds preflight to at most eight sampled shards per component; per-file row
  counts and source hashes are collected during the full streaming pass.
- Pilot `fc-01M22CANV1CRPVKY9KDME0AAC5` passed byte-digest checks and actual
  two-rank loader checks for both lengths. All 512 raw pilot rows fit 32k;
  16k retained 434, with 76 native and 2 reasoning rows excluded whole. These
  are early-file sample counts, not estimated overall drop rates.
- Full processing: 64 partitions, up to 32 CPU containers with eight workers each.
  Both output lengths share one raw tokenization pass.
- Original map inputs 9, 21, and 30 reached terminal `FAILURE`; the original
  coordinator continued healthy queued/running work. Global app logs confirm
  three Modal preemptions at 02:09:46, 02:09:52, and 02:12:22 EDT. The original
  exclusive `STARTED.json` guard does not allow Modal's same-input retry.
  Repair v1 partition9 reproduced this at 02:23:05: explicit preemption followed
  by `ValueError: Repair already launched` on restart. This is a preemption/retry
  handling defect, not evidence of a malformed input or OOM. Individual map-call
  exception retrieval returned `NotFound`; global logs supplied the diagnosis.
  The original coordinator subsequently surfaced the deferred
  `FileExistsError: .../partitions/part-009/STARTED.json` and exited; its own
  finalization did not run. Modal then terminated the 29 remaining map inputs:
  final original state is 32 success, 3 failure, 29 terminated. The 32 completed
  partition reports are preserved. The separate coordinator owns completion.
- Isolated repair app: `lclm-stage3-grouped-repair-20260909-v1`,
  `ap-bjVnKW4YuOQ0GkRwgYxXtH`, frozen at `/tmp/lclm-grouped-repair-v1-frozen`.
  Bounded scheduling smoke passed. Repair calls:
  9=`fc-01M22D8P6TGEBP7J34CQQ16EQF`,
  21=`fc-01M22D8P9BTSRX2A59A5N58HFV`,
  30=`fc-01M22D8PBP9WYTQJ82MM5TAXSA`.
  Repairs archive only those failed partials under `repair-audit/part-###`, then
  invoke the exact v3 processor with at most 16 in-flight 16-row chunks, 64GiB
  RAM, and explicit persisted traceback capture. Source selection, raw row IDs,
  category policies, tokenizers, length limits, and loss masks are unchanged.
  Successful reports enter the same v3 layout, so finalization can combine them
  with healthy original partitions. V3 itself was not redeployed or interrupted.
- Retry-safe repair app: `lclm-stage3-grouped-repair-20260909-v2`,
  `ap-flE1qYXMsUhr6OG4VSMAqJ`, frozen at `/tmp/lclm-grouped-repair-v2-frozen`.
  Same-call owner validation permits platform retries while rejecting duplicate
  independent calls; each attempt archives partial outputs separately under
  `repair-audit-v2/part-###/attempt-*`. Owner and bounded scheduler smoke passed.
  Partition 9 call=`fc-01M22DHK1W9EKTNMQF758WTJH9`; healthy v1 repairs 21/30 remain
  untouched. Larger memory/bounded scheduling are resource precautions, not a
  claimed fix for a confirmed OOM. Packing/data semantics remain identical.
- General completion app: `lclm-stage3-grouped-completion-20260909-v1`,
  `ap-1HUxoSjS86U7j4HdlKdA8U`, frozen at `/tmp/lclm-grouped-completion-v1-frozen`;
  coordinator `fc-01M22DP3S7Z08PVCP1G08HZTXS`. Smoke checks passed for bounded
  scheduling, same-call retry ownership, and rejecting repair while original
  inputs remain active. It checks every 45 seconds, waits for all 64 original
  map inputs to become terminal and for any earlier repair of a missing
  partition, then repairs any missing partition 0..63 with at most 3 active
  repairs. Completed reports are no-ops. Durable attempt audit/quarantine is
  `repair-completion-v1`. Once 64 reports exist, it finalizes both lengths.
  This conservative terminal boundary avoids relying on public call-graph
  positional ordering, which does not explicitly expose map input arguments.
- Completion v1 was superseded after the original map cancelled its remaining
  29 inputs: a three-repair cap would have unnecessarily serialized that larger
  recovery. Its coordinator was cancelled only after verifying it had zero
  repair children. Existing active repairs were not cancelled.
- **Active completion v2**: `lclm-stage3-grouped-completion-20260909-v2`,
  `ap-w8AV9badmWfSzYZQ4DhMLQ`; call `fc-01M22DXV1TJP5Q75NQJ8J3GG6N`;
  frozen `/tmp/lclm-grouped-completion-v2-frozen`. Smoke passed. It retains the
  same data processing, safe retry ownership, and per-container bounded queue;
  only the cap increases to 16 simultaneous repairs. Audit namespace is
  `repair-completion-v2`. Existing completed reports and active earlier repairs
  are respected. Both versions will be finalized after all 64 reports exist.
- Safe retry was exercised by a real platform preemption: partition 41 received
  `KeyboardInterrupt` at 02:36:08 and explicit Modal preemption at 02:36:11 EDT.
  The same call `fc-01M22DYAXHY67J75RJYFXMTP8G` restarted successfully, archived
  its interrupted partial, and returned to processing (30,000 rows by 02:38:04).
  Parent call graphs can retain the old attempt as `FAILURE`; actual result
  polling still timed out while resumed progress advanced. Querying a child call
  graph returns the enclosing root graph, so its first node is not authoritative
  child status. Completed reports drive slot release; external result polling
  supplies terminal-failure diagnosis. No independent duplicate was launched.
- Partition 51 later stalled at 130,000 raw rows (03:02:42 EDT). Container
  inspection found seven original sleeping pool workers and one newly spawned
  worker, consistent with a lost worker leaving an unresolved async result.
  Current memory was about 7GB; no OOM or original worker exit cause was proven.
  Its partial snapshot was committed before only that stalled call/container was
  cancelled. Healthy partition 63 and all completed reports were preserved.
- Isolated final repair: `lclm-grouped-final-repair-20260909-v1`,
  `ap-Ee5nXMYhqQktqTeA1ZdxWQ`, call `fc-01M22GPR92WVC4FPHNY4QFPGYX`,
  frozen `/tmp/lclm-grouped-final-repair-v1-frozen`. Coverage/worker-loss/fatal
  exception smoke passed. The bounded scheduler checks for exited workers every
  10 seconds, transports BaseException failures with raw-row IDs, and permits
  one bounded worker-loss retry. It reuses the unchanged v3 processor and
  existing per-attempt quarantine/report implementation under `final-repair-v1`.
  Partition 51 completed successfully in 944 seconds; all 64 reports are now
  complete. Original v3 finalizer `fc-01M22HM0HEAJWQP4PF16GBF4ZB` is checking
  every output byte digest, aggregate accounting, and representative actual
  loaders for both lengths. Final manifests remain pending this audit.
- Serial finalization was replaced with bounded parallel content auditing after
  measuring 322.67 GiB across 6,940 shards. At 03:46 EDT the serial reader was
  still on 16k partition 7, making its one-hour timeout inadequate at the observed
  volume throughput. Its progress and absent-manifest evidence is preserved in
  `parallel-audit-v1/original-serial-audit-evidence.json`. The old coordinator was
  cancelled; its result is a terminal `RemoteError`, and the exact old finalizer
  node is `TERMINATED`. Neither had published a final manifest.
- Parallel audit app: `lclm-grouped-parallel-audit-20260909-v1`,
  `ap-QLQRq0gNfWfz8Q345Slzc9`; coordinator
  `fc-01M22J4J3G3KM6ZR362DNVGE94`; immutable snapshot
  `/tmp/lclm-grouped-parallel-audit-v1-frozen`. At most 16 read-only checksum
  workers audit both lengths per partition. Certificates bind every full relative
  path, byte size, per-file SHA256, original concatenated output digest, exact
  partition report bytes, and exact base configuration. The gated single final
  writer retains aggregate invariants and both actual-loader checks. It verifies
  the old finalizer and coordinator are terminal again immediately before
  publication. The original processor/worker/tokenizer/loader files are identical
  to frozen v3; output data is unchanged. Test call
  `fc-01M22J2HHMX199M0K0F632RGEB` passed all 9 CPU tests in 15.56 seconds,
  including same-size corruption, binding changes, terminal-child selection,
  both runtime loaders, and the expansion chain's `base_snapshot` consumer.

## Provisional complete-partition totals (final audit pending)

| Length | Category | Retained examples | Packed sequences |
|---|---|---:|---:|
| 16,384 | Agent | 1,194,048 | 249,645 |
| 16,384 | Reasoning | 5,241,060 | 712,993 |
| 16,384 | Other | 14,876,080 | 1,292,364 |
| 32,768 | Agent | 1,230,344 | 147,725 |
| 32,768 | Reasoning | 5,402,133 | 465,966 |
| 32,768 | Other | 14,884,449 | 651,616 |

The 16k candidate has 21,311,188 examples in 2,255,002 sequences, 4,437 shards,
167,717,878,958 bytes (156.20 GiB), and 259,096 whole-example exclusions. The 32k
candidate has 21,516,926 examples in 1,265,307 sequences, 2,503 shards,
178,746,155,249 bytes (166.47 GiB), and 53,358 whole-example exclusions. Its base
count exactly matches the earlier corrected base-plus-recovery total 20,286,582,
and its native count matches 1,230,344. These totals become final only when the
content and actual-loader audits finish.

## Inputs, categories, and policy

- Base: `/data/stage3-final-mixture-cot50-v1`, 20,326,114 rows in 2,033 shards.
- Native: `/data/stage3-build-20260906/agents-transport`, 1,244,170 raw trajectories.
- Each raw row is read once. Old baseline and legacy-prefix-recovery packs are
  not inputs, avoiding duplicate recovered rows. The corrected SFT worker handles
  those prefix boundaries directly.
- Agent category: native trajectories only at this checkpoint.
- Reasoning category: `reasoning_data`, `dolci_think` only.
- Existing `reasoning_data` and `dolci_think` CoT50 targets are preserved exactly;
  their prompts must match the original, uncompressed prompts.
- `nemotron_math_4plus` is **other**, retaining existing content and compression.
  Actual inspection found **347/347 sampled rows** across 12 deterministic files
  and three positions per file are passage-reconstruction tasks: the target
  already appears verbatim in the prompt. A mathematical source name does not
  make this a problem-to-CoT dataset. No new CoT rewrite is applied.
  Evidence: `/data/stage3-build-20260906/nemotron-math-task-semantics-20260909.json`.
- Other category uses an explicit allowlist of all 35 remaining source labels;
  unknown source labels fail closed instead of silently being assigned.
- No trajectory is truncated to fit. Every processing/short/overlength exclusion
  is recorded with source file, raw row number, category, and length variant.

## New output locations on `lclm-stage3-data`

Root: `/data/stage3-build-20260906/grouped-packing-20260909-v3`

- `packed-cs16-16384/data/mixed/part-###/*.parquet`
- `packed-cs16-32768/data/mixed/part-###/*.parquet`
- `partitions/part-###/report.json` and `exclusions.jsonl`
- `source-manifest.json`, `preflight.json`, and `pilot/summary.json`
- Each completed version has its own `manifest.json`, including counts per
  category/source and representative actual-loader verification.

Each Parquet row is one category-pure sequence. Additional Parquet columns expose
category, example count, and exact length at ratio16; each packed example records
raw-row provenance and reasoning policy. Independent category bin packers feed a
512-sequence shuffle buffer. Base/native rows are proportionally interleaved
before tokenization, avoiding a large native-only tail. Source-file SHA256 hashes
are captured when read; all packed output byte hashes are verified at completion. The existing `DynamicPackedDataset` discovers this
layout and provides additional file/row shuffle at training time.

Decoder tokenizer: `Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`.
Encoder tokenizer: `Qwen/Qwen3-Embedding-0.6B` at
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
Decoder IDs and loss masks are pretokenized; encoder memory strings remain
runtime-tokenized. Pack lengths apply at compression ratio16.

No HF publication or model training is part of this packing job. Original raw
and packed data are preserved. Source provenance/terms review remains separate
from these technical packing and loader checks.

## Later reviewed expansion append

The available-data snapshot explicitly excludes unfinished/unreviewed expansion.
After that source passes its generation, grounding, and export gates, it can be
packed independently into each version under `data/expansion/part-###/*.parquet`,
with every sequence labeled `agent`. The existing loader discovers this component
alongside `data/mixed` and shuffles complete sequences. Base/native files do not
need repacking. The combined manifest must be updated and integration checked;
the current raw-source worker intentionally rejects unapproved expansion inputs.

Compatibility check: `discover_packed_parquet_files` explicitly includes
`data/*/part-*/*.parquet`; the existing discovery test already includes an
expansion component and excludes raw/staging/quarantine folders. The dynamic
loader shuffles the combined file list and saved sequence rows. No loader change
is required. Appending changes the dataset fingerprint, so training must open
the completed combined snapshot rather than silently resume a stale assignment.

The small later append adapter must process reviewed expansion through the same
pinned worker once for both lengths, record task/selection provenance, apply
whole-trajectory length exclusions, and tag each sequence `agent`. Unlike plain
native traces, expansion inputs intentionally contain memory segments. Therefore
the current native-only `validate_example` must not be reused unchanged for
expansion; the separate adapter validates input memory explicitly. The existing
legacy expansion packer is 32k-only and is not an unchanged dual-length adapter.

The old `expansion_release_selection.py` is explicitly tied to V6/235B source
generations and the historical multidoc2dial replacement. Newly reviewed 27B
traces need their own immutable selection and transport digest/path. The adapter
must consume that approved export, not reinterpret the old selector or combine
old automatic acceptances blindly. No new source selection or export has been
performed by this packing task. Combined manifests should retain the completed
base/native manifest digest and add the reviewed expansion selection/counts.

## Implemented append adapter (not launched on production data)

`data/grouped_stage3_expansion_append.py` provides `pack_partition` and
`finalize`. The adapter keeps all expansion shards in
`packed-cs16-<length>/expansion-staging/<manifest-sha>/part-###` until both
lengths pass aggregate ID/count/hash/category/length and actual-loader checks.
Then it publishes each complete component as `data/expansion`, retains each
original manifest as `base-native-manifest.json`, and writes combined manifests.
It permits input-segment memory for expansion without changing the native-only
validation or any deployed grouped-packing snapshot. Task IDs are unique across
all append partitions; interrupted attempts are quarantined under an owner ID.

Required reviewed transport manifest (all SHA256 values are lowercase hex):

```json
{
  "schema": "reviewed-expansion-transport-v1",
  "status": "reviewed_complete",
  "selection_sha256": "<reviewed selection hash>",
  "approval": {
    "status": "approved",
    "selection_sha256": "<same reviewed selection hash>",
    "artifacts": [{"path": "/absolute/review.json", "sha256": "<file hash>"}]
  },
  "transport_root": "/absolute/immutable/transport",
  "rows": 123,
  "task_ids_sha256": "<SHA256 of sorted task IDs joined by newline, with trailing newline>",
  "files": [{"name": "accepted-00000.parquet", "rows": 123, "sha256": "<file hash>"}],
  "base_snapshot": {
    "root": "/data/stage3-build-20260906/grouped-packing-20260909-v3",
    "manifest_sha256": {"16384": "<base manifest file hash>", "32768": "<base manifest file hash>"}
  }
}
```

Each transport row needs an explicit nonempty `task_id`, `source_dataset`,
`sub_dataset`, `compression_scope="input_segments"`,
`data_type="agent_trajectory"`, and lossless `messages`/`tools` (lists or JSON
strings). The adapter rejects unharvested assistant thought/preamble. The old
exporter does not always preserve task_id separately, so the new reviewed export
must include that field.

On Modal CPU workers call
`pack_partition(manifest_path, manifest_sha256, partition, partitions, owner_id,
checkpoint=volume.commit)`, with a stable owner ID across platform retries;
then `finalize(manifest_path, manifest_sha256, partitions,
checkpoint=volume.commit)`. No generation/source-selector/export is invoked.
Fixture tests are in `tests/test_grouped_stage3_expansion_append.py`; their
unique test-only CPU app is `lclm-grouped-expansion-append-tests-20260909-v1`,
app `ap-EUaFggLQWL6pXF08EATFhU`, call `fc-01M22EXBVRMJM2W5XXKMSNZHNA`.
All 5 tests passed in 43.13 seconds on Modal CPU, including actual pinned worker
and loader checks for both lengths, input-memory/loss-mask preservation,
staging invisibility, unchanged mixed-shard bytes, combined manifests, and
idempotent resume. Only generated test fixtures were used; no production
selection/export/append was launched.

Final append metadata revision also extends the combined raw-file hash inventory
and digest, policy counts, and runtime sample audit records. The same 5 tests
passed again in 32.72 seconds on unique test app
`lclm-grouped-expansion-append-tests-20260909-v2`,
`ap-lQ57YEZtrxg1EjfT1OPuZR`, call `fc-01M22G0B3PF6CTCGNWJ9FG5YX3`.
The tested source snapshot is `/tmp/lclm-grouped-expansion-append-tests-v2-frozen`.
