# Private Volume-v2 storage successor

The original fast backlog has **273,183** exact pending IDs: **896** source probes and **272,287** continuation tasks. The successor preserves the original task bytes, model/revision, eight TP1 H200 replicas, nonthinking request settings, teacher-only prompt guidance, tool schema, and all validators. Its new code hashes and storage configuration are explicit; this is not an unchanged-code claim.

The original `lclm-stage3-data` Volume is version1, whose 500,000-inode hard limit cannot fit the per-task generation layout. The new private output Volume `lclm-stage3-agent-outputs-v2-20260909` is verified version2 (`vo-LgQkHA7vrXR4VsXQbJMq2A`). Existing inputs, original accepts, pilot candidates, and completed original probes remain on `/data`; new artifacts go to `/runs/stage3-build-20260906/sglang27-continuation-storage-20260909-v3`.

Original probe lineage:

- Manifest SHA256: `96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe`.
- Root probe decision SHA256: `fbb29de5edf2572c840a97f6955f39ad939abbe7f3508e347b654d240d53bbcd`.
- Original coordinator: `fc-01M22EVC23780KBFQPQSQCX9DW`, now returned and app stopped by root.
- MAUD's64 completed original probes are reused by exact report/result/raw hashes. Other13 sources failed at Volume reload before reservations/model requests; CPU reconciliation verifies that they contain no reserved or committed attempts before a root-reviewed resume decision can run their original probe IDs.

Active SGLang logs stay on local disk. Startup/chunk/exit snapshots are atomically copied with all output descriptors closed before Volume reload/commit. This fixes the observed open-log Volume-busy failure without changing model serving parameters or restarting servers between sources.

App: `lclm-sglang27-continuation-storage-20260909-v3`. Immutable snapshot: `_modal_run/sglang-storage-successor-20260909-v3-rc1`. Deployment and CPU preparation cannot allocate GPU generation. CPU functions are `tests`, `prepare`, `reconcile_original_probes`, `check_gate`, and `status`; root alone writes reviewed decisions and dispatches `drive_plan(decision_sha256, stage)`.

The bounded512-task continuous queue preserves durable reservations, one committed final result per task ID, and maximum3 attempts only for identified same-input infrastructure restarts. Unknown dispatch/attempt outcomes and exhausted retries remain explicit source holds. Other sources continue. A CPU coordinator automatically hands the same finite plan and known in-flight child IDs to its successor after17hours, before the24hour cap. It commits a dispatch intent and successor call ID; an uncertain spawn is recovered only from an unambiguous call graph and never duplicated.

Terminal output is `coordinators/{stage}-{decision_sha}/completion-report.json`. It binds the new manifest/config/decision, original probe lineage, coordinator handoffs, per-source completed/unresolved/unattempted/held counts, and per-chunk report hashes plus result/raw directories. Chunk reports bind every committed result hash, which binds raw capture bytes. Partial worker failures remain unresolved, never silently completed or relabeled unattempted. The report is intended for root's finite format-audit → internal accepted export →16k/32k agent-group append chain; `approved_for_release` remains false.

## Root decision schema

Write `/runs/.../resume-probe-decision.json` only after CPU reconciliation and original-app shutdown are verified:

```json
{
  "decision": "resume_original_source_probes_on_volume_v2",
  "reviewed_by": "root",
  "approved_for_generation": true,
  "approved_for_release": false,
  "scope": "source_probes_only",
  "backlog_manifest_sha256": "<new prepared manifest SHA256>",
  "origin_probe_manifest_sha256": "96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe",
  "origin_probe_decision_sha256": "fbb29de5edf2572c840a97f6955f39ad939abbe7f3508e347b654d240d53bbcd",
  "generation_config": "<full new manifest generation_config object>",
  "original_coordinator_stopped": true,
  "original_coordinator_call_id": "fc-01M22EVC23780KBFQPQSQCX9DW",
  "reconciliation_review": "<root's concrete stopped-app and no-attempt evidence>",
  "reused_probe_reports": {"maud": "<original MAUD report SHA256>"},
  "never_attempted_probe_sources": ["<exact13 remaining source names from reconciliation>"]
}
```

Run CPU `check_gate(hash, "probe")` before `drive_plan(hash, "probe")`. The gate independently revalidates the original completed-pilot/serving evidence and original probe approval. Reused reports and raw/result bytes are checked. The two source sets must partition the original14 approved probe sources exactly once.

After mixed-storage probe review, write `/runs/.../continuation-decision.json`:

```json
{
  "decision": "continue_reviewed_sources_with_27b_on_volume_v2",
  "reviewed_by": "root",
  "approved_for_generation": true,
  "approved_for_release": false,
  "scope": "reviewed_sources",
  "backlog_manifest_sha256": "<new prepared manifest SHA256>",
  "origin_probe_manifest_sha256": "96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe",
  "origin_probe_decision_sha256": "fbb29de5edf2572c840a97f6955f39ad939abbe7f3508e347b654d240d53bbcd",
  "resume_probe_decision_sha256": "<exact root resume decision SHA256>",
  "generation_config": "<full new manifest generation_config object>",
  "source_reviews": {
    "<approved source>": {
      "approved_for_generation": true,
      "probe_report_sha256": "<exact original-or-successor complete64-row probe report hash>",
      "evidence_review": "<source-specific grounded quality and format/subgroup review>"
    }
  }
}
```

Run CPU `check_gate(hash, "continuation")` before `drive_plan(hash, "continuation")`. MAUD continuation chunk1 uses the original probe report; other sources' chunk1 uses the successor probe report. Later predecessors always come from the new output Volume. No source continuation is approved merely by its self-judge acceptance count.

## CPU validation and preparation completed

- App ID: `ap-PaoTiXTzZGxtSdAQolzarg`.
- CPU tests: **84 passed in0.52seconds**, call `fc-01M22FJ1XC0VSK65Z0BRE5Q93V`.
- Read-only original-probe reconciliation: call `fc-01M22FJ26HEZ5BTQWX51RYJYXR`; MAUD reused report SHA `93c0c696a01738888215147c5d45cc1c16ffc827cb92151d4defc7acecddf346`, all other13 sources verified never reserved, no unresolved source.
- CPU preparation: `fc-01M22FKTDKG9AHG6X82EZPJWPQ`, complete.
- New manifest SHA256: `16598d3f60dec5e4463fe1ce52a8802a1f90e1c7f997ac524aecea664005423b`.
- Exact272,287 continuation tasks:65,352 old rejects plus206,935 previously unattempted. Sorted remaining task-ID digest: `8d7bd7b31bdec5d39f198848616448619e90e7cabd56cac0f524b21df4423848`.
- Exact manifest bytes and validation/reconciliation results: `_modal_run/storage-successor-v3/` and the immutable deployment snapshot.

Root review and dispatch remain separate. This subtask has not written an approval record, spawned successor GPU generation, exported accepted data, published to HuggingFace, or started training.
