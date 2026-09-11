# Finite private expansion completion chain

**Final checkpoint, Sep10 18:32 UTC:** the reviewed build, including expansion, is complete. Both16k/32k versions are verified and trainable. Final counts and paths: [STAGE3_PACKED_DATA_20260910.md](STAGE3_PACKED_DATA_20260910.md). Earlier checkpoints below are historical.

Separate app `lclm-expansion-completion-chain-20260909-v1` implements CPU orchestration only. Deployment and fixture tests do not create approvals or run production selection, export, append, HuggingFace upload, or model training.

The chain follows the actual v3 continuation call and recorded coordinator handoffs, then reads and freezes its terminal report. Every selected original source must have all approved continuation chunks complete, with exact input/result bindings and zero unresolved or unattempted IDs. Its reviewed source probe is included separately by the actual source-review report hash. Both completed base manifests are required at `/data/stage3-build-20260906/grouped-packing-20260909-v3/packed-cs16-{16384,32768}/manifest.json`; the adapter checks full byte-audit status, tokenizer revisions and lengths.

Stages are committed separately: frozen27B selection → strict27B transport → additional corrected-source readiness → typed union selection → union transport → independent bounded append partitions → dual-length finalization. A completed strict transport is reused if a later union stage retries. Failed partial attempts remain separate; completed markers and stage reports are preserved. The CPU coordinator hands the same finite chain to a durable successor before its24-hour cap, and records all actual child IDs. Uncertain dispatches are never blindly duplicated.

Root's `no-repeated-immutable-expansion-v1` policy applies uniformly to accepted235B, pilot, v2/v3 and corrected-source expansion traces. Entire redundant traces and explicit manual IDs are excluded only from training. Reports separate dynamic exclusions, manual exclusions, overlap, union exclusions and eligible counts by source/teacher. Original verdicts, prompts, calls, observations and raw traces are unchanged. Native agent data are outside this expansion filter. Union identity uses global unique exported task IDs; teacher provenance and unavailable legacy raw capture remain explicit.

**Original opaque-label LexGLUE is blocked in all legacy235B, pilot and original v3 streams.** An additional corrected-Lex dependency must become ready before the one final union/append. The chain can wait for this reviewed artifact while retaining every completed earlier stage; it has no old-Lex fallback.

## Initial root policy

Root alone writes an actual policy, then calls CPU `check_policy(path, sha)` and `drive(path, sha)`. The schema is `finite-reviewed-expansion-packing-v1`:

```json
{
  "schema": "finite-reviewed-expansion-packing-v1",
  "reviewed_by": "root",
  "approved_for_internal_packing": true,
  "approved_for_release": false,
  "configuration": "<full configuration object from final CPU tests>",
  "reviewed_sources": ["<all selected sources, including explicitly deferred corrected sources>"],
  "sglang27_sources": ["<approved original v3 sources; NEVER lex_glue>"],
  "source_reviews": {
    "<each original selected source>": {
      "approved_for_internal_packing": true,
      "evidence_review": "<actual source-specific review>",
      "artifacts": [{"path": "<absolute frozen review>", "sha256": "<actual SHA>"}]
    }
  },
  "approval_artifacts": [{"path": "<absolute root review>", "sha256": "<actual SHA>"}],
  "manual_exclusions": {"path": "<absolute root exclusions JSON>", "sha256": "<actual SHA>"},
  "generation": {
    "app": "lclm-sglang27-continuation-storage-20260909-v3",
    "initial_call_id": "<actual continuation drive_plan call ID>",
    "manifest": {"path": "<absolute v3 manifest>", "sha256": "<actual SHA>"},
    "continuation_decision": {"path": "<actual reviewed generation decision>", "sha256": "<actual SHA>"},
    "resume_probe_decision": {"path": "<actual v3 resume decision>", "sha256": "<actual SHA>"}
  },
  "base": {
    "root": "/data/stage3-build-20260906/grouped-packing-20260909-v3",
    "manifest_sha256": {"16384": "<optional known final base SHA>", "32768": "<optional known final base SHA>"}
  },
  "append_partitions": 16,
  "retained_streams": ["<explicit typed235B/pilot specs below; NEVER opaque lex_glue>"],
  "additional_readiness_dependencies": [
    {
      "stream_id": "corrected_lexglue",
      "source_allowlist": ["lex_glue"],
      "task_version": "<actual corrected task version>",
      "ontology_artifact": {"path": "<frozen primary-source ontology/correction artifact>", "sha256": "<actual SHA>"},
      "readiness_path": "<absolute future corrected-source ready.json>"
    }
  ]
}
```

Omit optional `base.manifest_sha256` until those manifests exist; the chain still waits and freezes/verifies their real hashes. Additional source names must be absent from all original streams. Their separate review is deferred to the readiness artifact, so they must not be listed in initial `source_reviews`. All other selected sources require an initial source-specific review.

`retained_streams` follows `reviewed_expansion_adapters.py` exactly. Each includes a unique `stream_id`, `kind` and explicit `source_allowlist`. Root supplies all input/review paths and hashes; the chain derives final `counts` and `task_ids_sha256`:

- `legacy_accepted_jsonl`: source, input{path,sha256,rows}, generation_manifest, source_report, format_audit, review_artifacts, raw_capture_status=unavailable_legacy, optional source_tasks{path,sha256,rows} if original source-row identity is absent.
- `frozen_pilot_jsonl`: manifest, report, results{path,sha256,rows}, format_audit, review_artifacts, ALL per-source inputs{source,path,sha256}, raw_root. Only its explicit source_allowlist contributes rows; all original1000 results/inputs/raw bindings are validated.

The12-ID manual exclusion file and uniform repeat-policy code hash are frozen in the root policy. Mixed union candidate denominators are explicitly typed; they are not combined model accuracy.

## Corrected-source readiness contract

The corrected generation/export workflow writes its readiness JSON LAST, after its independently reviewed generation, strict selection/transport and format audit complete. The chain does not select a readiness file by glob or infer approval from its filename. It uses only the explicit future path and validates:

```json
{
  "schema": "reviewed-corrected-expansion-ready-v1",
  "status": "complete",
  "chain_policy_sha256": "<exact initial chain policy SHA>",
  "task_version": "<same version as dependency>",
  "ontology_artifact": {"path": "<same frozen artifact>", "sha256": "<same SHA>"},
  "source_allowlist": ["lex_glue"],
  "training_quality_code_sha256": "<exact expansion_training_quality.py SHA from chain configuration>",
  "manual_exclusions_sha256": "<exact manual file SHA from chain policy>",
  "source_review": {"path": "<separate root corrected-source review>", "sha256": "<actual SHA>"},
  "generation_manifest": {"path": "<corrected generation manifest>", "sha256": "<actual SHA>"},
  "generation_terminal": {"path": "<completed corrected generation terminal>", "sha256": "<actual SHA>"},
  "selection": {"path": "<strict reviewed-sglang27-export-v1 selection>", "sha256": "<actual SHA>"},
  "transport": {"path": "<completed reviewed-expansion-transport-v1 manifest>", "sha256": "<actual SHA>"},
  "format_audit": {"path": "<complete accepted transport audit>", "sha256": "<actual SHA>"}
}
```

Its source review must contain `reviewed_by=root`, `approved_for_internal_packing=true`, `approved_for_release=false`, exact `task_version`, `ontology_artifact`, `source_allowlist`, and `generation_manifest_sha256`. The actual generation manifest must contain the same task version and ontology artifact. The terminal uses `manifest_sha256`, `successor_terminal_status=terminal` and `sources[source]` with approved_for_stage=true, held=false, unresolved=0, unattempted=0 and positive completed=expected_rows.

The strict selection additionally binds `generation_manifest`, `generation_terminal` and `training_quality_accounting`; the latter includes the uniform quality `code_sha256` and exact accepted/manual/dynamic/overlap/excluded_union/eligible `counts`. Its `base_snapshot` must equal the primary chain's frozen base snapshot, available in the primary chain's committed strict selection/state once ready. The completed transport must bind that selection and base. The format audit contains `status=passed`, `failed_rows=0`, `selection_sha256` and `transport_manifest_sha256`. Actual transport messages are inspected again for repeated immutable expansion during union freezing. All reused shard hashes, source counts and exported-ID digests are checked again by the typed adapter.

## Final artifacts

Chain state and completion are under `/runs/stage3-build-20260906/expansion-completion-chain-20260909-v1/<policy SHA>/`. `completion.json` binds the generation terminal/handoffs, both base hashes, each frozen selection and completed transport, added-source readiness hashes, and final packed16k/32k manifest hashes plus append summary/counts. Original base/native manifests are archived by the append adapter, original categories remain intact, and expansion sequences are agent-only. External release remains false; no HF upload or model training is included.

## CPU validation completed

Final immutable snapshot: `_modal_run/expansion-completion-chain-20260909-v1-rc2`. Final CPU call `fc-01M22GHR7HH8ASKXB0MKG9Y4FK` passed **64 tests in1.99seconds**. Cases cover actual three-teacher export, source/ID/exclusion accounting, unchanged generation bytes, exact exclusion overlap, changed/held/incomplete generation rejection, completed component reuse, old-Lex rejection and deferred corrected-source readiness policy/version checks. The grouped append adapter separately passed its5 actual worker/runtime-loader fixture tests. Final configuration and test evidence are in `_modal_run/expansion-completion-chain/`; root should copy that exact configuration into its policy.

No production chain, approval record, accepted selection/export, expansion append, HF publication or training has been launched by this subtask.
