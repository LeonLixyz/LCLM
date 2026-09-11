# Corrected LexGLUE generation integration

Separate app `lclm-sglang27-lexglue-defined-20260909-v4` freezes a new task definition for the same186,444 original pending LexGLUE IDs. It preserves the balanced64 original probe IDs and186,380 continuation IDs. The old opaque-label v3 probes remain diagnostic records and are never relabeled as corrected results or included in training.

The exact task version is `lexglue-upstream-ontology-20260909-v1`. The tested transform changes question/raw_question and teacher/training prompts together using primary upstream definitions. IDs, source documents, gold labels, support IDs, tools and original attempt provenance are preserved. CaseHOLD content is unchanged because its choices were already explicit. Input ledgers record original/corrected task SHA values, config and changed status. Every new manifest and raw/result artifact binds the new task bytes, version and ontology.

Shared ontology: `/runs/stage3-build-20260906/reviewed-internal-packing-20260909-v1/lexglue_task_definitions.v1.json`, SHA256 `9fd5814221ddc80c087a6b6a07f78a0397bad0e662a2ef8703f11034d66f6c3e`. It is committed. Source evidence/validation is documented in `LEXGLUE_DEFINITION_20260909_STATUS.md`.

Serving and verification remain the validated pinned Qwen3.8-27B nonthinking setup: eight TP1 H200 replicas, CUDA graphs enabled, native tool parsing with raw token capture, bounded continuous queue, unchanged teacher-only source guidance, strict original validators, per-config counts and source circuit breakers. Active server logs are local; durable snapshots avoid Volume reload failures. Maximum3 attempts are allowed only for identified same-input infrastructure restarts, with one committed result per task/version. New outputs use the private version2 Volume; original data remain unchanged. The finite coordinator keeps known child IDs across automatic handoffs.

Root-owned review gates are separate for corrected64 probes and continuation. No GPU corrected-task call is launched by deployment or CPU preparation. CPU functions: `tests`, `prepare`, `prepare_one`, `finalize_preparation`, `check_gate`, `status`; root dispatches only `drive_plan(decision_sha256, stage)` after actual review.

Preparation completed on call `fc-01M22H0K7EQNXKQVD8H3544RFM`, app `ap-7gIYYPgi7NL8oMljtSmfjY`. The actual manifest at `/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4/manifest.json` has SHA256 `c90e0f49da167c97ccba5d9ceac1f80d659d0661f0785e2828d5a0c7a1c5f860`. All367 corrected chunk reports were verified before publication. Exact coverage is186,444 IDs:64 probes plus186,380 continuation. There are141,444 amended prompts and45,000 unchanged CaseHOLD tasks. Totals are CaseHOLD45,000, ECHR-A8,898, ECHR-B8,898, EURLEX54,079, LEDGAR60,000, SCOTUS4,037, and UNFAIR-ToS5,532. Exact manifest bytes and every generation code hash were independently compared with the immutable rc2 snapshot. Local evidence is `_modal_run/lexglue-defined-v4/manifest.json` and `preparation-result.json`; root separately owns its `preparation.json`, `tests.json`, decisions, and submissions.

The corrected probe task file is `/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4/inputs/chunk-00000/attempt-c7b04e7f6dcd4d04b69b7b5165af5a71/tasks.jsonl`, SHA256 `31930783936a69c7bd75acffb3cd54923072303675f5208243e083dc02772177`. It contains10 CaseHOLD tasks and9 from each other config. The current immutable v4 continuation gate is source-level, not config-selective. Any later per-config plan must use an explicit successor manifest and retain excluded configs as held IDs; it cannot silently reuse the all-source approval.

Final snapshot: `_modal_run/sglang-lexglue-defined-20260909-v4-rc2`. CPU tests passed64 in0.49seconds, call `fc-01M22GZ57KNXD2KMKNCHH4GF2N`. Full corrected input preparation is running on four CPU workers, call `fc-01M22H0K7EQNXKQVD8H3544RFM`. New output root is `/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4`.

## Root probe decision

After preparation, write `<new root>/probe-decision.json`:

```json
{
  "decision": "generate_corrected_lexglue_probe",
  "reviewed_by": "root",
  "approved_for_generation": true,
  "approved_for_release": false,
  "scope": "source_probes_only",
  "probe_sources": ["lex_glue"],
  "backlog_manifest_sha256": "<actual corrected input manifest SHA>",
  "generation_config": "<complete corrected manifest generation_config object>",
  "task_version": "lexglue-upstream-ontology-20260909-v1",
  "ontology_artifact": {
    "path": "/runs/stage3-build-20260906/reviewed-internal-packing-20260909-v1/lexglue_task_definitions.v1.json",
    "sha256": "9fd5814221ddc80c087a6b6a07f78a0397bad0e662a2ef8703f11034d66f6c3e"
  },
  "evidence_review": "<root's actual primary-definition, transform and diagnostic review>"
}
```

Run CPU `check_gate(hash, "probe")`, then `drive_plan(hash, "probe")`. The corrected64 are a prompt/task-definition canary, not evidence of general model superiority.

After actual per-config quality/raw/format review, the continuation decision at `<new root>/continuation-decision.json` retains the exact manifest/config/version/ontology fields and replaces decision/scope/probe_sources with:

```json
{
  "decision": "continue_corrected_lexglue",
  "scope": "reviewed_sources",
  "probe_decision_sha256": "<exact corrected probe decision SHA>",
  "source_reviews": {
    "lex_glue": {
      "approved_for_generation": true,
      "evidence_review": "<actual source/config review>",
      "probe_report_sha256": "<exact complete corrected64 report SHA>"
    }
  }
}
```

Run CPU `check_gate(hash, "continuation")`, then `drive_plan(hash, "continuation")`. All remaining186,380 IDs are finite bounded chunks; no manual32-chunk resubmission is needed. Incomplete or held corrected probes cannot open the continuation gate.

The primary packing chain's readiness target is `<new root>/internal-ready.json`, under the contract in `EXPANSION_COMPLETION_CHAIN_20260909_STATUS.md`. It must be written only after the corrected generation terminal, separate root internal source review, frozen accepted selection, strict transport, uniform manual/repeat filtering and final format audit exist. No readiness or approval record is fabricated by input preparation. No HF upload or model training is included.
