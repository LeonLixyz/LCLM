# SGLang 27B backlog continuation

Generation is prepared as a separate route and remains conditional on the root agent reviewing the completed 1,000-task pilot. No scale-decision record or GPU generation call has been created by this task.

Implementation: `data/sglang_backlog.py`, `data/sglang_backlog_modal.py`, and `tests/test_sglang_backlog.py`. The app is `lclm-sglang27-backlog-20260909-v1`; outputs are under `/data/stage3-build-20260906/sglang27-backlog-20260909-v1` on `lclm-stage3-data`.

The frozen input is `retry-and-remaining-qwen38-27b-v2-tasks`, including corrected MultiDoc2Dial, the full MAUD label ontology, and the PubMedQA training split. Expected selection is 274,183 original pending tasks (67,088 prior failures + 207,095 never attempted), minus all 1,000 pilot-reserved IDs: 273,183 additional tasks (66,088 prior failures + 207,095 never attempted). Original 235B accepted artifacts and pilot candidates stay separate with original provenance.

Every source starts with a 64-task chunk, followed by chunks of at most 512. A reusable class keeps one pinned Qwen3.8-27B SGLang server warm on eight H200s, with 16 concurrent requests, thinking disabled, the pilot's recommended sampling, 16 tool calls, and a 64-request total task bound. Harvesting removes assistant reasoning/prose and preserves native calls/results. Synthetic tasks use their exact verifier; real tasks use unchanged structural/native-answer and strict semantic checks. Automatic acceptance is not final release approval.

The journal reserves tasks and commits before submission. A reservation without a saved result remains an unresolved attempt requiring explicit reconciliation. It is neither automatically retried nor counted as completed. Source circuit breakers stop on zero acceptance after 64 completed tasks or at least 25% infrastructure/capture errors after 16. These holds do not relax verification or permanently delete pending tasks.

The CPU `prepare` function runs tests, builds frozen chunks, and verifies exact counts and global task-ID separation. `check_gate` only reads and validates an externally provided decision. `dispatch(decision_sha256, max_chunks=8)` checks the gate before GPU allocation and schedules a bounded continuation (1–32 chunks); it never creates authorization itself. `status` reports per-source completion, holds, and unresolved attempts.

Before root launches generation, it must write `scale-decision.json` in the new output directory with these exact fields, using real reviewed evidence:

```json
{
  "decision": "generate_failed_and_unattempted_with_27b",
  "reviewed_by": "root",
  "approved_for_generation": true,
  "approved_for_release": false,
  "backlog_manifest_sha256": "<SHA256 of backlog manifest.json>",
  "pilot_manifest_sha256": "<SHA256 of completed pilot manifest.json>",
  "pilot_results_sha256": "<SHA256 of completed pilot results.jsonl>",
  "pilot_report_sha256": "<SHA256 of completed pilot report.json>",
  "generation_config": "<copy the complete generation_config object from backlog manifest>",
  "evidence_review": {
    "recovery": "<reviewed recovery findings>",
    "speed": "<measured throughput and comparison limits>",
    "quality": "<source-grounded recovered-example review>",
    "parser_failures": "<raw-output versus native-parser findings>",
    "limitations": "<failed-only selection and same-model judge limitations>",
    "reviewed_task_ids": ["<actual reviewed pilot task IDs>"]
  }
}
```

The gate requires exactly 1,000 completed pilot results matching the frozen pilot IDs and report hashes, the fixed backlog counts, and the reviewed generation configuration/code hashes. Placeholder values above are documentation only, not an authorization record.

CPU validation passed: **54 tests in 0.29 seconds**, including selection, changed-input rejection, completed-evidence gating, unknown-attempt continuation, nonthinking harvesting, exact synthetic verification, and truncation/error separation. Deployment used immutable snapshot `_modal_run/sglang-backlog-20260909-v1`; app ID `ap-qZAKkTCYwhmwqB5vI5BLNF`. CPU preparation call `fc-01M22C7Y7ZZYXP932NDFBXTCSZ` completed with exact accounting: **273,183 rows = 66,088 prior failures + 207,095 unattempted**, in 554 chunks. Manifest SHA256: `6f6f7f929e78901da9147439841ec70455b49869832c9e2a864c760ce633e873`.

Before scaling beyond source probes, inspect the first 64 outcomes from each nonempty source, especially LexGLUE and procedural synthetic. The current runner uses 16-task reservation/commit batches, which wait for the slowest task; throughput may be lower than the pilot's steady queue. Source quality review and any scheduler change belong in the root's scale decision and require matching code/configuration hashes.

The new nested candidate outputs are intentionally not fed into the final training packer. A later reviewed export must select the cleaned `trace` fields and retain teacher/judge provenance; no final-release selection is produced here.
