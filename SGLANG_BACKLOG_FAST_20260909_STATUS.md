# Qwen3.8-27B pending-source probes and continuation

This v2 route reuses the exact 273,183 pending v1 task IDs: 66,088 prior failures and 207,095 unattempted tasks. It preserves all 15 source entries, including zero-pending Watsonx. Its 64-case first probes balance LexGLUE's seven configs, procedural synthetic families, or prior-failure/unattempted provenance for the other sources. Probe tasks plus disjoint continuation references must equal the original pending IDs exactly. Frozen pilot IDs and original accepted IDs are excluded explicitly. No original accepted artifacts or pilot candidates are merged or relabeled.

The serving layout matches the completed smoke: eight independent BF16 TP1 SGLang replicas in one H200:8 container, CUDA graphs enabled, up to 16 active requests per replica. The same pinned model, revision, image, tool parser and reasoning parser are used. A continuously refilled task queue reserves durable attempt records before submission and preserves each task's replica across its rollout/judge requests.

Root requested a versioned teacher-only prompt canary for these pending probes: concise direct answers; preserve conditions, exceptions, dates and deadlines; avoid inferred eligibility/procedural exclusions; numerical answers in stated units without calculation prose. BillSum additionally requires one paragraph after FINAL:, no headings/bullets/newlines, and retention of distinct dates and earlier/later-of conditions. Full text/hashes and raw requests are saved. This modifies request copies only. Stored task/training prompts, tool schema, validators, and the frozen 1,000-task pilot remain unchanged. This is a source prompt canary, not an isolated model comparison.

Identified same-Modal-call/input infrastructure restarts can recover missing committed results with new attempt journals/raw paths, at most three attempts per task. Committed results—including failures—are never retried or overwritten. Interrupted attempts are not counted as completed. Different logical calls, changed checksums, exhausted attempts, and unclassified failures remain visible for reconciliation; other sources can continue. Source outcome reports include LexGLUE config/synthetic family counts.

Implementation: `data/sglang_backlog_fast.py`, `data/sglang_backlog_fast_modal.py`, `data/sglang_27b_serving.py`, and `tests/test_sglang_backlog_fast.py`. The balanced probe helper is `data/expansion_source_probes.py`.

App: `lclm-sglang27-backlog-fast-20260909-v2`. Output root: `/data/stage3-build-20260906/sglang27-backlog-fast-20260909-v2` on `lclm-stage3-data`. CPU prepare/test operations do not start GPUs or write approval records. Runtime GPU methods independently validate the root-written review gate before serving requests.

Deployment uses immutable snapshot `_modal_run/sglang-fast-backlog-20260909-v2-rc1`, app ID `ap-ytktCvOAnPqROQOi1HH7j0`. CPU validation passed **74 tests in 0.29 seconds**, covering exact coverage, provenance, prompt-copy preservation, continuously refilled work, missing-result recovery limits, immutable references and source review gates. Test call: `fc-01M22E0SW7VXY8DYPTR81WE4KW`. CPU preparation call `fc-01M22E2RA6TM7M36WC79MR7N52` is running; no probe decision has been written.

Root must write `probe-decision.json` under the output root before invoking `dispatch_source_probes(decision_sha256)`. The exact schema is:

```json
{
  "decision": "generate_failed_and_unattempted_with_27b",
  "reviewed_by": "root",
  "approved_for_generation": true,
  "approved_for_release": false,
  "scope": "source_probes_only",
  "probe_sources": ["<explicit nonempty source names with pending rows>"],
  "backlog_manifest_sha256": "<exact v2 manifest.json SHA256>",
  "pilot_manifest_sha256": "8f4845352decc8bed612302fac8d962d700d8206ad93404b5392bfdd8e663ce9",
  "pilot_results_sha256": "417363d886f1afa02ceaf086125f51f73239082ff229b6101ae89816cd8a4e23",
  "pilot_report_sha256": "<exact completed pilot report.json SHA256>",
  "serving_smoke_report_sha256": "b548e5a5cd806e5b76645a89fd4bcce529fa4436d8993d372509f7ecf74abf13",
  "generation_config": "<copy complete v2 manifest generation_config object>",
  "evidence_review": {
    "recovery": "<root's reviewed recovery evidence>",
    "speed": "<root's throughput assessment and comparison limits>",
    "quality": "<root's source-grounded quality review>",
    "parser_failures": "<root's raw/native parser review>",
    "limitations": "<failed-only selection, correlated judge and prompt-canary limits>",
    "reviewed_task_ids": ["<real reviewed frozen pilot task IDs>"]
  }
}
```

`check_gate(decision_sha256, "probe")` validates without GPU allocation. Probe calls are bounded to each allowed source's chunk 0. No continuation follows automatically.

After reviewing source probes, root can write `continuation-decision.json`, retaining the base fields above, replacing `scope` with `"reviewed_sources"`, and replacing `probe_sources` with:

```json
{
  "probe_decision_sha256": "<exact reviewed probe-decision.json SHA256>",
  "source_reviews": {
    "<approved source>": {
      "approved_for_generation": true,
      "probe_report_sha256": "<exact outputs/source/chunk-00000/report.json SHA256>",
      "evidence_review": "<source-specific quality/subgroup review>"
    }
  }
}
```

Each probe report binds its exact input, decision, manifest, and final result hashes. A held or incomplete source cannot enter continuation. `continue_approved_sources(decision_sha256)` drives the remaining finite allowed backlog through bounded chunks. The CPU coordinator has a 24-hour cap, keeps durable child call IDs across preemption, and returns `continuation_required` when its safe dispatch budget is exhausted. Reinvoke the same function with the same decision hash to continue from that durable state. It does not require manual 32-chunk dispatches. `status()` includes committed, unresolved/inflight, unattempted, subgroup and hold counts, including partially written chunks.

No approval record, GPU probe/production call, release/export, Hugging Face publication, or training run has been created by this task.


## Completed preparation and probe storage successor

CPU preparation completed with exact273,183 pending IDs,896 disjoint probes and272,287 continuation IDs. Manifest SHA256 is `96584acb7b8ba93e45b79310c9793408aa7d5b18176880136db3ef511b5a3cfe`. Root launched probes after independent review; original MAUD completed64/64,32 automatic accepts. Later source calls failed at pre-request Volume reload because live server logs held Volume files open. Root stopped the original app after its coordinator returned. Full generation must use the private Volume-v2 storage successor documented in `SGLANG_STORAGE_SUCCESSOR_20260909_STATUS.md`, preserving completed MAUD outputs and original task IDs. The version1 input Volume cannot hold the final per-task output inode count. No full continuation has been dispatched by this subtask.
