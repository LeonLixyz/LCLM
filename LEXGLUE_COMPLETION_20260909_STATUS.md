# Selected corrected LexGLUE continuation — current plan

**Final checkpoint, Sep10 18:32 UTC:** the reviewed build, including expansion, is complete. Both16k/32k versions are verified and trainable. Final counts and paths: [STAGE3_PACKED_DATA_20260910.md](STAGE3_PACKED_DATA_20260910.md). Earlier checkpoints below are historical.

At Sep10 16:06 UTC, generation terminal checksum accounting is complete for all
123,421 tasks /84,928 automatic accepts. Original export and coordinator failed
on a platform restart because full retry-specific input identities were compared.
Their saved artifacts and frozen source decisions remain unchanged.

CPU execution successor `lclm-lexglue-export-recovery-20260910-v4` passed38 tests
in2.29s, including byte-identical Parquet export and unchanged quality filtering,
corrupt rejected-record detection and logical retry ownership. It uses bounded
parallel frozen reads; all original validation/selection bodies remain intact
apart from the explicit read replacements. Additional recovery lineage is bound
in a new immutable selection document, preserving the original selection.

Recovery worker `fc-01M261336XKVG6738KXTQZWZND` and resumed original completion
`fc-01M261351DC0KXQ693J1VCXR68` are active after the actual CPU gate passed. The original
native/token/label audits and watched readiness consumer are still mandatory.
Root launcher completed; do not duplicate dispatch. Records and exact bindings:
`_modal_run/lexglue-selected-v5/recovery/launch-summary.json`.


At16:19 UTC, preparation and both actual gates have completed. Generation call
`fc-01M23CHS7754139BJ2QXNXP2B0` is active with4,160 completed tasks and2,839
automatic accepts across9 complete chunks, plus80 in-flight attempts. CPU
completion `fc-01M23CHSD69R8F86WE1S49CAGZ` is active and waiting for generation.
Exact manifest: `53416eaaa5367e4ef31707d5534e2744b635cdef29e2bc8490bc073d532c8f99`.
Exact continuation decision: `71d4a5fd9860124ceb320be32fbee9c365c78686d731c2d9ee28f13ce199adfc`.
The recorded launcher completed successfully; do not repeat dispatch. Counts are
before repeat/manual/export checks. No final corrected readiness or append yet.


The all-config v4 completion below is historical and must not be launched for
this plan. Root holds ECHR-B and EURLEX after their corrected64 canary has no
usable accepts. The selected v5 path uses123,421 unattempted corrected tasks
across CaseHOLD, ECHR-A, LEDGAR, SCOTUS and UNFAIR-ToS. All64 canaries remain
diagnostics, excluded from both new generation and training. The62,959 held
unattempted tasks remain outside this finite plan.

New generation app: `lclm-sglang27-lexglue-selected-20260909-v5`,
`ap-RFeqiqlXNV3Sk9jHzaAU9N`. New CPU completion app:
`lclm-selected-lexglue-completion-20260909-v1`, `ap-4hpz4wLHBdAKGhkKLvtoA3`.
Immutable snapshot: `/tmp/lclm-selected-lexglue-20260909-v5-frozen`.
44 generation/preparation tests and34 completion/export/quality tests passed
on Modal; all returned code hashes match the frozen snapshot and working files.
The completion validates staged readiness through the unchanged original
packing-chain consumer before publishing its expected v4 readiness path.

Root source review:
`_modal_run/lexglue-selected-v5/root-semantic-review.json`, SHA
`d3650fc850cd25f7214352290efdd210aece984a8d7455a94596678f4c555542`.
Config selection SHA:
`d07210a4fda05d13d2582f55054885c3f368332df0160a91919ae40639b1a5ae`.
Review is bounded, grounded in source-first notes and actual corrected outputs;
classification plausibility is not exhaustive legal correctness. In particular,
UNFAIR-ToS's six nonrepeated accepts all have the none label; no balanced-accuracy
claim is made. Every future selected accept still requires exact raw/native/
prompt/token/label checks and the unchanged whole-trace repeat/manual filters.

At15:12 UTC preparation `fc-01M23BHTXZM1FT1D3VNHE9A7VT` is active. Root's
recorded launcher `_modal_run/launch_selected_lexglue.py` waits for exact-byte
parent-ID accounting, binds the actual resulting manifest, runs both actual
CPU gates, then dispatches generation and its finite completion dependency.
See `_modal_run/lexglue-selected-v5/launch-summary.json` for actual successful
submissions; do not infer dispatch from deployment or tests alone.

---

# Corrected LexGLUE finite CPU completion

`data/lexglue_completion.py` and `data/lexglue_completion_modal.py` implement a separate CPU-only completion app, `lclm-corrected-lexglue-completion-20260909-v2`. Deployment and tests do not approve or dispatch generation, export production data, append training data, upload to Hugging Face, or train a model. The earlier inert v1 app returned an older warm test snapshot after a same-name redeploy; v2 uses a fresh app name and verifies every returned configuration hash against its immutable final snapshot.

The root-owned corrected continuation decision is the only corrected-source review required. Alongside the existing v4 generation fields it must contain `approved_for_internal_packing: true`, `approved_for_release: false`, `source_allowlist: ["lex_glue"]`, `generation_manifest_sha256` equal to the actual corrected manifest hash, and the exact tested `completion_configuration` object. An optional `additional_manual_exclusions: {path, sha256}` binds a separate root-reviewed JSON object with `reviewed_by: "root"`, `count`, and unique `entries[].task_id`. These extra exclusions do not rewrite or replace the main policy’s common exclusion artifact. No source approval is generated by this app.

Root calls `check_inputs(policy_path, policy_sha, review_spec, manifest_spec)` for a CPU gate, then separately spawns `drive(policy_path, policy_sha, review_spec, manifest_spec, generation_call_id)`. Artifact specs are absolute `{path, sha256}` pairs. The main policy already binds the watched path `/runs/stage3-build-20260906/sglang27-lexglue-defined-20260909-v4/internal-ready.json` and the shared ontology artifact. The drive follows the actual corrected-generation coordinator handoffs and requires a completed, approved source with every manifest continuation chunk complete and zero held, unresolved, or unattempted rows. V4 currently covers all186,380 continuation tasks; it is not a per-config approval gate. The64 corrected probes are included from the separately reviewed chunk0 report. Original opaque LexGLUE traces cannot enter this path.

After both exact base manifests are ready, a CPU stage freezes every actual task/report/result hash and applies the common no-repeated-immutable-expansion policy plus the exact manual-ID union. Generation verdicts and raw outputs remain unchanged. Counts distinguish accepted candidates, repeated expansions, manual exclusions, overlap, and eligible rows, with per-config totals. The strict exporter checks every raw capture and the saved corrected task prompts, preserves native calls/tools and provenance, and exports only eligible accepts.

Each bounded transport shard receives an independent CPU audit using the existing pinned decoder/encoder and assistant-only label checks. All selected rows must pass canonical native call, evidence-body, prompt, source verification, and token/label checks. A staged readiness candidate is validated through the exact frozen primary-chain consumer before the watched final marker is published. The primary chain then includes this typed completed transport in its single final union and16k/32k append.

Durable state lives under the v4 root’s `private-completion/<policy-sha>/`. Completed selection, export, and shard-audit stages survive same-input platform retries. Partial attempts use separate paths. Spawned CPU calls and generation/CPU handoffs are recorded, with bounded automatic handoffs before the24-hour CPU timeout. Unknown dispatches or failed format checks remain explicit errors rather than being replayed or counted complete. The app uses at most four8-CPU audit containers and one4-CPU selection/export container; it uses no GPU.

All42 CPU tests passed in2.33 seconds on Modal call `fc-01M22HSC9TZK3VDQYSX2TEC570`. Final app ID: `ap-44nn0irKPgzIHPSA6NP6V2`; immutable snapshot: `_modal_run/lexglue-completion-20260909-v2-frozen`. Every returned completion, generator, and main-chain code hash was compared with the actual snapshot bytes. Configuration JSON is `_modal_run/lexglue-completion/configuration.json`, SHA256 `dd9ddfe06dfe25e19960ec6d00e45ae4016d933f65a3a734e59326f37084bc84`. Test and deployment records share that directory. Root owns the actual reviewed continuation and completion dispatch.
