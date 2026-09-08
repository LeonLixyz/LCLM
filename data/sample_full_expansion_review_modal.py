"""Prepare hash-selected single/multi-expansion examples for manual QA.

Never changes generation files or marks a release approved. Full training rows
are retained in the sample artifact; evidence views omit only the user context,
which duplicates the original documents, and never truncate tool observations.
"""
import hashlib
import json

import modal

from data.audit_full_expansion_modal import AUDITS, GENERATED
from data.stage3_full_modal import ROOT, image, volume

app = modal.App("lclm-full-expansion-review-samples")


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={"/data": volume})
def sample(source: str):
    from data.build_full_expansion_tasks_modal import SOURCES
    if source not in SOURCES:
        raise ValueError("Unknown source")
    audit = json.loads((AUDITS / f"{source}.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("Full source format audit must pass first")
    selected = {}; digest = hashlib.sha256(); count = 0
    with (GENERATED / f"{source}.accepted.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line); row = json.loads(line); count += 1
            segments = {c["function"]["arguments"]["segment_id"]
                        for m in row["messages"] for c in m.get("tool_calls", [])}
            category = "single" if len(segments) == 1 else "multi"
            score = hashlib.sha256(("final-review-v1:" + row["task_id"]).encode()).hexdigest()
            if category not in selected or score < selected[category][0]:
                selected[category] = (score, row)
    if digest.hexdigest() != audit["accepted_file_sha256"] or count != audit["rows"]:
        raise ValueError("Accepted file changed since full audit")
    examples = [{"category": k, "selection_hash": v[0], "training_row": v[1]}
                for k, v in sorted(selected.items())]
    artifact = {"status": "awaiting_manual_review", "approved": False,
                "source": source, "accepted_file_sha256": digest.hexdigest(),
                "selection": "lowest SHA256(final-review-v1:task_id) per single/multi bucket",
                "examples": examples}
    destination = ROOT / "full-expansion-manual-review-samples"
    destination.mkdir(exist_ok=True)
    (destination / f"{source}.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    views = [{"category": x["category"], "task_id": x["training_row"]["task_id"],
              "question": x["training_row"]["task"], "reference": x["training_row"]["gold_answer"],
              "verification": x["training_row"]["verification"],
              "tools": x["training_row"]["tools"],
              "messages_without_user_context": [m for m in x["training_row"]["messages"]
                                                 if m["role"] != "user"]} for x in examples]
    (destination / f"{source}.evidence.json").write_text(json.dumps(views, ensure_ascii=False, indent=2))
    volume.commit()
    return {"source": source, "status": artifact["status"], "path": str(destination),
            "samples": [{"category": x["category"], "task_id": x["training_row"]["task_id"]}
                        for x in examples]}


@app.local_entrypoint()
def main(source: str):
    print(json.dumps(sample.remote(source), indent=2))
