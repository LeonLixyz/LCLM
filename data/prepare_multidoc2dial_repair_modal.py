"""Build a separate corrected source shard; never modify live V6 inputs/outputs."""
import hashlib
import json
from pathlib import Path
import modal
from data.stage3_full_modal import image, volume

app = modal.App("lclm-prepare-multidoc2dial-repair")
BASE = Path("/data/stage3-agent/real-expansion")
OUTPUT = BASE / "pilots/multidoc2dial-chronological-v1-inputs"


@app.function(image=image, cpu=8, memory=16384, timeout=1800, volumes={"/data": volume})
def prepare():
    import subprocess
    from data.full_expansion_tasks import load
    from data.multidoc2dial_dialogue import VERSION, chronological_question
    from data.synthetic_expansion_agent import format_training_user_prompt, format_rollout_user_prompt
    subprocess.run(["python", "-m", "pytest", "/opt/lclm/tests/test_multidoc2dial_dialogue.py", "-q"], check=True)
    snapshot = json.loads((BASE / "sources/multidoc2dial/snapshot-manifest.json").read_text())
    revision = "1108a969d076f04c7367f0c2427d1c5d6d6bdaa0"
    if snapshot["revision"] != revision:
        raise ValueError("Unknown source revision")
    report_path = OUTPUT / "multidoc2dial.build.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["version"] != VERSION:
            raise ValueError("Incompatible prior repair")
        return report
    dialogues = {r["dial_id"]: r for r in load("multidoc2dial", "materialized/dialogue_domain/train")}
    rows = {r["id"]: r for r in load("multidoc2dial", "materialized/multidoc2dial/train")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "multidoc2dial.tasks.jsonl"
    if target.exists():
        raise ValueError("Incomplete prior repair output requires inspection")
    seen = set(); parent_hash = hashlib.sha256(); output_hash = hashlib.sha256(); previews = []
    parent = BASE / "pilots/full-20260906-v3/multidoc2dial.tasks.jsonl"
    with parent.open("rb") as stream, target.with_suffix(".tmp").open("wb") as output:
        for line in stream:
            parent_hash.update(line); task = json.loads(line)
            source_id = task["source_row_id"]
            if source_id in seen:
                raise ValueError("Duplicate source task")
            seen.add(source_id); row = rows[source_id]
            dialogue = dialogues[source_id.rsplit("_", 1)[0]]
            new_question = chronological_question(row, dialogue)
            prefix = f'Use these source documents: {row["title"]}.\n'
            suffix = "\nReturn exactly FINAL: followed by your answer."
            if task["question"] != prefix + row["question"].replace("[SEP]", "\n") + suffix:
                raise ValueError("Unexpected original task rendering")
            if task["gold_answer"] != row["utterance"]:
                raise ValueError("Original task reference changed")
            task["parent_task_id"] = task["task_id"]
            task["task_id"] = "rea4-md2d-" + hashlib.sha256((VERSION + ":" + source_id).encode()).hexdigest()[:24]
            task["question_rendering_version"] = VERSION
            task["question"] = task["raw_question"] = prefix + new_question + suffix
            task["training_user_prompt"] = format_training_user_prompt(task["segments"], task["question"])
            task["user_prompt"] = task["training_user_prompt"]
            task["rollout_user_prompt"] = format_rollout_user_prompt(task["segments"], task["question"])
            encoded = (json.dumps(task, ensure_ascii=False) + "\n").encode()
            output.write(encoded); output_hash.update(encoded)
            if len(previews) < 3:
                previews.append({k: task[k] for k in ("task_id", "parent_task_id", "question", "gold_answer")})
    if seen != set(rows) or len(seen) != 21451:
        raise ValueError("Corrective task coverage differs from original training source")
    target.with_suffix(".tmp").replace(target)
    report = {"status": "prepared", "approved": False, "version": VERSION,
              "source": "multidoc2dial", "source_revision": revision, "counts": {"tasks": len(seen)},
              "parent_tasks_sha256": parent_hash.hexdigest(), "tasks_sha256": output_hash.hexdigest(),
              "tests_passed": True, "raw_dialogue_roundtrip_rows": len(seen),
              "preserved": ["original documents", "segment IDs", "support IDs", "reference answers"],
              "next_gate": "Separate corrective Qwen235B pilot and review before full regeneration"}
    report_path.write_text(json.dumps(report, indent=2))
    (OUTPUT / "preview-questions.json").write_text(json.dumps(previews, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(prepare.remote(), indent=2))
