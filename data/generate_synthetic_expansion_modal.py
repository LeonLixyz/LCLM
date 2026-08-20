#!/usr/bin/env python3
"""Generate and verify synthetic selective-expansion traces on Modal.

The 235B model sees compressed ``seg_i`` blocks, the native expand tool, and
original segment text returned only after a valid tool call. Programmatic task
state and gold answers remain outside its context. Only traces that expand every
supporting segment and emit the exact final answer are harvested.

Pilot example:

    modal run --detach data/generate_synthetic_expansion_modal.py \
        --count 25 --run-name pilot-long-seg-v2
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lclm-synthetic-expansion"
MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_REVISION = "ac9c66cc9b46af7306746a9250f23d47083d689e"
SERVED_MODEL_NAME = "qwen3-235b-a22b-instruct-2507"
MODEL_PORT = 8000
MODEL_STARTUP_TIMEOUT_SECONDS = 90 * 60

PROJECT_ROOT = Path("/opt/lclm")
DATA_ROOT = Path("/data/stage3-agent/synthetic-expansion")
CACHE_ROOT = "/cache"
HF_CACHE_PATH = f"{CACHE_ROOT}/huggingface"
VLLM_CACHE_PATH = f"{CACHE_ROOT}/vllm"
FLASHINFER_CACHE_PATH = Path(CACHE_ROOT) / ".cache" / "flashinfer"

hf_cache_volume = modal.Volume.from_name("lclm-hf-cache")
data_volume = modal.Volume.from_name("lclm-stage3-data")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "openai>=2.0,<3",
    )
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            "_modal_run",
        ],
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "FLASHINFER_WORKSPACE_BASE": CACHE_ROOT,
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME)


def _read_completed_task_ids(*paths: Path) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("task_id")
                if isinstance(task_id, str):
                    completed.add(task_id)
    return completed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _normalize_flashinfer_cache_paths() -> None:
    """Make a copied FlashInfer Ninja cache portable to its volume mount."""

    old_prefix = "/root/.cache/flashinfer"
    new_prefix = str(FLASHINFER_CACHE_PATH)
    if not FLASHINFER_CACHE_PATH.exists():
        return
    for build_file in FLASHINFER_CACHE_PATH.rglob("build.ninja"):
        content = build_file.read_text()
        normalized = content.replace(old_prefix, new_prefix)
        if normalized != content:
            build_file.write_text(normalized)


@app.cls(
    image=image,
    gpu="H200:8",
    timeout=24 * 60 * 60,
    startup_timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={
        CACHE_ROOT: hf_cache_volume,
        "/data": data_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
class QwenExpansionAgentGenerator:
    @modal.enter()
    def start_model(self) -> None:
        import subprocess

        _normalize_flashinfer_cache_paths()
        command = [
            "vllm",
            "serve",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            str(MODEL_PORT),
            "--tensor-parallel-size",
            "8",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.90",
            # The persistent Modal volume is exposed as 9P. Prefetching the
            # checkpoint into the container's ample RAM avoids highly variable
            # shard-by-shard random reads during the eight-rank model load.
            "--safetensors-load-strategy",
            "prefetch",
            # Full graph compilation for this 235B MoE can exceed Modal's
            # stalled-input window and trigger a duplicate speculative
            # container. Eager mode avoids that startup race; generation is
            # still tensor-parallel over all eight H200s.
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--uvicorn-log-level",
            "warning",
        ]
        print("Starting model server:", " ".join(command), flush=True)
        self.model_process = subprocess.Popen(command, cwd=PROJECT_ROOT)

        deadline = time.monotonic() + MODEL_STARTUP_TIMEOUT_SECONDS
        health_url = f"http://127.0.0.1:{MODEL_PORT}/health"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.model_process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self.model_process.returncode}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
                        print("vLLM health check passed", flush=True)
                        # Persist any newly compiled FlashInfer/vLLM artifacts
                        # before generation so later containers start warm.
                        hf_cache_volume.commit()
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
            time.sleep(5)
        self.model_process.terminate()
        raise TimeoutError(f"vLLM did not become healthy: {last_error}")

    @modal.exit()
    def stop_model(self) -> None:
        process = getattr(self, "model_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except Exception:
                process.kill()

    @modal.method()
    def generate(
        self,
        *,
        count: int,
        seed: int,
        distractors: int,
        run_name: str,
        max_tool_calls: int,
        temperature: float,
    ) -> dict[str, Any]:
        from collections import Counter

        from openai import OpenAI

        from data.synthetic_expansion_agent import (
            generate_tasks,
            messages_for_openai_api,
            run_agent_rollout,
        )

        if count <= 0:
            raise ValueError("count must be positive")
        if not run_name or "/" in run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be a simple nonempty directory name")

        output_dir = DATA_ROOT / run_name
        tasks_path = output_dir / "tasks.jsonl"
        accepted_path = output_dir / "accepted.jsonl"
        rejected_path = output_dir / "rejected.jsonl"
        report_path = output_dir / "report.json"
        manifest_path = output_dir / "manifest.json"
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks = generate_tasks(count, seed=seed, distractors=distractors)
        manifest = {
            "schema_version": 2,
            "run_name": run_name,
            "count": count,
            "seed": seed,
            "distractors": distractors,
            "max_tool_calls": max_tool_calls,
            "temperature": temperature,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
        if manifest_path.exists():
            with manifest_path.open() as handle:
                existing_manifest = json.load(handle)
            if existing_manifest != manifest:
                raise RuntimeError(
                    "run_name already exists with a different manifest; choose a new run name"
                )
        else:
            _write_json_atomic(manifest_path, manifest)
        if not tasks_path.exists():
            for task in tasks:
                _append_jsonl(tasks_path, task)
        existing_accepted = _read_jsonl(accepted_path)
        existing_rejected = _read_jsonl(rejected_path)
        completed = _read_completed_task_ids(accepted_path, rejected_path)
        client = OpenAI(
            api_key="not-needed",
            base_url=f"http://127.0.0.1:{MODEL_PORT}/v1",
            timeout=180,
            max_retries=2,
        )

        counters: Counter[str] = Counter()
        family_accepted: Counter[str] = Counter()
        compression_accepted: Counter[str] = Counter()
        for trace in existing_accepted:
            counters["accepted"] += 1
            family = trace.get("sub_dataset")
            arm = trace.get("compression_arm")
            if isinstance(family, str):
                family_accepted[family] += 1
            if isinstance(arm, str):
                compression_accepted[arm] += 1
        for trace in existing_rejected:
            verification = trace.get("verification", {})
            reason = verification.get("reason", "unknown_rejection")
            counters[str(reason)] += 1
        started = time.time()

        for task_index, task in enumerate(tasks):
            if task["task_id"] in completed:
                counters["resumed_skip"] += 1
                continue

            def complete(messages, tools):
                response = client.chat.completions.create(
                    model=SERVED_MODEL_NAME,
                    messages=messages_for_openai_api(messages),
                    tools=list(tools),
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=512,
                    seed=seed + task_index,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                message = response.choices[0].message
                return {
                    "content": message.content or "",
                    "tool_calls": [call.model_dump() for call in (message.tool_calls or [])],
                }

            try:
                trace = run_agent_rollout(
                    task,
                    complete,
                    max_tool_calls=max_tool_calls,
                )
                trace["model"] = MODEL_ID
                trace["model_revision"] = MODEL_REVISION
                trace["generation"] = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "max_tokens_per_turn": 512,
                }
            except Exception as exc:
                trace = {
                    "schema_version": 2,
                    "data_type": "synthetic_expansion_agent",
                    "source_dataset": "synthetic-expansion-v2",
                    "sub_dataset": task["family"],
                    "task_id": task["task_id"],
                    "task": task["question"],
                    "gold_answer": task["gold_answer"],
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "verification": {
                        "accepted": False,
                        "reason": f"generation_exception:{type(exc).__name__}:{exc}",
                    },
                }

            verification = trace["verification"]
            reason = verification["reason"]
            counters[reason] += 1
            if verification["accepted"]:
                _append_jsonl(accepted_path, trace)
                family_accepted[task["family"]] += 1
                compression_accepted[trace["compression_arm"]] += 1
            else:
                _append_jsonl(rejected_path, trace)

            if (task_index + 1) % 5 == 0:
                data_volume.commit()
                print(
                    json.dumps(
                        {
                            "processed": task_index + 1,
                            "accepted": counters["accepted"],
                            "reasons": dict(counters),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        accepted = counters["accepted"]
        newly_processed = count - counters["resumed_skip"]
        total_processed = len(existing_accepted) + len(existing_rejected) + newly_processed
        report = {
            "status": "complete",
            "run_name": run_name,
            "output_dir": str(output_dir),
            "requested_tasks": count,
            "newly_processed": newly_processed,
            "resumed_skip": counters["resumed_skip"],
            "accepted": accepted,
            "rejected": total_processed - accepted,
            "acceptance_rate": accepted / total_processed if total_processed else None,
            "failure_reasons": {
                reason: value
                for reason, value in sorted(counters.items())
                if reason not in {"accepted", "resumed_skip"}
            },
            "accepted_by_family": dict(sorted(family_accepted.items())),
            "accepted_by_compression_arm": dict(sorted(compression_accepted.items())),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": seed,
            "distractors": distractors,
            "max_tool_calls": max_tool_calls,
            "temperature": temperature,
            "elapsed_seconds": time.time() - started,
        }
        _write_json_atomic(report_path, report)
        data_volume.commit()
        return report


@app.local_entrypoint()
def main(
    count: int = 25,
    seed: int = 20260818,
    distractors: int = 48,
    run_name: str = "pilot-long-seg-v2",
    max_tool_calls: int = 6,
    temperature: float = 0.1,
) -> None:
    generator = QwenExpansionAgentGenerator()
    # Submit asynchronously before waiting. If the local CLI loses its network
    # connection, Modal retains the server-side function call instead of
    # propagating cancellation into a long, resumable harvest.
    call = generator.generate.spawn(
        count=count,
        seed=seed,
        distractors=distractors,
        run_name=run_name,
        max_tool_calls=max_tool_calls,
        temperature=temperature,
    )
    print(f"Submitted generator call: {call.object_id}", flush=True)
    report = call.get()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
