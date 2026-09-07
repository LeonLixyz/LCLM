#!/usr/bin/env python3
"""Generate verified native-expansion trajectories for real-source pilot tasks."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import modal


APP_NAME = "lclm-real-expansion-rollout"
MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_REVISION = "ac9c66cc9b46af7306746a9250f23d47083d689e"
SERVED_MODEL_NAME = "qwen3-235b-a22b-instruct-2507"
MODEL_PORT = 8000
PROJECT_ROOT = Path("/opt/lclm")
DATA_ROOT = Path("/data/stage3-agent/real-expansion/pilots")
CACHE_ROOT = "/cache"
FLASHINFER_CACHE_PATH = Path(CACHE_ROOT) / ".cache" / "flashinfer"

hf_cache_volume = modal.Volume.from_name("lclm-hf-cache")
data_volume = modal.Volume.from_name("lclm-stage3-data")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "openai>=2.0,<3")
    .add_local_dir(
        ".",
        str(PROJECT_ROOT),
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.pyc", "_modal_run"],
    )
    .env(
        {
            "HF_HOME": f"{CACHE_ROOT}/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": f"{CACHE_ROOT}/vllm",
            "FLASHINFER_WORKSPACE_BASE": CACHE_ROOT,
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)
app = modal.App(APP_NAME)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _normalize_flashinfer_cache_paths() -> None:
    old_prefix = "/root/.cache/flashinfer"
    if not FLASHINFER_CACHE_PATH.exists():
        return
    for build_file in FLASHINFER_CACHE_PATH.rglob("build.ninja"):
        content = build_file.read_text()
        normalized = content.replace(old_prefix, str(FLASHINFER_CACHE_PATH))
        if normalized != content:
            build_file.write_text(normalized)


@app.cls(
    image=image,
    gpu="H200:8",
    timeout=24 * 60 * 60,
    startup_timeout=90 * 60,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={CACHE_ROOT: hf_cache_volume, "/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
class RealExpansionGenerator:
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
            "--safetensors-load-strategy",
            "prefetch",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--uvicorn-log-level",
            "warning",
        ]
        print("Starting model server:", " ".join(command), flush=True)
        self.model_process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        deadline = time.monotonic() + 90 * 60
        health_url = f"http://127.0.0.1:{MODEL_PORT}/health"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.model_process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.model_process.returncode}")
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    if response.status == 200:
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
    def generate_full(self) -> dict[str, Any]:
        from openai import OpenAI
        from data.full_expansion_rollouts import generate_all
        data_volume.reload()
        client=OpenAI(api_key='not-needed',base_url=f'http://127.0.0.1:{MODEL_PORT}/v1',timeout=300,max_retries=2)
        return generate_all(client,SERVED_MODEL_NAME,MODEL_REVISION,
            DATA_ROOT/'full-20260906-v3',data_volume.commit,data_volume.reload)

    @modal.method()
    def generate(
        self,
        *,
        run_name: str,
        seed: int,
        max_tool_calls: int,
        temperature: float,
    ) -> dict[str, Any]:
        from openai import OpenAI

        from data.synthetic_expansion_agent import (
            TEACHER_SYSTEM_PROMPT,
            messages_for_openai_api,
            run_agent_rollout,
        )
        from data.real_expansion_agent import verify_real_trace

        output_dir = DATA_ROOT / run_name
        tasks = _read_jsonl(output_dir / "tasks.jsonl")
        if not tasks:
            raise FileNotFoundError(f"no source tasks at {output_dir}")
        accepted_path = output_dir / "accepted.jsonl"
        rejected_path = output_dir / "rejected.jsonl"
        report_path = output_dir / "report.json"
        generation_manifest_path = output_dir / "generation-manifest.json"
        generation_manifest = {
            "run_name": run_name,
            "tasks": len(tasks),
            "seed": seed,
            "max_tool_calls": max_tool_calls,
            "temperature": temperature,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "teacher_system_prompt": TEACHER_SYSTEM_PROMPT,
            "teacher_prompt_saved_in_training_messages": False,
        }
        if generation_manifest_path.exists():
            with generation_manifest_path.open() as handle:
                if json.load(handle) != generation_manifest:
                    raise RuntimeError("generation settings changed for an existing run")
        else:
            _write_json(generation_manifest_path, generation_manifest)

        existing = _read_jsonl(accepted_path) + _read_jsonl(rejected_path)
        completed = {row.get("task_id") for row in existing}
        client = OpenAI(
            api_key="not-needed",
            base_url=f"http://127.0.0.1:{MODEL_PORT}/v1",
            timeout=180,
            max_retries=2,
        )
        counters: Counter[str] = Counter()
        accepted_by_source: Counter[str] = Counter()
        for row in existing:
            reason = str(row.get("verification", {}).get("reason", "unknown"))
            counters[reason] += 1
            if row.get("verification", {}).get("accepted"):
                accepted_by_source[str(row.get("source_dataset"))] += 1

        started = time.time()
        for task_index, task in enumerate(tasks):
            if task["task_id"] in completed:
                counters["resumed_skip"] += 1
                continue

            def complete(messages, tools):
                response = client.chat.completions.create(
                    model=SERVED_MODEL_NAME,
                    messages=messages_for_openai_api(messages),
                    tools=list(tools) if tools else None,
                    tool_choice="auto" if tools else None,
                    parallel_tool_calls=True,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=768,
                    seed=seed + task_index,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                message = response.choices[0].message
                return {
                    "content": message.content or "",
                    "tool_calls": [call.model_dump() for call in (message.tool_calls or [])],
                }

            try:
                trace = run_agent_rollout(task, complete, max_tool_calls=max_tool_calls)
                if not trace.get("rollout_failure_reason"):
                    trace["verification"] = verify_real_trace(task, trace["messages"])
                trace.update(
                    data_type="real_expansion_agent",
                    source_dataset=task["source_dataset"],
                    source_row_id=task["source_row_id"],
                    sub_dataset=task["family"],
                    model=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    generation={
                        "temperature": temperature,
                        "top_p": 0.95,
                        "max_tokens_per_turn": 768,
                        "teacher_prompt_saved_in_training_messages": False,
                    },
                )
            except Exception as exc:
                trace = {
                    "schema_version": 2,
                    "data_type": "real_expansion_agent",
                    "source_dataset": task["source_dataset"],
                    "source_row_id": task["source_row_id"],
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
            reason = str(trace["verification"]["reason"])
            counters[reason] += 1
            if trace["verification"]["accepted"]:
                _append_jsonl(accepted_path, trace)
                accepted_by_source[task["source_dataset"]] += 1
            else:
                _append_jsonl(rejected_path, trace)
            data_volume.commit()
            print(
                json.dumps(
                    {
                        "processed": task_index + 1,
                        "source": task["source_dataset"],
                        "reason": reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        accepted = len(_read_jsonl(accepted_path))
        rejected = len(_read_jsonl(rejected_path))
        report = {
            "status": "complete",
            "run_name": run_name,
            "requested_tasks": len(tasks),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / (accepted + rejected) if accepted + rejected else None,
            "failure_reasons": {
                key: value
                for key, value in sorted(counters.items())
                if key not in {"accepted", "resumed_skip"}
            },
            "accepted_by_source": dict(sorted(accepted_by_source.items())),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "elapsed_seconds": time.time() - started,
            "output_dir": str(output_dir),
        }
        _write_json(report_path, report)
        data_volume.commit()
        return report


@app.local_entrypoint()
def main(
    run_name: str = "real-pilot-20260825-v1",
    seed: int = 20260825,
    max_tool_calls: int = 4,
    temperature: float = 0.0,
    full: bool = False,
) -> None:
    if full:
        print(json.dumps(RealExpansionGenerator().generate_full.remote(),indent=2))
        return
    call = RealExpansionGenerator().generate.spawn(
        run_name=run_name,
        seed=seed,
        max_tool_calls=max_tool_calls,
        temperature=temperature,
    )
    print(f"Submitted generator call: {call.object_id}", flush=True)
    print(json.dumps(call.get(), ensure_ascii=False, indent=2, sort_keys=True))
