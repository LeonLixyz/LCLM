"""Bounded 32-task corrective pilot, isolated from the active full generator.

Deploy this app and spawn pilot() once. This module deliberately provides no
full-generation entrypoint: corrective source review must precede scaling.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal
from data.generate_real_expansion_modal import (
    image, data_volume, hf_cache_volume, CACHE_ROOT, PROJECT_ROOT,
    MODEL_ID, MODEL_REVISION, SERVED_MODEL_NAME,
)

APP_NAME = "lclm-md2d-corrective-pilot-v1"
app = modal.App(APP_NAME)
INPUTS = Path("/data/stage3-agent/real-expansion/pilots/multidoc2dial-chronological-v1-inputs")
OUTPUTS = INPUTS.parent / "multidoc2dial-chronological-v1-pilot"
EXPECTED_SHA = "ed8c72b55cd790cdc009b60e59088174ba8dfd35fb9f4d76d594379e1b005649"


@app.function(image=image, gpu="H200:8", cpu=16, memory=65536, timeout=7200,
              max_containers=1, scaledown_window=60,
              volumes={"/data": data_volume, CACHE_ROOT: hf_cache_volume},
              secrets=[modal.Secret.from_name("huggingface")])
def pilot():
    import subprocess
    from openai import OpenAI
    from data.full_expansion_rollouts import generate_all
    from data.multidoc2dial_dialogue import VERSION
    data_volume.reload()
    report = json.loads((INPUTS / "multidoc2dial.build.json").read_text())
    digest = hashlib.sha256()
    with (INPUTS / "multidoc2dial.tasks.jsonl").open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if (report["status"] != "prepared" or report["version"] != VERSION
            or report["counts"]["tasks"] != 21451 or not report["tests_passed"]
            or report["raw_dialogue_roundtrip_rows"] != 21451
            or report["tasks_sha256"] != EXPECTED_SHA or digest.hexdigest() != EXPECTED_SHA):
        raise ValueError("Corrective input receipt/hash mismatch")
    provenance = {"source": "multidoc2dial", "question_rendering_version": VERSION,
                  "source_revision": report["source_revision"], "tasks_sha256": EXPECTED_SHA,
                  "parent_tasks_sha256": report["parent_tasks_sha256"]}
    command = ["vllm", "serve", MODEL_ID, "--revision", MODEL_REVISION,
               "--served-model-name", SERVED_MODEL_NAME, "--host", "127.0.0.1", "--port", "8000",
               "--tensor-parallel-size", "8", "--max-model-len", "32768",
               "--gpu-memory-utilization", "0.90", "--safetensors-load-strategy", "prefetch",
               "--enforce-eager", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--uvicorn-log-level", "warning"]
    print("Starting isolated corrective-pilot server", flush=True)
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        deadline = time.monotonic() + 5400
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Corrective server exited: {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Corrective pilot server startup timed out")
            time.sleep(5)
        client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1",
                        timeout=300, max_retries=2)
        return generate_all(client, SERVED_MODEL_NAME, MODEL_REVISION, INPUTS,
                            data_volume.commit, data_volume.reload, output_root=OUTPUTS,
                            sources=["multidoc2dial"], pilot_limit=32, strict_semantics=True,
                            input_provenance=provenance)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=30)
