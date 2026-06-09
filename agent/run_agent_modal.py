"""Modal: run all agent-app benchmarks for cs16-causal-mlp-3e5.

Spawns parallel containers:
  * RULER agent  — 3 containers (one per context length: 4k, 8k, 16k)
                   each loops the 13 RULER subtasks
  * LongBench    — 1 container, all subtasks
  * LongHealth5  — 1 container

Run:
  modal run agent/run_agent_modal.py
"""

import modal

PROJECT_ROOT = "/app"
VOLUME_PATH = "/vol"

vol = modal.Volume.from_name("kv-cache-compression", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.5.0-devel-ubuntu22.04")
    .run_commands(
        "apt-get update",
        "apt-get install -y software-properties-common",
        "add-apt-repository ppa:deadsnakes/ppa -y",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-dev python-is-python3",
        "update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2",
        "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2",
        "rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        "apt-get update && apt-get install -y python3-pip git curl bash build-essential"
    )
    .add_local_dir(
        ".",
        f"{PROJECT_ROOT}/",
        copy=True,
        ignore=[
            "__pycache__", ".git", "*.pyc",
            ".venv", ".venv-verify", ".cache",
            "checkpoints",
            "*.safetensors", "*.bin", "*.pt", "*.ckpt",
            "wandb", "outputs", "results",
            "**/.ipynb_checkpoints",
        ],
    )
    .run_commands(
        "/bin/bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "export PATH=/root/.local/bin:$PATH && "
        f"cd {PROJECT_ROOT} && uv sync && "
        f"uv pip install rouge fuzzywuzzy python-Levenshtein jieba'",
    )
    .env(
        {
            "PATH": f"{PROJECT_ROOT}/.venv/bin:/root/.local/bin:$PATH",
            "VIRTUAL_ENV": f"{PROJECT_ROOT}/.venv",
            "PYTHONPATH": f"{PROJECT_ROOT}",
            "HF_HOME": f"{VOLUME_PATH}/hf_cache",
        }
    )
)

# Model selection — required via env var:
#   LCLM_MODEL_REPO=org/name  LCLM_MODEL_SHORT=name modal run ...
# ``MODEL_SHORT`` is used to derive the on-volume checkpoint directory
# and results root, so two runs with different ``MODEL_SHORT`` values
# share neither.
import os as _os
MODEL_REPO = _os.environ.get("LCLM_MODEL_REPO")
MODEL_SHORT = _os.environ.get("LCLM_MODEL_SHORT")
if not MODEL_REPO or not MODEL_SHORT:
    raise RuntimeError(
        "Set LCLM_MODEL_REPO=<hf-repo-id> and LCLM_MODEL_SHORT=<short-name> before invoking."
    )
CKPT_DIR = f"{VOLUME_PATH}/checkpoints/{MODEL_SHORT}"
RESULTS_ROOT = f"{VOLUME_PATH}/{MODEL_SHORT.replace('-', '_')}_results/agent"

app = modal.App(name=f"agent-{MODEL_SHORT}", image=image)

RULER_CONTEXT_LENGTHS = [4096, 8192, 16384]

NIAH_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery",
]


def download_checkpoint(repo: str, ckpt_dir: str) -> bool:
    import json, os, re, subprocess, urllib.request
    d = os.path.join(ckpt_dir, "decoder")
    if os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d)):
        return True
    print(f"  Downloading {repo}...")
    url = f"https://huggingface.co/api/models/{repo}/refs"
    with urllib.request.urlopen(url) as resp:
        refs = json.loads(resp.read())
    branches = [b["name"] for b in refs.get("branches", [])]
    branch = None
    for c in ["stage3-hf", "stage3_final_hf", "stage3_final"]:
        if c in branches:
            branch = c
            break
    if not branch:
        latest = 0
        for b in branches:
            m = re.match(r"^stage3_step_(\d+)_hf$", b)
            if m and int(m.group(1)) > latest:
                latest = int(m.group(1))
                branch = b
    if not branch:
        print(f"  No stage3 branch: {branches}")
        return False
    print(f"  Using branch: {branch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    subprocess.run(
        ["huggingface-cli", "download", repo, "--revision", branch, "--local-dir", ckpt_dir],
        check=True,
    )
    return True


def _run(cmd):
    import subprocess, os
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    print(f"  CMD: {' '.join(cmd[-12:])}")
    return subprocess.run(cmd, env=env, cwd=PROJECT_ROOT).returncode


@app.function(
    gpu="H100:1",
    volumes={VOLUME_PATH: vol},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_niah_subtask(ctx_len: int, subtask: str, n_samples: int = 0,
                     results_subdir: str = "", chunk_size: int = 256,
                     max_rounds: int = 5):
    import os
    os.chdir(PROJECT_ROOT)
    if not download_checkpoint(MODEL_REPO, CKPT_DIR):
        raise RuntimeError("checkpoint download failed")
    vol.commit()
    output_dir = f"{RESULTS_ROOT}{('/' + results_subdir) if results_subdir else ''}/niah"
    python = f"{PROJECT_ROOT}/.venv/bin/python"
    cmd = [
        python, "-u", "agent/agent_niah.py",
        "--checkpoint", CKPT_DIR,
        "--context_length", str(ctx_len),
        "--output_dir", output_dir,
        "--subtasks", subtask,
        "--chunk_size", str(chunk_size),
        "--max_rounds", str(max_rounds),
    ]
    if n_samples > 0:
        cmd += ["--n_samples", str(n_samples)]
    rc = _run(cmd)
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"agent_niah failed (rc={rc}) ctx={ctx_len} subtask={subtask}")


@app.function(
    gpu="H100:1",
    volumes={VOLUME_PATH: vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_ckpt():
    import os
    if not download_checkpoint(MODEL_REPO, CKPT_DIR):
        raise RuntimeError("checkpoint download failed")
    d = os.path.join(CKPT_DIR, "decoder")
    if os.path.isdir(d):
        shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
        print(f"Checkpoint ready. decoder/ shards: {shards}")
    vol.commit()


@app.local_entrypoint()
def main(
    ctx: str = "",             # comma-separated ints; empty = all (4096,8192,16384)
    subtasks: str = "",        # comma-separated NIAH subtask names; empty = all 8
    n_samples: int = 0,        # 0 = full; positive = small smoke run
    results_subdir: str = "",  # under /vol/.../agent/<subdir>/...; useful for "smoke"
    chunk_size: int = 256,     # chunk_size for the agent flow (sweep across {128,256,512,1024})
    max_rounds: int = 5,       # max EXPAND rounds before forced ANSWER
    max_concurrent: int = 32,  # cap on concurrent containers
):
    print(f"Step 1: ensure checkpoint for {MODEL_SHORT} on volume...")
    download_ckpt.remote()
    print("Checkpoint ready.")

    ctx_list = [int(x) for x in ctx.split(",") if x.strip()] if ctx else RULER_CONTEXT_LENGTHS
    subs = [x.strip() for x in subtasks.split(",") if x.strip()] if subtasks else NIAH_SUBTASKS

    jobs: list[tuple[str, callable, tuple]] = []
    for c in ctx_list:
        for st in subs:
            jobs.append((
                f"niah/{st}@{c}",
                run_niah_subtask.spawn,
                (c, st, n_samples, results_subdir, chunk_size, max_rounds),
            ))

    print(f"  total jobs: {len(jobs)} | concurrency cap: {max_concurrent} H100s")

    # Rolling-window scheduler — keep at most `max_concurrent` containers in flight,
    # spawn the next one as soon as a slot frees.
    import time
    import modal as _modal_mod
    in_flight: list[tuple[str, object]] = []  # (label, FunctionCall)
    next_idx = 0
    failures: list[tuple[str, str]] = []
    while next_idx < len(jobs) or in_flight:
        # Refill the window
        while len(in_flight) < max_concurrent and next_idx < len(jobs):
            label, spawn_fn, args = jobs[next_idx]
            print(f"  [+spawn {next_idx+1}/{len(jobs)}] {label}")
            fc = spawn_fn(*args)
            in_flight.append((label, fc))
            next_idx += 1
        # Try to drain at least one finished
        still: list[tuple[str, object]] = []
        drained = 0
        for label, fc in in_flight:
            try:
                fc.get(timeout=1.0)
                drained += 1
                print(f"  [+done] {label}  (in_flight: {len(in_flight) - drained})")
            except _modal_mod.exception.OutputExpiredError as e:
                failures.append((label, str(e))); drained += 1
                print(f"  [+fail] {label}: {e}")
            except _modal_mod.exception.FunctionTimeoutError as e:
                failures.append((label, str(e))); drained += 1
                print(f"  [+fail] {label}: {e}")
            except _modal_mod.exception.RemoteError as e:
                failures.append((label, str(e))); drained += 1
                print(f"  [+fail] {label}: {e}")
            except TimeoutError:
                # Still running; keep it
                still.append((label, fc))
            except Exception as e:
                # Other error — record as failure to avoid wedging
                failures.append((label, repr(e))); drained += 1
                print(f"  [+fail] {label}: {e!r}")
        in_flight = still
        if drained == 0 and in_flight:
            time.sleep(5)  # nothing finished — wait a bit before re-checking

    if failures:
        print(f"\n{len(failures)} job(s) failed:")
        for label, msg in failures[:20]:
            print(f"  - {label}: {msg[:200]}")
        if len(failures) > 20:
            print(f"  ... +{len(failures) - 20} more")
    else:
        print("\nAll agent runs complete.")
