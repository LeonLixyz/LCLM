"""LCLM end-to-end matrix test on Modal H200s.

What this app does:
  1. prep_data()          - generate synthetic packed-parquet (small CPU job)
  2. matrix_single_gpu()  - 5 cases × stage-0 training + conversion + HF inference
                             on 1×H200. Verifies adapter shapes for mean/eos/concat
                             across W ∈ {chunk_size, 1024} and adapter_type ∈ {mlp, attn_mlp}.
  3. multi_gpu()          - 1 case × stage-0 on 8×H200 (DeepSpeed ZeRO-2)
  4. vllm_infer()         - load one of the converted checkpoints with vLLM

Run detached:
    modal run --detach _modal_run/lclm_matrix.py
"""
from __future__ import annotations

import modal
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("lclm-matrix-test")
vol = modal.Volume.from_name("lclm-test", create_if_missing=True)

LCLM_IGNORE = [
    "__pycache__/", ".git/", "*.safetensors", "*.bin", "*.pt",
    "_smoketest_ckpts/", "_matrix_ckpts/", "_modal_run/",
    "data/packed_batches/", "wandb/", ".venv/", "*.parquet",
]

# Training image: torch + flash-attn + accelerate + deepspeed + LCLM repo
train_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch==2.9.1", "transformers>=4.45", "accelerate", "deepspeed",
        "peft", "liger-kernel", "torchdata", "wandb", "pyarrow",
        "safetensors", "huggingface-hub", "pyyaml", "numpy",
        "wheel", "packaging", "ninja",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

# vLLM inference image: needs nvcc (in /usr/local/cuda) so vLLM can JIT
# its flashinfer kernels at first launch.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.7", "transformers", "peft", "safetensors", "pyyaml", "torchdata")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)


CASES = [
    {"name": "mean_W1024",     "pooling": "mean",   "W": 1024, "adapter_type": "mlp"},
    {"name": "eos_W1024",      "pooling": "eos",    "W": 1024, "adapter_type": "mlp"},
    {"name": "concat_W1024",   "pooling": "concat", "W": 1024, "adapter_type": "mlp"},
    {"name": "mean_W16",       "pooling": "mean",   "W": 16,   "adapter_type": "mlp"},
    {"name": "mean_attn_mlp",  "pooling": "mean",   "W": 1024, "adapter_type": "attn_mlp"},
]


# ---------------------------------------------------------------------------
# Synthetic data generation (~5min on CPU)
# ---------------------------------------------------------------------------

@app.function(image=train_image, gpu=None, timeout=10*60, volumes={"/vol": vol})
def prep_data():
    import os, pickle, random, subprocess
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer

    out_dir = "/vol/packed_batches/smoke"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "shard-000.parquet")):
        print(f"data already present at {out_dir}")
        return

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for t in ("<|memory|>", "<|memory_start|>", "<|memory_end|>"):
        if t not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [t]})
    mem_start = tok.convert_tokens_to_ids("<|memory_start|>")
    mem_tok = tok.convert_tokens_to_ids("<|memory|>")
    mem_end = tok.convert_tokens_to_ids("<|memory_end|>")

    rng = random.Random(42)
    words = ["compression", "context", "model", "transformer", "encoder", "decoder",
             "latent", "vector", "memory", "attention", "tokens"]

    def make_one():
        ex = {"base_input_ids": [], "base_labels": [],
              "memory_strings": [], "memory_positions": []}
        prefix = tok.encode("Read and answer:\n", add_special_tokens=False)
        ex["base_input_ids"] += prefix; ex["base_labels"] += [-100]*len(prefix)
        for _ in range(2):
            ex["memory_positions"].append(len(ex["base_input_ids"]))
            ex["base_input_ids"] += [mem_start, mem_tok, mem_end]
            ex["base_labels"] += [-100]*3
            ex["memory_strings"].append(
                " ".join(rng.choice(words) for _ in range(rng.randint(100, 300)))
            )
            sep = tok.encode("\n", add_special_tokens=False)
            ex["base_input_ids"] += sep; ex["base_labels"] += [-100]*len(sep)
        q = tok.encode("Answer: ", add_special_tokens=False)
        ex["base_input_ids"] += q; ex["base_labels"] += [-100]*len(q)
        a = tok.encode("This discusses compression.", add_special_tokens=False)
        ex["base_input_ids"] += a; ex["base_labels"] += a
        return ex

    for f in range(2):
        rows = []
        for _ in range(64):
            rows.append(pickle.dumps([make_one(), make_one()]))
        pq.write_table(pa.table({"packed_batch_bytes": rows}),
                       os.path.join(out_dir, f"shard-{f:03d}.parquet"))
        print(f"wrote shard-{f:03d}.parquet")
    vol.commit()
    print("data ready")


# ---------------------------------------------------------------------------
# Train + convert one case (single GPU)
# ---------------------------------------------------------------------------

def _write_smoke_yaml(case, vol_path="/vol/packed_batches/smoke"):
    p = case
    yml = f"""experiment: {p['name']}
models:
  encoder: Qwen/Qwen3-Embedding-0.6B
  decoder: Qwen/Qwen3-0.6B
  pooling: {p['pooling']}
  encoder_window_size: {p['W']}
  adapter_type: {p['adapter_type']}
  num_adapter_layers: 1
  encoder_mask_type: bidirectional
  mask: bidirectional
  decoder_attn_implementation: flash_attention_2
  embed_attn_implementation: flash_attention_2
  boundary_overlap: 0
  use_fused_ce: false
  random_init_decoder: false
  random_init_encoder: false
training:
  chunk_size: 16
  max_encode_batch_size: 1024
  gradient_accumulation_steps: 1
  num_epochs: 1
  max_steps: 10
  warmup_steps: 5
  max_grad_norm: 1.0
  save_steps: 5
  delete_old_checkpoints: false
  seed: 42
  auto_resume: false
  preempt_save_minutes: 0
  optimizer: adamw
  scheduler: cosine
  adam_epsilon: 1.0e-8
  decoder_betas: "0.9,0.95"
  encoder_betas: "0.9,0.95"
  adapter_betas: "0.9,0.95"
  decoder_gradient_checkpointing: false
  use_liger_kernel: false
data:
  num_workers: 0
  use_packing: true
  use_memory_wrapping: true
  packed_attention_backend: flash
  train_wrap_tokens: true
  max_packed_length: 16384
distributed:
  type: deepspeed
  config: ./scripts/distributed_configs/deepspeed_zero2_multi_node.yaml
stages:
  0:
    train_batch_size: 1
    warmup_steps: 5
    train_decoder: false
    train_encoder: false
    adapter_lr: 1.0e-3
    encoder_lr: 6.0e-5
    decoder_lr: 6.0e-5
    min_lr: 1.0e-6
    decoder_weight_decay: 0.0
    encoder_weight_decay: 0.0
    adapter_weight_decay: 0.01
    encoder_gradient_checkpointing: false
    dataset: {vol_path}
    distributed:
      type: deepspeed
      config: ./scripts/distributed_configs/deepspeed_zero2_multi_node.yaml
logging:
  wandb_project: lclm-smoke
  wandb_name: {p['name']}
  log_interval: 5
  log_token_counts: false
"""
    path = f"/tmp/smoke_{p['name']}.yaml"
    open(path, "w").write(yml)
    return path


def _run_pipeline(yaml_path, output_dir):
    import os, subprocess
    env = os.environ.copy()
    env["OUTPUT_DIR"] = output_dir
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = "/root/LCLM"
    return subprocess.run(
        ["bash", "scripts/run_pipeline.sh", yaml_path],
        cwd="/root/LCLM", env=env, capture_output=True, text=True, timeout=20*60,
    )


def _verify_hf_load(hf_dir, case):
    import json, sys, torch
    sys.path.insert(0, "/root/LCLM")
    cfg = json.load(open(f"{hf_dir}/model_config.json"))
    print(f"  cfg = {cfg}", flush=True)
    expected_keys = {"chunk_size", "encoder_window_size", "pooling", "encoder_mask_type",
                     "boundary_overlap", "adapter_type", "num_adapter_layers",
                     "encoder_name", "decoder_name", "model_type"}
    missing = expected_keys - set(cfg)
    assert not missing, f"missing keys: {missing}"

    from inference.hf import load_model, generate_text
    model, tok, proc = load_model(hf_dir, device="cuda", dtype="bf16")
    enc_h = model.encoder.embed_model.config.hidden_size
    expected_in = 16 * enc_h if case['pooling'] == "concat" else enc_h
    got_in = model.adapter.fc1.in_features
    assert got_in == expected_in, f"adapter fc1.in {got_in} != expected {expected_in}"

    prompt = "<|memory_start|>The fox jumps. <|memory_end|> What does the fox do?"
    out = generate_text(model, tok, proc, prompt, device="cuda", max_tokens=10, temperature=0.0)
    del model
    torch.cuda.empty_cache()
    return {
        "adapter_type_persisted": cfg["adapter_type"],
        "pooling_persisted": cfg["pooling"],
        "encoder_window_size_persisted": cfg["encoder_window_size"],
        "adapter_fc1_in": got_in,
        "adapter_fc1_in_expected": expected_in,
        "generated": out[:80] if out else None,
    }


@app.function(image=train_image, gpu="H200", timeout=2*60*60,
              volumes={"/vol": vol})
def train_and_verify(case: dict):
    """One matrix cell on 1×H200. Returns a result dict."""
    import os, traceback
    case_dir = f"/vol/matrix_ckpts/{case['name']}"
    os.makedirs(case_dir, exist_ok=True)
    yml = _write_smoke_yaml(case)
    print(f">>> CASE: {case['name']}", flush=True)
    r = _run_pipeline(yml, case_dir)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        return {"name": case["name"], "ok": False, "step": "pipeline", "rc": r.returncode,
                "stderr": r.stderr[-2000:]}
    # Find HF dir
    hf_dir = f"{case_dir}/{case['name']}/stage0-hf"
    if not os.path.isdir(hf_dir):
        return {"name": case["name"], "ok": False, "step": "no_hf_dir"}
    try:
        v = _verify_hf_load(hf_dir, case)
    except Exception as e:
        traceback.print_exc()
        return {"name": case["name"], "ok": False, "step": "verify",
                "error": str(e)}
    vol.commit()
    return {"name": case["name"], "ok": True, **v}


# ---------------------------------------------------------------------------
# Multi-GPU training (8 × H200)
# ---------------------------------------------------------------------------

@app.function(image=train_image, gpu="H200:8", timeout=2*60*60,
              volumes={"/vol": vol})
def multi_gpu_train():
    """Stage-0 training on 8×H200, DeepSpeed ZeRO-2."""
    import os, subprocess, traceback
    case = {"name": "multigpu_mean_W1024", "pooling": "mean", "W": 1024, "adapter_type": "mlp"}
    case_dir = f"/vol/matrix_ckpts/{case['name']}"
    os.makedirs(case_dir, exist_ok=True)
    yml = _write_smoke_yaml(case)
    print(f">>> MULTI-GPU CASE: {case['name']}", flush=True)
    nvidia = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print(nvidia.stdout, flush=True)
    r = _run_pipeline(yml, case_dir)
    print(r.stdout[-3000:], flush=True)
    if r.returncode != 0:
        return {"name": case["name"], "ok": False, "step": "pipeline", "rc": r.returncode,
                "stderr": r.stderr[-3000:]}
    hf_dir = f"{case_dir}/{case['name']}/stage0-hf"
    try:
        v = _verify_hf_load(hf_dir, case)
    except Exception as e:
        traceback.print_exc()
        return {"name": case["name"], "ok": False, "step": "verify", "error": str(e)}
    vol.commit()
    return {"name": case["name"], "ok": True, **v}


# ---------------------------------------------------------------------------
# vLLM inference on a converted checkpoint
# ---------------------------------------------------------------------------

@app.function(image=vllm_image, gpu="H200", timeout=30*60, volumes={"/vol": vol})
def vllm_infer(case_name: str = "mean_W1024"):
    import sys
    sys.path.insert(0, "/root/LCLM")
    hf_dir = f"/vol/matrix_ckpts/{case_name}/{case_name}/stage0-hf"
    print(f"Loading {hf_dir} via LCLMVLLMDecoder", flush=True)
    from inference.vllm import LCLMVLLMDecoder
    runner = LCLMVLLMDecoder(checkpoint_path=hf_dir, tensor_parallel_size=1)
    prompt = "<|memory_start|>The fox jumps over the lazy dog. <|memory_end|> What does the fox do?"
    outputs = runner.generate(
        prompts=[[{"role": "user", "content": prompt}]],
        max_tokens=20, temperature=0.0,
    )
    print(f"vLLM output: {outputs[0]!r}", flush=True)
    return {"ok": True, "output": outputs[0][:200]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@app.function(image=train_image, cpu=2, timeout=4*60*60, volumes={"/vol": vol})
def orchestrate():
    """Single serverside orchestrator — survives local disconnect.

    Use ``modal run --detach _modal_run/lclm_matrix.py`` to fire-and-forget.
    """
    print("Step 1: prep synthetic data...", flush=True)
    prep_data.remote()

    print("Step 2: 5-case single-GPU matrix (mapped, parallel)...", flush=True)
    matrix_results = list(train_and_verify.map(CASES))
    for r in matrix_results:
        mark = "OK" if r.get("ok") else "FAIL"
        print(f"  [{mark}] {r['name']}: {r}", flush=True)

    print("Step 3: multi-GPU (8× H200) training...", flush=True)
    mg = multi_gpu_train.remote()
    mark = "OK" if mg.get("ok") else "FAIL"
    print(f"  [{mark}] multi_gpu: {mg}", flush=True)

    print("Step 4: vLLM inference on the converted mean_W1024 checkpoint...", flush=True)
    try:
        vl = vllm_infer.remote(case_name="mean_W1024")
        mark = "OK" if vl.get("ok") else "FAIL"
        print(f"  [{mark}] vllm: {vl}", flush=True)
    except Exception as e:
        print(f"  [FAIL] vllm: {e}", flush=True)
        vl = {"ok": False, "error": str(e)}

    print("\n========== FINAL ==========", flush=True)
    for r in matrix_results + [mg]:
        mark = "✓" if r.get("ok") else "✗"
        print(
            f"  {mark} {r['name']}: pooling={r.get('pooling_persisted')} "
            f"W={r.get('encoder_window_size_persisted')} "
            f"adapter={r.get('adapter_type_persisted')} "
            f"fc1_in={r.get('adapter_fc1_in')} (exp {r.get('adapter_fc1_in_expected')})",
            flush=True,
        )
    return {"matrix": matrix_results, "multi_gpu": mg, "vllm": vl}


@app.local_entrypoint()
def main():
    print("Dispatching orchestrate() — survives local disconnect with --detach", flush=True)
    orchestrate.remote()
