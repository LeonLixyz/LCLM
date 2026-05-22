"""Migrate 3 already-downloaded checkpoints (kv-cache-compression volume) to
the new schema, then test HF + vLLM inference on each.

Run detached:
    modal run --detach _modal_run/lclm_hub_test.py
"""
from __future__ import annotations
import modal, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
app = modal.App("lclm-hub-test")
kvc_vol = modal.Volume.from_name("kv-cache-compression")
lclm_vol = modal.Volume.from_name("lclm-test")

LCLM_IGNORE = [
    "__pycache__/", ".git/", "*.safetensors", "*.bin", "*.pt",
    "_smoketest_ckpts/", "_matrix_ckpts/", "_modal_run/",
    "data/packed_batches/", "wandb/", ".venv/", "*.parquet",
]

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

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm>=0.7", "transformers", "peft", "safetensors", "pyyaml", "torchdata")
    .add_local_dir(str(REPO), remote_path="/root/LCLM", copy=True, ignore=LCLM_IGNORE)
)

CASES = [
    {"name": "cs8-causal-mlp-3e5",  "src": "/kvc/checkpoints/cs8-causal-mlp-3e5"},
    {"name": "cs4-causal-mlp-3e5",  "src": "/kvc/checkpoints/cs4-causal-mlp-3e5"},
    {"name": "cs16-causal-mlp-3e5", "src": "/kvc/checkpoints/cs16-causal-mlp-3e5"},
]


def _translate_config(old: dict) -> dict:
    pooling_map = {
        "summary_mean": "mean", "summary_concat": "concat",
        "summary": "eos", "mean": "mean", "eos": "eos",
    }
    pooling = pooling_map.get(old.get("pooling_token", "eos"), "eos")
    use_attn = old.get("use_adapter_attention", False)
    order = old.get("adapter_order", "attention_mlp")
    if not use_attn:
        adapter_type = "mlp"
    elif order == "attention_mlp":
        adapter_type = "attn_mlp"
    else:
        adapter_type = "mlp_attn"
    return {
        "model_type": "LCLM", "schema_version": 1,
        "encoder_name": old.get("embed_model_name", "Qwen/Qwen3-Embedding-0.6B"),
        "decoder_name": old.get("llm_model_name", "Qwen/Qwen3-4B-Instruct-2507"),
        "chunk_size": old.get("chunk_size", 16),
        "encoder_window_size": old.get("batch_summary_tokens", 1024),
        "pooling": pooling,
        "encoder_mask_type": old.get("embed_mask_type", "causal"),
        "boundary_overlap": old.get("overlap_tokens", 0),
        "adapter_type": adapter_type,
        "num_adapter_layers": old.get("num_adapter_layers", 1),
    }


@app.function(image=train_image, cpu=4, timeout=30*60,
              volumes={"/kvc": kvc_vol, "/lclm": lclm_vol})
def migrate(case: dict):
    """Copy src/ to /lclm/migrated/<name>/ with new layout + new config."""
    import os, shutil, json
    src = case["src"]
    dst = f"/lclm/migrated/{case['name']}"
    if os.path.isdir(dst) and os.path.isfile(os.path.join(dst, "model_config.json")):
        print(f"  already migrated: {dst}")
        return {"name": case["name"], "ok": True, "dst": dst, "skipped": True}

    os.makedirs(dst, exist_ok=True)

    # 1) subdir renames + recursive copy
    renames = [("llm", "decoder"), ("embedder", "encoder"), ("projectors", "adapter")]
    for old_name, new_name in renames:
        old_path = os.path.join(src, old_name)
        if not os.path.isdir(old_path):
            raise RuntimeError(f"missing {old_path}")
        new_path = os.path.join(dst, new_name)
        if os.path.isdir(new_path):
            shutil.rmtree(new_path)
        print(f"  copying {old_path} -> {new_path}", flush=True)
        shutil.copytree(old_path, new_path)

    # 2) rename code_adapter.safetensors -> adapter.safetensors
    old_w = os.path.join(dst, "adapter", "code_adapter.safetensors")
    new_w = os.path.join(dst, "adapter", "adapter.safetensors")
    if os.path.isfile(old_w):
        os.rename(old_w, new_w)
        print(f"  renamed adapter weight", flush=True)

    # 3) rewrite model_config.json
    cfg_path = os.path.join(src, "model_config.json")
    with open(cfg_path) as f:
        old_cfg = json.load(f)
    print(f"  old cfg: {old_cfg}", flush=True)
    new_cfg = _translate_config(old_cfg)
    print(f"  new cfg: {new_cfg}", flush=True)
    with open(os.path.join(dst, "model_config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)
    # also copy processor_config.json if present
    pc = os.path.join(src, "processor_config.json")
    if os.path.isfile(pc):
        shutil.copy(pc, os.path.join(dst, "processor_config.json"))

    lclm_vol.commit()
    return {"name": case["name"], "ok": True, "dst": dst}


@app.function(image=train_image, gpu="H200", timeout=20*60, volumes={"/lclm": lclm_vol})
def test_hf(name: str):
    import sys, torch, traceback
    sys.path.insert(0, "/root/LCLM")
    dst = f"/lclm/migrated/{name}"
    print(f">>> HF test: {name}", flush=True)
    try:
        from inference.hf import load_model, generate_text
        model, tok, proc = load_model(dst, device="cuda", dtype="bf16")
        prompt = ("<|memory_start|>Compression of long context using latent vectors "
                  "produced by an encoder-decoder transformer model. "
                  "The encoder reduces the input by a factor of N tokens per "
                  "latent. The decoder consumes these latents.<|memory_end|> "
                  "Summarize.")
        out = generate_text(model, tok, proc, prompt, device="cuda",
                            max_tokens=50, temperature=0.0)
        return {"name": name, "ok": True, "backend": "hf", "output": out[:300]}
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "ok": False, "backend": "hf", "error": str(e)}


@app.function(image=vllm_image, gpu="H200", timeout=30*60, volumes={"/lclm": lclm_vol})
def test_vllm(name: str):
    import sys, traceback
    sys.path.insert(0, "/root/LCLM")
    dst = f"/lclm/migrated/{name}"
    print(f">>> vLLM test: {name}", flush=True)
    try:
        from inference.vllm import LCLMVLLMDecoder
        runner = LCLMVLLMDecoder(checkpoint_path=dst, tensor_parallel_size=1)
        prompt = ("<|memory_start|>Compression of long context using latent vectors "
                  "produced by an encoder-decoder transformer model. "
                  "The encoder reduces the input by a factor of N tokens per "
                  "latent. The decoder consumes these latents.<|memory_end|> "
                  "Summarize.")
        outputs = runner.generate(
            prompts=[[{"role": "user", "content": prompt}]],
            max_tokens=50, temperature=0.0,
        )
        return {"name": name, "ok": True, "backend": "vllm", "output": outputs[0][:300]}
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "ok": False, "backend": "vllm", "error": str(e)}


@app.function(image=train_image, cpu=2, timeout=2*60*60,
              volumes={"/kvc": kvc_vol, "/lclm": lclm_vol})
def orchestrate():
    print("Step 1: migrate 3 checkpoints (parallel)...", flush=True)
    migrate_results = list(migrate.map(CASES))
    for r in migrate_results:
        print(f"  migrate {r['name']}: ok={r['ok']} dst={r.get('dst')}", flush=True)

    names = [r["name"] for r in migrate_results if r["ok"]]
    print(f"\nStep 2: HF inference on {len(names)} checkpoints (parallel)...", flush=True)
    hf_results = list(test_hf.map(names))
    for r in hf_results:
        mark = "OK" if r.get("ok") else "FAIL"
        print(f"  [HF {mark}] {r['name']}: {r.get('output', r.get('error'))[:200]}", flush=True)

    print(f"\nStep 3: vLLM inference on {len(names)} checkpoints (parallel)...", flush=True)
    vllm_results = list(test_vllm.map(names))
    for r in vllm_results:
        mark = "OK" if r.get("ok") else "FAIL"
        print(f"  [vLLM {mark}] {r['name']}: {r.get('output', r.get('error'))[:200]}", flush=True)

    print("\n========== FINAL ==========", flush=True)
    for hf, vl in zip(hf_results, vllm_results):
        m_hf = "✓" if hf.get("ok") else "✗"
        m_vl = "✓" if vl.get("ok") else "✗"
        print(f"  {hf['name']}: HF {m_hf}  vLLM {m_vl}", flush=True)
    return {"migrate": migrate_results, "hf": hf_results, "vllm": vllm_results}


@app.local_entrypoint()
def main():
    print("Dispatching orchestrate()...", flush=True)
    orchestrate.remote()
