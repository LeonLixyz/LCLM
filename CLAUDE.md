# Claude Instructions

Project-specific instructions for Claude Code.

## Repo shape

This is a production-leaning LCLM codebase. Six top-level concerns:

| Path             | What lives here |
| ---------------- | --------------- |
| `latent_context/`| Model definition (encoder, adapter, decoder, processor, `from_pretrained` loader, config-compat shim) |
| `train/`         | Training loop (`launch_train.py` + `trainer.py`) |
| `scripts/`       | Launch scripts (`train.sh`, `run_pipeline.sh`, `convert_checkpoint.sh`) + experiment/pretrain YAML configs + accelerate `distributed_configs/` |
| `inference/`     | HF (`hf.py`) and vLLM (`vllm.py`) inference, plus runnable `examples/` |
| `agent/`         | Agent app — `EXPAND(i)` over compressed segments, RULER/LongBench/LongHealth5 runners, Modal launcher |
| `data/`          | Training-time dataset + collator + packing utilities only |
| `utils/`         | Helpers: env, seed, scheduler, nan checks, FSDP/DeepSpeed → HF checkpoint converters in `utils/checkpoints/` |

## GPU code
- For interactive prototyping see `.claude/skills/` (Modal).
- To use Modal: `conda activate modal`.

## What's intentionally not in this repo
- No benchmark / eval pipeline. The repo is the *model*; evals live elsewhere.
- No in-training loss eval — `eval/` and all `eval_logloss_*` flags removed.
- No baseline (KV-cache / compaction) code. Out of scope for the shipped model.
- No figure-reproduction code (`figures_repro/` removed).
- No dataset-creation utilities — `data/` only contains what training actually loads.
