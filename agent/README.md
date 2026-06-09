# LCLM agent evaluations (RULER NIAH)

Implementation and runner for the agentic retrieval experiments described in §6
of the LCLM paper. The agent compresses the haystack into latent tokens with
LCLM, then exposes an `EXPAND(i)` tool the model can call to read the raw text
of any single chunk it wants to inspect more closely.

Scope: **RULER NIAH only** (8 subtasks).

## Layout

| File / dir              | What it does |
| ----------------------- | ------------ |
| `__init__.py`           | Resolves `PROJECT_ROOT` once and pushes it onto `sys.path`. |
| `_lib.py`               | Shared building blocks: `load_agent_model()`, chunking, triage → expand generation, scoring. Imports from `latent_context`. |
| `agent_niah.py`         | Runs the 8 NIAH subtasks at one context length. |
| `run_agent_modal.py`    | Modal orchestrator. Spawns each (subtask × ctx-length) cell as its own H100 container. |

## Running

### Local (single GPU, smoke run)

```bash
python agent/agent_niah.py \
  --checkpoint /path/to/lclm-cr16 \
  --context_length 4096 \
  --output_dir _runs/niah_4k \
  --subtasks niah_single_3 \
  --n_samples 8
```

### Modal (full sweep, parallel containers)

The Modal runner picks the LCLM checkpoint via env vars so you can switch models
without editing code:

```bash
LCLM_MODEL_REPO=<org>/<lclm-id> \
LCLM_MODEL_SHORT=<short-tag> \
modal run agent/run_agent_modal.py
```

Useful flags on `main()`:

- `--ctx 4096,8192,16384` — RULER context lengths to run (default: all three)
- `--subtasks niah_single_3,niah_multikey_1` — NIAH subtasks (default: all 8)
- `--n_samples 50` — small smoke run (0 = full)
- `--results_subdir smoke` — writes to a sibling dir so it doesn't shadow full results
- `--chunk_size 256` — agent chunk_size (sweep across {128, 256, 512, 1024})
- `--max_rounds 5` — max EXPAND rounds before forced ANSWER
- `--max_concurrent 32` — H100 container budget

Results land under `/vol/<MODEL_SHORT>_results/agent/niah/...` on the
`kv-cache-compression` Modal volume.

## Note on the eval dataset

`agent_niah.py` loads `SaylorTwift/RULER-{ctx}-llama-3.2-tokenizer`, which ships
raw RULER text (`input` + `outputs` columns). Upstream sized those haystacks to
fit `ctx` llama tokens; we retokenize at runtime with the Qwen encoder, so the
effective Qwen-token count drifts by ~5% from the standard `run_eval` baseline
(which uses `latent-context/lclm-eval (config="ruler")` — Qwen-resized — but only ships
chat-templated prompts, not raw text the agent can chunk).
