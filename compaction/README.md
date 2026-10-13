# Online latent compaction

Code for Section 7.2 / Figure 7 of *End-to-End Context Compression at Scale*:
an LCLM compresses its **own** reasoning while it generates, so the decoder's
active context stays short no matter how long the answer gets.

## How it works

With the default settings (`block_size=512`, `keep_recent=256`):

1. Generate the first 768 tokens normally.
2. Encode the oldest 512 generated tokens with the LCLM encoder into a latent
   block (32 latents for the 16x model), keep the most recent 256 tokens raw,
   and continue generating from
   `[prompt, <|memory_start|> latents <|memory_end|>, recent raw tokens]`.
3. After every further 512 generated tokens, compact the oldest 512 raw tokens
   again. Earlier latent blocks are reused unchanged; the prompt is never compressed.

Cuts never split a multi-token Unicode character, so a block can be a few tokens
shorter than 512. The full-context reference generates the same answer in one
request. Both modes get the same output budget (`max_context` minus the prompt
length, 32,768 by default), so compaction never buys extra tokens.

**Context statistics** count active decoder positions, including the prompt,
latent tokens, memory delimiters and recent raw tokens. A request with `p`
prompt positions that emits `n` tokens occupies `p+1, ..., p+n` positions.

- `avg_context_positions` (avg KV): mean context over a problem's generated
  tokens. Reported numbers average this per-problem value over problems.
- `peak_context_positions`: largest `p+n` in the problem. Reported as the mean
  over problems ("mean peak") and the maximum over problems ("max peak").

## Python API

```python
from latent_context import from_pretrained
from compaction import HFBackend, generate_with_compaction, generate_full_context

model, tokenizer, _ = from_pretrained("latent-context/0.6b-4b-LCLM-16x")
backend = HFBackend(model, tokenizer)   # sampling defaults: T=0.7, top-p 0.8, top-k 20, min-p 0, seed 0
out = generate_with_compaction(backend, "Find all real x such that ...", block_size=512, keep_recent=256)
print(out["text"])
print(out["generated_tokens"], len(out["compactions"]), out["avg_context_positions"], out["peak_context_positions"])

ref = generate_full_context(backend, "Find all real x such that ...")
```

`HFBackend` runs one trajectory at a time with `transformers` generation and is
meant for experimentation. The paper's results use `VLLMBackend` (HF encoder +
vLLM decoder with `prompt_embeds` on one GPU) driven by
`compaction.scheduler.serve`, which keeps up to 16 trajectories in flight per GPU.
`compaction.controller.trajectory` is backend-agnostic if you want to plug in
your own decoder.

## MATH evaluation

`compaction/eval_math.py` runs both modes on the 5,000 MATH test problems
(`EleutherAI/hendrycks_math`, all seven subject test splits, pinned revision and
file hashes), with one vLLM worker per GPU. Each problem is prompted as

```
<problem>

Solve the problem step by step. Finish with your final answer in \boxed{}.
```

and scored by the last `\boxed{}` answer in the output, compared symbolically
with the last `\boxed{}` answer of the reference solution using math-verify.

Requirements: a CUDA GPU, `vllm` (results were produced with vLLM 0.11.0 and
transformers 4.57.1) and `pip install math-verify==0.9.0`.

Smoke test (20 fixed problems spanning all subjects, both modes, one GPU):

```bash
python -m compaction.eval_math --checkpoint latent-context/0.6b-4b-LCLM-16x \
    --out runs/math_smoke --limit 20 --num-gpus 1
```

Full reproduction (all 5,000 problems, both modes, all visible GPUs; about
1.3 hours on 8xH200):

```bash
python -m compaction.eval_math --checkpoint latent-context/0.6b-4b-LCLM-16x --out runs/math_full
```

or on [Modal](https://modal.com) (H200:8 by default; set `LCLM_MODAL_GPU` and
`LCLM_MODAL_VOLUME` to change the GPU or output volume):

```bash
modal run compaction/modal_app.py --run-name math-full
modal run compaction/modal_app.py --run-name smoke --limit 20
```

Use `--modes compaction` or `--modes full` to run a single mode. Interrupted runs
resume when the same command is re-run. Outputs in `--out`:

| File | Contents |
|---|---|
| `generations.rank*.jsonl` | Raw per-trajectory results: text, requests, compactions, context statistics |
| `scored.jsonl` | One row per (problem, mode) with the extracted answer and correctness |
| `metrics.json` | Per-mode and per-subject accuracy and context statistics |

### Expected results

16x LCLM, 5,000 problems, temperature 0.7, top-p 0.8, top-k 20, min-p 0, seed 0,
32,768-token context limit:

| Mode | Accuracy | Avg KV | Mean peak | Max peak |
|---|---:|---:|---:|---:|
| Full context | 92.46% | 1,730 | 3,366 | 32,768 |
| Online compaction | 90.52% | 551 | 935 | 4,135 |

Compaction was triggered on 4,166 of 5,000 problems. Sampling and batched
vLLM execution are not bitwise deterministic, so a rerun will differ slightly
from these numbers.

## Tests

```bash
python -m pytest tests/test_compaction_*.py -q
```

The tests run on CPU with fake backends and a tiny random decoder; the math-verify
scoring tests are skipped if `math_verify` is not installed.
