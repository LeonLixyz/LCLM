# Agent with latent context

Code for the agent in Section 7.1 of the paper (Figure 6). The haystack is split
into segments of at most 1,024 encoder tokens; every segment is compressed by
the LCLM (16x) and labelled with a segment id. The model receives the whole
compressed sequence in one prompt together with one native tool, `expand`,
which returns the original text of a segment:

```
[Segment seg_8]
<|memory_start|>...segment text, compressed to latents...<|memory_end|>
[/Segment seg_8]
...
assistant: <tool_call>{"name": "expand", "arguments": {"segment_id": "seg_8"}}</tool_call>
tool:      ...exact original text of seg_8...
assistant: The special magic number for ... is 4207615.
```

On RULER needle-in-a-haystack, the agent is compared with the same LCLM 16x
given the same compressed segments and no tool. Protocol details are in
[docs/agent.md](../docs/agent.md).

## Layout

| File | What it does |
|---|---|
| `ruler_tools.py` | Prompt parsing, lossless segmentation, question selection, the compressed-only episode, the agent loop and scoring. CPU only. |
| `backend.py` | `Backend`: LCLM encoder/adapter + vLLM decoder from prompt embeddings for multi-turn chats; `run_jobs`: one model replica per GPU. |
| `run_niah_agent.py` | Runs both conditions on one context length and writes `episodes.jsonl` and `summary.json`. |
| `aggregate_niah.py` | Per-length averages (compressed-only vs. with expansion) next to the Figure 6 values. |
| `run_niah_agent_modal.py` | Same run on [Modal](https://modal.com), one multi-GPU container per context length. |
| `figure6_questions.json` | The RULER source indices behind Figure 6 (50 per subtask and length). |
| `tool_tasks.py`, `tool_eval.py` | Native tool-use diagnostic (direct, chained and recovery tool calls, memory expansion); see [docs/agent.md](../docs/agent.md#native-tool-use-diagnostic). |

## Setup

Install the repo as in the [top-level README](../README.md). The runs use vLLM
prompt embeddings (`enable_prompt_embeds`, vLLM >= 0.10); the paper numbers
were produced with vLLM 0.11.0 and transformers 4.57.1. Data comes from the
`ruler` config of [`latent-context/lclm-eval`](https://huggingface.co/datasets/latent-context/lclm-eval).
`--checkpoint` takes a local LCLM directory or a Hugging Face repo id and
defaults to `latent-context/0.6b-4b-LCLM-16x`.

## Smoke test

Two questions of one subtask at 4k (four episodes) on one GPU:

```bash
python -m agent.run_niah_agent --context-length 4096 --subtasks niah_single_3 \
    --n-questions 2 --output-dir runs/niah_agent_smoke/4096
```

The log prints each question's score in both conditions and ends with a
per-subtask table. Add `--save-transcripts` to keep the full conversations
(`transcripts.jsonl`).

## Reproduce Figure 6

Defaults are the paper settings: all 8 NIAH subtasks, the 50 Figure 6 questions
per subtask, 1,024-token segments, an expansion cap of 16 calls at 4k and 8k and
32 at 16k, and 512 generated tokens per turn.

```bash
for n in 4096 8192 16384; do
  python -m agent.run_niah_agent --context-length $n --gpus 8 --output-dir runs/niah_agent/$n
done
python -m agent.aggregate_niah runs/niah_agent --per-task
```

`--gpus N` runs one independent model replica per GPU (`--gpus 1` works too,
just slower). Each length is 400 questions and 800 episodes; on 8 H200 GPUs one
length took about 10 minutes. Runs resume from an existing `episodes.jsonl` and
refuse to mix configurations in one output directory.

On Modal (set `LCLM_MODAL_GPU`, default `H100:8`):

```bash
modal run agent/run_niah_agent_modal.py                  # 4k, 8k, 16k -> runs/niah_agent/
python -m agent.aggregate_niah runs/niah_agent --per-task
```

Expected averages over the 8 subtasks (Figure 6, LCLM 0.6b-4b 16x):

| Context | Compressed only | With expansion |
|---|---:|---:|
| 4k | 80.94 | 99.44 |
| 8k | 75.88 | 99.06 |
| 16k | 69.56 | 95.06 |

Scores are the mean case-insensitive coverage of the reference answers in the
final answer (partial credit for multi-answer subtasks). Unfinished episodes,
malformed tool calls and episodes that hit a cap score zero. Small differences
from these numbers are expected on other GPUs or library versions.

To evaluate other questions, pass `--selection hash --seed <seed>`; questions are
then ranked by a hash of the seed, subtask and RULER source index, so the same
source questions are selected at every context length. `--max-calls`,
`--segment-tokens` and `--n-questions` override the other defaults.

## Tests

CPU only, with scripted model outputs:

```bash
python -m pytest tests/test_ruler_native_tools.py tests/test_ruler_full.py tests/test_native_tool_eval.py
```
