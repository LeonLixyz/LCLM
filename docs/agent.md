# Agent with latent context: protocol

This describes the RULER NIAH protocol behind Figure 6 of the paper, implemented
in [`agent/ruler_tools.py`](../agent/ruler_tools.py), and a smaller native
tool-use diagnostic. Usage is in [`agent/README.md`](../agent/README.md).

## Data and questions

Inputs are the chat-templated RULER prompts in the `ruler` config of
`latent-context/lclm-eval` (categories `memwrap/ruler/<subtask>_<length>`), for
the eight NIAH subtasks `niah_single_{1,2,3}`, `niah_multikey_{1,2,3}`,
`niah_multivalue` and `niah_multiquery`. Each has 500 questions per context
length. The 4k/8k/16k labels are the upstream RULER lengths; actual token
counts under the Qwen tokenizers vary.

Figure 6 uses 50 questions per subtask and length (400 per length), listed in
[`agent/figure6_questions.json`](../agent/figure6_questions.json) by RULER source
index. Questions are ranked by SHA-256 of the seed, subtask and source index,
independently of answers or scores, so the same source questions appear at
every length. For five subtasks the list is exactly the seed-2026092001 sample
(`--selection hash`); `niah_multikey_2`, `niah_multivalue` and `niah_multiquery`
use an earlier sample drawn the same way. The expansion caps were chosen on those
three subtasks and then fixed for the other five.

## Prompt

The source prompt is split into system message, instruction prefix, haystack
and question. In `niah_multivalue` and `niah_multiquery` the question sits
inside the memory span; it is moved after the haystack unchanged. System
instructions and questions are never compressed.

The haystack is cut into lossless character slices of at most 1,024 encoder
tokens, preferring a newline within the last 256 characters before the cut.
Concatenating the segments recovers the haystack exactly, and boundaries depend
only on the text. Each segment is wrapped as

```
[Segment seg_3]
<|memory_start|>segment text<|memory_end|>
[/Segment seg_3]
```

and encoded separately into `ceil(tokens / 16)` latents. Both conditions see the
same segments. The instruction appended to the system message asks the model to
answer using the segmented context and to return only the requested values.

- **Compressed only**: no tools. A tool call scores zero.
- **Agent**: the `expand` tool is provided through the tokenizer's native chat
  template (Qwen `<tool_call>` JSON). The system message adds rules for selective
  expansion and the call budget (`Expansion budget remaining: 16 of 16.`), and
  after every turn with expansions a user message states the remaining budget
  until it is spent.

## Agent loop

Each assistant turn either answers or emits one or more `expand` calls. Calls are
parsed strictly (no JSON repair, duplicate keys rejected); a well-formed call
with a bad or unknown segment id returns an error the model can recover from and
still counts against the budget. `expand` returns the exact original segment text.

The cap is 16 calls at 4k and 8k, and 32 at 16k (16 at 32k). When the budget is
spent, or a turn requests more calls than remain (none of that turn's calls run),
the model is told that expansion is disabled and must answer without tools.
Generation is limited to 512 tokens per turn with a 65,536-token runtime context;
nothing is truncated.

## Scoring

Only the final assistant answer is scored, never tool calls or returned text.
The score is the percentage of reference strings contained in the answer,
ignoring case (RULER `string_match_all`; partial credit for multi-answer
subtasks). Episodes that end without a final answer (output cap, malformed tool
call, tool call after expansion was disabled) score zero. Figure 6 reports the
mean over all 400 questions per length, i.e. the average of the eight subtask
scores.

`episodes.jsonl` also records stop reasons, tool calls, expanded segments,
`expanded_answer_coverage` (whether the expanded text contained every answer,
computed after generation) and decoder prompt positions.

## Native tool-use diagnostic

[`agent/tool_eval.py`](../agent/tool_eval.py) executes the model's native tool
calls against deterministic, in-memory tools and feeds the results back. It is a
procedural diagnostic, not a BFCL or ToolBench score. With the default 50 tasks
per family there are 200 tasks and 300 episodes:

| Family | What must work |
|---|---|
| `direct` | Tool selection, typed arguments, multiple calls; 10 no-tool controls |
| `chain` | Resolve an alias, use the returned id, inspect inventory; half also make a stateful reservation |
| `recovery` | Observe a retryable tool error, retry and use the result |
| `memory` | Recover exact identifiers from 8 or 16 documents, half requiring a two-hop join; run as `plain`, `compressed` and `compressed_expand` |

Tools never run code, shell commands or network requests. Success requires the
correct answer, the required tool calls and, for reservations, the correct
state change; answer content and the `FINAL: {"answer": ...}` format are scored
separately. Episodes allow 8 turns, 12 tool calls and 1,024 tokens per turn.

```bash
python -m agent.tool_eval --output-dir runs/native_tools --gpus 8
```
