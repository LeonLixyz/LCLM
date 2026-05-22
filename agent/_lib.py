"""Shared building blocks for agent-style evals: chunking, two-pass generation, scoring."""

from __future__ import annotations

import os
import re
import sys
from typing import Iterable, Sequence

import torch

# PROJECT_ROOT is set up by agent.__init__; importing it ensures
# the repo root is on sys.path before downstream imports run.
from agent import PROJECT_ROOT  # noqa: F401

from latent_context import from_pretrained  # noqa: E402
from benchmark.scoring_function import get_scorer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"
DEFAULT_BASE_TOKENIZER = "Qwen/Qwen3-4B-Instruct-2507"


def load_agent_model(checkpoint: str, device: str = "cuda", dtype: str = "bf16", compression_ratio: int = 16):
    """Load model + processor + LLM tokenizer + a base tokenizer for chunking.

    Note: ``compression_ratio`` is the LCLM's token→latent ratio and is
    distinct from the haystack-chunking ``chunk_size`` used by
    ``chunk_by_tokens`` below.
    """
    model, decoder_tok, processor = from_pretrained(
        checkpoint, device=device, compression_ratio=compression_ratio, dtype=dtype,
    )
    base_tok = AutoTokenizer.from_pretrained(DEFAULT_BASE_TOKENIZER)
    return model, decoder_tok, processor, base_tok


def chunk_by_tokens(text: str, tokenizer, chunk_size: int = 450) -> list[str]:
    """Split text into ~chunk_size-token pieces along sentence boundaries when possible."""
    if not text or not text.strip():
        return []
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for s in sentences:
        n = len(tokenizer.encode(s))
        if cur_n + n > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur, cur_n = [s], n
        else:
            cur.append(s)
            cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    # Defensive: drop any empty / whitespace-only chunks (can happen with code).
    return [c for c in chunks if c and c.strip()]


def chunk_by_paragraphs(text: str, tokenizer, chunk_size: int = 450) -> list[str]:
    """Like chunk_by_tokens but prefers double-newline boundaries first."""
    if not text or not text.strip():
        return []
    tokens = tokenizer.encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for p in paras:
        n = len(tokenizer.encode(p))
        if n > chunk_size:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_n = [], 0
            chunks.extend(chunk_by_tokens(p, tokenizer, chunk_size))
        elif cur_n + n > chunk_size and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_n = [p], n
        else:
            cur.append(p)
            cur_n += n
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c and c.strip()]


def parse_chunk_selection(response: str, num_chunks: int) -> list[int]:
    """Extract 0-indexed chunk indices from a triage response."""
    m = re.search(r"SELECTED:\s*([\d,\s]+)", response)
    nums = re.findall(r"\d+", m.group(1)) if m else None
    if not nums:
        lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
        nums = re.findall(r"\d+", lines[-1] if lines else response)
    out = []
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < num_chunks:
            out.append(idx)
    return sorted(set(out)) if out else [0]


def generate(model, decoder_tokenizer, processor, prompt: str, *,
             device: str = "cuda", max_tokens: int = 128, temperature: float = 0.0,
             label: str = "", assistant_prefix: str = "",
             pre_formatted: bool = False) -> tuple[str, dict]:
    """Run one generation; return (decoded_text, stats).

    If `pre_formatted=True`, the `prompt` argument is treated as the full
    chat-template string already (used for multi-turn chat); otherwise it is
    wrapped as a single user-then-assistant turn.

    `assistant_prefix` is inserted after `<|im_start|>assistant\\n` so the
    model continues from it (e.g. "Answer:" / "Summary:" for LongBench).
    """
    if pre_formatted:
        formatted = prompt
    else:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"
    with torch.inference_mode():
        proc = processor.process_wrapped_batch(
            prompts=[formatted], targets=None, padding="longest",
            truncation=True, return_tensors="pt",
        )
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        seq_len = int(input_ids.shape[1])
        ec = sum(proc["embed_token_counts"][0]) if proc["embed_token_counts"][0] else 0
        n_start = int((input_ids[0] == processor.memory_start_id).sum())
        n_end = int((input_ids[0] == processor.memory_end_id).sum())
        n_mem = int((input_ids[0] == processor.memory_token_id).sum())
        n_real = seq_len - n_mem - n_start - n_end
        equiv = n_real + ec + n_start + n_end
        savings_pct = round((1 - seq_len / equiv) * 100, 1) if equiv > 0 else 0.0

        do_sample = temperature > 0
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_token_ids=proc["memory_token_ids"],
            memory_positions=proc["memory_positions"],
            latent_counts=proc["latent_counts"],
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            do_sample=do_sample,
            repetition_penalty=1.0,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
        )
        text = decoder_tokenizer.decode(out[0], skip_special_tokens=False)

    stats = {
        "compressed_tokens": seq_len,
        "uncompressed_tokens": equiv,
        "savings_pct": savings_pct,
    }
    if label:
        print(f"  [{label}] compressed={seq_len} equiv={equiv} savings={savings_pct}%")
    return text.strip(), stats


def clean_response(text: str) -> str:
    """Strip Qwen chat artifacts from a generation."""
    return text.split("<|im_end|>")[0].strip()


def build_triage_prompt(prefix: str, chunks: Sequence[str], question: str,
                        chunk_label: str = "Chunk", min_hint: int = 1) -> str:
    parts = [prefix.rstrip() + "\n"] if prefix else []
    for i, c in enumerate(chunks):
        parts.append(f"{chunk_label} {i+1}: {MEMORY_START}{c}{MEMORY_END}")
    parts.append("")
    parts.append(question.strip())
    bias = ""
    if min_hint > 1:
        bias = (f" Lean toward INCLUDING chunks rather than excluding. "
                f"If multiple chunks could plausibly contain a relevant value, key, or candidate, "
                f"select ALL of them. Aim to select at least {min_hint} chunks unless you are highly confident "
                f"a smaller set covers the answer.")
    parts.append(
        f"\nIdentify EVERY chunk above that could plausibly contain information needed to answer the question.{bias} "
        "Briefly explain your reasoning, then on the LAST line write "
        "`SELECTED: <comma-separated chunk numbers>` (e.g. `SELECTED: 3,5,9,12`)."
    )
    return "\n".join(parts)


def apply_min_select(selected: list[int], n_chunks: int, min_select: int) -> list[int]:
    """Pad a triage selection up to min_select chunks if needed.

    Strategy: keep the model's selections, then add chunks not yet selected,
    spreading them roughly evenly across the body so we don't bias to one side.
    """
    if min_select <= 0 or len(selected) >= min_select or n_chunks == 0:
        return selected
    target = min(min_select, n_chunks)
    sel = set(selected)
    # Spread the remaining picks evenly across un-selected chunk indices.
    remaining = [i for i in range(n_chunks) if i not in sel]
    need = target - len(sel)
    if need >= len(remaining):
        sel.update(remaining)
    else:
        # Even-stride sample so we cover the body, not just the start.
        step = len(remaining) / need
        for k in range(need):
            sel.add(remaining[int(k * step)])
    return sorted(sel)


def build_answer_prompt(prefix: str, chunks: Sequence[str], selected: Iterable[int],
                        question: str, chunk_label: str = "Chunk") -> str:
    sel = set(selected)
    parts = [prefix.rstrip() + "\n"] if prefix else []
    for i, c in enumerate(chunks):
        if i in sel:
            parts.append(f"{chunk_label} {i+1} [EXPANDED]:\n{c}")
        else:
            parts.append(f"{chunk_label} {i+1} [compressed]: {MEMORY_START}{c}{MEMORY_END}")
    parts.append("")
    parts.append(question.strip())
    return "\n".join(parts)


def build_compressed_only_prompt(prefix: str, chunks: Sequence[str], question: str,
                                 chunk_label: str = "Chunk") -> str:
    """No triage; every chunk stays compressed. Used for summarization-style tasks."""
    return build_answer_prompt(prefix, chunks, selected=[], question=question, chunk_label=chunk_label)


def build_single_block_prompt(prefix: str, body: str, question: str) -> str:
    """Wrap the entire body in ONE memory block — no chunk labels, no segmentation.

    Matches the standard memwrap eval's single-block format. Use for aggregation /
    tracking tasks (vt, cwe, passage_count) where inter-segment relationships matter
    and per-chunk compression breaks the chain of reasoning across segments.
    """
    pre = (prefix.rstrip() + "\n") if prefix else ""
    return f"{pre}{MEMORY_START}\n{body}\n{MEMORY_END}\n\n{question.strip()}"


# ──────────────────────────────────────────────────────────────────────────
# Multi-round agent: model decides EXPAND vs ANSWER at each round, up to N rounds.
# ──────────────────────────────────────────────────────────────────────────

def build_round_prompt(prefix: str, chunks: Sequence[str], expanded: Iterable[int],
                       question: str, rounds_left: int) -> str:
    """Build a single-round prompt for the multi-round agent.

    Layout (concat-style — full body + relevant excerpts, then question + tool spec):

      <prefix>
      [Document — indexed chunks for reference]
      Chunk 1: <|memory_start|>...<|memory_end|>
      Chunk 2: <|memory_start|>...<|memory_end|>
      ...
      Chunk K: <|memory_start|>...<|memory_end|>

      [Already expanded — plain text, concatenated]   (omitted when none)
      {chunk_i text}

      {chunk_j text}

      <question>

      <tool spec + decision instructions>
    """
    expanded_set = sorted(set(expanded))
    parts: list[str] = []
    if prefix:
        parts.append(prefix.rstrip())
        parts.append("")
    parts.append("[Document — indexed chunks, compressed]")
    for i, c in enumerate(chunks):
        # Skip degenerate empty chunks: an empty <|memory_start|><|memory_end|>
        # region crashes the processor (chunk_count=0). The chunker already
        # filters these, but defend at the prompt level too.
        if not c or not c.strip():
            continue
        parts.append(f"Chunk {i+1}: {MEMORY_START}{c}{MEMORY_END}")
    parts.append("")
    if expanded_set:
        parts.append("[Already expanded passages]")
        for i in expanded_set:
            parts.append(chunks[i].strip())
            parts.append("")
    parts.append(f"Question: {question.strip()}")
    parts.append("")
    if not expanded_set:
        # First decision turn: force at least one EXPAND. Compressed memory
        # alone is too lossy for confident retrieval-style answers.
        parts.append(
            "Each chunk above is shown in compressed form (lossy 16x summary). The "
            "compressed view is fine for skimming structure, but it is NOT reliable "
            "for exact-string retrieval --- compressed memory produces near-miss "
            "typos (right shape, wrong characters) for UUIDs, hashes, names, code, "
            "and any verbatim string longer than ~8 characters.\n\n"
            "RULE: this is the first decision turn. You must EXPAND at least one "
            "chunk before committing an ANSWER. Pick the chunks that look most "
            "likely to contain information needed by the question; multiple chunks "
            "per call are encouraged if several look plausible.\n\n"
            "ANSWER FORMAT NOTES (for later rounds):\n"
            "- The 'Chunk N' labels are an INDEX for the EXPAND tool, not part of "
            "the document. If your answer references a label inside the document "
            "(e.g. 'Paragraph N'), copy it exactly from the expanded text - do NOT "
            "return the Chunk number.\n"
            "- For multiple-choice, write only the letter (A / B / C / D / E).\n"
            "- For summarization, write the summary directly with no preamble.\n\n"
            f"You have {rounds_left} round(s) of expansion left after this one.\n\n"
            "Output a single line:\n"
            "  EXPAND: <comma-separated chunk numbers>"
        )
        return "\n".join(parts)
    parts.append(
        "Each chunk above is shown in compressed form (lossy 16x summary). To read exact "
        "characters of a chunk you must EXPAND it. You have already expanded some "
        "chunks (shown above as plain text under [Already expanded passages]).\n\n"
        "HARD RULE — answer fidelity:\n"
        "If the answer to the question requires copying an exact string verbatim (UUID, "
        "hash, ID, name, code, quoted phrase, anything longer than ~8 characters or with "
        "non-dictionary tokens) and the expanded passages above do NOT yet contain that "
        "string, EXPAND more chunks rather than guessing from compressed memory. "
        "Compressed memory produces near-miss typos (right shape, wrong characters) for "
        "any exact-string answer. Examples that REQUIRE expansion before answering: 'what "
        "is the magic uuid for X', 'what is the password / hash / token', 'what name "
        "appears in the document next to Y', 'fill in the exact phrase that follows Z'.\n\n"
        "When NOT to EXPAND:\n"
        "- The compressed view already contains what you need (summary / overall topic / "
        "yes-no question / count of duplicates / classification).\n"
        "- You have already expanded the relevant chunks in a previous round.\n\n"
        "When you DO EXPAND, prefer multiple chunks per call if multiple chunks could "
        "plausibly contain the answer. Cheaper to expand 5 candidates and pick the right "
        "one in the next round than to guess wrong now.\n\n"
        "ANSWER FORMAT NOTES:\n"
        "- The 'Chunk N' labels above are an INDEX for the EXPAND tool. They are NOT part "
        "of the document. If your answer is supposed to reference a label that exists in "
        "the document itself (e.g. 'Paragraph N', 'Section X', 'Document 3'), copy the "
        "label exactly as it appears INSIDE the expanded chunk text — do NOT return the "
        "Chunk number.\n"
        "- For multiple-choice questions, write only the letter (A / B / C / D / E) after "
        "ANSWER:.\n"
        "- For summarization, write the summary directly after ANSWER: (one or more "
        "sentences, no preamble).\n\n"
        f"You have {rounds_left} round(s) of expansion left after this one.\n\n"
        "Output exactly ONE of the following on a new line:\n"
        "  EXPAND: <comma-separated chunk numbers>     (request more chunks)\n"
        "  ANSWER: <your final answer>                 (commit to an answer)"
    )
    return "\n".join(parts)


_EXPAND_RE = re.compile(r"EXPAND\s*:\s*([0-9,\s]+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_round_output(text: str, n_chunks: int) -> dict:
    """Parse a round response into either an EXPAND request or an ANSWER commit.

    Returns one of:
      {"action": "expand", "indices": [0-indexed ints], "raw": text}
      {"action": "answer", "answer": str, "raw": text}

    Logic:
      - If both EXPAND and ANSWER appear, pick whichever appears LAST in the text
        (i.e. the model's most-recent decision).
      - Indices are clipped to [0, n_chunks) and de-duplicated.
      - If neither directive is present, treat the whole text as the answer.
    """
    text = text.strip()
    expand_m = list(_EXPAND_RE.finditer(text))
    answer_m = list(_ANSWER_RE.finditer(text))
    last_expand = expand_m[-1] if expand_m else None
    last_answer = answer_m[-1] if answer_m else None

    pick_expand = last_expand is not None and (
        last_answer is None or last_expand.start() > last_answer.start()
    )
    if pick_expand:
        nums = re.findall(r"\d+", last_expand.group(1))
        idxs = sorted({int(n) - 1 for n in nums if 0 <= int(n) - 1 < n_chunks})
        return {"action": "expand", "indices": idxs, "raw": text}
    if last_answer is not None:
        ans = last_answer.group(1).strip()
        # ANSWER text is everything after "ANSWER:" up to optional trailing tags
        ans = ans.split("<|im_end|>")[0].strip()
        return {"action": "answer", "answer": ans, "raw": text}
    # Fallback: no directive — accept whole text as answer
    return {"action": "answer", "answer": text, "raw": text}


def run_multi_round_agent(model, decoder_tok, processor, *, prefix: str,
                          chunks: Sequence[str], question: str,
                          max_rounds: int = 5, max_tokens: int = 256,
                          device: str = "cuda", label: str = "") -> dict:
    """Run the multi-round expand-or-answer loop.

    Returns dict with:
      response:       final answer string
      expanded:       sorted list of 0-indexed chunks ever expanded
      n_rounds:       number of rounds actually run
      transcripts:    per-round {prompt_summary, action, indices/answer} list
      answer_stats:   stats from the round that committed the answer
    """
    expanded: set[int] = set()
    transcripts: list[dict] = []
    last_stats: dict = {}
    last_text = ""
    n_chunks = len(chunks)

    for r in range(max_rounds):
        rounds_left = max_rounds - r - 1  # rounds AFTER this one
        prompt = build_round_prompt(prefix, chunks, expanded, question, rounds_left)
        raw, stats = generate(model, decoder_tok, processor, prompt,
                              device=device, max_tokens=max_tokens, temperature=0.0,
                              label=f"{label}R{r+1}")
        text = clean_response(raw)
        last_text = text
        last_stats = stats
        parsed = parse_round_output(text, n_chunks)
        transcripts.append({
            "round": r + 1,
            "action": parsed["action"],
            "indices": parsed.get("indices"),
            "answer": parsed.get("answer"),
            "raw": text,  # full model output, for debugging
            "n_expanded_before": len(expanded),
            "stats": stats,
        })

        # Hard constraint: at least one EXPAND must precede ANSWER.  If the
        # model ignores the prompt and emits ANSWER on the first turn, demote
        # to a forced expansion so we don't accept an answer drawn purely
        # from compressed memory.
        if parsed["action"] == "answer" and not expanded:
            # Try to recover any chunk indices the model may have mentioned
            # in its raw output (it sometimes lists candidates while answering).
            forced_idx = re.findall(r"\b([0-9]{1,3})\b", text)
            forced = sorted({int(n) - 1 for n in forced_idx
                             if 1 <= int(n) <= n_chunks})[:5]  # cap at 5
            if not forced:
                forced = [0]  # nothing parseable; expand chunk 1 as a fallback
            expanded.update(forced)
            transcripts[-1]["action"] = "answer_demoted_to_expand"
            transcripts[-1]["indices"] = sorted(forced)
            if rounds_left == 0:
                break
            continue

        if parsed["action"] == "answer":
            return {
                "response": parsed["answer"],
                "expanded": sorted(expanded),
                "n_rounds": r + 1,
                "transcripts": transcripts,
                "answer_stats": stats,
            }
        # action == "expand"
        new_idxs = [i for i in parsed["indices"] if i not in expanded]
        if not new_idxs:
            # Model wants to expand but selected nothing new — break and force answer
            break
        if rounds_left == 0:
            # No more rounds; treat this round's text as the answer (best-effort)
            break
        expanded.update(new_idxs)

    # Out of rounds (or self-loop): force a final answer pass with everything currently expanded.
    final_prompt = build_round_prompt(prefix, chunks, expanded, question, rounds_left=0)
    # Strip the EXPAND option from the prompt by appending an explicit instruction.
    final_prompt += "\n\nNo more expansion rounds available. Write `ANSWER: <your final answer>` now."
    raw, stats = generate(model, decoder_tok, processor, final_prompt,
                          device=device, max_tokens=max_tokens, temperature=0.0,
                          label=f"{label}FINAL")
    text = clean_response(raw)
    parsed = parse_round_output(text, n_chunks)
    answer = parsed.get("answer", text)
    transcripts.append({
        "round": len(transcripts) + 1,
        "action": "forced_answer",
        "answer": answer,
        "raw": text,
        "n_expanded_before": len(expanded),
        "stats": stats,
    })
    return {
        "response": answer,
        "expanded": sorted(expanded),
        "n_rounds": len(transcripts),
        "transcripts": transcripts,
        "answer_stats": stats,
    }


def score_response(response: str, scoring_function: str, ground_truth: dict) -> dict:
    """Wrap benchmark.scoring_function.get_scorer for one response."""
    scorer = get_scorer(scoring_function)
    return scorer(response, ground_truth)


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Average all numeric metric keys across per-sample dicts."""
    if not per_sample:
        return {}
    keys = set()
    for d in per_sample:
        keys.update(k for k, v in d.items() if isinstance(v, (int, float)))
    out = {}
    for k in keys:
        vals = [d[k] for d in per_sample if isinstance(d.get(k), (int, float))]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    out["__n__"] = len(per_sample)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Multi-turn chat-format agent
#
# Instead of rebuilding a full prompt every round (the older single-turn
# approach), this variant maintains a real conversation history:
#
#   system    -> agent role + tool spec
#   user      -> [Document — indexed chunks, compressed] + Question
#   assistant -> "EXPAND: 7, 9"            (model's tool call)
#   user      -> "[EXPAND result] Chunk 7: ...  Chunk 9: ..."  (tool return)
#   assistant -> "EXPAND: 11"
#   user      -> "[EXPAND result] Chunk 11: ..."
#   assistant -> "ANSWER: ..."
#
# The compressed document appears in the first user turn ONLY, so the
# compressor runs exactly once per sample regardless of round count.
# Tool results are appended (only the new chunks each turn), so the
# conversation grows linearly with how much the agent chose to expand
# rather than re-rendering everything every round.  The model also
# sees its own prior reasoning across rounds, which is the natural
# multi-step agent pattern.
# ──────────────────────────────────────────────────────────────────────────

def build_initial_chat_messages(prefix: str, chunks: Sequence[str], question: str,
                                 max_rounds: int = 5) -> list[dict]:
    """Build the seed message list (system + first user turn).

    The first user turn is the only one that contains <|memory_start|>...<|memory_end|>
    regions, so the compressor runs once on those and never again for this sample.
    """
    system = (
        "You are an agent reasoning over a long document. Parts of the document "
        "are shown to you in compressed form using <|memory_start|>...<|memory_end|> "
        "tags; each tagged region is a 16x lossy summary. The compressed view is "
        "fine for skimming structure, but you cannot read exact characters from it.\n\n"
        "You have a single tool, EXPAND, which returns the original plain text of "
        "chunks you specify. To call the tool, write a single line of the form "
        "`EXPAND: <comma-separated chunk numbers>`. The next user turn will return "
        "the requested chunks as plain text. To finish, write a single line of the "
        "form `ANSWER: <your final answer>`.\n\n"
        f"You have {max_rounds} rounds total.\n\n"
        "RULE: You must EXPAND at least one chunk before committing an ANSWER. "
        "Compressed memory produces near-miss typos for any verbatim string longer "
        "than ~8 characters (UUIDs, hashes, names, code, quoted phrases) - do not "
        "answer such questions from the compressed view alone.\n\n"
        "Format notes:\n"
        "- 'Chunk N' is an INDEX for the EXPAND tool, not part of the document. "
        "If the answer references a label that exists inside the document text "
        "(e.g. 'Paragraph N', 'Section X'), copy it from the expanded chunk text "
        "exactly; do NOT return the Chunk number.\n"
        "- For multiple-choice questions, write only the letter (A / B / C / D / E) "
        "after ANSWER:.\n"
        "- For summarization, write the summary directly after ANSWER: with no "
        "preamble.\n"
        "- Multiple chunks per EXPAND call are encouraged when several plausibly "
        "contain the answer."
    )
    parts: list[str] = []
    if prefix:
        parts.append(prefix.rstrip())
        parts.append("")
    parts.append("[Document — indexed chunks, compressed]")
    for i, c in enumerate(chunks):
        if not c or not c.strip():
            continue
        parts.append(f"Chunk {i+1}: {MEMORY_START}{c}{MEMORY_END}")
    parts.append("")
    parts.append(f"Question: {question.strip()}")
    return [
        {"role": "system",  "content": system},
        {"role": "user",    "content": "\n".join(parts)},
    ]


def format_tool_result(chunks: Sequence[str], new_indices: Iterable[int],
                        rounds_left: int) -> str:
    """Format a tool-result user turn for newly expanded chunks."""
    parts = ["[EXPAND result]"]
    for i in sorted(new_indices):
        parts.append(f"Chunk {i+1}:")
        parts.append(chunks[i].strip() if chunks[i] else "")
        parts.append("")
    if rounds_left > 0:
        parts.append(f"({rounds_left} round(s) of expansion left.)")
    return "\n".join(parts)


def render_chat(messages: list[dict], assistant_prefix: str = "") -> str:
    """Render a message list into the Qwen3 chat-template string, ending at the
    assistant cursor for next-turn generation."""
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    out.append(f"<|im_start|>assistant\n{assistant_prefix}")
    return "".join(out)


def run_multi_round_agent_chat(model, decoder_tok, processor, *, prefix: str,
                                 chunks: Sequence[str], question: str,
                                 max_rounds: int = 5, max_tokens: int = 256,
                                 device: str = "cuda", label: str = "") -> dict:
    """Multi-turn chat-format agent loop.

    Returns the same dict shape as `run_multi_round_agent`:
      response, expanded, n_rounds, transcripts, answer_stats
    """
    messages = build_initial_chat_messages(prefix, chunks, question, max_rounds)
    expanded: set[int] = set()
    transcripts: list[dict] = []
    last_stats: dict = {}
    n_chunks = len(chunks)

    for r in range(max_rounds):
        rounds_left = max_rounds - r - 1  # remaining AFTER this turn
        formatted = render_chat(messages)
        raw, stats = generate(
            model, decoder_tok, processor, formatted,
            device=device, max_tokens=max_tokens, temperature=0.0,
            label=f"{label}R{r+1}", pre_formatted=True,
        )
        text = clean_response(raw)
        last_stats = stats
        parsed = parse_round_output(text, n_chunks)

        # Append the assistant turn to the conversation history
        messages.append({"role": "assistant", "content": text})

        transcripts.append({
            "round": r + 1,
            "action": parsed["action"],
            "indices": parsed.get("indices"),
            "answer": parsed.get("answer"),
            "raw": text,
            "n_expanded_before": len(expanded),
            "stats": stats,
        })

        # Hard constraint: at least one EXPAND must precede ANSWER. If the model
        # ignores the system rule and emits ANSWER on the first turn, demote to
        # a forced expansion so we don't accept compressed-memory hallucinations.
        if parsed["action"] == "answer" and not expanded:
            forced_idx = re.findall(r"\b([0-9]{1,3})\b", text)
            forced = sorted({int(n) - 1 for n in forced_idx
                             if 1 <= int(n) <= n_chunks})[:5]
            if not forced:
                forced = [0]
            expanded.update(forced)
            transcripts[-1]["action"] = "answer_demoted_to_expand"
            transcripts[-1]["indices"] = sorted(forced)
            # Append a tool-result user turn telling the model what was expanded.
            messages.append({
                "role": "user",
                "content": (
                    "Reminder: you must EXPAND at least one chunk before answering. "
                    f"Expanding chunks {sorted([i+1 for i in forced])} based on the "
                    "indices you mentioned.\n\n" +
                    format_tool_result(chunks, forced, rounds_left)
                ),
            })
            if rounds_left == 0:
                break
            continue

        if parsed["action"] == "answer":
            return {
                "response": parsed["answer"],
                "expanded": sorted(expanded),
                "n_rounds": r + 1,
                "transcripts": transcripts,
                "answer_stats": stats,
            }

        # action == "expand"
        new_idxs = [i for i in parsed["indices"] if i not in expanded]
        if not new_idxs:
            # Self-loop without progress; force final answer
            break
        if rounds_left == 0:
            break
        expanded.update(new_idxs)
        # Append the tool-result user turn (only the new chunks)
        messages.append({
            "role": "user",
            "content": format_tool_result(chunks, new_idxs, rounds_left),
        })

    # Out of rounds — force a final answer pass (with EXPAND disabled).
    messages.append({
        "role": "user",
        "content": ("No more expansion rounds available. "
                    "Write `ANSWER: <your final answer>` now."),
    })
    formatted = render_chat(messages)
    raw, stats = generate(
        model, decoder_tok, processor, formatted,
        device=device, max_tokens=max_tokens, temperature=0.0,
        label=f"{label}FINAL", pre_formatted=True,
    )
    text = clean_response(raw)
    parsed = parse_round_output(text, n_chunks)
    answer = parsed.get("answer", text)
    transcripts.append({
        "round": len(transcripts) + 1,
        "action": "forced_answer",
        "answer": answer,
        "raw": text,
        "n_expanded_before": len(expanded),
        "stats": stats,
    })
    return {
        "response": answer,
        "expanded": sorted(expanded),
        "n_rounds": len(transcripts),
        "transcripts": transcripts,
        "answer_stats": stats,
    }


__all__ = [
    "load_agent_model",
    "chunk_by_tokens",
    "chunk_by_paragraphs",
    "parse_chunk_selection",
    "apply_min_select",
    "generate",
    "clean_response",
    "build_triage_prompt",
    "build_answer_prompt",
    "build_compressed_only_prompt",
    "build_single_block_prompt",
    "build_round_prompt",
    "parse_round_output",
    "run_multi_round_agent",
    "build_initial_chat_messages",
    "format_tool_result",
    "render_chat",
    "run_multi_round_agent_chat",
    "score_response",
    "aggregate_metrics",
    "MEMORY_START",
    "MEMORY_END",
]
