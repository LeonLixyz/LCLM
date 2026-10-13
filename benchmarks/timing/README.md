# Timing benchmarks

Code for the efficiency results in Section 6 and the "Timing Implementation"
appendix. All runs use one GPU (an H200 141 GB in the paper), batch size 1.
LCLM timings use the optimized encoder in
[`inference/optimized_encoder.py`](../../inference/optimized_encoder.py); see
[`docs/timing.md`](../../docs/timing.md).

| Script | Figure | Measures |
|---|---|---|
| `lclm_hf_scaling.py` | 4, and the LCLM TTFT in 1 | LCLM TTFT + peak memory, 4k to 1M tokens, Hugging Face decoder |
| `kv_baselines_hf.py` | 4, and the baseline TTFT in 1 | KV-cache compression baselines, same context grid |
| `vllm_ttft.py` | 5 | LCLM and uncompressed Qwen3 / Llama 3 models served with vLLM |
| `modal_app.py` | | runs any of the above on a Modal H200 |

Run every command from the repository root. Each script appends one JSON
record per measurement to `--output` (JSONL), including every trial.

## Dependencies

| Script | Needs beyond `uv sync` |
|---|---|
| `lclm_hf_scaling.py` | nothing (PyTorch SDPA; `flash-attn` not required) |
| `vllm_ttft.py` | `vllm==0.11.0` |
| `kv_baselines_hf.py` | `flash-attn` 2.x; KVPress at commit `fa7a0dc1d6d5332a8ab00d03c67de7182c6846b0` (`pip install "kvpress @ git+https://github.com/NVIDIA/kvpress@fa7a0dc1d6d5332a8ab00d03c67de7182c6846b0"`); for Attention Matching, a clone of the [official repository](https://github.com/adamzweiger/compaction) at `2020f6d37940fe013e7e7f1ebf409e5d107c5670` with its `requirements.txt` installed |

The paper's runs used the `vllm/vllm-openai:v0.11.0` image (PyTorch 2.8) with
`transformers==4.57.1`, `accelerate==1.10.1` and `peft==0.17.1`;
`modal_app.py` builds the same environment.

## Commands

```bash
# Figure 4 / Figure 1, LCLM side (one run per compression ratio).
for r in 4 8 16; do
  python -m benchmarks.timing.lclm_hf_scaling \
      --checkpoint latent-context/0.6b-4b-LCLM-${r}x \
      --output runs/timing/lclm_hf_scaling.jsonl
done
# The paper's 4x point at 1M tokens was measured in a separate run with
#   --lengths 1M --expandable-segments

# Figure 4 / Figure 1, KV-cache baselines.
python -m benchmarks.timing.kv_baselines_hf \
    --methods all --attention-matching-repo /path/to/compaction \
    --output runs/timing/kv_baselines_hf.jsonl

# Figure 5: one process per model.
for m in lclm-4x lclm-8x lclm-16x qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b \
         qwen3-4b-instruct-2507 llama3.2-1b-instruct llama3.2-3b-instruct llama3.1-8b-instruct; do
  python -m benchmarks.timing.vllm_ttft --model $m --output runs/timing/vllm_ttft.jsonl
done
```

Useful options: `--lengths 4k,64k` (scaling scripts), `--benchmarks
ruler_4096,longbench` (`vllm_ttft.py`), `--warmups` / `--repeats`,
`--checkpoint` (any local LCLM directory or Hugging Face id).

On Modal (results land in the `lclm-timing` volume and `runs/timing/`):

```bash
modal run benchmarks/timing/modal_app.py --script lclm_hf_scaling \
    --args "--checkpoint latent-context/0.6b-4b-LCLM-16x"
modal run benchmarks/timing/modal_app.py --script vllm_ttft --args "--model lclm-16x"
modal run benchmarks/timing/modal_app.py --script kv_baselines_hf --args "--methods all"
```

Set `LCLM_TIMING_VOLUME` to use another volume and `LCLM_HF_SECRET` to the
name of a Modal secret with `HF_TOKEN` for the gated Llama models.

## Protocol

**Figures 1 and 4 (vs. KV-cache compression).** Every decoder runs in
Hugging Face Transformers, batch size 1. Synthetic contexts of N tokens,
N = 4k, 8k, ..., 1M. The accuracy-vs-TTFT plots (Figure 1 and its appendix
counterpart) take the TTFT at 4k, 8k and 16k for RULER and at 64k for
LongBench and LongHealth5.

- LCLM (`lclm_hf_scaling.py`): N encoder tokens (`" the"` repeated) through
  the optimized encoder (Flash SDPA, `torch.compile`, CUDA graphs, 32 windows
  of W=1024 per forward, i.e. 32,768 tokens per pass), decoder prefill over
  the N/ratio latents with PyTorch SDPA and logits at every position, argmax
  of the last position. TTFT = encoder + adapter + decoder prefill + first
  token. Two warmups, five measured trials; fastest trial reported. Peak
  memory is `torch.cuda.max_memory_allocated`, reset before each trial,
  maximum over measured trials. Compilation and graph capture happen during
  the warmups. The script also records the relative L2 difference between
  optimized and stock encoder latents at 4k.
- Baselines (`kv_baselines_hf.py`): Qwen3-4B-Instruct-2507 with
  FlashAttention-2, N copies of token id 0, one prefill forward inside the
  KVPress press (logits at every position); for Attention Matching, KV
  extraction over the repeat prompt plus compaction. Fresh model per context
  length, 30 warmups, then 20 timed trials; fastest trial reported, peak
  memory over the timed trials. Contexts above a method's first OOM are
  marked OOM without running.

**Figure 5 (vs. uncompressed models).** Every decoder is served with vLLM
0.11 in BF16 on one GPU: batch size 1 (`max_num_seqs=1`), prefix caching
off, chunked prefill with an 8,192-token budget, piecewise CUDA graphs
captured for every prefill chunk size used, `gpu_memory_utilization=0.65`,
one greedy output token. Contexts: 4k, 8k, 16k (RULER) and 64k (LongBench,
LongHealth5). Qwen3 base models (native 40,960 positions) are served with
YaRN factor 4 at all lengths. For an LCLM, the optimized encoder encodes the
N real token ids (LongHealth5: five segments, 64 windows in total) and vLLM
prefills a dummy-token request of N/ratio tokens; TTFT is their sum within
the same trial. Engine start-up, encoder compilation and the hand-off of
latents to vLLM are excluded. Two warmups, five measured trials; fastest
trial reported.

## Reference numbers (H200, from the paper's runs)

LCLM, `lclm_hf_scaling.py` (TTFT ms / peak GiB):

| ratio | 4k | 64k | 256k | 1M |
|---|---|---|---|---|
| 4x | 38 / 9.2 | 684 / 16.0 | 5,330 / 37.1 | 69,676 / 121.4 |
| 8x | 29 / 9.0 | 374 / 12.5 | 2,134 / 23.0 | 19,654 / 65.2 |
| 16x | 27 / 8.9 | 255 / 10.7 | 1,159 / 16.0 | 7,326 / 37.1 |

Baselines, `kv_baselines_hf.py` (time ms / peak GiB; all OOM at 512k except
Attention Matching, which fails numerically at 512k):

| method | 4k | 64k | 256k |
|---|---|---|---|
| no compression | 100 / 9.3 | 4,471 / 35.4 | 61,416 / 119.0 |
| SnapKV | 121 / 8.8 | 4,597 / 27.0 | 61,783 / 85.3 |
| KVzip | 1,250 / 10.7 | 10,301 / 36.5 | 81,729 / 120.6 |
| Attention Matching | 2,369 / 8.2 | 33,824 / 22.8 | 358,179 / 66.2 |

`vllm_ttft.py`, TTFT ms at RULER 4k / LongBench 64k: LCLM 4x 28 / 470,
8x 22 / 278, 16x 21 / 211; Qwen3-4B-Instruct-2507 56 / 2,840;
Qwen3-0.6B 15 / 965; Llama-3.1-8B-Instruct 89 / 3,228.
