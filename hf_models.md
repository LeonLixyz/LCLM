# HuggingFace Models

> All checkpoints below load through the unified API:
>
> ```python
> from latent_context import LCLM
> model = LCLM.from_pretrained("smcleish/<model-id>")
> ```
>
> The loader auto-detects legacy and new config-key layouts (see `latent_context/_config_compat.py`), so legacy IDs in the tables below still work as-is.

Sources:
- Collection: https://huggingface.co/collections/smcleish/compression
- User: https://huggingface.co/Leon-Sean-Dev

Conventions used in the tables below (decoded from the codebase):
- **Compression** — `cs<N>` / `chunksize-N` is `compression_ratio = N`. Default = **16x** when not specified in the ID.
- **Pooling** — three values today: `mean`, `eos`, `concat`. Historic IDs containing `summary_mean` are `pooling=mean` with `encoder_window_size > compression_ratio`; `summary_concat` is `pooling=concat`; `summary` is `pooling=eos` with `encoder_window_size > compression_ratio`. `bst<W>` in old IDs = `encoder_window_size = W`.
- **Adapter** — `attn-mlp` / `mlp-attn` / `mlp` denote `adapter_order`; `ov<K>` is `boundary_overlap = K` (ov0 = no overlap); `causal` / `bidirectional` is the encoder attention mask.
- **LRs** — first `<rate>` = pretraining LR, second `<rate>` after `post-train-` = post-training LR. Omitted when not in the model ID.

---

## `smcleish/compression` Collection

25 models — Qwen3-Embedding-0.6B student, Qwen3-4B-Instruct teacher.

### Leon-Sean-Dev models in the collection

| Date | Model ID | Compression | Pooling | Adapter | Pretrain LR | Post-train LR |
|---|---|---|---|---|---|---|
| Feb 19 | `Qwen3-Embedding-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-attn-mlp-ov256-stage-3-1e-5` | 16x | summary_mean | attn+mlp, ov256 (stage-3) | 1e-5 | — |
| Feb 15 | `Qwen3-Embedding-0.6B-Qwen3-4B-Inst-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data-run-3` | 16x | summary_mean | — | 1e-5 | — |
| Feb 12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Inst-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data-run-2` | 16x | summary_mean | — | 1e-5 | — |
| Feb 9  | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data` | 16x | summary_mean | — | 1e-5 | — |
| Feb 6  | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn-mlp-ov256` | 16x | summary_mean | attn+mlp, ov256 | — | — |
| Feb 6  | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn-mlp-ov256-chunksize-8` | 8x¹ | summary_mean | attn+mlp, ov256 | — | — |
| Jan 19 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn` | 16x | summary_mean | attn only | — | — |
| Jan 12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-1e5` | 16x | summary_mean | mlp, ov0, bidirectional | 1e-5 | — |
| Jan 12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024` | 16x | summary_mean | mlp, ov0, bidirectional | 1e-6 | — |
| Jan 12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-3e6` | 16x | summary_mean | mlp, ov0, bidirectional | 3e-6 | — |
| Dec 30, 2025 | `0.6_4b_eos_causal_embed` | 16x | eos | causal embedder | — | — |
| Dec 30, 2025 | `0.6_4b_eos_causal_instruct` | 16x | eos | causal embedder (instruct variant) | — | — |
| Dec 30, 2025 | `0.6_4b_eos_bidirectional_embed` | 16x | eos | bidirectional embedder | — | — |
| Dec 30, 2025 | `0.6_4b_summary_mean_causal_embed` | 16x | summary_mean | causal embedder | — | — |

¹ ID contains both `cs16` and `chunksize-8`; `chunksize-8` is the operative compression setting.

### smcleish models in the collection

| Date | Model ID | Compression | Pooling | Adapter | Pretrain LR | Post-train LR |
|---|---|---|---|---|---|---|
| Mar 4  | `0.6b-embed-4b-instruct-cs-8-summary-mean-1024-attn-mlp-ov256-stage3-lr-1e-5` | 8x | summary_mean | attn+mlp, ov256 (stage3) | 1e-5 | — |
| Mar 1  | `0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov256` | 16x | summary_mean | mlp, ov256 | — | — |
| ~30d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-causal-1e-5` | 16x | summary_mean | mlp, ov0, causal | 1e-5 | — |
| ~30d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-1e-5` | 16x | summary_mean | mlp, ov0 | 1e-5 | — |
| ~28d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-causal-1e-5-post-train-2e-5` | 16x | summary_mean | mlp, ov0, causal | 1e-5 | 2e-5 |
| ~17d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-8-summary-mean-1024-mlp-ov0-causal-2e-5` | 8x | summary_mean | mlp, ov0, causal | 2e-5 | — |
| ~16d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-causal-1e-5-post-train-3e-5` | 16x | summary_mean | mlp, ov0, causal | 1e-5 | 3e-5 |
| ~14d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-causal-1e-5-post-train-5e-5` | 16x | summary_mean | mlp, ov0, causal | 1e-5 | 5e-5 |
| ~12d ago | `tuo-prod-0.6b-embed-4b-instruct-cs-4-summary-mean-1024-mlp-ov0-causal-3e-5` | 4x | summary_mean | mlp, ov0, causal | 3e-5 | — |
| ~12d ago | `0.6b-embed-4b-instruct-cs-8-summary-mean-1024-mlp-ov0-causal-1e-5-post-train-3e-5` | 8x | summary_mean | mlp, ov0, causal | 1e-5 | 3e-5 |
| ~8d ago  | `tuo-prod-0.6b-embed-4b-instruct-cs-4-summary-mean-1024-mlp-ov0-causal-1e-5-post-train-2e-5` | 4x | summary_mean | mlp, ov0, causal | 1e-5 | 2e-5 |

---

## `Leon-Sean-Dev` Profile (all 15 models)

Team: Sean McLeish, Leon Li. Site: https://mcleish7.github.io/. 14 of these also appear in the `smcleish/compression` collection above; `4_4b_eos_causal_embed` is profile-only.

| Date | Model ID | Compression | Pooling | Adapter | Pretrain LR | Post-train LR |
|---|---|---|---|---|---|---|
| 2026-02-19 | `Qwen3-Embedding-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-attn-mlp-ov256-stage-3-1e-5` | 16x | summary_mean | attn+mlp, ov256 (stage-3) | 1e-5 | — |
| 2026-02-15 | `Qwen3-Embedding-0.6B-Qwen3-4B-Inst-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data-run-3` | 16x | summary_mean | — | 1e-5 | — |
| 2026-02-12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Inst-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data-run-2` | 16x | summary_mean | — | 1e-5 | — |
| 2026-02-09 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-1e5-16384-short-data` | 16x | summary_mean | — | 1e-5 | — |
| 2026-02-06 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn-mlp-ov256-chunksize-8` | 8x¹ | summary_mean | attn+mlp, ov256 | — | — |
| 2026-02-06 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn-mlp-ov256` | 16x | summary_mean | attn+mlp, ov256 | — | — |
| 2026-01-19 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-attn` | 16x | summary_mean | attn only | — | — |
| 2026-01-12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-1e5` | 16x | summary_mean | mlp, ov0, bidirectional | 1e-5 | — |
| 2026-01-12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-lr-3e6` | 16x | summary_mean | mlp, ov0, bidirectional | 3e-6 | — |
| 2026-01-12 | `Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024` | 16x | summary_mean | mlp, ov0, bidirectional | 1e-6 | — |
| 2025-12-30 | `0.6_4b_eos_causal_embed` | 16x | eos | causal embedder | — | — |
| 2025-12-30 | `0.6_4b_eos_causal_instruct` | 16x | eos | causal embedder (instruct variant) | — | — |
| 2025-12-30 | `0.6_4b_eos_bidirectional_embed` | 16x | eos | bidirectional embedder | — | — |
| 2025-12-30 | `0.6_4b_summary_mean_causal_embed` | 16x | summary_mean | causal embedder | — | — |
| 2025-12-26 | `4_4b_eos_causal_embed` | 16x | eos | causal embedder (4B student) | — | — |

¹ ID contains both `cs16` and `chunksize-8`; `chunksize-8` is the operative compression setting.
