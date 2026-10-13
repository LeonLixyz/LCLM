# Timing and the optimized encoder

The paper's TTFT and memory results (Figures 1, 4 and 5) use an
inference-only encoder fast path, `inference/optimized_encoder.py`. The
benchmarks themselves, their commands and the measurement protocol are in
[`benchmarks/timing/README.md`](../benchmarks/timing/README.md).

## Usage

```python
import torch
from inference.hf import load_model
from inference.optimized_encoder import enable_encoder_optimization

model, tokenizer, processor = load_model("latent-context/0.6b-4b-LCLM-16x", device="cuda:0")
model.eval()
optimization = enable_encoder_optimization(model, compile_encoder=True, batch_windows=32)
token_ids = model.encoder.embed_tokenizer.encode(long_text, add_special_tokens=False)
with torch.inference_mode():
    latents = model._process_latent_embeddings([token_ids])  # first call per shape compiles
optimization.restore()  # optional: back to the stock encoder
```

`enable_encoder_optimization` raises if the model is in training mode, is not
on a CUDA device, or uses an unsupported encoder configuration.

## What changes

| Stock encoder | Optimized encoder |
|---|---|
| windows left-padded, padding mask | windows right-padded, no mask (causal attention never sees later padding) |
| attention backend from `load_model` (FlashAttention-2 if installed, else SDPA) | PyTorch SDPA restricted to its FlashAttention backend |
| eager PyTorch | `torch.compile(fullgraph=True, dynamic=True)` and one CUDA graph per batch shape |
| `max_encode_batch_size` from the model | `batch_windows` windows per forward (32, i.e. 32,768 tokens at W=1024) |
| per-group pooling loop | vectorized mean pooling |

The windowing, adapter and decoder are unchanged. Supported checkpoints: causal
encoder mask, `pooling="mean"`, `boundary_overlap=0`, no sliding-window layers;
the released `latent-context/0.6b-4b-LCLM-{4x,8x,16x}` checkpoints qualify.

## Costs and caveats

- The first call with a new batch shape compiles the encoder and captures a
  CUDA graph (seconds to minutes). Benchmarks run warmups first and exclude
  this setup. Up to two shapes are kept (full batches of `batch_windows`
  windows and the trailing partial batch); a third shape frees them and
  captures again, so inputs whose window counts vary from call to call keep
  paying capture cost.
- Graph outputs are copied before they are returned, so latents from
  different batches never alias.
- Use `torch.inference_mode()` for every call once the graphs are captured.
- Latents are not bit-identical to the stock encoder. In BF16, the Flash kernel
  and `torch.compile` change numerics: on the synthetic 4k timing input the
  relative L2 difference between optimized and stock latents was 5.5 to 6.4%
  across the three checkpoints. The paper's accuracy results use the stock
  encoder; `lclm_hf_scaling.py` records this difference with every run.
