"""LCLM inference.

Two backends:

- ``inference.hf`` — HuggingFace Transformers. Reference path, single device.
- ``inference.vllm_inference`` — two-stage CLI: HF encoder writes a ``.pt``
  blob of latent tokens, then a separate vLLM process reads it and
  generates. Use this for batched eval / serving. Running both in one
  process OOMs (vLLM grabs all GPU memory at init).

Runnable examples live in ``inference/examples/``.
"""

from .hf import generate_text, load_model

__all__ = ["load_model", "generate_text"]
