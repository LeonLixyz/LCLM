"""LCLM inference.

Two backends:

- ``inference.hf`` — HuggingFace Transformers. Reference path, single device.
- ``inference.vllm`` — vLLM decoder + HF encoder. Production path with paged
  attention, continuous batching, fp8.

Runnable examples live in ``inference/examples/``.
"""

from .hf import generate_text, load_model
from .vllm import LCLMVLLMDecoder

__all__ = ["load_model", "generate_text", "LCLMVLLMDecoder"]
