"""Online latent compaction for long generations (paper Section 7.2).

During generation the oldest generated tokens are periodically encoded by the
LCLM encoder into latent tokens, keeping only the most recent tokens raw:

    from latent_context import from_pretrained
    from compaction import HFBackend, generate_with_compaction

    model, tokenizer, _ = from_pretrained("latent-context/0.6b-4b-LCLM-16x")
    out = generate_with_compaction(HFBackend(model, tokenizer), "Prove that ...")
    print(out["text"], out["avg_context_positions"], out["peak_context_positions"])

``VLLMBackend`` with ``compaction.scheduler.serve`` runs many trajectories in
parallel; ``python -m compaction.eval_math`` reproduces the MATH results.
"""
from .backend import SAMPLING, HFBackend, VLLMBackend
from .controller import (
    BLOCK_SIZE,
    KEEP_RECENT,
    MAX_CONTEXT,
    context_stats,
    generate_full_context,
    generate_with_compaction,
    run,
    trajectory,
)


__all__ = [
    'BLOCK_SIZE', 'KEEP_RECENT', 'MAX_CONTEXT', 'HFBackend', 'VLLMBackend', 'SAMPLING',
    'context_stats', 'generate_full_context', 'generate_with_compaction', 'run', 'trajectory',
]
