# Rename the stage-2 checkpoint to LCLM names

Please perform this on Sean's cluster in `/usr/workspace/mcleish1/LCLM`.
Use the latest `train` branch, preserving local changes. It must include
`fda3d0b` (checkpoint validation/microbatch fix) and `d4db69a` (source naming cleanup).
Read `AGENTS.md` first.

The new LCLM code already implements the old `summary_mean` behavior as
`mean` with encoder window 1024. Outputs and gradients matched in GPU tests.
This task changes checkpoint names/metadata, without changing weights or training behavior.

## Locations

Current checkpoint, referenced by both YAMLs in `sean_shells/experiment_config/`:

```text
/usr/workspace/mcleish1/LCLM/tuo_outputs_prod/tuo-prod-0.6b-embed-4b-instruct-cs-16-summary-mean-1024-mlp-ov0-causal-1e-5/checkpoints/Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-summary_mean-bst1024-mlp-ov0/stage2-hf
```

New checkpoint location:

```text
/usr/workspace/mcleish1/LCLM/checkpoints/Qwen3-Embedding-0.6B-Qwen3-4B-Instruct-2507-cs16-mean-w1024-mlp-ov0-causal/stage2-hf
```

## Changes

1. Inspect the source's actual `model_config.json`. Validate it against the
   existing stage-3 config using `utils.checkpointing.validate_resume_model_config`.
   If it disagrees, report the mismatch; do not overwrite metadata to force a match.
2. Preserve the original checkpoint. Copy/reflink it into a temporary sibling
   of the new destination, then rename these entries there:

   | Old | New |
   |---|---|
   | `llm/` | `decoder/` |
   | `embedder/` | `encoder/` |
   | `projectors/` | `adapter/` |
   | `adapter/code_adapter.safetensors` | `adapter/adapter.safetensors` |

3. Write the validated normalized metadata as the new `model_config.json`:

   ```json
   {
     "compression_ratio": 16,
     "encoder_window_size": 1024,
     "pooling": "mean",
     "encoder_mask_type": "causal",
     "boundary_overlap": 0,
     "adapter_type": "mlp"
   }
   ```

   Remove the old metadata keys from this new file. Keep all model/tokenizer
   files, token IDs, shard indexes and weight bytes unchanged; no re-export or retraining.
4. Compare SHA-256 hashes for every source file against its mapped destination,
   excluding only the intentionally rewritten root `model_config.json`. Validate
   the new metadata too. Publish the completed directory atomically; never overwrite
   an existing destination without checking it.
5. Update `stages[3].resume_from_checkpoint` in both the 16k and 32k YAMLs to the
   new path. Keep the run/output names, datasets and all hyperparameters unchanged:
   encoder batch cap 1024, microbatch 2, accumulation 1, ZeRO-1, BF16, LR `3e-5`.

Commit and push the two YAML updates to `train`, preserving other people's work.
Report the destination, hash/metadata checks and commit. Do not launch production
training or remove the loader's support for other older checkpoints as part of this rename.
