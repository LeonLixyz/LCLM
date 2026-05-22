#!/usr/bin/env python3
"""
Entrypoint for training Code LLaVA using the organized modules.
Parses CLI args into a TrainingConfig and invokes the trainer.

Usage:
    # With individual args
    accelerate launch train/launch_train.py --train_decoder true --decoder_lr 1e-6 ...

    # With YAML config (recommended)
    accelerate launch train/launch_train.py --config scripts/experiment_config/pipeline_config.yaml --stage 0
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import argparse
import os
import sys
import json
import torch
import yaml
from transformers import HfArgumentParser
from train.trainer import LCLMTrainer


def generate_experiment_name(cfg: dict) -> str:
    """Encode {encoder}-{decoder}-cs{N}-w{W}-{pooling}-{adapter_type}-ov{O}-{mask}."""
    encoder_short = cfg["models"]["encoder"].split("/")[-1]
    decoder_short = cfg["models"]["decoder"].split("/")[-1]
    compression_ratio = cfg["training"]["compression_ratio"]
    pooling = cfg["models"].get("pooling", "mean")
    encoder_window_size = cfg["models"].get("encoder_window_size", 1024)
    adapter_type = cfg["models"].get("adapter_type", "mlp")
    boundary_overlap = cfg["models"].get("boundary_overlap", 0)
    mask = cfg["models"].get("encoder_mask_type", cfg["models"].get("mask", "causal"))

    name = f"{encoder_short}-{decoder_short}-cs{compression_ratio}-w{encoder_window_size}-{pooling}-{adapter_type}"
    if boundary_overlap:
        name = f"{name}-ov{boundary_overlap}"
    return f"{name}-{mask}"


def load_config_to_argv(config_path: str, stage: int, output_dir: str | None = None) -> list[str]:
    """Load YAML config and convert to command line arguments.

    Config structure:
    - models: encoder, decoder, pooling
    - training: common training settings
    - data: common data settings (num_workers, etc.)
    - distributed: global distributed config (fallback)
    - output: output directory
    - logging: wandb settings
    - stages: per-stage overrides (dataset, distributed, lr, etc.)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    t = cfg["training"]
    s = cfg["stages"][stage]
    d = cfg.get("data", {})

    # Stage-specific (required)
    dataset = s["dataset"]
    dist = s["distributed"]

    # Generate experiment name (auto-generate if "auto" or empty)
    experiment = cfg.get("experiment", "auto")
    if not experiment or experiment == "auto":
        experiment = generate_experiment_name(cfg)

    # Output directory: CLI arg > config > default
    if output_dir is None:
        output_dir = cfg.get("output", {}).get("dir", "./checkpoints")

    # Compute resume_from_checkpoint (stage N loads stage N-1's HF checkpoint)
    if stage > 0:
        resume = f'{output_dir}/{experiment}/stage{stage-1}-hf'
    else:
        resume = None

    # Stage-specific resume_from_checkpoint overrides auto-computed one
    # This allows skipping stages or loading from a different checkpoint
    if "resume_from_checkpoint" in s:
        resume = s["resume_from_checkpoint"]

    args = [
        # Models
        "--embed_model_name", cfg["models"]["encoder"],
        "--decoder_name", cfg["models"]["decoder"],
        "--pooling", cfg["models"].get("pooling", "mean"),
        "--encoder_window_size", str(cfg["models"].get("encoder_window_size", 1024)),
        "--encoder_mask_type", cfg["models"].get("mask", "causal"),
        "--decoder_attn_implementation", cfg["models"]["decoder_attn_implementation"],
        "--embed_attn_implementation", cfg["models"]["embed_attn_implementation"],
        "--adapter_type", cfg["models"].get("adapter_type", "mlp"),
        "--num_adapter_layers", str(cfg["models"].get("num_adapter_layers", 1)),
        "--use_fused_ce", str(cfg["models"].get("use_fused_ce", False)).lower(),
        "--random_init_decoder", str(cfg["models"].get("random_init_decoder", False)).lower(),
        "--random_init_encoder", str(cfg["models"].get("random_init_encoder", False)).lower(),

        # Stage-specific training
        "--train_decoder", str(s["train_decoder"]).lower(),
        "--train_encoder", str(s["train_encoder"]).lower(),
        "--adapter_lr", str(s["adapter_lr"]),
        "--encoder_lr", str(s["encoder_lr"]),
        "--decoder_lr", str(s["decoder_lr"]),
        "--min_lr", str(s["min_lr"]),
        "--decoder_weight_decay", str(s["decoder_weight_decay"]),
        "--encoder_weight_decay", str(s["encoder_weight_decay"]),
        "--adapter_weight_decay", str(s["adapter_weight_decay"]),
        "--encoder_gradient_checkpointing", str(s["encoder_gradient_checkpointing"]).lower(),

        # Common training (from training section)
        "--compression_ratio", str(t["compression_ratio"]),
        "--train_batch_size", str(s["train_batch_size"]),
        "--max_encode_batch_size", str(t["max_encode_batch_size"]),
        "--gradient_accumulation_steps", str(t["gradient_accumulation_steps"]),
        "--num_epochs", str(t["num_epochs"]),
        "--max_steps", str(t.get("max_steps", -1)),
        "--warmup_steps", str(s.get("warmup_steps", t["warmup_steps"])),
        "--max_grad_norm", str(t["max_grad_norm"]),
        "--save_steps", str(t["save_steps"]),
        "--delete_old_checkpoints", str(t.get("delete_old_checkpoints", True)).lower(),
        "--seed", str(t["seed"]),
        "--auto_resume", str(t["auto_resume"]).lower(),
        "--optimizer_type", t["optimizer"],
        "--scheduler_type", t["scheduler"],
        "--adam_epsilon", str(t["adam_epsilon"]),
        "--decoder_betas", t["decoder_betas"],
        "--encoder_betas", t["encoder_betas"],
        "--adapter_betas", t["adapter_betas"],
        "--decoder_gradient_checkpointing", str(t["decoder_gradient_checkpointing"]).lower(),
        "--use_liger_kernel", str(t["use_liger_kernel"]).lower(),

        # SLURM preemption checkpoint (queries SLURM for remaining time)
        "--preempt_save_minutes", str(t.get("preempt_save_minutes", 0)),

        # Data (stage-specific dataset, rest from data section with defaults)
        "--dataset_name", dataset,
        "--dataloader_num_workers", str(d.get("num_workers", 4)),
        "--use_packing", str(d.get("use_packing", True)).lower(),
        "--packed_attention_backend", d.get("packed_attention_backend", "flash"),
        "--use_memory_wrapping", str(d.get("use_memory_wrapping", True)).lower(),
        "--train_wrap_tokens", str(d.get("train_wrap_tokens", True)).lower(),

        # Output
        "--output_dir", output_dir,
        "--experiment", experiment,
        "--stage", str(stage),

        # Logging
        "--wandb_project", cfg["logging"]["wandb_project"],
        "--log_interval", str(cfg["logging"].get("log_interval", 1)),
        "--log_token_counts", str(cfg["logging"].get("log_token_counts", False)).lower(),
    ]

    # Optional wandb_name (if not specified, uses auto-generated experiment name)
    if cfg["logging"].get("wandb_name"):
        args.extend(["--wandb_name", cfg["logging"]["wandb_name"]])

    # Optional skip_step_ranges (for skipping problematic data points) - stage-specific only
    if s.get("skip_step_ranges"):
        args.extend(["--skip_step_ranges", s["skip_step_ranges"]])

    # Optional max_packed_length (for padding to consistent tensor shapes with flex_attention)
    if s.get("max_packed_length"):
        args.extend(["--max_packed_length", str(s["max_packed_length"])])

    # Optional boundary_overlap (0 = no overlap; default 0)
    if cfg["models"].get("boundary_overlap") is not None:
        args.extend(["--boundary_overlap", str(cfg["models"]["boundary_overlap"])])

    args.extend([
        # Distributed (stage-specific, required)
        "--distributed_type", dist["type"],
    ])

    if resume:
        args.extend(["--resume_from_checkpoint", resume])

    return args


@dataclass
class ModelArguments:
    embed_model_name: str = field(
        default="Qwen/Qwen3-Embedding-0.6B",
        metadata={"help": "HF model ID or path for the embedding model used by the code chunker."},
    )
    pooling: str = field(
        default="mean",
        metadata={"help": "Within-chunk pooling. One of: 'mean', 'eos', 'concat'."},
    )
    encoder_mask_type: str = field(
        default="causal",
        metadata={"help": "Encoder attention mask: 'causal' or 'bidirectional'."},
    )
    max_encode_batch_size: int = field(
        default=0,
        metadata={"help": "Mini-batch size for encoder forward passes. 0 = no batching."},
    )
    encoder_window_size: int = field(
        default=1024,
        metadata={"help": "Encoder window size W in tokens. Must be a multiple of compression_ratio. W=compression_ratio encodes each chunk independently; W>compression_ratio lets chunks attend to each other."},
    )
    decoder_name: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "HF model ID or path for the base LLM."},
    )
    decoder_tokenizer_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "HF model ID or path for the LLM tokenizer. Uses same tokenizer for all LLM models."},
    )
    train_decoder: bool = field(
        default=True,
        metadata={"help": "Whether to fine-tune the LLM."},
    )
    train_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to fine-tune the embedding model."},
    )
    decoder_attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "LLM attention implementation: 'sdpa', 'flash_attention_2', 'flex_attention', or 'eager'."},
    )
    embed_attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Embedder attention implementation: 'sdpa', 'flash_attention_2', 'flex_attention', or 'eager'."},
    )

    # LLM partial training (embed_tokens + first N layers)
    train_decoder_embed_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to train the LLM's embed_tokens layer (input embeddings)."},
    )
    train_decoder_num_layers: int = field(
        default=0,
        metadata={"help": "Number of initial transformer layers to train in the LLM (0 = none). E.g., 4 trains layers 0-3."},
    )

    teacher_decoder_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional HF model ID for the teacher LLM. Defaults to decoder_name when unset."},
    )

    # LoRA for LLM
    decoder_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA adapters on the base LLM."},
    )
    decoder_lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank for LLM adapters."},
    )
    decoder_lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (scaling) for LLM adapters."},
    )
    decoder_lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LLM LoRA adapters."},
    )
    decoder_lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module name substrings to target for LLM LoRA injection. If None, uses a sensible default."},
    )

    # LoRA for embedder
    embed_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA adapters on the embedding model used by the code chunker."},
    )
    embed_lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank for embedding model adapters."},
    )
    embed_lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (scaling) for embedding model adapters."},
    )
    embed_lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for embedding model LoRA adapters."},
    )
    embed_lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module name substrings to target for embedding model LoRA injection."},
    )

    # Adapter
    adapter_type: str = field(
        default="mlp",
        metadata={"help": "Adapter shape: 'mlp' | 'mlp_attn' | 'attn_mlp'. 'mlp' is pure projection; the others add transformer layers before/after the MLP."},
    )
    num_adapter_layers: int = field(
        default=1,
        metadata={"help": "Number of transformer layers when adapter_type != 'mlp'."},
    )
    use_fused_ce: bool = field(
        default=False,
        metadata={"help": "Use LigerFusedLinearCrossEntropyLoss in LCLM.forward. Avoids materializing the [packed_seq, vocab] fp32 logits tensor. Required when HF path OOMs (bs≥4 at packed seq=16384) but ~70% slower than HF CE at small bs — leave off by default."},
    )
    boundary_overlap: int = field(
        default=0,
        metadata={"help": "Context tokens added before/after each encoder window for boundary attention. 0 = no overlap."},
    )

    # Random initialization (pretraining from scratch)
    random_init_decoder: bool = field(
        default=False,
        metadata={"help": "Initialize LLM with random weights (for pretraining from scratch). Uses model config but no pretrained weights."},
    )
    random_init_encoder: bool = field(
        default=False,
        metadata={"help": "Initialize embedder with random weights (for pretraining from scratch). Uses model config but no pretrained weights."},
    )


@dataclass
class TrainingConfig:
    """Training configuration"""
    use_liger_kernel: bool = field(
        default=True,
        metadata={"help": "Whether to use Liger kernel."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps."},
    )
    decoder_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing for the LLM."},
    )
    encoder_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing for the embedder."},
    )
    scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Default scheduler type."},
    )
    optimizer_type: str = field(
        default="adamw",
        metadata={"help": "Optimizer type, one of: adamw, adafactor, sgd"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for Adam optimizer."},
    )
    # Different learning rates for different components
    decoder_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for LLM model. If None, uses main lr."},
    )
    encoder_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for embedder model. If None, uses main lr."},
    )
    adapter_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for adapter/projection layers. If None, uses main lr."},
    )
    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate for all parameter groups."},
    )
    # Different beta values for different components (for AdamW) - can be provided as "beta1,beta2" string
    decoder_betas: Optional[str] = field(
        default=None,
        metadata={"help": "Beta values for LLM model AdamW optimizer as 'beta1,beta2'. If None, uses (0.9, 0.995)."},
    )
    encoder_betas: Optional[str] = field(
        default=None,
        metadata={"help": "Beta values for embedder model AdamW optimizer as 'beta1,beta2'. If None, uses (0.9, 0.995)."},
    )
    adapter_betas: Optional[str] = field(
        default=None,
        metadata={"help": "Beta values for adapter AdamW optimizer as 'beta1,beta2'. If None, uses (0.9, 0.995)."},
    )

    # Different weight decay for different components
    decoder_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Weight decay for LLM model. If None, uses main weight_decay."},
    )
    encoder_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Weight decay for embedder model. If None, uses main weight_decay."},
    )
    adapter_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Weight decay for adapter layers. If None, uses main weight_decay."},
    )
    num_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs."},
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Maximum number of training steps. -1 means no limit (train for full num_epochs)."},
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Warmup steps for LR scheduler."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping norm."},
    )
    save_steps: int = field(
        default=10000,
        metadata={"help": "Save checkpoint every N steps."},
    )
    delete_old_checkpoints: bool = field(
        default=True,
        metadata={"help": "Delete old checkpoints, keeping only the latest one. If False, keep all checkpoints."},
    )
    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "Directory to save checkpoints and logs."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Whether to automatically resume training from the latest checkpoint."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )
    skip_steps: int = field(
        default=0,
        metadata={"help": "Number of training steps to skip (for resuming from HF checkpoint at a specific step). Dataloader will skip this many batches."},
    )
    skip_step_ranges: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated step ranges to skip (e.g., '6500-6510,7000-7005'). These steps will advance the LR scheduler but skip forward/backward pass."},
    )
    load_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether to load the optimizer from the checkpoint."},
    )
    load_scheduler: bool = field(
        default=True,
        metadata={"help": "Whether to load the scheduler from the checkpoint."},
    )
    model_only: bool = field(
        default=False,
        metadata={"help": "Whether to load only the model from the checkpoint."},
    )
    wandb_project: str = field(
        default="code-llava",
        metadata={"help": "WandB project name."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "WandB run name."},
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Print training progress every N gradient steps (1 = every step)."},
    )
    log_token_counts: bool = field(
        default=False,
        metadata={"help": "Log token counts (before/after compression) and throughput per gradient step."},
    )
    experiment: str = field(
        default="default",
        metadata={"help": "Experiment name for grouping checkpoints. Checkpoints saved to output_dir/experiment/"},
    )
    stage: int = field(
        default=0,
        metadata={"help": "Training stage (0=adapter, 1=encoder, 2=decoder, 3=final). Used in checkpoint naming."},
    )
    distributed_type: str = field(
        default="fsdp",
        metadata={"help": "Whether to run evaluation only."},
    )
    # SLURM preemption checkpoint: queries SLURM for remaining job time
    preempt_save_minutes: int = field(
        default=0,
        metadata={"help": "Save checkpoint this many minutes before SLURM job ends. 0 disables. Queries SLURM via squeue."},
    )
    # Chunk size 
    compression_ratio: int = field(
        default=128,
        metadata={"help": "Number of code chunks to extract per example."},
    )
    use_memory_wrapping: bool = field(
        default=True,
        metadata={"help": "Wrap code region with <|memory_start|> and <|memory_end|> tokens. If False, use only repeated <|memory|> tokens."},
    )
    train_wrap_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to train the code tokens."},
    )

@dataclass
class DataArguments:
    """Data configuration, separate from TrainingConfig."""
    # Either a single CSV file path or a list of dataset specs (as JSON string or @file)
    dataset_name: str = field(
        default="dataset",
        metadata={"help": "Path to a CSV dataset or base dataset name."},
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Number of packed sequences per batch (increase for better GPU utilization)."},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data loading."},
    )
    max_memory_length: int = field(
        default=8192,
        metadata={"help": "Maximum tokenized length for code embeddings."},
    )

    # Packing (uses dynamic packing with StatefulDataLoader for auto-resume)
    use_packing: bool = field(
        default=True,
        metadata={"help": "Use sequence packing for training (2-3x throughput). Requires preprocessing with data/preprocess_for_dynamic_packing.py"},
    )
    packed_attention_backend: str = field(
        default="flash",
        metadata={"help": "Backend for packed attention: 'flash' (FlashAttention varlen) or 'flex' (PyTorch flex_attention)"},
    )
    max_packed_length: Optional[int] = field(
        default=None,
        metadata={"help": "If provided, pad all packed batches to this length for consistent tensor shapes (avoids torch.compile recompilation with flex_attention). Should match the max_packed_length used during preprocessing."},
    )

    # Memory masking arguments (optional data augmentation)
    memory_mask_prompts: bool = field(
        default=False,
        metadata={"help": "Apply memory masking to prompt messages before tokenization."},
    )
    memory_mask_ratio_min: float = field(
        default=0.8,
        metadata={"help": "Lower bound (0-1) of memory masking ratio."},
    )
    memory_mask_ratio_max: float = field(
        default=1.0,
        metadata={"help": "Upper bound (0-1) of memory masking ratio."},
    )

def main():
    """Main entry point for training"""
    # Filter out --local_rank argument added by DeepSpeed/torch.distributed
    # HfArgumentParser doesn't recognize it, but it's handled by the distributed backend
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--local_rank')]

    # Check for --config, --stage, --output_dir to load from YAML
    # This must happen before HfArgumentParser runs
    config_path = None
    stage = None
    output_dir = None
    new_argv = [sys.argv[0]]  # Keep the script name
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--stage" and i + 1 < len(sys.argv):
            stage = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output_dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        else:
            new_argv.append(sys.argv[i])
            i += 1

    # If config is provided, load it and prepend args
    if config_path is not None:
        if stage is None:
            raise ValueError("--stage is required when using --config")
        config_args = load_config_to_argv(config_path, stage, output_dir)
        # Config args come first, then any CLI overrides
        sys.argv = new_argv[:1] + config_args + new_argv[1:]
        print(f"Loaded config from {config_path} (stage {stage})")
    else:
        sys.argv = new_argv

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create trainer
    trainer = LCLMTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
