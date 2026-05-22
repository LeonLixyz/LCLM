#!/usr/bin/env python3
"""
Load converted PyTorch checkpoint (from zero_to_fp32.py) and map to LCLM model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dataclasses import dataclass
import os
import json
import random
import numpy as np
from safetensors.torch import save_file as save_safetensors
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel

# Import local modules
from latent_context.model import LCLM
from latent_context.processor import LCLMProcessor

# Import checking functions from convert_deepspeed_to_hf
from utils.convert_deepspeed_to_hf import (
    check_embedding_changes_quick,
    check_parameter_differences,
    set_seed,
    _wrap_with_lora,
)


@dataclass
class LoadArgs:
    pytorch_checkpoint_dir: str  # Directory containing pytorch_model*.bin files
    output_dir: str | None = None  # Optional: save as HuggingFace format
    embed_model: str | None = None
    decoder: str | None = None
    compression_ratio: int = 32
    max_memory_length: int = 8192
    wrap_code: bool = False
    decoder_lora_rank: int | None = None
    decoder_lora_alpha: float | None = None
    decoder_lora_dropout: float | None = None
    decoder_lora_target_modules: str | None = None
    embed_lora_rank: int | None = None
    embed_lora_alpha: float | None = None
    embed_lora_dropout: float | None = None
    embed_lora_target_modules: str | None = None
    adapter_type: str = "mlp"
    num_adapter_layers: int = 1
    pooling: str = "mean"
    encoder_mask_type: str = "bidirectional"
    encoder_window_size: int = 1024
    boundary_overlap: int = 0


def load_sharded_checkpoint(checkpoint_dir):
    """
    Load sharded PyTorch checkpoint files (pytorch_model*.bin)
    
    Args:
        checkpoint_dir: Directory containing pytorch_model*.bin and pytorch_model.bin.index.json
    
    Returns:
        Complete state dict
    """
    print(f"Loading PyTorch checkpoint from: {checkpoint_dir}")
    
    # Check for index file
    index_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
    
    if os.path.exists(index_file):
        # Load sharded checkpoint
        print("Found sharded checkpoint index file")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get('weight_map', {})
        if not weight_map:
            raise ValueError("Index file does not contain weight_map")
        
        # Get unique shard files
        shard_files = sorted(set(weight_map.values()))
        print(f"Loading {len(shard_files)} shard files...")
        
        # Load all shards
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard_file)
            print(f"  Loading {shard_file}...")
            shard_state = torch.load(shard_path, map_location='cpu', weights_only=False)
            state_dict.update(shard_state)
            del shard_state  # Free memory
        
        print(f"✓ Loaded {len(state_dict)} parameters from {len(shard_files)} shards")
        
    else:
        # Try single file
        single_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(single_file):
            print("Loading single checkpoint file")
            state_dict = torch.load(single_file, map_location='cpu', weights_only=False)
            print(f"✓ Loaded {len(state_dict)} parameters")
        else:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Remove 'module.' prefix if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned_key = key[7:]
        else:
            cleaned_key = key
        cleaned_state_dict[cleaned_key] = value
    
    return cleaned_state_dict


def load_and_map_checkpoint(
    pytorch_checkpoint_dir,
    output_dir=None,
    encoder_name=None,
    decoder_name=None,
    compression_ratio=None,
    max_memory_length=None,
    wrap_code=False,
    decoder_lora_rank=None,
    decoder_lora_alpha=None,
    decoder_lora_dropout=None,
    decoder_lora_target_modules=None,
    embed_lora_rank=None,
    embed_lora_alpha=None,
    embed_lora_dropout=None,
    embed_lora_target_modules=None,
    adapter_type="mlp",
    num_adapter_layers=1,    pooling="summary_mean",
    encoder_mask_type="bidirectional",
    encoder_window_size=1024,
    boundary_overlap=0,
):
    """
    Load PyTorch checkpoint and map to LCLM model
    
    Args:
        pytorch_checkpoint_dir: Directory containing pytorch_model*.bin files
        output_dir: Optional output directory for HuggingFace format
        encoder_name: Name of the embedding model
        decoder_name: Name of the LLM model
        compression_ratio: Chunk size for code processing
        max_memory_length: Maximum code length
        wrap_code: Whether to add <|memory_start|>/<|memory_end|> special tokens
    """
    # Set seed for reproducible initialization
    set_seed(42)
    
    print(f"Loading PyTorch checkpoint: {pytorch_checkpoint_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    
    # Model configuration - use provided values or defaults
    if encoder_name is None:
        encoder_name = "Qwen/Qwen3-Embedding-8B"
    if decoder_name is None:
        decoder_name = "Qwen/Qwen3-32B"
    if compression_ratio is None:
        compression_ratio = 32
    if max_memory_length is None:
        max_memory_length = 4096
    
    print(f"\nModel configuration:")
    print(f"  Embed model: {encoder_name}")
    print(f"  LLM model: {decoder_name}")
    print(f"  Chunk size: {compression_ratio}")
    print(f"  Max code length: {max_memory_length}")
    print(f"  Wrap code: {wrap_code}")

    # Step 1: Load PyTorch checkpoint
    print("\nStep 1: Loading PyTorch checkpoint...")
    state_dict = load_sharded_checkpoint(pytorch_checkpoint_dir)
    print(f"Loaded {len(state_dict)} parameters from checkpoint")
    
    # Step 2: Create model architecture and load base models
    print("\nStep 2: Creating model architecture...")
    
    ### TOKENIZERS ###
    # Always use the same tokenizer regardless of LLM model
    LLM_TOKENIZER_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    decoder_tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_NAME)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    embed_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    ### LLM ###
    print("Loading LLM model in fp32")
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        torch_dtype=torch.float32,
    )

    ### EMBEDDER ###
    print("Loading embedder model in fp32")
    embed_model = AutoModel.from_pretrained(
        encoder_name,
        torch_dtype=torch.float32,
    )
    
    ### ADD NEW TOKENS ###
    if wrap_code:
        new_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
    else:
        new_tokens = ['<|memory|>']
    num_added_tokens = decoder_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    # Resize token embeddings to accommodate new tokens
    set_seed(42)
    if num_added_tokens > 0:
        old_vocab_size = decoder.get_input_embeddings().weight.size(0)
        decoder.resize_token_embeddings(len(decoder_tokenizer))
        print(f"Resized token embeddings from {old_vocab_size} to {len(decoder_tokenizer)}")
    
    ### CREATE PROCESSOR ###
    processor = LCLMProcessor(
        decoder_tokenizer=decoder_tokenizer,
        embed_tokenizer=embed_tokenizer,
        compression_ratio=compression_ratio,
        max_memory_length=max_memory_length,
    )
    
    ### CREATE CODELLAVA MODEL ###
    model = LCLM(
        decoder=decoder,
        decoder_tokenizer=decoder_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
        processor=processor,
        compression_ratio=compression_ratio,
        max_memory_length=max_memory_length,
        train_decoder=False,
        train_encoder=False,
        adapter_type=adapter_type,
        num_adapter_layers=num_adapter_layers,
        pooling=pooling,
        encoder_mask_type=encoder_mask_type,
        encoder_window_size=encoder_window_size,
        boundary_overlap=boundary_overlap,
    )

    # Snapshot pretrained weights for later comparison
    base_pretrained_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _parse_modules(modules_str):
        if not modules_str:
            return None
        modules = [m.strip() for m in modules_str.split(',') if m.strip()]
        return modules or None

    decoder_target_modules = _parse_modules(decoder_lora_target_modules)
    embed_target_modules = _parse_modules(embed_lora_target_modules)

    if decoder_target_modules is None and decoder_lora_rank is not None and decoder_lora_alpha is not None:
        decoder_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    if decoder_lora_rank is not None and decoder_lora_alpha is not None and decoder_target_modules:
        print("Wrapping LLM model with provided LoRA configuration")
        model.decoder = _wrap_with_lora(
            model.decoder,
            task_type="CAUSAL_LM",
            rank=decoder_lora_rank,
            alpha=decoder_lora_alpha,
            dropout=decoder_lora_dropout or 0.0,
            target_modules=decoder_target_modules,
            modules_to_save=["embed_tokens", "lm_head"],
        )

    if embed_lora_rank is not None and embed_lora_alpha is not None and embed_target_modules:
        print("Wrapping embedder model with provided LoRA configuration")
        model.encoder.embed_model = _wrap_with_lora(
            model.encoder.embed_model,
            task_type="FEATURE_EXTRACTION",
            rank=embed_lora_rank,
            alpha=embed_lora_alpha,
            dropout=embed_lora_dropout or 0.0,
            target_modules=embed_target_modules,
            modules_to_save=["embed_tokens"],
        )

    # Step 3: Load the checkpoint weights
    print("\nStep 3: Loading checkpoint weights into model...")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys during loading: {len(missing_keys)}")
        for key in missing_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Unexpected keys during loading: {len(unexpected_keys)}")
        for key in unexpected_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")

    print("✓ Weights loaded successfully!")

    # Merge LoRA adapters if present
    if isinstance(model.decoder, PeftModel):
        print("Merging LoRA adapters into LLM base model...")
        merged_llm = model.decoder.merge_and_unload()
        model.decoder = merged_llm.to(dtype=torch.float32)

    if isinstance(model.encoder.embed_model, PeftModel):
        print("Merging LoRA adapters into embedder base model...")
        merged_embed = model.encoder.embed_model.merge_and_unload()
        model.encoder.embed_model = merged_embed.to(dtype=torch.float32)

    # Step 4: Run diagnostic checks
    print("\nStep 4: Running diagnostic checks...")

    final_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    check_embedding_changes_quick(base_pretrained_state_dict, final_state_dict, decoder_tokenizer, wrap_code=wrap_code)

    check_parameter_differences(
        pretrained_state_dict=base_pretrained_state_dict,
        loaded_state_dict=final_state_dict,
        model_name="LCLM",
        tokenizer=decoder_tokenizer,
    )

    # Step 5: Optionally save in HuggingFace format
    if output_dir:
        print("\nStep 5: Saving in HuggingFace format...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create structured output directories
        decoder_dir = os.path.join(output_dir, "decoder")
        encoder_dir = os.path.join(output_dir, "encoder")
        projectors_dir = os.path.join(output_dir, "adapter")
        
        os.makedirs(decoder_dir, exist_ok=True)
        os.makedirs(encoder_dir, exist_ok=True)
        os.makedirs(projectors_dir, exist_ok=True)
        
        # Save LLM model and tokenizer
        print("Saving LLM model and tokenizer...")
        model.decoder.save_pretrained(
            decoder_dir,
            safe_serialization=True,
        )
        decoder_tokenizer.save_pretrained(decoder_dir)
        
        # Save embedder model and tokenizer
        print("Saving embedder model and tokenizer...")
        model.encoder.embed_model.save_pretrained(
            encoder_dir,
            safe_serialization=True,
        )
        embed_tokenizer.save_pretrained(encoder_dir)
        
        # Save code adapter
        print("Saving code adapter...")
        adapter_state = {k: v.detach().cpu() for k, v in model.adapter.state_dict().items()}
        save_safetensors(adapter_state, os.path.join(projectors_dir, "adapter.safetensors"))
        
        # Save model configuration
        print("Saving model configuration...")
        from utils.checkpointing import save_model_config
        save_model_config(
            output_dir,
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            compression_ratio=compression_ratio,
            encoder_window_size=encoder_window_size,
            pooling=pooling,
            encoder_mask_type=encoder_mask_type,
            boundary_overlap=boundary_overlap,
            adapter_type=adapter_type,
            num_adapter_layers=num_adapter_layers,
        )
        # Save processor config
        processor_config = {
            "compression_ratio": compression_ratio,
            "max_memory_length": max_memory_length,
            "use_memory_wrapping": wrap_code,
            "memory_placeholder": "<|memory|>",
            "memory_start": "<|memory_start|>",
            "memory_end": "<|memory_end|>"
        }
        
        with open(os.path.join(output_dir, "processor_config.json"), 'w') as f:
            json.dump(processor_config, f, indent=2)
        
        print(f"\n✓ Conversion complete! HuggingFace format saved to: {output_dir}")
    else:
        print("\n✓ Model loaded and checked successfully!")
        print("Note: No output directory specified, so not saving to disk.")
    
    return model


def main():
    parser = HfArgumentParser(LoadArgs)
    args = parser.parse_args_into_dataclasses()[0]
    
    load_and_map_checkpoint(
        pytorch_checkpoint_dir=args.pytorch_checkpoint_dir,
        output_dir=args.output_dir,
        encoder_name=args.embed_model,
        decoder_name=args.decoder,
        compression_ratio=args.compression_ratio,
        max_memory_length=args.max_memory_length,
        wrap_code=args.wrap_code,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
        decoder_lora_target_modules=args.decoder_lora_target_modules,
        embed_lora_rank=args.embed_lora_rank,
        embed_lora_alpha=args.embed_lora_alpha,
        embed_lora_dropout=args.embed_lora_dropout,
        embed_lora_target_modules=args.embed_lora_target_modules,
        adapter_type=args.adapter_type,
        num_adapter_layers=args.num_adapter_layers,        pooling=args.pooling,
        encoder_mask_type=args.encoder_mask_type,
        encoder_window_size=args.encoder_window_size,
        boundary_overlap=args.boundary_overlap,
    )


if __name__ == "__main__":
    main()

