#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to HuggingFace format for easy loading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dataclasses import dataclass
import os
import json
import random
import numpy as np
from typing import List
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model, PeftModel
import glob

# Import local modules
from latent_context.model import LCLM
from latent_context.processor import LCLMProcessor


@dataclass
class ConvertArgs:
    deepspeed_checkpoint: str
    output_dir: str
    embed_model: str | None = None
    decoder: str | None = None
    compression_ratio: int = 32
    max_memory_length: int = 8192
    wrap_code: bool = False
    pooling: str = "eos"
    encoder_mask_type: str = "causal"
    encoder_window_size: int = 256
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
    boundary_overlap: int = 0


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wrap_with_lora(
    model,
    task_type: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
    modules_to_save: list[str] | None = None,
):
    """Wrap model with LoRA using explicitly provided hyperparameters."""

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
        modules_to_save=modules_to_save,
    )

    wrapped_model = get_peft_model(model, lora_cfg)
    return wrapped_model


def check_new_tokens_only(pretrained_state_dict, loaded_state_dict, tokenizer, new_tokens):
    """
    Check if only the new tokens have changed in the embedding layers
    
    Args:
        pretrained_state_dict: State dict from pretrained model
        loaded_state_dict: State dict loaded from checkpoint
        tokenizer: The tokenizer with new tokens added
        new_tokens: List of new tokens that were added
    """
    print(f"\n=== Checking if new tokens changed ===")
    print(f"New tokens added: {new_tokens}")
    print(f"Total tokenizer vocabulary size: {len(tokenizer)}")
    
    # Calculate expected indices for new tokens (should be at the end)
    num_new_tokens = len(new_tokens)
    original_vocab_size = len(tokenizer) - num_new_tokens
    new_token_start_idx = original_vocab_size
    
    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"New tokens should be at indices: {new_token_start_idx} to {len(tokenizer)-1}")
    
    # Get the actual token IDs for the new tokens
    actual_new_token_ids = []
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        actual_new_token_ids.append(token_id)
        print(f"  '{token}' -> token_id: {token_id}")
    
    # Validate that the new tokens occupy contiguous indices at the end
    expected_new_token_ids = list(range(new_token_start_idx, new_token_start_idx + num_new_tokens))
    if actual_new_token_ids != expected_new_token_ids:
        print(f"\n✗ WARNING: New token IDs are not at expected contiguous positions!")
        print(f"  Expected IDs: {expected_new_token_ids}")
        print(f"  Actual IDs:   {actual_new_token_ids}")
    else:
        print(f"\n✓ New tokens are at the expected contiguous positions: {expected_new_token_ids}")
    
    # Verify round-trip token strings for those IDs
    roundtrip_tokens = [tokenizer.convert_ids_to_tokens(i) for i in expected_new_token_ids]
    for idx, (expected_tok, rt_tok) in enumerate(zip(new_tokens, roundtrip_tokens)):
        check_id = expected_new_token_ids[idx]
        if rt_tok != expected_tok:
            print(f"  ✗ Mismatch at id {check_id}: expected '{expected_tok}', got '{rt_tok}'")
        else:
            print(f"  ✓ Verified token '{expected_tok}' at id {check_id}")
    
    layers_to_check = ['decoder.model.embed_tokens.weight', 'decoder.lm_head.weight']
    
    for layer_name in layers_to_check:
        # Find the actual keys containing these layer names
        embed_keys = [k for k in loaded_state_dict.keys() if layer_name in k]
        
        for key in embed_keys:
            if key not in pretrained_state_dict:
                print(f"\nSkipping {key} (not in pretrained model)")
                continue
                
            print(f"\nChecking layer: {key}")
            
            pretrained_param = pretrained_state_dict[key].cpu().float()
            loaded_param = loaded_state_dict[key].cpu().float()
            
            # Check shapes
            print(f"  Pretrained shape: {pretrained_param.shape}")
            print(f"  Loaded shape: {loaded_param.shape}")
            
            # Check if original tokens are unchanged
            original_tokens_unchanged = torch.equal(
                pretrained_param[:original_vocab_size], 
                loaded_param[:original_vocab_size]
            )
            
            if original_tokens_unchanged:
                print(f"  ✓ Original tokens (0 to {original_vocab_size-1}) are UNCHANGED")
            else:
                # Find which original tokens changed
                diff = torch.abs(pretrained_param[:original_vocab_size] - loaded_param[:original_vocab_size])
                changed_mask = torch.any(diff > 0, dim=1) if len(diff.shape) > 1 else diff > 0
                changed_indices = torch.nonzero(changed_mask).flatten()
                
                print(f"  ✗ WARNING: {len(changed_indices)} original tokens have changed!")
                if len(changed_indices) <= 10:
                    for idx in changed_indices:
                        idx_val = idx.item()
                        token_str = tokenizer.decode([idx_val])
                        max_diff = torch.max(diff[idx_val]).item() if len(diff.shape) > 1 else diff[idx_val].item()
                        print(f"    - Token {idx_val}: '{token_str}' (max diff: {max_diff:.6e})")
                else:
                    print(f"    Showing first 10 of {len(changed_indices)} changed tokens:")
                    for idx in changed_indices[:10]:
                        idx_val = idx.item()
                        token_str = tokenizer.decode([idx_val])
                        max_diff = torch.max(diff[idx_val]).item() if len(diff.shape) > 1 else diff[idx_val].item()
                        print(f"    - Token {idx_val}: '{token_str}' (max diff: {max_diff:.6e})")
            
            # Check new token region
            if loaded_param.shape[0] >= len(tokenizer):
                print(f"\n  New token region ({new_token_start_idx} to {len(tokenizer)-1}):")
                
                # Check if new tokens are different from initialization (same approach as original tokens)
                new_tokens_unchanged = torch.equal(
                    pretrained_param[new_token_start_idx:len(tokenizer)], 
                    loaded_param[new_token_start_idx:len(tokenizer)]
                )
                
                if new_tokens_unchanged:
                    print(f"  ✗ WARNING: New tokens ({new_token_start_idx} to {len(tokenizer)-1}) are UNCHANGED from initialization")
                else:
                    # Find which new tokens changed
                    diff = torch.abs(pretrained_param[new_token_start_idx:len(tokenizer)] - loaded_param[new_token_start_idx:len(tokenizer)])
                    changed_mask = torch.any(diff > 0, dim=1) if len(diff.shape) > 1 else diff > 0
                    changed_indices = torch.nonzero(changed_mask).flatten()
                    
                    print(f"  ✓ {len(changed_indices)} new tokens have changed from initialization")
                    for idx in changed_indices:
                        abs_idx = new_token_start_idx + idx.item()
                        token_str = tokenizer.decode([abs_idx]) if abs_idx < len(tokenizer) else f"<token_{abs_idx}>"
                        max_diff = torch.max(diff[idx]).item() if len(diff.shape) > 1 else diff[idx].item()
                        print(f"    - Token {abs_idx}: '{token_str}' (max diff: {max_diff:.6e})")
    
    print("\n=== End of new token check ===")


def check_embedding_changes_quick(pretrained_state_dict, loaded_state_dict, decoder_tokenizer, wrap_code=False):
    """
    Quick check focusing only on the new token additions
    """
    if wrap_code:
        new_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
    else:
        new_tokens = ['<|memory|>']
    check_new_tokens_only(pretrained_state_dict, loaded_state_dict, decoder_tokenizer, new_tokens)


def check_parameter_differences(pretrained_state_dict, loaded_state_dict, model_name="model", tokenizer=None):
    """
    Check and report differences between pretrained and loaded parameter names
    
    Args:
        pretrained_state_dict: State dict from pretrained model
        loaded_state_dict: State dict loaded from checkpoint
        model_name: Name of the model for logging purposes
        tokenizer: Tokenizer for decoding token IDs in embedding layers (optional)
    """
    pretrained_keys = set(pretrained_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    
    # Find differences
    only_in_pretrained = pretrained_keys - loaded_keys
    only_in_loaded = loaded_keys - pretrained_keys
    common_keys = pretrained_keys & loaded_keys
    
    print(f"\n=== Parameter Analysis for {model_name} ===")
    print(f"Total pretrained parameters: {len(pretrained_keys)}")
    print(f"Total loaded parameters: {len(loaded_keys)}")
    print(f"Common parameters: {len(common_keys)}")
    print(f"Only in pretrained: {len(only_in_pretrained)}")
    print(f"Only in loaded: {len(only_in_loaded)}")
    
    if only_in_pretrained:
        print(f"\nParameters only in pretrained {model_name}:")
        for key in sorted(only_in_pretrained):
            print(f"  - {key}")
    
    if only_in_loaded:
        print(f"\nParameters only in loaded checkpoint:")
        for key in sorted(only_in_loaded):
            print(f"  - {key}")
    
    # Check for shape mismatches in common parameters
    shape_mismatches = []
    value_mismatches = []
    identical_parameters = []
    
    for key in common_keys:
        pretrained_param = pretrained_state_dict[key]
        loaded_param = loaded_state_dict[key]
        
        # Check shape mismatch
        if pretrained_param.shape != loaded_param.shape:
            shape_mismatches.append((key, pretrained_param.shape, loaded_param.shape))
        else:
            # Move to same device and dtype for comparison
            pretrained_param = pretrained_param.to(device='cpu', dtype=torch.float32)
            loaded_param = loaded_param.to(device='cpu', dtype=torch.float32)
            
            # Check if tensors are exactly equal
            if torch.equal(pretrained_param, loaded_param):
                identical_parameters.append(key)
            else:
                # Calculate difference statistics
                diff = torch.abs(pretrained_param - loaded_param)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                value_mismatches.append((key, max_diff, mean_diff))
    
    if shape_mismatches:
        print(f"\nShape mismatches in common parameters:")
        for key, pretrained_shape, loaded_shape in shape_mismatches:
            print(f"  - {key}: pretrained {pretrained_shape} vs loaded {loaded_shape}")
    
    print(f"Identical parameters: {len(identical_parameters)}, Different parameters: {len(value_mismatches)}")

    print(f"=== End Parameter Analysis for {model_name} ===\n")
    
    if value_mismatches:
        print(f"\nParameters with value differences:")
        for key, max_diff, mean_diff in value_mismatches:
            print(f"  - {key}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if identical_parameters:
        print(f"\nParameters identical to pretrained:")
        for key in identical_parameters:
            print(f"  - {key}")


def load_deepspeed_checkpoint(checkpoint_dir, use_fp32_from_optimizer=True):
    """
    Load DeepSpeed checkpoint and convert to a single state dict.

    Args:
        checkpoint_dir: Path to checkpoint directory containing pytorch_model subfolder
        use_fp32_from_optimizer: If True, use official DeepSpeed fp32 reconstruction from optimizer states

    Returns:
        Merged state dict
    """
    print(f"Loading DeepSpeed checkpoint from: {checkpoint_dir}")

    # Check for DeepSpeed checkpoint directory
    deepspeed_dir = os.path.join(checkpoint_dir, "pytorch_model")
    if not os.path.isdir(deepspeed_dir):
        raise ValueError(f"DeepSpeed checkpoint directory not found: {deepspeed_dir}")

    if use_fp32_from_optimizer:
        print("Using official DeepSpeed fp32 reconstruction from optimizer states...")
        try:
            # Import the official DeepSpeed conversion utility
            zero_to_fp32_path = os.path.join(os.path.dirname(__file__), "zero_to_fp32.py")

            # Import the function directly
            import importlib.util
            spec = importlib.util.spec_from_file_location("zero_to_fp32", zero_to_fp32_path)
            zero_to_fp32_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(zero_to_fp32_module)

            # Use the official function to get fp32 state dict
            state_dict = zero_to_fp32_module.get_fp32_state_dict_from_zero_checkpoint(
                checkpoint_dir=os.path.dirname(deepspeed_dir),  # Parent dir
                tag="pytorch_model",
                exclude_frozen_parameters=False
            )

            print(f"✓ Successfully loaded fp32 state dict with {len(state_dict)} parameters")

            # Remove 'module.' prefix if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_key = key[7:]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value

            return cleaned_state_dict

        except Exception as e:
            print(f"⚠️  Warning: Failed to use fp32 reconstruction: {e}")
            print("Falling back to loading model states directly (may be bf16)...")

    # Fallback: Load model states directly (original method)
    # Find model states file
    model_states_pattern = os.path.join(deepspeed_dir, "*model_states.pt")
    model_states_files = glob.glob(model_states_pattern)

    if not model_states_files:
        raise ValueError(f"No model states file found in: {deepspeed_dir}")

    model_states_file = model_states_files[0]
    print(f"Loading model states from: {model_states_file}")

    # Load the model states
    checkpoint_data = torch.load(model_states_file, map_location='cpu', weights_only=False)

    # DeepSpeed saves the model state dict in different formats depending on ZeRO stage
    # For ZeRO stage 1 and 2, the full model is in 'module' key
    # For ZeRO stage 3, we need to gather shards

    if 'module' in checkpoint_data:
        state_dict = checkpoint_data['module']
        print(f"Loaded state dict with {len(state_dict)} parameters (ZeRO stage 1/2)")
    elif 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
        print(f"Loaded state dict with {len(state_dict)} parameters")
    else:
        # If neither, the checkpoint_data itself might be the state dict
        state_dict = checkpoint_data
        print(f"Loaded state dict with {len(state_dict)} parameters (direct format)")

    # Remove 'module.' prefix if present (from DDP/DeepSpeed wrapping)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned_key = key[7:]  # Remove 'module.' prefix
        else:
            cleaned_key = key
        cleaned_state_dict[cleaned_key] = value

    print(f"Cleaned state dict contains {len(cleaned_state_dict)} parameters")
    print(f"⚠️  Note: Loaded from model states - weights may be in training precision (bf16), not fp32")

    return cleaned_state_dict


def convert_deepspeed_to_hf(
    deepspeed_checkpoint_path,
    output_dir,
    encoder_name=None,
    decoder_name=None,
    compression_ratio=None,
    max_memory_length=None,
    wrap_code=False,
    pooling="eos",
    encoder_mask_type="causal",
    encoder_window_size=256,
    decoder_lora_rank=None,
    decoder_lora_alpha=None,
    decoder_lora_dropout=None,
    decoder_lora_target_modules=None,
    embed_lora_rank=None,
    embed_lora_alpha=None,
    embed_lora_dropout=None,
    embed_lora_target_modules=None,
    adapter_type="mlp",
    num_adapter_layers=1,
    boundary_overlap=0,
):
    """
    Convert DeepSpeed checkpoint to HuggingFace format
    
    Args:
        deepspeed_checkpoint_path: Path to DeepSpeed checkpoint directory
        output_dir: Output directory for HuggingFace format checkpoint
        encoder_name: Name of the embedding model (default: "Qwen/Qwen3-Embedding-8B")
        decoder_name: Name of the LLM model (default: "Qwen/Qwen3-32B")
        compression_ratio: Chunk size for code processing (default: 32)
        max_memory_length: Maximum code length (default: 4096)
        wrap_code: Whether to add <|memory_start|>/<|memory_end|> special tokens
    """
    # Set seed for reproducible initialization
    set_seed(42)
    
    print(f"Converting DeepSpeed checkpoint: {deepspeed_checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.isdir(deepspeed_checkpoint_path):
        raise ValueError(f"DeepSpeed checkpoint path must be a directory: {deepspeed_checkpoint_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Model configuration - use provided values or defaults
    if encoder_name is None:
        encoder_name = "Qwen/Qwen3-Embedding-8B"
    if decoder_name is None:
        decoder_name = "Qwen/Qwen3-32B"
    if compression_ratio is None:
        compression_ratio = 32
    if max_memory_length is None:
        max_memory_length = 4096
    
    print(f"Model configuration:")
    print(f"  Embed model: {encoder_name}")
    print(f"  LLM model: {decoder_name}")
    print(f"  Chunk size: {compression_ratio}")
    print(f"  Max code length: {max_memory_length}")
    print(f"  Wrap code: {wrap_code}")

    # Step 1: Load DeepSpeed checkpoint
    print("Step 1: Loading DeepSpeed checkpoint...")
    print("Note: Using fp32 reconstruction from optimizer states (recommended for ZeRO)")
    merged_state_dict = load_deepspeed_checkpoint(
        deepspeed_checkpoint_path,
        use_fp32_from_optimizer=True  # Set to False to use model states directly
    )
    print(f"Loaded {len(merged_state_dict)} parameters from DeepSpeed checkpoint")
    
    # Step 2: Create model architecture and load base models
    print("Step 2: Creating model architecture...")
    
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
        attn_implementation="flash_attention_2",
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

    # Step 3: Load the merged weights
    print("Step 3: Loading merged weights...")

    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

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

    print("Weights loaded successfully!")

    if isinstance(model.decoder, PeftModel):
        print("Merging LoRA adapters into LLM base model...")
        merged_llm = model.decoder.merge_and_unload()
        # Keep in fp32 for proper precision
        model.decoder = merged_llm.to(dtype=torch.float32)

    if isinstance(model.encoder.embed_model, PeftModel):
        print("Merging LoRA adapters into embedder base model...")
        merged_embed = model.encoder.embed_model.merge_and_unload()
        # Keep in fp32 for proper precision
        model.encoder.embed_model = merged_embed.to(dtype=torch.float32)

    # Step 4: Compare merged model to original pretrained weights for diagnostics
    print("Step 4: Checking parameter differences...")

    final_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    check_embedding_changes_quick(base_pretrained_state_dict, final_state_dict, decoder_tokenizer, wrap_code=wrap_code)

    check_parameter_differences(
        pretrained_state_dict=base_pretrained_state_dict,
        loaded_state_dict=final_state_dict,
        model_name="LCLM",
        tokenizer=decoder_tokenizer,
    )

    # Step 5: Save in HuggingFace format
    print("Step 5: Saving in HuggingFace format...")
    
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
    
    return output_dir


def main():
    parser = HfArgumentParser(ConvertArgs)
    args = parser.parse_args_into_dataclasses()[0]

    convert_deepspeed_to_hf(
        deepspeed_checkpoint_path=args.deepspeed_checkpoint,
        output_dir=args.output_dir,
        encoder_name=args.embed_model,
        decoder_name=args.decoder,
        compression_ratio=args.compression_ratio,
        max_memory_length=args.max_memory_length,
        wrap_code=args.wrap_code,
        pooling=args.pooling,
        encoder_mask_type=args.encoder_mask_type,
        encoder_window_size=args.encoder_window_size,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_lora_dropout=args.decoder_lora_dropout,
        decoder_lora_target_modules=args.decoder_lora_target_modules,
        embed_lora_rank=args.embed_lora_rank,
        embed_lora_alpha=args.embed_lora_alpha,
        embed_lora_dropout=args.embed_lora_dropout,
        embed_lora_target_modules=args.embed_lora_target_modules,
        adapter_type=args.adapter_type,
        num_adapter_layers=args.num_adapter_layers,
        boundary_overlap=args.boundary_overlap,
    )


if __name__ == "__main__":
    main()

