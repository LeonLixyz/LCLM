#!/usr/bin/env python3
"""
Code LLaVA Trainer with Accelerate support
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from liger_kernel.transformers import (
    LigerFusedLinearCrossEntropyLoss,
)
from transformers import Adafactor
from torch.optim import AdamW, SGD
from accelerate import Accelerator
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import os
import json
import random
import hashlib
import socket
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
# Import local modules
from latent_context.model import LCLM
from latent_context.processor import LCLMProcessor
from data.dataset import prepare_datasets
from utils.nan_checks import has_non_finite_loss_and_gradients
from utils.checkpointing import save_checkpoint, find_latest_checkpoint
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModel
from utils.seed import set_seed
from utils.scheduler import CosineWithMinLRScheduler
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedStateDictConfig, ShardedOptimStateDictConfig, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import BackwardPrefetch                                                                                                   
from utils.env import load_env

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(True)

class LCLMTrainer:
    """
    Trainer class for Code LLaVA model using Accelerate
    """
    
    def __init__(
        self,
        model_args,
        data_args, 
        training_args,
        accelerator: Optional[Accelerator] = None
    ):
        """
        Initialize the trainer with arguments
        
        Args:
            model_args: Model configuration arguments
            data_args: Data configuration arguments  
            training_args: Training configuration arguments
            accelerator: Optional accelerator instance
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # Set seed
        set_seed(training_args.seed)
        # Load .env (OPENAI_API_KEY, HUGGINGFACE_HUB_TOKEN, WANDB_API_KEY, etc.)
        load_env()
            
        # Initialize components
        self.model = None
        self.decoder_tokenizer = None
        self.embed_tokenizer = None
        
        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.param_group_names = []

        # Dataloaders
        self.train_dataloader = None
        self._stateful_dataloader = None  # Reference to StatefulDataLoader (if using dynamic packing)

        # Training state
        self.global_step = 0
        self.num_examples_seen = 0

        # Running average tracker for training loss
        self.training_loss_window = []
        self.window_size = 100

        # Token count accumulators (reset after each gradient step)
        self.accum_memory_tokens = 0  # Code tokens before compression
        self.accum_decoder_tokens = 0  # Total LLM sequence tokens
        self.last_step_time = None  # Wallclock time of last gradient step

        # Checkpointing
        self.auto_resume_checkpoint_path = None
        self.should_resume_from_checkpoint = (self.training_args.resume_from_checkpoint is not None)

        # SLURM preemption checkpoint tracking
        self._preemption_checkpoint_saved = False

        # Parse skip step ranges (e.g., "6500-6510,7000-7005")
        self._skip_step_ranges = self._parse_skip_step_ranges(training_args.skip_step_ranges)
    
        # Build model first (without accelerator)
        # Note: We'll check auto_resume AFTER building the model, so for now just build normally
        if self.should_resume_from_checkpoint:
            print("=" * 80)
            print("RESUME FROM CHECKPOINT: Loading model weights from HuggingFace format checkpoint")
            print(f"Checkpoint path: {self.training_args.resume_from_checkpoint}")
            print("This loads: model weights only (decoder/, encoder/, adapter/)")
            print("This does NOT load: optimizer, scheduler, or dataloader state")
            print("Training will start from step 0 with fresh optimizer/scheduler")
            if self.training_args.auto_resume:
                print("Note: Auto-resume is also enabled. If an auto-resume checkpoint exists,")
                print("      it will take priority over this checkpoint.")
            print("=" * 80)
            self._resume_from_checkpoint(self.training_args.resume_from_checkpoint)
        else:
            self.build_model_and_tokenizer()

        # Now setup accelerator with model's FSDP ignored modules
        self.setup_accelerator()

        ### CHECK MULTI-NODE ###
        print(
            f"[rank {self.accelerator.process_index}] "
            f"host={socket.gethostname()} "
            f"local_rank={self.accelerator.local_process_index} "
            f"num_procs={self.accelerator.num_processes} ",
            flush=True,
        )

        if self.accelerator.is_main_process:
            print("=== Accelerator state ===")
            print(self.accelerator.state)
        
        # Set accelerator on the model components separately
        self.model.setup_accelerator(self.accelerator)
        self.model.encoder.setup_accelerator(self.accelerator)

        self.load_datasets()
        self.setup_optimizer_and_scheduler()

        self.prepare_for_training()

        self._install_special_token_embedding_grad_mask()

        # Create output directory
        if self.accelerator.is_main_process:
            os.makedirs(training_args.output_dir, exist_ok=True)
        
    def setup_accelerator(self):
        # Initialize accelerator if not provided

        if self.training_args.distributed_type == "fsdp":

            fsdp_wrap_classes = ["Qwen3DecoderLayer", "Adapter"]

            fsdp_ignored_modules = []
            if self.training_args.train_wrap_tokens and not self.model_args.train_decoder:
                emb_mod = self.model.decoder.get_input_embeddings()
                fsdp_ignored_modules.append(emb_mod)

            print(f"FSDP wrap classes: {fsdp_wrap_classes}")
            print(f"FSDP ignored modules: {fsdp_ignored_modules}")


            fsdp_mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,  
                buffer_dtype=torch.bfloat16,
            )

            fsdp_plugin = FullyShardedDataParallelPlugin(
                transformer_cls_names_to_wrap=fsdp_wrap_classes,
                state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
                optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                ignored_modules=fsdp_ignored_modules,
                mixed_precision_policy=fsdp_mixed_precision,
            )

            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                log_with="wandb",
                project_dir=self.training_args.output_dir,
                fsdp_plugin=fsdp_plugin,
            )
            
            if fsdp_ignored_modules is not None:
                for mod in fsdp_ignored_modules:
                    mod.to(self.accelerator.device)

        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                log_with="wandb",
                project_dir=self.training_args.output_dir,
            )
      
        # Generate run name (no batch size since using packing)
        # Shortened name for HF (max 96 chars for repo name)
        embed_parts = self.model_args.embed_model_name.split('/')[-1].split('-')
        decoder_parts = self.model_args.decoder_name.split('/')[-1].split('-')
        
        # Extract embed model name with size: e.g., "Qwen3-Embedding-0.6B" -> "Qwen3-Embedding-0.6B"
        embed_short = embed_parts[0]  # e.g., "Qwen3"
        for part in embed_parts[1:]:
            if 'Embedding' in part or 'embedding' in part:  # Keep "Embedding" in name
                embed_short += f"-{part}"
            elif 'B' in part:  # Found size like "0.6B", "8B", etc.
                embed_short += f"-{part}"
                break
        
        # Extract LLM name with size and Instruct: e.g., "Qwen3-4B-Instruct-2507" -> "Qwen3-4B-Instruct"
        decoder_short = decoder_parts[0]  # e.g., "Qwen3"
        for part in decoder_parts[1:]:
            if 'B' in part:  # Found size like "4B"
                decoder_short += f"-{part}"
            elif 'Instruct' in part:  # Found "Instruct"
                decoder_short += f"-{part}"
                break
        
        train_mode = "adapter"
        if self.model_args.train_decoder:
            train_mode = "decoder"
        elif self.model_args.train_encoder:
            if self.model_args.train_decoder_num_layers > 0:
                train_mode = f"encoder-decoder-{self.model_args.train_decoder_num_layers}-layers"
            else:
                train_mode = "encoder"
        
        # Select appropriate learning rate based on training mode
        if train_mode == "decoder":
            selected_lr = self.training_args.decoder_lr
        elif train_mode == "encoder":
            selected_lr = self.training_args.encoder_lr
        else:  # adapter
            selected_lr = self.training_args.adapter_lr

        if self.model_args.decoder_lora and self.model_args.embed_lora:
            lora_suffix = "-decoder-lora-embed-lora"
        elif self.model_args.decoder_lora:
            lora_suffix = "-decoder-lora"
        elif self.model_args.embed_lora:
            lora_suffix = "-embed-lora"
        else:
            lora_suffix = ""

        # Stage name for checkpoint naming
        stage_num = self.training_args.stage

        # Simple checkpoint naming: stage{N}
        self.run_name_short = f"stage{stage_num}"

        # Update output_dir to include experiment name
        experiment = self.training_args.experiment
        if experiment and experiment != "default":
            self.training_args.output_dir = os.path.join(self.training_args.output_dir, experiment)
            os.makedirs(self.training_args.output_dir, exist_ok=True)
            self.accelerator.print(f"Output directory (with experiment): {self.training_args.output_dir}")

        # Wandb run name: custom_name_stage{N} or experiment_stage{N} if "auto" or not specified
        if self.training_args.wandb_name and self.training_args.wandb_name != "auto":
            self.run_name = f"{self.training_args.wandb_name}_stage{stage_num}"
        else:
            self.run_name = f"{experiment}_stage{stage_num}"
        
        # Initialize tracking
        self.accelerator.init_trackers(
            # project_name="compression-training",
            project_name=self.training_args.wandb_project,
            config={
                **vars(self.model_args),
                **vars(self.data_args), 
                **vars(self.training_args)
            },
            init_kwargs={
                "wandb": {
                    "name": self.run_name,  # {experiment}_stage{N}
                }
            }
        )

        self.accelerator.print(self.model)
        
        # Check for auto-resume after accelerator setup
        # Note: auto_resume can work together with resume_from_checkpoint
        # Priority: auto_resume checkpoints > resume_from_checkpoint > from scratch
        if self.training_args.auto_resume:
            self._check_auto_resume()

    def _check_auto_resume(self):
        """Check for latest checkpoint and set it for auto-resume
        
        Note: Auto-resume loads the full accelerator state (model, optimizer, scheduler, dataloader)
        from checkpoints saved during training to recover from GPU cluster preemptions.
        
        Priority order:
        1. If auto-resume checkpoint found (from current run): Use that - overrides resume_from_checkpoint
        2. If no auto-resume checkpoint: Will use resume_from_checkpoint if provided (already loaded)
        3. If neither: Start from scratch (already initialized)
        """
        self.accelerator.print("Checking for auto-resume checkpoints from current run...")
        
        result = find_latest_checkpoint(
            output_dir=self.training_args.output_dir,
            run_name=self.run_name_short
        )
        
        if result is not None:
            checkpoint_path, global_step = result
            self.auto_resume_checkpoint_path = checkpoint_path
            self.accelerator.print(f"Found auto-resume checkpoint at step {global_step}: {checkpoint_path}")
            
            if self.should_resume_from_checkpoint:
                self.accelerator.print(f"Note: Auto-resume checkpoint found. This overrides resume_from_checkpoint.")
                self.accelerator.print(f"      (resume_from_checkpoint was: {self.training_args.resume_from_checkpoint})")
            
            self.accelerator.print("Will resume training from this checkpoint after model preparation")
            
            # Store the step number for validation - actual state will be loaded by accelerator.load_state()
            self.global_step = global_step
            
            # Load metadata to restore other training state
            metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.num_examples_seen = metadata.get("num_examples_seen", 0)
                    self.accelerator.print(f"Will resume from step {self.global_step}, examples seen: {self.num_examples_seen}")
        else:
            if self.should_resume_from_checkpoint:
                self.accelerator.print("No auto-resume checkpoint found.")
                self.accelerator.print(f"✓ Starting training from resume_from_checkpoint: {self.training_args.resume_from_checkpoint}")
                self.accelerator.print("  (This will be the first run, auto-resume will protect subsequent runs)")
            else:
                self.accelerator.print("No auto-resume checkpoint found. Starting training from scratch.")
    

    def build_model_and_tokenizer(self):
        """Build model and tokenizer"""

        ### TOKENIZERS ###
        # Use configured tokenizer (can be different from LLM model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.model_args.decoder_tokenizer_name)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.model_args.embed_model_name)

        ### LLM ###
        decoder_attn = self.model_args.decoder_attn_implementation
        print(f"LLM attention implementation: {decoder_attn}")

        if getattr(self.model_args, 'random_init_decoder', False):
            print(f"Randomly initializing LLM from config: {self.model_args.decoder_name}")
            decoder_config = AutoConfig.from_pretrained(self.model_args.decoder_name)
            decoder_config._attn_implementation = decoder_attn
            if self.training_args.use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM
                self.decoder = AutoLigerKernelForCausalLM.from_config(decoder_config)
            else:
                self.decoder = AutoModelForCausalLM.from_config(decoder_config)
        elif self.training_args.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            self.decoder = AutoLigerKernelForCausalLM.from_pretrained(
                self.model_args.decoder_name,
                attn_implementation=decoder_attn,
            )
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.model_args.decoder_name,
                attn_implementation=decoder_attn,
            )

        ### EMBEDDER ###
        embed_attn = self.model_args.embed_attn_implementation
        print(f"Embedder attention implementation: {embed_attn}")

        if getattr(self.model_args, 'random_init_encoder', False):
            print(f"Randomly initializing embedder from config: {self.model_args.embed_model_name}")
            embed_config = AutoConfig.from_pretrained(self.model_args.embed_model_name)
            embed_config._attn_implementation = embed_attn
            self.embed_model = AutoModel.from_config(embed_config)
        else:
            self.embed_model = AutoModel.from_pretrained(
                self.model_args.embed_model_name,
                attn_implementation=embed_attn,
            )

        ### NEW TOKENS ###
        new_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
        num_added_tokens = self.decoder_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        
        # Resize token embeddings to accommodate new tokens
        if num_added_tokens > 0:
            old_vocab_size = self.decoder.get_input_embeddings().weight.size(0)
            self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
            print(f"Resized token embeddings from {old_vocab_size} to {len(self.decoder_tokenizer)}")

        ### FREEZE ###
        if not self.model_args.train_decoder:
            # Freeze all LLM parameters
            for param in self.decoder.parameters():
                param.requires_grad = False

            # Enable gradients for the embedding layer (needed for special tokens)
            if self.training_args.train_wrap_tokens:
                embedding_layer = self.decoder.get_input_embeddings()
                embedding_layer.weight.requires_grad = True

            # Enable training for LLM embed_tokens if requested (full embeddings, not just special tokens)
            if self.model_args.train_decoder_embed_tokens:
                embedding_layer = self.decoder.get_input_embeddings()
                embedding_layer.weight.requires_grad = True
                print(f"Enabled training for LLM embed_tokens (full embeddings)")

            # Enable training for first N transformer layers if requested
            if self.model_args.train_decoder_num_layers > 0:
                self._unfreeze_llm_layers(self.decoder, self.model_args.train_decoder_num_layers)

        if not self.model_args.train_encoder:
            for param in self.embed_model.parameters():
                param.requires_grad = False

        ### PACKED ATTENTION (must be applied BEFORE LoRA) ###
        if self.data_args.use_packing:
            backend = self.data_args.packed_attention_backend
            print(f"Enabling packed attention (backend: {backend}, before LoRA)")
            if backend == "flash":
                from latent_context.packed_flash import replace_with_packed_attention
            elif backend == "flex":
                from latent_context.packed_flex import replace_with_packed_attention
            else:
                raise ValueError(f"Unknown packed_attention_backend: {backend}")
            replace_with_packed_attention(self.decoder)
            print(f"✓ Packed attention enabled ({backend})")

        ### LORA ###
        if self.model_args.decoder_lora:
            print("Applying LoRA to LLM model")
            self.decoder = self._apply_lora(
                model=self.decoder, 
                train_model=self.model_args.train_decoder, 
                lora=self.model_args.decoder_lora, 
                lora_r=self.model_args.decoder_lora_r, 
                lora_alpha=self.model_args.decoder_lora_alpha, 
                lora_dropout=self.model_args.decoder_lora_dropout, 
                lora_target_modules=self.model_args.decoder_lora_target_modules,
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"],
            )

        if self.model_args.embed_lora:
            print("Applying LoRA to embedder model")
            self.embed_model = self._apply_lora(
                model=self.embed_model, 
                train_model=self.model_args.train_encoder, 
                lora=self.model_args.embed_lora, 
                lora_r=self.model_args.embed_lora_r, 
                lora_alpha=self.model_args.embed_lora_alpha, 
                lora_dropout=self.model_args.embed_lora_dropout, 
                lora_target_modules=self.model_args.embed_lora_target_modules,
                task_type="FEATURE_EXTRACTION",
                modules_to_save=["embed_tokens"],
            )

        ### PROCESSOR ###
        self.processor = LCLMProcessor(
            decoder_tokenizer=self.decoder_tokenizer,
            embed_tokenizer=self.embed_tokenizer,
            compression_ratio=self.training_args.compression_ratio,
            max_memory_length=self.data_args.max_memory_length,
            use_memory_wrapping=self.training_args.use_memory_wrapping,
        )

        ### Code LLaVA Model ###
        self.model = LCLM(
            decoder=self.decoder,
            decoder_tokenizer=self.decoder_tokenizer,
            embed_model=self.embed_model,
            embed_tokenizer=self.embed_tokenizer,
            processor=self.processor,
            compression_ratio=self.training_args.compression_ratio,
            max_memory_length=self.data_args.max_memory_length,
            train_decoder=self.model_args.train_decoder,
            train_encoder=self.model_args.train_encoder,
            max_encode_batch_size=self.model_args.max_encode_batch_size,
            pooling=self.model_args.pooling,
            encoder_mask_type=self.model_args.encoder_mask_type,
            encoder_window_size=self.model_args.encoder_window_size,
            packed_attention_backend=self.data_args.packed_attention_backend if self.data_args.use_packing else None,
            accelerator=None,  # Will be set after accelerator is created
            adapter_type=self.model_args.adapter_type,
            num_adapter_layers=self.model_args.num_adapter_layers,
            boundary_overlap=self.model_args.boundary_overlap,
            use_fused_ce=self.model_args.use_fused_ce,
        )

        ### GRADIENT CHECKPOINTING ###
        if self.training_args.decoder_gradient_checkpointing:
            print("Enabling gradient checkpointing for LLM model")
            self.model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.decoder.config.use_cache = False
        if self.training_args.encoder_gradient_checkpointing:
            print("Enabling gradient checkpointing for embedder model")
            self.model.encoder.embed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.encoder.embed_model.config.use_cache = False
        

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a HuggingFace-format checkpoint directory.
        
        This loads model weights from a previously trained checkpoint in HF format:
        - decoder/ directory containing decoder model and tokenizer
        - embedder/ directory containing embedder model and tokenizer  
        - projectors/ directory containing projection layer weights
        
        Note: This does NOT load optimizer/scheduler state. Use auto_resume for that.
        This is typically used to start training from a pre-trained model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory with decoder/, encoder/, adapter/ subdirs
        """
        print(f"Resuming training from HF-format checkpoint: {checkpoint_path}")

        # Load tokenizer from checkpoint LLM dir
        decoder_dir = os.path.join(checkpoint_path, "decoder")
        embed_dir = os.path.join(checkpoint_path, "encoder")
        projectors_dir = os.path.join(checkpoint_path, "adapter")
        print(f"LLM dir: {decoder_dir}")
        print(f"Embedder dir: {embed_dir}")
        print(f"Projectors dir: {projectors_dir}")

        # Load tokenizer from checkpoint (contains special tokens)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_dir)
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        decoder_attn = self.model_args.decoder_attn_implementation
        if self.training_args.use_liger_kernel:
            print(f"Using Liger kernel for LLM model (resume), attn={decoder_attn}")
            self.decoder = AutoLigerKernelForCausalLM.from_pretrained(
                decoder_dir,
                attn_implementation=decoder_attn,
            )
        else:
            print(f"Using AutoModelForCausalLM for LLM model (resume), attn={decoder_attn}")
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_dir,
                attn_implementation=decoder_attn,
            )

        if not self.model_args.train_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

            # Enable gradients for the embedding layer (needed for special tokens)
            if self.training_args.train_wrap_tokens:
                embedding_layer = self.decoder.get_input_embeddings()
                embedding_layer.weight.requires_grad = True

            # Enable training for LLM embed_tokens if requested (full embeddings, not just special tokens)
            if self.model_args.train_decoder_embed_tokens:
                embedding_layer = self.decoder.get_input_embeddings()
                embedding_layer.weight.requires_grad = True
                print(f"Enabled training for LLM embed_tokens (full embeddings)")

            # Enable training for first N transformer layers if requested
            if self.model_args.train_decoder_num_layers > 0:
                self._unfreeze_llm_layers(self.decoder, self.model_args.train_decoder_num_layers)

        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_dir)

        embed_attn = self.model_args.embed_attn_implementation
        print(f"Loading embedder model from checkpoint, attn={embed_attn}")
        resumed_embed_model = AutoModel.from_pretrained(
            embed_dir,
            attn_implementation=embed_attn,
        )

        self.embed_model = resumed_embed_model

        if not self.model_args.train_encoder:
            for param in self.embed_model.parameters():
                param.requires_grad = False

        ### PACKED ATTENTION (must be applied BEFORE LoRA) ###
        if self.data_args.use_packing:
            backend = self.data_args.packed_attention_backend
            print(f"Enabling packed attention (backend: {backend}, before LoRA)")
            if backend == "flash":
                from latent_context.packed_flash import replace_with_packed_attention
            elif backend == "flex":
                from latent_context.packed_flex import replace_with_packed_attention
            else:
                raise ValueError(f"Unknown packed_attention_backend: {backend}")
            replace_with_packed_attention(self.decoder)
            print(f"✓ Packed attention enabled ({backend})")

        ### LORA ###
        if self.model_args.decoder_lora:
            print("Applying LoRA to LLM model")
            self.decoder = self._apply_lora(
                model=self.decoder, 
                train_model=self.model_args.train_decoder, 
                lora=self.model_args.decoder_lora, 
                lora_r=self.model_args.decoder_lora_r, 
                lora_alpha=self.model_args.decoder_lora_alpha, 
                lora_dropout=self.model_args.decoder_lora_dropout, 
                lora_target_modules=self.model_args.decoder_lora_target_modules,
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"],
            )

        if self.model_args.embed_lora:
            print("Applying LoRA to embedder model")
            self.embed_model = self._apply_lora(
                model=self.embed_model, 
                train_model=self.model_args.train_encoder, 
                lora=self.model_args.embed_lora, 
                lora_r=self.model_args.embed_lora_r, 
                lora_alpha=self.model_args.embed_lora_alpha, 
                lora_dropout=self.model_args.embed_lora_dropout, 
                lora_target_modules=self.model_args.embed_lora_target_modules,
                task_type="FEATURE_EXTRACTION",
                modules_to_save=["embed_tokens"],
            ) 

        ### NEW TOKENS HANDLING (Resume) ###
        # When resuming, the tokenizer already contains the special tokens from the checkpoint
        # Gradient mask hooks will be installed after model preparation

        ### PROCESSOR ###
        self.processor = LCLMProcessor(
            decoder_tokenizer=self.decoder_tokenizer,
            embed_tokenizer=self.embed_tokenizer,
            compression_ratio=self.training_args.compression_ratio,
            max_memory_length=self.data_args.max_memory_length,
            use_memory_wrapping=self.training_args.use_memory_wrapping,
        )

        # Create LCLM
        self.model = LCLM(
            decoder=self.decoder,
            decoder_tokenizer=self.decoder_tokenizer,
            embed_model=self.embed_model,
            embed_tokenizer=self.embed_tokenizer,
            processor=self.processor,
            compression_ratio=self.training_args.compression_ratio,
            max_memory_length=self.data_args.max_memory_length,
            train_decoder=self.model_args.train_decoder,
            train_encoder=self.model_args.train_encoder,
            max_encode_batch_size=self.model_args.max_encode_batch_size,
            pooling=self.model_args.pooling,
            encoder_mask_type=self.model_args.encoder_mask_type,
            encoder_window_size=self.model_args.encoder_window_size,
            packed_attention_backend=self.data_args.packed_attention_backend if self.data_args.use_packing else None,
            accelerator=None,  # Will be set after accelerator is created
            adapter_type=self.model_args.adapter_type,
            num_adapter_layers=self.model_args.num_adapter_layers,
            boundary_overlap=self.model_args.boundary_overlap,
            use_fused_ce=self.model_args.use_fused_ce,
        )

        # Load adapter weights
        adapter_path = os.path.join(projectors_dir, "adapter.safetensors")
        if os.path.isfile(adapter_path):
            adapter_state = load_safetensors(adapter_path)
            self.model.adapter.load_state_dict(adapter_state, strict=True)
            print("Loaded code adapter weights")
        else:
            print(f"Warning: adapter weights not found at {adapter_path}")

        # Apply gradient checkpointing settings
        if self.training_args.decoder_gradient_checkpointing:
            print("Enabling gradient checkpointing for LLM model")
            self.model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.decoder.config.use_cache = False
        if self.training_args.encoder_gradient_checkpointing:
            print("Enabling gradient checkpointing for embedder model")
            self.model.encoder.embed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.encoder.embed_model.config.use_cache = False

        
    def load_datasets(self):
        """Load and prepare datasets using data module functions"""
        self.accelerator.print("Loading datasets...")
        
        # Use the comprehensive data loading function from data module
        processor = self.model.processor if hasattr(self.model, 'processor') else None
        self.train_dataloader = prepare_datasets(
            data_args=self.data_args,
            training_args=self.training_args,
            accelerator=self.accelerator,
            processor=processor,
            pooling=self.model_args.pooling,
        )
        
    def setup_optimizer_and_scheduler(self):
        """Setup single optimizer with multiple parameter groups and corresponding schedulers"""
        # Create parameter groups
        param_groups, group_info = self._create_parameter_groups()
        self._print_trainable_parameters()
        self._print_model_config()

        # Calculate total steps
        total_steps = int(len(self.train_dataloader) * self.training_args.num_epochs // self.accelerator.gradient_accumulation_steps)
        if self.training_args.max_steps > 0:
            total_steps = min(total_steps, self.training_args.max_steps)
        
        # Create single optimizer with multiple parameter groups
        if self.training_args.optimizer_type == "adamw":
            self.optimizer = AdamW(
                param_groups,
                eps=self.training_args.adam_epsilon,
            )
        elif self.training_args.optimizer_type == "adafactor":
            self.optimizer = Adafactor(
                param_groups,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        elif self.training_args.optimizer_type == "sgd":
            self.optimizer = SGD(
                param_groups,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.training_args.optimizer_type}")
        
        # Create scheduler based on scheduler_type
        # Note: Both warmup_steps and total_steps need to be multiplied by num_processes
        # for correct behavior with distributed training (scheduler steps per process)
        warmup_steps = self.training_args.warmup_steps * self.accelerator.num_processes
        total_steps_scaled = total_steps * self.accelerator.num_processes
        self.accelerator.print(f"Scheduler setup: warmup_steps={warmup_steps}, total_steps={total_steps_scaled} (user: {self.training_args.warmup_steps}, {total_steps})")
        if self.training_args.scheduler_type == "cosine":
            self.scheduler = CosineWithMinLRScheduler(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps_scaled,
                min_lr=self.training_args.min_lr,  # All parameter groups decay to decoder_lr
            )
        elif self.training_args.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.training_args.scheduler_type}. Supported types: cosine, constant")
        
        # Store parameter group names for logging
        self.param_group_names = [info[1] for info in group_info]
        
        # Log what we're doing
        self.accelerator.print(f"Using {self.training_args.scheduler_type} scheduler for all parameter groups:")
        for group_index, group_name in group_info:
            start_lr = self.optimizer.param_groups[group_index]['lr']
            if self.training_args.scheduler_type == "cosine":
                self.accelerator.print(f"  {group_name}: {start_lr} -> {self.training_args.min_lr}")
            elif self.training_args.scheduler_type == "constant":
                self.accelerator.print(f"  {group_name}: {start_lr} (constant after warmup)")
        

    def prepare_for_training(self):
        """Prepare model, optimizer, scheduler, and dataloaders with accelerator."""
        from torch.utils.data import IterableDataset
        is_iterable = isinstance(self.train_dataloader.dataset, IterableDataset)

        if is_iterable:
            # For IterableDataset (dynamic packing with StatefulDataLoader):
            # - Don't wrap dataloader with accelerate (causes prefetch state mismatch)
            # - Move batches to device manually in training loop
            if self.accelerator.state.deepspeed_plugin is not None:
                self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler)

            # Store reference if using StatefulDataLoader
            self._stateful_dataloader = self.train_dataloader if hasattr(self.train_dataloader, 'state_dict') else None
        else:
            self._stateful_dataloader = None
            self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_dataloader)

        # Load checkpoint if auto-resume is enabled
        if self.auto_resume_checkpoint_path is not None:
            self._load_auto_resume_checkpoint()

        # Print FSDP structure for debugging
        # if self.training_args.distributed_type == "fsdp":
        #     self._print_fsdp_structure()

    def _print_fsdp_structure(self):
        """Print FSDP wrapping structure and parameter shapes for debugging."""

        self.accelerator.print("=" * 80)
        self.accelerator.print("FSDP STRUCTURE DEBUG INFO")
        self.accelerator.print("=" * 80)

        def _get_fsdp_info(module, prefix=""):
            """Recursively get FSDP wrapping info."""
            lines = []
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                is_fsdp = isinstance(child, FSDP)

                if is_fsdp:
                    # Get flat_param info
                    flat_param_info = ""
                    if hasattr(child, '_flat_param') and child._flat_param is not None:
                        fp = child._flat_param
                        flat_param_info = f" [flat_param: {fp.shape}, requires_grad={fp.requires_grad}]"
                    elif hasattr(child, 'params') and child.params:
                        # Alternative way to access flat params
                        for p in child.params:
                            flat_param_info = f" [flat_param: {p.shape}, requires_grad={p.requires_grad}]"
                            break

                    lines.append(f"FSDP({full_name}): {type(child._fsdp_wrapped_module).__name__}{flat_param_info}")
                    # Recurse into FSDP module
                    lines.extend(_get_fsdp_info(child._fsdp_wrapped_module, full_name))
                else:
                    # Check if this is a significant module
                    param_count = sum(p.numel() for p in child.parameters(recurse=False))
                    if param_count > 0:
                        lines.append(f"  {full_name}: {type(child).__name__} (params: {param_count:,})")
                    # Recurse into children
                    lines.extend(_get_fsdp_info(child, full_name))

            return lines

        # Get the unwrapped model
        model = self.accelerator.unwrap_model(self.model)

        # Check if top-level is FSDP
        if isinstance(self.model, FSDP):
            self.accelerator.print("Top-level: FSDP wrapped")
            lines = _get_fsdp_info(self.model._fsdp_wrapped_module, "")
        else:
            self.accelerator.print("Top-level: Not FSDP wrapped")
            lines = _get_fsdp_info(model, "")

        for line in lines: 
            print(line)

        #### Print FSDP config info ####
        self.accelerator.print("########################################################")
        self.accelerator.print("FSDP CONFIG:")
        if isinstance(self.model, FSDP):
            self.accelerator.print(f"  Sharding Strategy: {self.model.sharding_strategy}")
            self.accelerator.print(f"  Mixed Precision: {self.model.mixed_precision}")
            self.accelerator.print(f"  Backward Prefetch: {self.model.backward_prefetch}")
            self.accelerator.print(f"  Use Orig Params: {self.model._fsdp_use_orig_params}")

        #### Print all parameter shapes (shows sharded vs unsharded) ####
        self.accelerator.print("########################################################")
        self.accelerator.print("")
        self.accelerator.print("PARAMETER SHAPES:")
        self.accelerator.print("-" * 80)
        for name, param in self.model.named_parameters():
            grad_status = "Train" if param.requires_grad else "Frozen"
            shape_str = str(list(param.shape))
            numel = param.numel()
            print(f"  [{grad_status}] {name}: {shape_str} ({numel:,})")
        self.accelerator.print("=" * 80)

    def _load_auto_resume_checkpoint(self):
        """Load training state from auto-resume checkpoint."""
        self.accelerator.print("=" * 80)
        self.accelerator.print(f"AUTO-RESUME: Loading from {self.auto_resume_checkpoint_path}")
        self.accelerator.print(f"Resuming from step {self.global_step}, examples seen: {self.num_examples_seen}")
        self.accelerator.print("=" * 80)

        # Load model, optimizer, scheduler state
        self.accelerator.load_state(self.auto_resume_checkpoint_path)

        # Load StatefulDataLoader state (saved separately, not tracked by accelerator)
        if self._stateful_dataloader is not None:
            self._load_dataloader_state()

        self.accelerator.print("✓ Auto-resume complete")

    def _load_dataloader_state(self):
        """Load StatefulDataLoader state from checkpoint."""
        dl_state_path = os.path.join(
            self.auto_resume_checkpoint_path,
            f"dataloader_state_{self.accelerator.process_index}.pt"
        )
        if not os.path.exists(dl_state_path):
            self.accelerator.print(f"Warning: No dataloader state found at {dl_state_path}")
            return

        dl_state = torch.load(dl_state_path)
        ds_state = dl_state.get('dataset_state', {})

        # Load state directly into dataset (StatefulDataLoader defers loading otherwise)
        if ds_state and hasattr(self._stateful_dataloader.dataset, 'load_state_dict'):
            self._stateful_dataloader.dataset.load_state_dict(ds_state)

        # Also load into StatefulDataLoader for completeness
        self._stateful_dataloader.load_state_dict(dl_state)

        self.accelerator.print(f"Restored dataloader state: file_idx={ds_state.get('file_idx')}, row_idx={ds_state.get('row_idx')}")

    def _get_slurm_remaining_seconds(self) -> float:
        """Query SLURM for remaining job time.

        Returns:
            Remaining seconds, or float('inf') if not in a SLURM job or query fails.
        """
        import subprocess

        try:
            if "tuolumne" in socket.gethostname():
                # tuo uses flux
                result = subprocess.run(
                    ["flux", "job", "timeleft"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True
                )
                return int(result.stdout.strip())
            else:
                slurm_job_id = os.environ.get('SLURM_JOB_ID')
                if not slurm_job_id:
                    return float('inf')

                # return the time format should be sth like: "23:45:30" or "1-12:00:00" (days-hours:min:sec)
                result = subprocess.run(
                    ['squeue', '-j', slurm_job_id, '-h', '-o', '%L'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0 or not result.stdout.strip():
                    return float('inf')

                time_str = result.stdout.strip()
                return self._parse_slurm_time(time_str)
        except Exception as e:
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Warning: Failed to query SLURM remaining time: {e}")
            return float('inf')

    def _parse_slurm_time(self, time_str: str) -> float:
        """Parse SLURM time format to seconds.

        Formats: "MM:SS", "HH:MM:SS", "D-HH:MM:SS"
        """
        try:
            days = 0
            if '-' in time_str:
                days_part, time_str = time_str.split('-')
                days = int(days_part)

            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = map(int, parts)
            else:
                return float('inf')

            return days * 86400 + hours * 3600 + minutes * 60 + seconds
        except (ValueError, AttributeError):
            return float('inf')

    def _should_save_preemption_checkpoint(self) -> bool:
        """Check if we should save a checkpoint due to approaching SLURM job time limit.

        Returns:
            True if preemption checkpoint should be saved, False otherwise.
        """
        # Skip if preemption checkpointing is disabled
        if self.training_args.preempt_save_minutes <= 0:
            return False

        # Skip if we already saved a preemption checkpoint
        if self._preemption_checkpoint_saved:
            return False

        # Query SLURM for actual remaining time
        remaining_seconds = self._get_slurm_remaining_seconds()
        preempt_threshold_seconds = self.training_args.preempt_save_minutes * 60

        # Check if we're within the preemption window
        if remaining_seconds <= preempt_threshold_seconds:
            self.accelerator.print(
                f"⚠️ SLURM preemption checkpoint: {remaining_seconds/60:.1f} minutes remaining "
                f"(threshold: {self.training_args.preempt_save_minutes} minutes)"
            )
            return True

        return False

    def train(self):
        """Main training loop."""
        batches_per_epoch = len(self.train_dataloader)
        grad_accum_steps = max(1, self.accelerator.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum_steps))
        total_steps = steps_per_epoch * self.training_args.num_epochs
        if self.training_args.max_steps > 0:
            total_steps = min(total_steps, self.training_args.max_steps)

        self.accelerator.print(f"Training for {self.training_args.num_epochs} epochs ({total_steps} steps, max_steps={self.training_args.max_steps})")
        self.accelerator.print(f"Batches per epoch: {batches_per_epoch}, Steps per epoch: {steps_per_epoch}")
        if self.global_step > 0:
            self.accelerator.print(f"Resuming from step {self.global_step}")
        if self._skip_step_ranges:
            ranges_str = ", ".join(f"{s}-{e}" for s, e in self._skip_step_ranges)
            self.accelerator.print(f"⏭️ Skip step ranges configured: {ranges_str}")

        # StatefulDataLoader handles resume automatically - just run epochs
        self._max_steps_reached = False
        for epoch in range(self.training_args.num_epochs):
            self._train_epoch(epoch)
            if self._max_steps_reached:
                break

        # Save final checkpoint
        self.accelerator.wait_for_everyone()
        self.accelerator.print("Saving final checkpoint...")
        save_checkpoint(
            accelerator=self.accelerator,
            model=self.model,
            decoder_tokenizer=self.decoder_tokenizer,
            embed_tokenizer=self.embed_tokenizer,
            global_step=self.global_step,
            output_dir=self.training_args.output_dir,
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=self.training_args,
            run_name=self.run_name_short,
            extra_metadata={
                "epoch": int(self.training_args.num_epochs),
                "num_examples_seen": int(self.num_examples_seen),
                "final_checkpoint": True,
            },
            is_final=True,
            train_dataloader=self._stateful_dataloader,
            delete_old_checkpoints=self.training_args.delete_old_checkpoints,
        )

        # Cleanup
        if self.accelerator.is_main_process:
            wandb.finish()
        self.accelerator.end_training()
        
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        # StatefulDataLoader handles resume automatically
        is_stateful = self._stateful_dataloader is not None
        optimizer_step_loss = 0.0

        # Calculate progress bar initial position from global_step (for resume)
        # global_step tracks total steps across all files, not just current file's row_idx
        pbar_initial = self.global_step if self.global_step > 0 else 0

        # Sanity check: verify global_step matches dataloader position
        if is_stateful and hasattr(self._stateful_dataloader.dataset, '_state') and self.global_step > 0:
            ds = self._stateful_dataloader.dataset
            state = ds._state
            file_idx = state.get('file_idx', 0)
            row_idx = state.get('row_idx', 0)
            # Estimate position: file_idx * rows_per_file + row_idx
            # Use first file to get rows_per_file (assumes all files have same size)
            if hasattr(ds, 'parquet_files') and len(ds.parquet_files) > 0:
                import pyarrow.parquet as pq
                first_file_rows = pq.read_metadata(ds.parquet_files[0]).num_rows
                # Account for multi-GPU: each rank sees subset of files
                estimated_position = (file_idx * first_file_rows + row_idx)
                batch_size = self.data_args.train_batch_size
                grad_accum = self.accelerator.gradient_accumulation_steps
                expected_samples = self.global_step * batch_size * grad_accum
                self.accelerator.print(
                    f"Resume sanity check: global_step={self.global_step}, "
                    f"expected_samples={expected_samples} (global_step * {batch_size} * {grad_accum}), "
                    f"estimated_from_dataloader={estimated_position} "
                    f"(file_idx={file_idx} * {first_file_rows} + row_idx={row_idx})"
                )

        # Progress bar (main process only)
        if self.accelerator.is_main_process:
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                initial=pbar_initial,
                total=len(self.train_dataloader)
            )
        else:
            pbar = self.train_dataloader

        for step, batch in enumerate(pbar, start=pbar_initial):
            # Move batch to device if using StatefulDataLoader (not wrapped by accelerate)
            if is_stateful:
                batch = self._move_batch_to_device(batch)

            # Check if this step should be skipped (problematic data points)
            # We check global_step + 1 because global_step hasn't been incremented yet
            next_step = self.global_step + 1
            should_skip = self._should_skip_step(next_step)

            # Always use accumulate() context to maintain proper gradient accumulation tracking
            with self.accelerator.accumulate(self.model):
                if should_skip:
                    # Skip the forward/backward pass but still advance scheduler
                    loss = None
                    if self.accelerator.sync_gradients:
                        self.accelerator.print(f"⏭️ Skipping step {next_step} (in skip_step_ranges)")
                        self.scheduler.step()
                else:
                    # Normal training step
                    loss = self._training_step(batch, step)
                    if loss is not None:
                        optimizer_step_loss += loss

            if self.accelerator.sync_gradients:
                self.global_step += 1

                # Early stop if max_steps reached
                if self.training_args.max_steps > 0 and self.global_step >= self.training_args.max_steps:
                    self.accelerator.print(f"Reached max_steps={self.training_args.max_steps}, stopping training.")
                    self._max_steps_reached = True
                    break

                do_save = (self.training_args.save_steps > 0 and self.global_step % self.training_args.save_steps == 0)

                # Handle skipped steps differently - don't log loss metrics
                if should_skip:
                    # For skipped steps, log minimal metrics
                    metrics = {
                        "train/step": self.global_step,
                        "train/skipped": 1,
                    }
                    # Log learning rates for all parameter groups
                    for i, group_name in enumerate(self.param_group_names):
                        lr = self.optimizer.param_groups[i]['lr']
                        metrics[f"train/learning_rate_{group_name.lower()}"] = lr

                    self.accelerator.log(metrics, step=self.global_step)
                    optimizer_step_loss = 0.0

                    # Still track data points seen
                    batch_size = batch['input_ids'].size(0)
                    self.num_examples_seen += int(batch_size) * self.accelerator.num_processes

                    # Update progress bar
                    if self.accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                        pbar.set_description(f"Step {self.global_step} (skipped)")

                    continue  # Skip the rest of the logging/checkpoint logic for this step

                # Normal step metrics
                # Convert accumulated loss to tensor for gather_for_metrics
                loss_tensor = torch.tensor(optimizer_step_loss / self.accelerator.gradient_accumulation_steps, device=self.accelerator.device)
                loss_value = self.accelerator.gather_for_metrics(loss_tensor).mean().item()

                running_avg_loss = self._update_training_loss_average(loss_value)

                # Only print to console every log_interval steps (wandb logs every step)
                if self.global_step % self.training_args.log_interval == 0:
                    self.accelerator.print(f"Step {self.global_step} - Loss: {loss_value:.4f} | Running Avg Loss: {running_avg_loss:.4f}")

                metrics = {
                    "train/loss": loss_value,
                    "train/loss_running_avg": running_avg_loss,
                    "train/step": self.global_step,
                    "train/compression_ratio": self.model.encoder.compression_ratio,
                    "train/skipped": 0,
                }

                # Token counts and throughput (if enabled)
                if self.training_args.log_token_counts:
                    current_time = time.time()

                    # Gather token counts across ranks
                    memory_tokens_tensor = torch.tensor(self.accum_memory_tokens, device=self.accelerator.device, dtype=torch.float32)
                    decoder_tokens_tensor = torch.tensor(self.accum_decoder_tokens, device=self.accelerator.device, dtype=torch.float32)

                    all_memory_tokens = self.accelerator.gather(memory_tokens_tensor)
                    all_decoder_tokens = self.accelerator.gather(decoder_tokens_tensor)

                    total_memory_tokens = all_memory_tokens.sum().item()
                    total_decoder_tokens = all_decoder_tokens.sum().item()

                    metrics.update({
                        "train/memory_tokens_per_step": total_memory_tokens,  # Code tokens before compression
                        "train/decoder_tokens_per_step": total_decoder_tokens,  # Total LLM sequence tokens
                    })

                    # Throughput (tokens/sec) - skip first step
                    if self.last_step_time is not None:
                        elapsed = current_time - self.last_step_time
                        if elapsed > 0:
                            metrics["train/memory_tokens_per_sec"] = total_memory_tokens / elapsed
                            metrics["train/decoder_tokens_per_sec"] = total_decoder_tokens / elapsed
                            metrics["train/step_time_sec"] = elapsed

                    # Per-rank token counts
                    for rank_idx, (mem_tok, decoder_tok) in enumerate(zip(
                        all_memory_tokens.tolist(), all_decoder_tokens.tolist()
                    )):
                        metrics[f"train/memory_tokens_rank{rank_idx}"] = mem_tok
                        metrics[f"train/decoder_tokens_rank{rank_idx}"] = decoder_tok

                    # Reset accumulators and update timestamp
                    self.accum_memory_tokens = 0
                    self.accum_decoder_tokens = 0
                    self.last_step_time = current_time

                # Log learning rates for all parameter groups
                for i, group_name in enumerate(self.param_group_names):
                    lr = self.optimizer.param_groups[i]['lr']
                    metrics[f"train/learning_rate_{group_name.lower()}"] = lr

                self.accelerator.log(metrics, step=self.global_step)

                optimizer_step_loss = 0.0

                # checkpoint
                if do_save:
                    self.accelerator.wait_for_everyone()
                    save_checkpoint(
                        accelerator=self.accelerator,
                        model=self.model,
                        decoder_tokenizer=self.decoder_tokenizer,
                        embed_tokenizer=self.embed_tokenizer,
                        global_step=self.global_step,
                        output_dir=self.training_args.output_dir,
                        model_args=self.model_args,
                        data_args=self.data_args,
                        training_args=self.training_args,
                        run_name=self.run_name_short,
                        extra_metadata={
                            "epoch": int(epoch),
                            "num_examples_seen": int(self.num_examples_seen),
                        },
                        train_dataloader=self._stateful_dataloader,
                        delete_old_checkpoints=self.training_args.delete_old_checkpoints,
                    )
                    self.accelerator.wait_for_everyone()
                    # Reset timing so checkpoint time isn't included in throughput
                    if self.training_args.log_token_counts:
                        self.last_step_time = time.time()

                # SLURM preemption checkpoint: save if approaching job time limit
                if self._should_save_preemption_checkpoint():
                    self.accelerator.print(f"💾 Saving preemption checkpoint at step {self.global_step}...")
                    self.accelerator.wait_for_everyone()
                    save_checkpoint(
                        accelerator=self.accelerator,
                        model=self.model,
                        decoder_tokenizer=self.decoder_tokenizer,
                        embed_tokenizer=self.embed_tokenizer,
                        global_step=self.global_step,
                        output_dir=self.training_args.output_dir,
                        model_args=self.model_args,
                        data_args=self.data_args,
                        training_args=self.training_args,
                        run_name=self.run_name_short,
                        extra_metadata={
                            "epoch": int(epoch),
                            "num_examples_seen": int(self.num_examples_seen),
                            "preemption_checkpoint": True,
                        },
                        train_dataloader=self._stateful_dataloader,
                        delete_old_checkpoints=self.training_args.delete_old_checkpoints,
                    )
                    self._preemption_checkpoint_saved = True
                    self.accelerator.print(f"✅ Preemption checkpoint saved. Training will continue until SLURM kills the job.")
                    self.accelerator.wait_for_everyone()
                    # Reset timing so checkpoint time isn't included in throughput
                    if self.training_args.log_token_counts:
                        self.last_step_time = time.time()

            # Track data points seen
            batch_size = batch['input_ids'].size(0)
            self.num_examples_seen += int(batch_size) * self.accelerator.num_processes
            
            # Update progress bar (respect log_interval)
            if self.accelerator.is_main_process and hasattr(pbar, 'set_postfix') and loss is not None:
                if self.global_step % self.training_args.log_interval == 0:
                    pbar.set_description(f"Step {self.global_step}")
                    pbar.set_postfix({'loss': f"{loss:.4f}"})
            
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the accelerator device.

        Used when StatefulDataLoader is not wrapped by accelerate's prepare_data_loader
        (to avoid prefetch-induced state mismatch on resume).
        """
        device = self.accelerator.device
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                moved[k] = [t.to(device, non_blocking=True) for t in v]
            else:
                moved[k] = v
        return moved

    def _training_step(self, batch: Dict[str, Any], step: int) -> Optional[float]:
        """Execute one training step

        Returns:
            loss value or None if step was skipped
        """

        skip_nan_grad_step = False

        # Handle packed batches
        sample_lens = batch.pop('sample_lens', None)  # Remove from batch to avoid duplicate

        # Accumulate token counts for this batch (if logging enabled)
        if self.training_args.log_token_counts:
            # Memory tokens = code tokens before compression (sum over all batch items)
            memory_token_ids = batch.get('memory_token_ids', [])
            for batch_item_codes in memory_token_ids:
                if batch_item_codes:
                    self.accum_memory_tokens += sum(len(seg) for seg in batch_item_codes)

            # LLM tokens = total sequence length
            if 'input_ids' in batch:
                self.accum_decoder_tokens += batch['input_ids'].numel()

        # DEBUG: Log chunk info periodically to verify compression_ratio is working
        if step % 100 == 0:
            latent_counts = batch.get('latent_counts', [])
            total_chunks = sum(sum(cc) for cc in latent_counts if cc)
            memory_positions = batch.get('memory_positions', [])
            num_regions = sum(len(cp) for cp in memory_positions if cp)
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
            seq_len = batch['input_ids'].shape[1] if 'input_ids' in batch else 0

            # Get embed token counts (before chunking) - sum over all batch items
            memory_token_ids = batch.get('memory_token_ids', [])
            total_embed_tokens = sum(
                sum(len(seg) for seg in batch_codes)
                for batch_codes in memory_token_ids if batch_codes
            )

            self.accelerator.print(
                f"[Step {step}] compression_ratio={self.model.encoder.compression_ratio}, "
                f"batch_size={batch_size}, seq_len={seq_len}, num_regions={num_regions}, "
                f"total_embed_tokens={total_embed_tokens}, total_memory_tokens={total_chunks}"
            )

        # If sample_lens is not None, we are using packed training, otherwise we are using regular training
        outputs = self.model(**batch, sample_lens=sample_lens)
        loss = outputs.loss

        # DEBUG: Sync and check after forward
        # import torch
        # torch.cuda.synchronize()
        # print(f"[DEBUG Step {step}] Forward done, loss={loss.item():.4f}")

        # self.accelerator.print(f"Loss: {loss}")
        self.accelerator.backward(loss)

        # DEBUG: Sync and check after backward
        # torch.cuda.synchronize()
        # print(f"[DEBUG Step {step}] Backward done")

        # DEBUG: Check gradients are valid
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         if not torch.isfinite(param.grad).all():
        #             print(f"[DEBUG Step {step}] Non-finite grad in {name}")
        #             break
        # torch.cuda.synchronize()
        # print(f"[DEBUG Step {step}] Grad check done")

        # Check for non-finite loss and gradients
        # Only check gradients when loss is non-finite to avoid expensive full parameter sweep
        if not torch.isfinite(loss).item():
            non_finite_any = has_non_finite_loss_and_gradients(loss=loss, model=self.model, accelerator=self.accelerator)
        else:
            non_finite_any = False
        
        if non_finite_any:
            skip_nan_grad_step = True
            self.accelerator.print(f"[Step {self.global_step}] Non-finite detected. Skipping optimizer step.")
            self.optimizer.zero_grad(set_to_none=True)
        else:
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.training_args.max_grad_norm)

            # Apply optimizer step (no-op during accumulation due to accelerator.accumulate())
            self.optimizer.step()

            # Step scheduler only at end of gradient accumulation
            if self.accelerator.sync_gradients:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()
            # torch.cuda.synchronize()
            # print(f"[DEBUG Step {step}] Zero grad done")
        
        # Return loss value
        if not skip_nan_grad_step:
            return loss.item()
            
        return None
        
    
            
    def _create_parameter_groups(self):
        """Create parameter groups for single optimizer with different learning rates"""
        # Collect parameters by component
        decoder_params = []
        encoder_params = []
        adapter_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if "decoder" in name:
                decoder_params.append(param)
            elif "encoder.embed_model" in name:
                encoder_params.append(param)
            elif "adapter" in name:
                adapter_params.append(param)
            else:
                decoder_params.append(param)
        
        # Parse beta values
        decoder_betas = self._parse_betas(self.training_args.decoder_betas) 
        encoder_betas = self._parse_betas(self.training_args.encoder_betas) 
        adapter_betas = self._parse_betas(self.training_args.adapter_betas) 
        
        # Create parameter groups for the optimizer
        param_groups = []
        group_info = []  # (group_index, group_name) for logging

        if decoder_params:
            decoder_lr = self.training_args.decoder_lr
            decoder_weight_decay = self.training_args.decoder_weight_decay or 0.0

            param_groups.append({
                'params': decoder_params,
                'lr': decoder_lr,
                'weight_decay': decoder_weight_decay,
                'betas': decoder_betas,
            })
            group_info.append((len(param_groups) - 1, "LLM"))
            self.accelerator.print(f"LLM parameter group: {len(decoder_params)} params, lr={decoder_lr}, weight_decay={decoder_weight_decay}")
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.training_args.encoder_lr,
                'weight_decay': self.training_args.encoder_weight_decay,
                'betas': encoder_betas,
            })
            group_info.append((len(param_groups) - 1, "Embedder"))
            self.accelerator.print(f"Embedder parameter group: {len(encoder_params)} params, lr={self.training_args.encoder_lr}, weight_decay={self.training_args.encoder_weight_decay}")
        
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': self.training_args.adapter_lr,
                'weight_decay': self.training_args.adapter_weight_decay,
                'betas': adapter_betas,
            })
            group_info.append((len(param_groups) - 1, "Adapter"))
            self.accelerator.print(f"Adapter parameter group: {len(adapter_params)} params, lr={self.training_args.adapter_lr}, weight_decay={self.training_args.adapter_weight_decay}")
        
        if not param_groups:
            raise ValueError("No trainable parameters found!")
            
        return param_groups, group_info
    
    def _parse_betas(self, betas_str: str) -> Tuple[float, float]:
        """Parse beta values from string format 'beta1,beta2'"""
        try:
            beta1, beta2 = betas_str.split(',')
            return (float(beta1.strip()), float(beta2.strip()))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid beta format '{betas_str}'. Expected 'beta1,beta2' (e.g., '0.9,0.999')") from e

    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                self.accelerator.print(f"{name}: {param.shape} {param.dtype}")
                trainable_params += num_params
        self.accelerator.print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def _print_model_config(self):
        """Print model configuration summary"""
        ma = self.model_args
        ta = self.training_args
        self.accelerator.print("=" * 50)
        self.accelerator.print("Model config:")
        self.accelerator.print(f"  pooling:          {ma.pooling}")
        self.accelerator.print(f"  encoder_mask_type:        {ma.encoder_mask_type}")
        self.accelerator.print(f"  encoder_window_size:   {ma.encoder_window_size}")
        self.accelerator.print(f"  boundary_overlap:         {ma.boundary_overlap}")
        self.accelerator.print(f"  num_adapter_layers:     {ma.num_adapter_layers}")
        self.accelerator.print(f"  compression_ratio:             {ta.compression_ratio}")
        self.accelerator.print("=" * 50)

    def _apply_lora(
        self,
        model,
        train_model: bool,
        lora: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Optional[List[str]],
        task_type: str = "CAUSAL_LM",
        modules_to_save: Optional[List[str]] = None,
    ):
        """Apply LoRA to model"""

        base_dtype = None
        if lora:
            # Capture existing floating point dtype so injected LoRA params can match it
            for param in model.parameters():
                if param.is_floating_point():
                    base_dtype = param.dtype
                    break

        if lora and train_model:
            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", 
                    "k_proj", 
                    "v_proj", 
                    "o_proj",
                    "gate_proj", 
                    "down_proj", 
                    "up_proj",
                ]
            if modules_to_save:
                # Deduplicate while preserving user intent order
                seen = set()
                modules_to_save = [m for m in modules_to_save if not (m in seen or seen.add(m))]

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=task_type,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, lora_cfg)

            if base_dtype is not None:
                for name, param in model.named_parameters():
                    if "lora_" in name and param.is_floating_point() and param.dtype != base_dtype:
                        param.data = param.data.to(dtype=base_dtype)

        if not train_model:
            for param in model.parameters():
                param.requires_grad = False
        return model
    
    def _update_training_loss_average(self, loss_value: float) -> float:
        """Update running average of training losses and return current average"""
        self.training_loss_window.append(loss_value)

        # Keep only the last window_size losses
        if len(self.training_loss_window) > self.window_size:
            self.training_loss_window.pop(0)

        # Calculate and return current average
        return sum(self.training_loss_window) / len(self.training_loss_window)

    def _unfreeze_llm_layers(self, decoder, num_layers: int):
        """Unfreeze the first N transformer layers of the LLM model.

        Args:
            decoder: The LLM model (Qwen3ForCausalLM)
            num_layers: Number of initial layers to unfreeze (e.g., 4 unfreezes layers 0-3)
        """
        # Qwen3/LLaMA structure: decoder.model.layers
        layers = decoder.model.layers
        total_layers = len(layers)

        if num_layers > total_layers:
            print(f"Warning: train_decoder_num_layers={num_layers} > total_layers={total_layers}. Training all {total_layers} layers.")
            num_layers = total_layers

        # Unfreeze the first N layers
        unfrozen_params = 0
        for i in range(num_layers):
            for name, param in layers[i].named_parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()

        print(f"Unfroze first {num_layers}/{total_layers} LLM transformer layers ({unfrozen_params:,} parameters)")

    def _parse_skip_step_ranges(self, skip_step_ranges: Optional[str]) -> List[Tuple[int, int]]:
        """Parse skip step ranges string into list of (start, end) tuples.

        Args:
            skip_step_ranges: Comma-separated ranges like "6500-6510,7000-7005"

        Returns:
            List of (start, end) tuples (inclusive on both ends)
        """
        if not skip_step_ranges:
            return []

        ranges = []
        for part in skip_step_ranges.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                ranges.append((int(start.strip()), int(end.strip())))
            else:
                # Single step
                step = int(part)
                ranges.append((step, step))

        return ranges

    def _should_skip_step(self, step: int) -> bool:
        """Check if the given step should be skipped.

        Args:
            step: The current global step

        Returns:
            True if this step should be skipped (no forward/backward pass)
        """
        for start, end in self._skip_step_ranges:
            if start <= step <= end:
                return True
        return False

    def _install_special_token_embedding_grad_mask(self) -> None:
        """Install a gradient mask hook so only special token rows receive gradients when LLM is frozen.

        Must be called after the model has been prepared/wrapped by the accelerator.
        """
        try:
            if self.model_args.train_decoder or not self.training_args.train_wrap_tokens:
                self.accelerator.print("Skipping special token embedding grad mask installation for LLM training or not training code tokens")
                return

            # If training full embed_tokens, don't mask - all embeddings should get gradients
            if self.model_args.train_decoder_embed_tokens:
                self.accelerator.print("Skipping special token embedding grad mask - training full LLM embed_tokens and first N layers")
                return

            # # Access the underlying model even if wrapped by DDP/Accelerate
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            decoder = unwrapped_model.decoder

            # Identify special token ids present in tokenizer
            special_tokens = ['<|memory_start|>', '<|memory_end|>', '<|memory|>']
            special_token_ids = [
                self.decoder_tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens
            ]
            special_token_ids = [tid for tid in special_token_ids]

            if not special_token_ids:
                return

            embedding_layer = decoder.get_input_embeddings()
            # print(f"Embedding layer shape: {embedding_layer.weight.shape}")

            def _make_grad_mask_hook(allowed_ids: torch.Tensor):
                def _hook(grad: torch.Tensor) -> torch.Tensor:
                    if grad is None:
                        return grad
                    mask = torch.zeros_like(grad)
                    mask[allowed_ids] = 1
                    # print(f"Mask shape: {mask.shape}")
                    # print(f"Grad shape: {grad.shape}")
                    # print(f"Last 5 mask values: {mask[-5:]}")
                    # print(f"Last 5 grad values: {grad[-5:]}")
                    return grad * mask
                return _hook

            allowed_ids_tensor = torch.tensor(
                special_token_ids,
                device=embedding_layer.weight.device,
                dtype=torch.long,
            )
            embedding_layer.weight.register_hook(_make_grad_mask_hook(allowed_ids_tensor))
            self.accelerator.print(
                f"Enabled grad mask for {len(special_token_ids)} special token embeddings after model prepare"
            )
        except Exception as exc:
            # Fail-soft: don't crash training if hook installation fails
            self.accelerator.print(f"Warning: failed to install grad mask hook: {exc}")
