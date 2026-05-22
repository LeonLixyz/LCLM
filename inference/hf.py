"""HuggingFace-Transformers inference for LCLM.

Exposes ``load_model`` and ``generate_text`` for loading an LCLM checkpoint
and running prompt-by-prompt generation on a single device.

For production serving with paged attention and continuous batching,
see ``inference.vllm.LCLMVLLMDecoder``.
"""

import json
import os

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from latent_context.model import LCLM
from latent_context.processor import LCLMProcessor


def load_model(
    checkpoint_path,
    device="cuda",
    max_memory_length=12288,
    dtype="bf16",
    **overrides,
):
    """Load an LCLM checkpoint.

    The directory must contain ``decoder/``, ``encoder/``, ``adapter/`` subdirs
    and a ``model_config.json`` written by ``utils.checkpointing.save_model_config``.
    All model-shape settings (``compression_ratio``, ``encoder_window_size``, ``pooling``,
    ``encoder_mask_type``, ``boundary_overlap``, ``adapter_type``, ``num_adapter_layers``)
    come from that config; pass ``**overrides`` only to override.
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Resolve HF repo IDs to local directories via snapshot_download.
    if not os.path.isdir(checkpoint_path):
        if "/" in checkpoint_path and not checkpoint_path.startswith("/"):
            from huggingface_hub import snapshot_download
            print(f"Downloading {checkpoint_path} from HF hub...")
            checkpoint_path = snapshot_download(repo_id=checkpoint_path, repo_type="model")
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Local directory with decoder/, encoder/, adapter/ subdirs.
    decoder_dir = os.path.join(checkpoint_path, "decoder")
    embed_dir = os.path.join(checkpoint_path, "encoder")
    projectors_dir = os.path.join(checkpoint_path, "adapter")

    if not all(os.path.exists(d) for d in [decoder_dir, embed_dir, projectors_dir]):
        candidates = [
            os.path.join(checkpoint_path, d) for d in os.listdir(checkpoint_path)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_path, d))
        ]
        if candidates:
            latest_checkpoint = sorted(candidates, key=lambda p: int(os.path.basename(p).split('_')[1]))[-1]
            decoder_dir = os.path.join(latest_checkpoint, "decoder")
            embed_dir = os.path.join(latest_checkpoint, "encoder")
            projectors_dir = os.path.join(latest_checkpoint, "adapter")
        else:
            raise FileNotFoundError(f"No valid checkpoint structure found in {checkpoint_path}")

    # Load model config and pull every shape setting from it (with `overrides`
    # taking precedence so callers can experiment without rewriting the file).
    from utils.checkpointing import load_model_config
    cfg = load_model_config(os.path.dirname(decoder_dir))
    cfg.update(overrides)

    compression_ratio          = cfg["compression_ratio"]
    encoder_window_size = cfg["encoder_window_size"]
    pooling             = cfg["pooling"]
    encoder_mask_type   = cfg["encoder_mask_type"]
    boundary_overlap    = cfg["boundary_overlap"]
    adapter_type        = cfg["adapter_type"]
    num_adapter_layers  = cfg.get("num_adapter_layers", 1)

    print(
        f"Loaded model_config: compression_ratio={compression_ratio}, encoder_window_size={encoder_window_size}, "
        f"pooling={pooling}, encoder_mask_type={encoder_mask_type}, "
        f"boundary_overlap={boundary_overlap}, adapter_type={adapter_type}, "
        f"num_adapter_layers={num_adapter_layers}"
    )

    print(f"Loading LLM from: {decoder_dir}")
    print(f"Loading embedder from: {embed_dir}")
    print(f"Loading projectors from: {projectors_dir}")

    # Set dtype and attention implementation. Flash attention is preferred
    # for bf16/fp16 but falls back to sdpa when the package isn't installed
    # (e.g. inside a vLLM-only image).
    if dtype == "fp32":
        torch_dtype = torch.float32
        attn_impl = "eager"
    else:
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
    print(f"dtype={dtype}, attn_implementation={attn_impl}")
    
    # Load tokenizers
    # Always use the same tokenizer regardless of LLM model.
    # Prefer local tokenizer files when available to support offline runs.
    LLM_TOKENIZER_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    decoder_tokenizer_source = decoder_dir
    if not os.path.isfile(os.path.join(decoder_dir, "tokenizer_config.json")):
        decoder_tokenizer_source = LLM_TOKENIZER_NAME
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_source)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    embed_tokenizer = AutoTokenizer.from_pretrained(embed_dir)
    
    # Load LLM model
    decoder_has_adapters = os.path.isfile(os.path.join(decoder_dir, "adapter_model.safetensors"))
    if decoder_has_adapters:
        # Load base model and then PEFT adapters
        config_path = os.path.join(decoder_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-0.6B')
        else:
            base_model_name = 'Qwen/Qwen3-0.6B'  # fallback
            
        decoder = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        decoder = PeftModel.from_pretrained(decoder, decoder_dir)
    else:
        # Load full model
        decoder = AutoModelForCausalLM.from_pretrained(
            decoder_dir,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    
    # Load embedding model
    embed_has_adapters = os.path.isfile(os.path.join(embed_dir, "adapter_model.safetensors"))
    if embed_has_adapters:
        # Load base model and then PEFT adapters
        config_path = os.path.join(embed_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-Embedding-0.6B')
        else:
            base_model_name = 'Qwen/Qwen3-Embedding-0.6B'  # fallback
            
        embed_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        embed_model = PeftModel.from_pretrained(embed_model, embed_dir)
    else:
        # Load full model
        embed_model = AutoModel.from_pretrained(
            embed_dir,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    
    # Create processor
    processor = LCLMProcessor(
        decoder_tokenizer=decoder_tokenizer,
        embed_tokenizer=embed_tokenizer,
        compression_ratio=compression_ratio,  # default, could be read from config
        max_memory_length=max_memory_length,  # default, could be read from config
        use_memory_wrapping=True,
    )
    
    # Create LCLM model
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
        pooling=pooling,
        encoder_mask_type=encoder_mask_type,
        encoder_window_size=encoder_window_size,
        boundary_overlap=boundary_overlap,
        adapter_type=adapter_type,
        num_adapter_layers=num_adapter_layers,
    )

    # Load adapter weights
    adapter_path = os.path.join(projectors_dir, "adapter.safetensors")
    if os.path.isfile(adapter_path):
        adapter_state = load_safetensors(adapter_path)
        model.adapter.load_state_dict(adapter_state, strict=True)
        print("Loaded code adapter weights")
    else:
        print(f"Warning: adapter weights not found at {adapter_path}")
    
    model.to(device)
    model.to(torch_dtype)
    model.eval()
    print("Model loaded successfully!")
    print(model)

    # Print model dtype info
    print(f"\n{'='*60}")
    print("Model dtype info:")
    print(f"  LLM dtype: {next(model.decoder.parameters()).dtype}")
    print(f"  Embedder dtype: {next(model.encoder.embed_model.parameters()).dtype}")
    print(f"  Adapter dtype: {next(model.adapter.parameters()).dtype}")
    print(f"{'='*60}\n")
    
    return model, decoder_tokenizer, processor


def generate_text(model, decoder_tokenizer, processor, prompt, device="cuda", max_tokens=512, temperature=0.7):
    """Generate text given a prompt that already contains <|memory_start|>...<|memory_end|> tags"""
    print(f"\n{'='*60}")

    # Prompt should already contain <|memory_start|>...<|memory_end|> tags
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    print(formatted_prompt)
    print(f"\n{'='*60}")
    
    with torch.inference_mode():
        # Use processor to handle code integration
        processed = processor.process_wrapped_batch(
            prompts=[formatted_prompt],
            targets=None,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = processed['input_ids'].to(device)
        attention_mask = processed['attention_mask'].to(device)
        memory_positions = processed['memory_positions']
        latent_counts = processed['latent_counts']
        memory_token_ids = processed['memory_token_ids']

        # Generate with code context
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_token_ids=memory_token_ids,
            memory_positions=memory_positions,
            latent_counts=latent_counts,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            do_sample=temperature > 0,
            top_k=20,
            min_p=0,
            repetition_penalty=1.1,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = decoder_tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Remove prompt from output
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
    
    print("OUTPUT:")
    print(generated_text)
    print(f"{'='*60}\n")
    
    return generated_text
