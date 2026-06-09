#!/usr/bin/env python3
"""
Clean checkpointing with safetensors and folder structure.

CHECKPOINT FORMAT:
------------------
This module saves checkpoints in Accelerator state format for auto-resume:
- Uses accelerator.save_state() to save full training state
- Includes: model weights, optimizer state, scheduler state, dataloader state
- Used by auto_resume to recover from GPU cluster preemptions

For HuggingFace format checkpoints (model weights only):
- Manually export with decoder/, encoder/, adapter/ subdirectories
- Used by resume_from_checkpoint to start from pre-trained model
- Does NOT include optimizer/scheduler state
"""

import os
import json
import shutil
import re
import glob
from typing import Optional, Dict, Any, Tuple
from dataclasses import asdict
from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoModel
from huggingface_hub import HfApi
import torch


def save_model_config(
    output_dir: str,
    *,
    compression_ratio: int,
    encoder_window_size: int,
    pooling: str,
    encoder_mask_type: str,
    boundary_overlap: int,
    adapter_type: str,
    num_adapter_layers: int = 1,
    # Accepted for API symmetry but unused — the actual encoder/decoder
    # weights live in the {encoder,decoder}/ subdirs.
    encoder_name: str | None = None,
    decoder_name: str | None = None,
) -> str:
    """Write the minimal ``model_config.json`` for an LCLM checkpoint.

    Only the fields the inference loader needs to reconstruct the model:
    pooling, window/overlap/compression sizing, encoder mask type, and
    adapter shape. ``num_adapter_layers`` is only included when the adapter
    actually has attention layers (``adapter_type != "mlp"``).
    """
    cfg = {
        "compression_ratio": compression_ratio,
        "encoder_window_size": encoder_window_size,
        "pooling": pooling,
        "encoder_mask_type": encoder_mask_type,
        "boundary_overlap": boundary_overlap,
        "adapter_type": adapter_type,
    }
    if adapter_type != "mlp":
        cfg["num_adapter_layers"] = num_adapter_layers
    path = os.path.join(output_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def load_model_config(checkpoint_path: str) -> Dict[str, Any]:
    """Read ``model_config.json`` from a checkpoint dir. Raises if missing."""
    p = os.path.join(checkpoint_path, "model_config.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No model_config.json at {p}")
    with open(p) as f:
        return json.load(f)

def save_checkpoint(
    *,
    accelerator,
    model,
    decoder_tokenizer,
    embed_tokenizer,
    global_step: int,
    output_dir: str,
    model_args: Any,
    data_args: Any,
    training_args: Any,
    run_name: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
    keep_latest_n: int = 1,
    is_final: bool = False,
    train_dataloader=None,  # Optional: for StatefulDataLoader state saving
    delete_old_checkpoints: bool = True,  # If False, keep all checkpoints
):
    """
    Save checkpoint and clean up old checkpoints to keep only the latest N.

    Args:
        keep_latest_n: Number of latest checkpoints to keep (default: 3)
        is_final: If True, save as {run_name}_final instead of {run_name}_step_{global_step}
        train_dataloader: Optional dataloader to save state (for StatefulDataLoader)
    """

    # Use run_name prefix to avoid overlap between experiments
    if is_final:
        suffix = "final"
    else:
        suffix = f"step_{global_step}"

    if run_name:
        checkpoint_dir = os.path.join(output_dir, f"{run_name}_{suffix}")
    else:
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_{suffix}")

    # Save the checkpoint with all states (model, optimizer, scheduler, dataloader)
    accelerator.save_state(checkpoint_dir)

    # Ensure zero_to_fp32.py is present for DeepSpeed checkpoints
    zero_script_dst = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if not os.path.exists(zero_script_dst):
        zero_script_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "zero_to_fp32.py")
        if os.path.exists(zero_script_src):
            shutil.copy(zero_script_src, zero_script_dst)

    # Save StatefulDataLoader state separately (not tracked by accelerator.prepare)
    if train_dataloader is not None and hasattr(train_dataloader, 'state_dict'):
        dl_state = train_dataloader.state_dict()
        dl_state_path = os.path.join(checkpoint_dir, f"dataloader_state_{accelerator.process_index}.pt")
        torch.save(dl_state, dl_state_path)

    # Save training metadata for resumption
    if accelerator.is_main_process:
        metadata = {
            "global_step": global_step,
            "run_name": run_name,
            **(extra_metadata or {}),
        }
        metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    accelerator.wait_for_everyone()
    # Clean up old checkpoints (only on main process to avoid race conditions)
    if accelerator.is_main_process and delete_old_checkpoints:
        ### UPLOAD TO HF
        # if global_step % 1000 == 0:
        #     try:
        #         token = _get_hf_token()
        #         username = _get_hf_username(token=token)
        #         # Use run_name as prefix if provided, otherwise fallback to component name
        #         if run_name:
        #             repo_id = f"{username}/{run_name}_checkpoint_{global_step}"
        #         else:
        #             component = _determine_component(model_args)
        #             repo_id = f"{username}/{component}_checkpoint_{global_step}"
        #         rank = getattr(accelerator, "process_index", 0)
                
        #         # Upload the entire checkpoint folder, preserving structure
        #         print(f"[rank {rank}] Uploading checkpoint folder to HF: {repo_id}")
        #         _upload_folder_to_hf(folder_path=checkpoint_dir, repo_id=repo_id, token=token)
        #     except Exception as e:
        #         print(f"Warning: Failed HF upload setup: {e}")
        
        _cleanup_old_checkpoints(output_dir, keep_latest_n, run_name_prefix=run_name)
    
    return checkpoint_dir


def find_latest_checkpoint(output_dir: str, run_name: str = "") -> Optional[Tuple[str, int]]:
    """
    Find the latest checkpoint directory in output_dir.
    
    Args:
        output_dir: Directory to search for checkpoints
        run_name: Optional run name prefix to filter checkpoints
    
    Returns:
        Tuple of (checkpoint_path, global_step) or None if no checkpoint found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Pattern to match checkpoints with optional run_name prefix
    if run_name:
        checkpoint_pattern = re.compile(rf'^{re.escape(run_name)}_step_(\d+)$')
    else:
        checkpoint_pattern = re.compile(r'^checkpoint_(\d+)$')
    
    checkpoints = []
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, item_path))
    
    if not checkpoints:
        return None
    
    # Return the checkpoint with the highest step number
    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoints[-1]
    
    return latest_path, latest_step

def _cleanup_old_checkpoints(output_dir: str, keep_latest_n: int, run_name_prefix: str = ""):
    """
    Remove old checkpoints, keeping only the latest N.
    Only keeps checkpoints with pattern checkpoint_{step} or {run_name}_step_{step}.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_latest_n: Number of latest checkpoints to keep
        run_name_prefix: Optional run name prefix to filter checkpoints
    """
    try:
        # Find all checkpoint directories with pattern matching run_name
        if run_name_prefix:
            checkpoint_pattern = re.compile(rf'^{re.escape(run_name_prefix)}_step_(\d+)$')
        else:
            checkpoint_pattern = re.compile(r'^checkpoint_(\d+)$')
        
        checkpoints = []
        
        if not os.path.exists(output_dir):
            return
            
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    step = int(match.group(1))
                    checkpoints.append((step, item_path))
        
        # Sort by step number (ascending)
        checkpoints.sort(key=lambda x: x[0])
        
        # Keep only the latest N checkpoints
        if len(checkpoints) > keep_latest_n:
            checkpoints_to_remove = checkpoints[:-keep_latest_n]
            
            for step, checkpoint_path in checkpoints_to_remove:
                print(f"Removing old checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path, ignore_errors=True)
                
        print(f"Checkpoint cleanup complete. Keeping {min(len(checkpoints), keep_latest_n)} most recent checkpoints.")
        
    except Exception as e:
        print(f"Warning: Failed to clean up old checkpoints: {e}")


def _determine_component(model_args: Any) -> str:
    """
    Determine which component is being trained to name the HF repo.
    - If both train_decoder and train_encoder are False -> adapter
    - If train_encoder True and train_decoder False -> embedder
    - Otherwise -> decoder
    """
    try:
        train_decoder = bool(getattr(model_args, "train_decoder"))
    except Exception:
        train_decoder = False
    try:
        train_encoder = bool(getattr(model_args, "train_encoder"))
    except Exception:
        train_encoder = False
    
    if not train_decoder and not train_encoder:
        return "adapter"
    if train_encoder and not train_decoder:
        return "encoder"
    return "decoder"


def _get_hf_username(token: Optional[str] = None) -> str:
    """Get the HF username from the current login, env var HF_USERNAME, or raise."""
    try:
        api = HfApi(token=token) if token else HfApi()
        info = api.whoami()
        if isinstance(info, dict) and "name" in info and info["name"]:
            return info["name"]
        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict) and info[0].get("name"):
            return info[0]["name"]
    except Exception:
        pass
    username = os.environ.get("HF_USERNAME")
    if not username:
        raise RuntimeError(
            "Could not resolve HF username. Set HF_USERNAME env var or `huggingface-cli login`."
        )
    return username


def _upload_folder_to_hf(*, folder_path: str, repo_id: str, token: Optional[str] = None):
    """
    Create (if needed) and upload a directory to a HF model repo.
    """
    try:
        api = HfApi(token=token) if token else HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder_path,
            commit_message=f"Add {os.path.basename(folder_path)}",
            token=token,
        )
        print(f"✓ Uploaded checkpoint to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Warning: Failed to upload checkpoint to HF: {e}")


def _upload_file_to_hf(*, file_path: str, repo_id: str, token: Optional[str] = None):
    """
    Create (if needed) and upload a single file to a HF model repo.
    """
    try:
        api = HfApi(token=token) if token else HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {os.path.basename(file_path)}",
            token=token,
        )
        print(f"✓ Uploaded {os.path.basename(file_path)} to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Warning: Failed to upload file to HF: {e}")


def _get_hf_token() -> Optional[str]:
    """
    Get HF token from common environment variable names, or None if not present.
    """
    return (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )
