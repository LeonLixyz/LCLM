#!/bin/bash
# Unified checkpoint conversion script (supports DeepSpeed and FSDP)
# Auto-detects checkpoint type and uses appropriate conversion method
#
# Usage: ./scripts/convert_checkpoint.sh <checkpoint_dir> <output_hf_dir> [options]
#
# Example:
#   ./scripts/convert_checkpoint.sh checkpoints/checkpoint_10000 checkpoints/0.6b-4b-adapter-hf

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <output_hf_dir> [options]"
    echo ""
    echo "Options:"
    echo "  --type TYPE              Checkpoint type: fsdp or deepspeed (auto-detected if not specified)"
    echo "  --embed_model MODEL      Embedding model name"
    echo "  --decoder MODEL        LLM model name"
    echo "  --compression_ratio N           Chunk size for code processing"
    echo "  --max_memory_length N      Maximum code length"
    echo "  --wrap_code BOOL         Whether to use code wrapping tokens"
    echo "  --adapter_type TYPE      Adapter shape: mlp | mlp_attn | attn_mlp"
    echo "  --num_adapter_layers N   Number of adapter attention layers"
    echo "  --pooling TOKEN          Pooling strategy: mean, eos, concat"
    echo "  --encoder_mask_type TYPE   Embedder attention: causal or bidirectional"
    echo "  --encoder_window_size N Tokens per batch for summary modes"
    echo "  --boundary_overlap N       Overlap tokens for tiled processing"
    echo ""
    echo "Example:"
    echo "  $0 checkpoints/checkpoint_10000 checkpoints/hf --type fsdp"
    exit 1
fi

CHECKPOINT_DIR="$1"
OUTPUT_HF_DIR="$2"
shift 2

# Default values
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-16}"
MAX_CODE_LENGTH="${MAX_CODE_LENGTH:-8192}"
WRAP_CODE="${WRAP_CODE:-True}"
ADAPTER_TYPE="${ADAPTER_TYPE:-mlp}"
NUM_ADAPTER_LAYERS="${NUM_ADAPTER_LAYERS:-1}"
POOLING="${POOLING:-eos}"
ENCODER_MASK_TYPE="${ENCODER_MASK_TYPE:-causal}"
ENCODER_WINDOW_SIZE="${ENCODER_WINDOW_SIZE:-256}"
BOUNDARY_OVERLAP="${BOUNDARY_OVERLAP:-0}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-}"  # fsdp or deepspeed, auto-detect if empty

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --embed_model)
            EMBED_MODEL="$2"
            shift 2
            ;;
        --decoder)
            LLM_MODEL="$2"
            shift 2
            ;;
        --compression_ratio)
            COMPRESSION_RATIO="$2"
            shift 2
            ;;
        --max_memory_length)
            MAX_CODE_LENGTH="$2"
            shift 2
            ;;
        --wrap_code)
            WRAP_CODE="$2"
            shift 2
            ;;
        --type)
            CHECKPOINT_TYPE="$2"
            shift 2
            ;;
        --adapter_type)
            ADAPTER_TYPE="$2"
            shift 2
            ;;
        --num_adapter_layers)
            NUM_ADAPTER_LAYERS="$2"
            shift 2
            ;;        --pooling)
            POOLING="$2"
            shift 2
            ;;
        --encoder_mask_type)
            ENCODER_MASK_TYPE="$2"
            shift 2
            ;;
        --encoder_window_size)
            ENCODER_WINDOW_SIZE="$2"
            shift 2
            ;;
        --boundary_overlap)
            BOUNDARY_OVERLAP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Checkpoint Conversion"
echo "=========================================="
echo "  Input checkpoint:  $CHECKPOINT_DIR"
echo "  Output HF dir:     $OUTPUT_HF_DIR"
echo "  Embed model:       $EMBED_MODEL"
echo "  LLM model:         $LLM_MODEL"
echo "  Chunk size:        $COMPRESSION_RATIO"
echo "  Max code length:   $MAX_CODE_LENGTH"
echo "  Wrap code:         $WRAP_CODE"
echo "  Pooling:           $POOLING"
echo "  Encoder mask:      $ENCODER_MASK_TYPE"
echo "  Encoder window W:  $ENCODER_WINDOW_SIZE"
echo "  Adapter type:      $ADAPTER_TYPE"
echo "  Adapter layers:    $NUM_ADAPTER_LAYERS"
echo "  Boundary overlap:  ${BOUNDARY_OVERLAP:-0}"
echo "=========================================="

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

# Check if already converted
if [ -d "$OUTPUT_HF_DIR" ] && [ -f "$OUTPUT_HF_DIR/model_config.json" ]; then
    echo "HF checkpoint already exists at $OUTPUT_HF_DIR, skipping conversion."
    exit 0
fi

# Determine checkpoint type (use provided --type or auto-detect)
if [ -n "$CHECKPOINT_TYPE" ]; then
    echo "Using specified checkpoint type: $CHECKPOINT_TYPE"
else
    # Auto-detect: FSDP has pytorch_model_fsdp_0/, DeepSpeed has zero_to_fp32.py
    if [ -d "$CHECKPOINT_DIR/pytorch_model_fsdp_0" ]; then
        CHECKPOINT_TYPE="fsdp"
        echo "Auto-detected checkpoint type: FSDP"
    elif [ -f "$CHECKPOINT_DIR/zero_to_fp32.py" ]; then
        CHECKPOINT_TYPE="deepspeed"
        echo "Auto-detected checkpoint type: DeepSpeed"
    else
        echo "ERROR: Could not detect checkpoint type. Use --type fsdp or --type deepspeed"
        exit 1
    fi
fi

if [ "$CHECKPOINT_TYPE" = "fsdp" ]; then
    ################################
    # FSDP Conversion
    ################################
    echo ""
    echo "Converting FSDP checkpoint..."
    echo "=========================================="

    # Build args for FSDP conversion
    FSDP_ARGS=(
        --fsdp_checkpoint "$CHECKPOINT_DIR"
        --output_dir "$OUTPUT_HF_DIR"
        --embed_model "$EMBED_MODEL"
        --decoder "$LLM_MODEL"
        --compression_ratio "$COMPRESSION_RATIO"
        --max_memory_length "$MAX_CODE_LENGTH"
        --wrap_code "$WRAP_CODE"
        --pooling "$POOLING"
        --encoder_mask_type "$ENCODER_MASK_TYPE"
        --encoder_window_size "$ENCODER_WINDOW_SIZE"
        --adapter_type "$ADAPTER_TYPE"
        --num_adapter_layers "$NUM_ADAPTER_LAYERS"
    )
    FSDP_ARGS+=(--boundary_overlap "$BOUNDARY_OVERLAP")

    python -m utils.convert_fsdp_to_hf "${FSDP_ARGS[@]}"

else
    ################################
    # DeepSpeed Conversion
    ################################
    PYTORCH_TEMP_DIR="${CHECKPOINT_DIR}/converted_output"

    echo ""
    echo "STEP 1: Extract fp32 weights from DeepSpeed"
    echo "=========================================="

    # Check if already extracted
    if [ -d "$PYTORCH_TEMP_DIR" ] && [ -f "$PYTORCH_TEMP_DIR/pytorch_model.bin.index.json" ]; then
        echo "PyTorch weights already extracted, skipping..."
    else
        cd "$CHECKPOINT_DIR" || exit 1
        python zero_to_fp32.py . ./converted_output --tag pytorch_model
        cd - > /dev/null || exit 1
    fi

    echo ""
    echo "STEP 2: Load, check, and save to HuggingFace"
    echo "=========================================="

    # Build args for DeepSpeed conversion
    DS_ARGS=(
        --pytorch_checkpoint_dir "$PYTORCH_TEMP_DIR"
        --output_dir "$OUTPUT_HF_DIR"
        --embed_model "$EMBED_MODEL"
        --decoder "$LLM_MODEL"
        --compression_ratio "$COMPRESSION_RATIO"
        --max_memory_length "$MAX_CODE_LENGTH"
        --wrap_code "$WRAP_CODE"
        --pooling "$POOLING"
        --encoder_mask_type "$ENCODER_MASK_TYPE"
        --encoder_window_size "$ENCODER_WINDOW_SIZE"
        --adapter_type "$ADAPTER_TYPE"
        --num_adapter_layers "$NUM_ADAPTER_LAYERS"
    )
    DS_ARGS+=(--boundary_overlap "$BOUNDARY_OVERLAP")

    python -m utils.load_zero_torch "${DS_ARGS[@]}"
fi

echo ""
echo "Done! HuggingFace model saved to: $OUTPUT_HF_DIR"
