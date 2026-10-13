#!/bin/bash
# Simple training script template
#
# Usage:
#   bash scripts/train.sh <config.yaml> <stage>
#
# Environment variables:
#   OUTPUT_DIR    - Checkpoint output directory (required)
#   DS_HOSTFILE   - DeepSpeed hostfile for multi-node (optional)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config.yaml> <stage>"
    echo "Example: OUTPUT_DIR=./checkpoints $0 scripts/config/pipeline_config.yaml 0"
    exit 1
fi

CONFIG_FILE="$1"
STAGE="$2"

# Required env vars
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR environment variable is required}"

################################
# Python path
################################
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"

################################
# Distributed env (from SLURM or manual, with auto-detect fallback)
################################
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"

# Detect local capacity before calculating the global process count.
if [ -z "${GPUS_PER_NODE:-}" ]; then
    GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

require_positive_int() {
    if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "$1 must be a positive integer (got '$2')" >&2
        exit 2
    fi
}
require_positive_int NUM_MACHINES "$NUM_MACHINES"
require_positive_int GPUS_PER_NODE "$GPUS_PER_NODE"
WORLD_SIZE="${WORLD_SIZE:-$((NUM_MACHINES * GPUS_PER_NODE))}"
require_positive_int WORLD_SIZE "$WORLD_SIZE"
if (( WORLD_SIZE % NUM_MACHINES != 0 )); then
    echo "WORLD_SIZE must be divisible by NUM_MACHINES" >&2
    exit 2
fi
if ! [[ "$MACHINE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( MACHINE_RANK >= NUM_MACHINES )); then
    echo "MACHINE_RANK must be between 0 and NUM_MACHINES - 1" >&2
    exit 2
fi
if (( NUM_MACHINES > 1 )) && [[ "$MASTER_ADDR" == localhost || "$MASTER_ADDR" == 127.0.0.1 || "$MASTER_ADDR" == ::1 ]]; then
    echo "Set MASTER_ADDR to a rank-0 address reachable from every node" >&2
    exit 2
fi

# Optional env vars
DS_HOSTFILE="${DS_HOSTFILE:-}"

echo "=========================================="
echo "Training Stage $STAGE"
echo "=========================================="
echo "Config:      $CONFIG_FILE"
echo "Output:      $OUTPUT_DIR"
echo ""
echo "Distributed:"
echo "  MASTER_ADDR   = $MASTER_ADDR"
echo "  MASTER_PORT   = $MASTER_PORT"
echo "  NUM_MACHINES  = $NUM_MACHINES"
echo "  WORLD_SIZE    = $WORLD_SIZE"
echo "  MACHINE_RANK  = $MACHINE_RANK"
echo "  GPUS_PER_NODE = $GPUS_PER_NODE"
echo "=========================================="

################################
# Triton Cache Setup
################################
JOB_TAG=${SLURM_JOB_ID:-${JOB_ID:-$$}}
RANK_ID=${SLURM_PROCID:-${LOCAL_RANK:-${RANK:-0}}}
TRITON_CACHE_BASE=${TRITON_CACHE_BASE:-/tmp/triton_cache}
TRITON_RUN_DIR="${TRITON_CACHE_BASE%/}/triton_${JOB_TAG}_rank${RANK_ID}"
export TRITON_HOME="$TRITON_RUN_DIR"
export TRITON_CACHE_DIR="$TRITON_RUN_DIR/cache"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TRITON_HOME/autotune"

export TOKENIZERS_PARALLELISM=false

################################
# Get accelerate config and distributed type from YAML (stage-specific with fallback to global)
################################
read DIST_CONFIG DIST_TYPE <<< $(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG_FILE'))
stage = cfg['stages'][$STAGE]
dist = stage.get('distributed', cfg.get('distributed', {}))
config = dist.get('config', './scripts/distributed_configs/deepspeed_zero1_multi_node.yaml')
dist_type = dist.get('type', 'deepspeed')
print(config, dist_type)
")

echo "  DIST_TYPE     = $DIST_TYPE"
echo "  DIST_CONFIG   = $DIST_CONFIG"
if [ "$DIST_TYPE" = "deepspeed" ] && [ -n "$DS_HOSTFILE" ]; then
    echo "  DS_HOSTFILE   = $DS_HOSTFILE"
fi
echo "=========================================="

################################
# Run training
################################
# Build accelerate command with optional hostfile
ACCELERATE_ARGS=(
    --config_file "$DIST_CONFIG"
    --num_machines "$NUM_MACHINES"
    --num_processes "$WORLD_SIZE"
    --machine_rank "$MACHINE_RANK"
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
)

# Add hostfile for multi-node DeepSpeed if specified
if [ -n "$DS_HOSTFILE" ] && [ "$DIST_TYPE" = "deepspeed" ]; then
    ACCELERATE_ARGS+=(--deepspeed_hostfile "$DS_HOSTFILE")
fi

accelerate launch "${ACCELERATE_ARGS[@]}" \
    train/launch_train.py \
    --config "$CONFIG_FILE" \
    --stage "$STAGE" \
    --output_dir "$OUTPUT_DIR"
