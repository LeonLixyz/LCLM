#!/bin/bash
# Full training pipeline: runs all 4 stages with checkpoint conversion
#
# Usage:
#   OUTPUT_DIR=./checkpoints bash scripts/run_pipeline.sh config.yaml
#
# Environment variables:
#   OUTPUT_DIR  - Checkpoint output directory (required)
#   DS_HOSTFILE - DeepSpeed hostfile for multi-node (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG_FILE="${1:-scripts/experiment_config/0.6b-4b-cs16-mean-w1024-bidirectional-mlp-O0.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Required env var
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR environment variable is required}"

############################
# Read config
############################
EMBED_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models']['encoder'])")
LLM_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models']['decoder'])")
COMPRESSION_RATIO=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['training']['compression_ratio'])")
ADAPTER_TYPE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models'].get('adapter_type', 'mlp'))")
NUM_ADAPTER_LAYERS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models'].get('num_adapter_layers', 1))")
POOLING=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models'].get('pooling', 'eos'))")
ENCODER_MASK_TYPE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models'].get('mask', 'causal'))")
ENCODER_WINDOW_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['models'].get('encoder_window_size', 256))")
BOUNDARY_OVERLAP=$(python -c "import yaml; v = yaml.safe_load(open('$CONFIG_FILE'))['models'].get('boundary_overlap'); print(0 if v is None else v)")

# Function to get distributed type for a specific stage
get_dist_type() {
    local stage="$1"
    python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG_FILE'))
stages = cfg.get('stages', {})
stage_cfg = stages.get($stage, stages.get('$stage', {}))
dist_type = stage_cfg.get('distributed', {}).get('type', 'deepspeed')
print(dist_type)
"
}

# Generate experiment name (auto-generate if "auto" or empty)
EXPERIMENT=$(PYTHONPATH=. python -c "import sys, yaml; sys.path.insert(0, '.'); from train.launch_train import generate_experiment_name; cfg = yaml.safe_load(open('$CONFIG_FILE')); exp = cfg.get('experiment', 'auto'); print(exp if exp and exp != 'auto' else generate_experiment_name(cfg))")

echo "=========================================="
echo "Training Pipeline"
echo "=========================================="
echo "Config:     $CONFIG_FILE"
echo "Experiment: $EXPERIMENT"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="

############################
# Helper functions
############################

find_latest_checkpoint() {
    local stage="$1"
    local prefix="${OUTPUT_DIR}/${EXPERIMENT}/stage${stage}"
    if [ -d "${prefix}_final" ]; then echo "${prefix}_final"; return; fi
    local latest="" latest_step=0
    for ckpt in "${prefix}_step_"*; do
        if [ -d "$ckpt" ]; then
            step=$(echo "$ckpt" | grep -oP '_step_\K\d+' || true)
            if [ -n "$step" ] && [ "$step" -gt "$latest_step" ]; then
                latest_step="$step"; latest="$ckpt"
            fi
        fi
    done
    echo "$latest"
}

stage_complete() {
    local stage="$1"
    local hf_path="${OUTPUT_DIR}/${EXPERIMENT}/stage${stage}-hf"
    [ -d "$hf_path" ] && [ -f "$hf_path/model_config.json" ]
}

run_conversion() {
    # Pick rank from whatever launcher set (torchrun, slurm, mpi, etc.)
    local rank="${RANK:-${LOCAL_RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}}}"

    local stage="$1"
    local checkpoint=$(find_latest_checkpoint "$stage")
    local hf_output="${OUTPUT_DIR}/${EXPERIMENT}/stage${stage}-hf"
    local dist_type=$(get_dist_type "$stage")

    if [ -z "$checkpoint" ]; then
        echo "ERROR: No checkpoint for stage $stage"; return 1
    fi

    # Barrier files (must be on a filesystem visible to ALL ranks/nodes)
    local barrier_dir="${OUTPUT_DIR}/${EXPERIMENT}/.barriers"
    mkdir -p "$barrier_dir"

    # Make barrier unique to this stage+checkpoint so reruns don't see stale markers
    local ck_tag="$(basename "$checkpoint" | tr '/: ' '___')"
    local done_file="${barrier_dir}/stage${stage}.${ck_tag}.done"
    local fail_file="${barrier_dir}/stage${stage}.${ck_tag}.fail"

    if [[ "$rank" == "0" ]]; then
        # Clear any previous markers for this exact stage+checkpoint
        rm -f "$done_file" "$fail_file"

        echo ""
        echo "Converting: $checkpoint -> $hf_output"

        # Build convert args
        CONVERT_ARGS=(
            "$checkpoint" "$hf_output"
            --type "$dist_type"
            --embed_model "$EMBED_MODEL"
            --decoder "$LLM_MODEL"
            --compression_ratio "$COMPRESSION_RATIO"
            --adapter_type "$ADAPTER_TYPE"
            --num_adapter_layers "$NUM_ADAPTER_LAYERS"
            --pooling "$POOLING"
            --encoder_mask_type "$ENCODER_MASK_TYPE"
            --encoder_window_size "$ENCODER_WINDOW_SIZE"
            --boundary_overlap "$BOUNDARY_OVERLAP"
        )
        bash "$SCRIPT_DIR/convert_checkpoint.sh" "${CONVERT_ARGS[@]}"
        local rc=$?

        if [[ $rc -eq 0 ]]; then
            touch "$done_file"
        else
            echo "$rc" > "$fail_file"
        fi
        return $rc
    fi

    # Non-zero ranks: wait until rank 0 finishes
    echo "Rank $rank waiting for conversion barrier (stage=$stage, ck=$ck_tag)..."
    while [[ ! -f "$done_file" && ! -f "$fail_file" ]]; do
        sleep 2
    done

    if [[ -f "$fail_file" ]]; then
        echo "Conversion failed on rank 0 (rc=$(cat "$fail_file" 2>/dev/null || echo 1))."
        return 1
    fi

    echo "Barrier passed (rank 0 finished conversion)."
    return 0
}

############################
# Run Pipeline
############################

# Get list of stages defined in config (sorted)
DEFINED_STAGES=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG_FILE'))
stages = cfg.get('stages', {})
# Sort and print space-separated
print(' '.join(str(s) for s in sorted(stages.keys())))
")

if [ -z "$DEFINED_STAGES" ]; then
    echo "ERROR: No stages defined in config file"
    exit 1
fi

echo "Stages defined in config: $DEFINED_STAGES"

# Convert to array
read -ra STAGES_ARRAY <<< "$DEFINED_STAGES"
FIRST_STAGE="${STAGES_ARRAY[0]}"

for stage in $DEFINED_STAGES; do
    echo ""
    echo "==================== STAGE $stage ===================="

    if stage_complete "$stage"; then
        echo "Stage $stage complete, skipping..."
        continue
    fi

    # For first defined stage, check if resume_from_checkpoint is specified (allows skipping earlier stages)
    # For subsequent stages, require previous stage to be complete
    if [ "$stage" -gt "$FIRST_STAGE" ] && ! stage_complete "$((stage-1))"; then
        echo "ERROR: Stage $((stage-1)) not complete"; exit 1
    fi

    # Training
    echo ""
    echo "=========================================="
    echo "TRAINING STAGE $stage"
    echo "=========================================="
    bash "$SCRIPT_DIR/train.sh" "$CONFIG_FILE" "$stage"

    # Conversion
    run_conversion "$stage"
    echo "Stage $stage done!"
done

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
