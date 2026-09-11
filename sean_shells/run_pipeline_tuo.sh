#!/usr/bin/env bash
set -euo pipefail

export NUM_MACHINES=$((WORLD_SIZE / 4))
export MACHINE_RANK=$((RANK / 4))

JOB_TAG=${SLURM_JOB_ID:-${JOB_ID:-$$}}
export TRITON_CACHE_BASE="/dev/shm/triton_cache_${JOB_TAG}_${MASTER_PORT}"
export TRITON_CACHE_DIR="$TRITON_CACHE_BASE/cache"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/inductor_cache_${JOB_TAG}_${MASTER_PORT}"
export TRITON_HOME="/dev/shm/triton_home_${JOB_TAG}_${MASTER_PORT}" # https://docs.cscs.ch/software/ml/pytorch/#running-pytorch-jobs-with-slurm

mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TRITON_HOME"

### hostfile creation for deepspeed multinode nossh launcher ###
HOSTFILE="$OUTPUT_DIR/ds_hostfile_${FLUX_JOB_ID}.txt"
echo "hostfile: $HOSTFILE"

if [[ $RANK == 0 ]]; then
    echo "rank 0 hit" 
    # pick a GPUs-per-node value (this varies by cluster config)
    echo "rank 0 statting"
    rm -f "$HOSTFILE"
    echo "rank 0 rm'd"

    flux hostlist -ed $'\n' "$(flux getattr hostlist)" |
    while IFS= read -r host; do
        echo "$host slots=$GPUS_PER_NODE" >> "$HOSTFILE"
    done
    echo "Wrote hostfile to $HOSTFILE"
fi
while [ ! -s $HOSTFILE ]
do
    sleep 10
done

export DS_HOSTFILE="$HOSTFILE"
export WANDB_DIR=$OUTPUT_DIR
export WANDB_NAME=$RUN_NAME
export OUTPUT_DIR=$OUTPUT_DIR/checkpoints

exec scripts/run_pipeline.sh "$@"
