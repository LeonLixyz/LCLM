python /usr/workspace/mcleish1/llnl-tools/launch_tuo.py \
    --env_act_style=conda_activate \
    --rccl_installdir=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.4.1/install/lib \
    --output_dir=/usr/workspace/mcleish1/LCLM/tuo_outputs_prod \
    --rocm_version=6.4.2 \
    --run_name=tuo-prod-0.6b-4b-cs16-mean-w1024-mlp-ov0-causal-stage3-final-mixture-20260910-16k-from-stage2-lr3e-5 \
    --nodes=32 \
    --minutes=1440 \
    --repetitions=2 \
    --launch_once_per_node=true \
    --custom_invocation='bash sean_shells/run_pipeline_tuo.sh /usr/workspace/mcleish1/LCLM/sean_shells/experiment_config/0.6b-4b-cs16-mean-w1024-mlp-ov0-causal-stage3-final-mixture-20260910-16k-from-stage2-lr3e-5.yaml' --bank=effml

python /usr/workspace/mcleish1/llnl-tools/launch_tuo.py \
    --env_act_style=conda_activate \
    --rccl_installdir=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.4.1/install/lib \
    --output_dir=/usr/workspace/mcleish1/LCLM/tuo_outputs_prod \
    --rocm_version=6.4.2 \
    --run_name=tuo-prod-0.6b-4b-cs16-mean-w1024-mlp-ov0-causal-stage3-final-mixture-20260910-32k-from-stage2-lr3e-5 \
    --nodes=32 \
    --minutes=1440 \
    --repetitions=2 \
    --launch_once_per_node=true \
    --custom_invocation='bash sean_shells/run_pipeline_tuo.sh /usr/workspace/mcleish1/LCLM/sean_shells/experiment_config/0.6b-4b-cs16-mean-w1024-mlp-ov0-causal-stage3-final-mixture-20260910-32k-from-stage2-lr3e-5.yaml' --bank=effml
