export CUDA_VISIBLE_DEVICES="0" ; python main.py \
    --name OffTRC-Cassie \
    --seed 1 \
    --env_name Cassie-v0 \
    --total_steps 4000000 \
    --limit_values "0.025, 0.025, 0.4" \
    --wandb