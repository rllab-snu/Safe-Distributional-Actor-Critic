export CUDA_VISIBLE_DEVICES="0" ; python main.py \
    --name OffTRC-MITCheetah \
    --seed 1 \
    --env_name MITCheetah-v0 \
    --total_steps 2000000 \
    --limit_values "0.025, 0.025, 0.4" \
    --wandb