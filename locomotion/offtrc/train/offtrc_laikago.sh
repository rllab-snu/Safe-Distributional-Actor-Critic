export CUDA_VISIBLE_DEVICES="0" ; python main.py \
    --name OffTRC-Laikago \
    --seed 1 \
    --env_name Laikago-v0 \
    --total_steps 2000000 \
    --limit_values "0.025, 0.025, 0.4" \
    --wandb