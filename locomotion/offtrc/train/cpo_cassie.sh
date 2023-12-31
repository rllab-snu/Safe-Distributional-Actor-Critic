export CUDA_VISIBLE_DEVICES="0" ; python main.py \
    --name CPO-Cassie \
    --seed 1 \
    --env_name Cassie-v0 \
    --total_steps 4000000 \
    --limit_values "0.025, 0.025, 0.4" \
    --n_steps 5000 \
    --n_past_steps 5000 \
    --n_update_steps 5000 \
    --len_replay_buffer 5000 \
    --wandb
