export CUDA_VISIBLE_DEVICES="0" ; python main.py --name CPO-PointGoal --seed 1 --env_name Safexp-PointGoal1-v0 --n_steps 5000 --n_update_steps 5000 --len_replay_buffer 5000 --cost_alpha 1.0 --test
