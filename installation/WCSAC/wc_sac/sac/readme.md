Portions of the code in saclag.py and wcsac.py are adapted from [Safety Starter Agents](https://github.com/openai/safety-starter-agents) and [Spinning Up in Deep RL](https://github.com/openai/spinningup/tree/master/spinup/utils).


## parameters
1. env_fn - lambda: gym.make(env_name).
2. actor_fn, critic_fn - mlp structure.
3. ac_kwargs - the configuration for actor-critics.
    - hidden_sizes=(512, 512).
4. seed.
5. steps_per_epoch - after "steps_per_epoch" steps, agents are trained.
    - have to set to "1000", which is the same as SDAC.
6. epochs
    - total epochs.
    - epochs*steps_per_epoch = total_steps.
7. replay_size - 1e6.
8. gamma - 0.99.
9. cl
    - cvar alpha.
    - have to set to "0.5" or "1.0".
10. polyak - 0.995.
11. lr - set to default.
12. batch_size - set to default.
13. local_start_steps - 10000/num_procs(), which is the same as SDAC.
14. max_ep_len - 1000.
15. save_freq - save networks after "save_freq" epochs.
16. update_freq - at every "update_freq" steps, nets are updated.
17. cost_lim - 0.025.
18. damp_scale - ??.