from safety_gym.envs.suite import SafexpEnvBase
from gym.envs.registration import register

# original
register(
    id='MITCheetah-v0',
    entry_point='utils.cheetah_env.env_v0:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

# 
register(
    id='MITCheetah-v1',
    entry_point='utils.cheetah_env.env_v1:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Laikago-v0',
    entry_point='utils.laikago_env.env_v0:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Laikago-v1',
    entry_point='utils.laikago_env.env_v1:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Cassie-v0',
    entry_point='utils.cassie_env.env_v0:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Cassie-v1',
    entry_point='utils.cassie_env.env_v1:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

# ============== for CVPO ============== #
bench_base = SafexpEnvBase(
    '', {
        'observe_goal_lidar': True,
        'observe_box_lidar': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16
    })

button_all = {
    'task': 'button',
    'buttons_num': 4,
    'buttons_size': 0.1,
    'buttons_keepout': 0.2,
    'observe_buttons': True,
    'hazards_size': 0.2,
    'hazards_keepout': 0.,
    'gremlins_travel': 0.3,
    'gremlins_keepout': 0.35,
    'hazards_locations': [(0, -1.9), (0., 1.9), (0, 0), (1.9, 0), (-1.9, 0)],
    'buttons_locations': [(-1.3, -1.3), (1.3, -1.3), (-1.3, 1.3), (1.3, 1.3)],
    'gremlins_locations': [(0, -1.3), (0., 1.3), (1.3, 0), (-1.3, 0)]
}

button_constrained = {
    'constrain_hazards': True,
    'constrain_buttons': True,
    'constrain_gremlins': True,
    'observe_hazards': True,
}

button1 = {
    'hazards_num': 5,
    'gremlins_num': 4,
    'observe_gremlins': True,
}
button1.update(button_constrained)

button2 = {
    'placements_extents': [-1.8, -1.8, 1.8, 1.8],
    'hazards_num': 11,
    'observe_gremlins': False,
    'gremlins_num': 0,
    'hazards_locations': [(0.1, -1.9), (-0.2, 1.7), (0.3, 0.1), (2, -0.1), (-1.8, 0.2),
                          (-0.1, -1.2), (0., 1.3), (1.1, 0), (-1., 0), (0.2, -0.9),
                          (-1.1, 1.1)],
    'buttons_locations': [(-1.1, -1.3), (1.8, -1.5), (-1.4, 0.6), (1.0, 1.3)],
}
button2.update(button_constrained)

bench_button_base = bench_base.copy('Button', button_all)
bench_button_base.register('3', button1)
bench_button_base.register('4', button2)
# ====================================== #