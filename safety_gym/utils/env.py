from scipy.spatial.transform import Rotation
from copy import deepcopy
import numpy as np
import safety_gym
import pickle
import random
import gym
import re


class GymEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        _, self.robot_name, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if self.task_name == 'goal':
            self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16
        elif self.task_name == 'button':
            self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16*3
        self.observation_space = gym.spaces.box.Box(
            -np.ones(self.obs_dim, dtype=np.float64), 
            np.ones(self.obs_dim, dtype=np.float64), 
            dtype=np.float64,
        )
        self.action_space = self._env.action_space
        self.goal_threshold = np.inf

        # set robot size
        if self.robot_name in ['point']:
            self.robot_size = 0.15
        elif self.robot_name in ['car']:
            self.robot_size = 0.2
        elif self.robot_name in ['doggo']:
            self.robot_size = 0.3

        # set hazard coefficient for cost function
        if self.task_name == 'goal':
            self.h_coeff = 10.0
        elif self.task_name == 'button':
            self.h_coeff = 10.0 
        else:
            self.h_coeff = 10.0

        # set hazard size
        self.hazard_size = 0.2
        self.gremlin_size = 0.1*np.sqrt(2.0)
        self.button_size = 0.1
        self.safety_confidence = 0.0


    def seed(self, num_seed):
        num_seed += random.randint(0, 10000)
        self._env.seed(num_seed)

    def _get_original_state(self):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = np.array([self._env.dist_goal()])
        goal_dist = np.clip(goal_dist, 0.0, self.goal_threshold)
        acc = self._env.world.get_sensor('accelerometer')[:2]
        vel = self._env.world.get_sensor('velocimeter')[:2]
        rot_vel = self._env.world.get_sensor('gyro')[2:]
        if self.task_name == 'goal':
            hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
            lidar = hazards_lidar
        elif self.task_name == 'button':
            hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
            gremlins_lidar = self._env.obs_lidar(self._env.gremlins_obj_pos, 3)
            if self.button_timer == 0:
                buttons_lidar = self._env.obs_lidar(self._env.buttons_pos, 3)
            else:
                buttons_lidar = np.zeros(self._env.unwrapped.lidar_num_bins)
            lidar = np.concatenate([hazards_lidar, gremlins_lidar, buttons_lidar])
        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (lidar - 0.3)/0.3], axis=0)
        return state

    def _get_cost(self, h_dist):
        cost = 1.0/(1.0 + np.exp((h_dist - self.safety_confidence)*self.h_coeff))
        return cost

    def _get_min_dist(self, hazard_pos_list, pos):
        pos = np.array(pos)
        min_dist = np.inf
        for hazard_pos in hazard_pos_list:
            dist = np.linalg.norm(hazard_pos[:2] - pos[:2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_hazard_dist(self):
        if self.task_name == "button":
            # button
            if self.button_timer == 0:
                pos_list = []
                for button_idx, button_pos in enumerate(self._env.buttons_pos):
                    if button_idx == self._env.goal_button: continue
                    pos_list.append(button_pos)
                h_dist = self._get_min_dist(pos_list, self._env.world.robot_pos()) - (self.button_size + self.robot_size)
            else:
                h_dist = np.inf

            # gremlin
            temp_dist = self._get_min_dist(self._env.gremlins_obj_pos, self._env.world.robot_pos()) - (self.gremlin_size + self.robot_size)
            h_dist = min(h_dist, temp_dist)

            # hazard
            temp_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
            h_dist = min(h_dist, temp_dist)
        elif self.task_name == "goal":
            # hazard
            h_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
        return h_dist
        
    def reset(self):
        self.t = 0
        if self.task_name == "button":
            self.button_timer = 0
        self._env.reset()
        state = self._get_original_state()
        return state

    def step(self, action):
        reward = 0
        is_goal_met = False
        num_cv = 0

        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)

            if info['cost'] > 0:
                num_cv += 1
            try:
                if info['goal_met']:
                    is_goal_met = True
            except:
                pass
                
            reward += r_t
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break

        state = self._get_original_state()
        h_dist = self._get_hazard_dist()

        # update button_timer
        if self.task_name == "button":
            self.button_timer = self._env.unwrapped.buttons_timer

        # update information
        info['goal_met'] = is_goal_met
        info['cost'] = self._get_cost(h_dist)
        info['cost_0'] = info['cost']
        info['num_cv'] = num_cv
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()



def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    if env_name in ['Safexp-PointButton3-v0', 'Safexp-CarButton3-v0']:
        return gym.make(env_name)
    elif 'safexp' in env_name.lower():
        # return gym.make(env_name)
        return GymEnv(env_name, seed, max_episode_length, action_repeat)
    else:
        raise ValueError("There is no env_name.")
