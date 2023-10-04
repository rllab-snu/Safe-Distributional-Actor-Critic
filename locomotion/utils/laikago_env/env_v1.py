# ===== add python path ===== #
import glob
from ntpath import join
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

from utils.laikago_env.env_v0 import Env as OriginalEnv

from scipy.spatial.transform import Rotation
from collections import OrderedDict
from collections import deque
from copy import deepcopy
from gym import spaces
import numpy as np
import xmltodict
import time
import json
import gym
import sys
import cv2
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class Env(OriginalEnv):
    def __init__(self, use_fixed_base=False, init_base_pos=[0.0, 0.0, 0.5], init_base_quat=[1.0, 0.0, 0.0, 0.0], max_episode_length=1000, lam1=0, lam2=0, lam3=0) -> None:
        super(Env, self).__init__(use_fixed_base, init_base_pos, init_base_quat, max_episode_length)
        self.lambda_vector = np.array([lam1, lam2, lam3], dtype=np.float32)
        self.num_costs += 1


    def _step(self, action, **kwargs):
        state, true_reward, done, info = super()._step(action, **kwargs)
        reward = (true_reward - np.sum(np.array(info['costs'])*self.lambda_vector))/(1.0 + np.sum(self.lambda_vector))
        info['costs'].append(true_reward)
        return state, reward, done, info


if __name__ == "__main__":
    env = Env(use_fixed_base=False)

    for i in range(10):
        env.reset()
        start_t = time.time()
        global_t = 0.0
        elapsed_t = 0.0
        action = np.zeros(env.action_space.shape[0])
        for i in range(100):
            s, r, d, info = env.step(action)
            env.render()
            global_t += env.env_dt

            elapsed_t = time.time() - start_t
            if elapsed_t < global_t:
                time.sleep(global_t - elapsed_t)

            if d: break

            sys.stdout.write("\rsim time : {:.3f} s, real time : {:.3f} s".format(global_t, elapsed_t))
            sys.stdout.flush()
    sys.stdout.write(f"\rgoodbye.{' '*50}\n")
