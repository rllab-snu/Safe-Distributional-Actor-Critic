from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from stable_baselines3.common.callbacks import BaseCallback
from utils.normalize import RunningMeanStd

from collections import deque
import numpy as np
import pickle
import wandb
import sys
import os

class Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, args, verbose=0):
        super(Callback, self).__init__(verbose)
        '''
        Those variables will be accessible in the callback
        (they are defined in the base class)

        ==== The RL model ====
        self.model = None  # type: BaseAlgorithm

        ==== An alias for self.model.get_env(), the environment used for training ====
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        ==== Number of time the callback was called ====
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int

        ==== local and global variables ====
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]

        ==== The logger object, used to report things in the terminal ====
        self.logger = None  # stable_baselines3.common.logger

        ==== parent class ====
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]
        '''
        self.name = args.name
        self.save_freq = args.save_freq
        self.save_path = args.save_path
        self.save_dir = args.save_dir

        # =========== for logging =========== #
        # wandb
        if args.wandb:
            project_name = f'[Constrained RL] {args.setting_name}'
            wandb.init(
                project=project_name, 
                config=args,
                sync_tensorboard=True,
            )
            run_idx = wandb.run.name.split('-')[-1]
            wandb.run.name = f"{args.name}-{run_idx}"
        
        self.n_envs = args.n_envs
        self.num_costs = args.num_costs
        self.use_wandb = args.wandb

        self.total_steps = 0
        self.score_logger = deque(maxlen=10)
        self.eplen_logger = deque(maxlen=10)
        self.cv_logger = deque(maxlen=10)
        self.cost_loggers = []
        for _ in range(args.num_costs):
            self.cost_loggers.append(deque(maxlen=10))

        self.scores = np.zeros(args.n_envs)
        self.eplens = np.zeros(args.n_envs)
        self.num_cvs = np.zeros(args.n_envs)
        self.costs = np.zeros((args.n_envs, args.num_costs))
        # =================================== #


    def _print(self, contents) -> None:
        print(f"[{self.name}] {contents}")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        for env_idx in range(self.n_envs):
            self.eplens[env_idx] += 1
            self.scores[env_idx] += self.locals['rewards'][env_idx]
            self.num_cvs[env_idx] += self.locals['infos'][env_idx].get('num_cv', 0)
            for cost_idx in range(self.num_costs):
                if 'costs' in self.locals['infos'][env_idx].keys():
                    self.costs[env_idx, cost_idx] += self.locals['infos'][env_idx]['costs'][cost_idx]
            if self.locals['dones'][env_idx]:
                self.eplen_logger.append(self.eplens[env_idx])
                self.score_logger.append(self.scores[env_idx])
                self.cv_logger.append(self.num_cvs[env_idx])
                for cost_idx in range(self.num_costs):
                    self.cost_loggers[cost_idx].append(self.costs[env_idx, cost_idx])
                    self.costs[env_idx, cost_idx] = 0.0
                self.eplens[env_idx] = 0
                self.scores[env_idx] = 0.0
                self.num_cvs[env_idx] = 0.0
            self.total_steps += 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.save_path)
            self._print('save success!')

        log_data = {
            "rollout/step": self.total_steps,
            "rollout/score": np.mean(self.score_logger), 
            "rollout/ep_len": np.mean(self.eplen_logger),
            "rollout/ep_cv": np.mean(self.cv_logger),
        }
        for cost_idx in range(self.num_costs):
            log_data[f'rollout/cost_{cost_idx}'] = np.mean(self.cost_loggers[cost_idx])
        if self.use_wandb:
            wandb.log(log_data)
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.model.save(self.save_path)
        self._print('save success!')
        pass

class CustomSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, args, start_method=None):
        super().__init__(env_fns, start_method)
        self.obs_rms = RunningMeanStd(args.save_dir, self.observation_space.shape[0])

    def reset(self):
        observations = super().reset()
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        return norm_observations

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        for info in infos:
            if 'terminal_observation' in info.keys():
                info['terminal_observation'] = self.obs_rms.normalize(info['terminal_observation'])
        if self.obs_rms.count % int(1e5) == 0:
            self.obs_rms.save()
        return norm_observations, rewards, dones, infos


class CustomSubprocVecEnv2(SubprocVecEnv):
    def __init__(self, env_fns, args, start_method=None):
        super().__init__(env_fns, start_method)

    def reset(self):
        obs = []
        for remote in self.remotes:
            remote.send(("reset", None))
            temp_obs = remote.recv()
            n_steps = np.random.randint(0, 1000)
            action = np.zeros(self.action_space.shape[0])
            for _ in range(n_steps):
                remote.send(("step", action))
                temp_obs, _, _, _ = remote.recv()
            obs.append(temp_obs)
        return _flatten_obs(obs, self.observation_space)


class CustomSubprocVecEnv3(SubprocVecEnv):
    def __init__(self, env_fns, args, start_method=None):
        super().__init__(env_fns, start_method)
        self.obs_rms = RunningMeanStd(args.save_dir, self.observation_space.shape[0])
        with env_fns[0]() as env:
            self.num_costs = env.unwrapped.num_costs

    def reset(self):
        obs = []
        for remote in self.remotes:
            remote.send(("reset", None))
            temp_obs = remote.recv()
            n_steps = np.random.randint(0, 1000)
            action = np.zeros(self.action_space.shape[0])
            for _ in range(n_steps):
                remote.send(("step", action))
                temp_obs, _, _, _ = remote.recv()
            obs.append(temp_obs)
        observations = _flatten_obs(obs, self.observation_space)
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        return norm_observations

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        for info in infos:
            if 'terminal_observation' in info.keys():
                info['terminal_observation'] = self.obs_rms.normalize(info['terminal_observation'])
        if self.obs_rms.count % int(1e5) == 0:
            self.obs_rms.save()
        return norm_observations, rewards, dones, infos
