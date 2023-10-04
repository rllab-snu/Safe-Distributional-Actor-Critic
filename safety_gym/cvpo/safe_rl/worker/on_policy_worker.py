import gym
import numpy as np
import torch
from safe_rl.policy.base_policy import Policy
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import to_tensor
from safe_rl.worker.buffer import OnPolicyBuffer


class OnPolicyWorker:
    r'''
    Collect data based on the policy and env, and store the interaction data to data buffer.
    '''
    def __init__(self,
                 env: gym.Env,
                 policy: Policy,
                 logger: EpochLogger,
                 interact_steps=2000,
                 timeout_steps=200,
                 gamma=0.99,
                 lam=0.97,
                 **kwargs) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger
        self.interact_steps = interact_steps
        self.timeout_steps = timeout_steps

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape

        if "Safe" in env.spec.id:
            self.SAFE_RL_ENV = True

        self.buffer = OnPolicyBuffer(obs_dim, act_dim, self.interact_steps + 1, gamma,
                                     lam)

    def work(self):
        '''
        Interact with the environment to collect data
        '''
        self.cost_list = []
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        for i in range(self.interact_steps):

            action, value, log_prob = self.policy.act(obs)
            obs_next, reward, done, info = self.env.step(action)

            if hasattr(self.policy, "get_qc_v"):
                cost_value = self.policy.get_qc_v(obs)
            else:
                cost_value = 0

            if done and 'TimeLimit.truncated' in info:
                done = False
                timeout_env = True
            else:
                timeout_env = False

            cost = info["cost"] if "cost" in info else 0

            self.buffer.store(obs, np.squeeze(action), reward, value, log_prob, done,
                              cost, cost_value)
            self.logger.store(VVals=value, CostVVals=cost_value, tab="worker")
            ep_reward += reward
            ep_cost += cost
            ep_len += 1
            obs = obs_next

            timeout = ep_len == self.timeout_steps - 1 or i == self.interact_steps - 1 or timeout_env and not done
            terminal = done or timeout
            if terminal:
                # after each episode
                if timeout:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    _, value, _ = self.policy.act(obs)
                    if hasattr(self.policy, "get_qc_v"):
                        cost_value = self.policy.get_qc_v(obs)
                    else:
                        cost_value = 0
                else:
                    value = 0
                    cost_value = 0
                self.buffer.finish_path(value, cost_value)
                if i < self.interact_steps - 1:
                    self.logger.store(EpRet=ep_reward,
                                      EpLen=ep_len,
                                      EpCost=ep_cost,
                                      tab="worker")
                obs = self.env.reset()
                self.cost_list.append(ep_cost)
                # episode reward and length
                ep_reward = 0
                ep_cost = 0
                ep_len = 0
        return self.interact_steps

    def eval(self):
        '''
        Evaluate the policy
        '''
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        for i in range(self.timeout_steps):
            action, _, _ = self.policy.act(obs, eval=True)
            obs_next, reward, done, info = self.env.step(action)
            if "cost" in info:
                cost = info["cost"]
                ep_cost += cost
            ep_reward += reward
            ep_len += 1
            obs = obs_next
            if done:
                break
        self.logger.store(TestEpRet=ep_reward,
                          TestEpLen=ep_len,
                          TestEpCost=ep_cost,
                          tab="eval")

    def get_sample(self):
        data = self.buffer.get()
        # torch.save(data, "buffer.pt")
        self.buffer.clear()
        data["ep_cost"] = to_tensor(np.mean(self.cost_list))
        return data
