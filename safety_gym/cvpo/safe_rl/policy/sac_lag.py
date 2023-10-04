from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
from safe_rl.policy import SAC, LagrangianPIDController
from safe_rl.policy.model.mlp_ac import EnsembleQCritic
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import (count_vars, get_device_name, to_device, to_ndarray,
                                     to_tensor)
from torch.optim import Adam


class SACLagrangian(SAC):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 num_qc=1,
                 cost_limit=40,
                 use_cost_decay=False,
                 cost_start=100,
                 cost_end=10,
                 decay_epoch=100,
                 timeout_steps=200,
                 KP=0,
                 KI=0.1,
                 KD=0,
                 per_state=True,
                 **kwargs) -> None:
        r'''
        Soft Actor Critic (SAC) with Lagrangian multiplier

        Args in kwargs:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        '''
        self.use_cost_decay = use_cost_decay
        self.cost_start = cost_start
        self.cost_end = cost_end
        self.decay_epoch = decay_epoch
        super().__init__(env, logger, **kwargs)
        '''
        Notice: The output action are normalized in the range [-1, 1], 
        so please make sure your action space's high and low are suitable
        '''
        qc = EnsembleQCritic(self.obs_dim,
                             self.act_dim,
                             self.hidden_sizes,
                             nn.ReLU,
                             num_q=num_qc)
        self._qc_training_setup(qc)

        if self.use_cost_decay:
            self.epoch = 0
            self.qc_start = self.cost_start * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
            self.qc_end = self.cost_end * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
            self.decay_func = lambda x: self.qc_end + (
                self.qc_start - self.qc_end) * np.exp(-5. * x / self.decay_epoch)
            self._step_qc_thres()
        else:
            self.qc_thres = cost_limit * (1 - self.gamma**timeout_steps) / (
                1 - self.gamma) / timeout_steps
        print("Cost constraint: ", self.qc_thres)

        self.controller = LagrangianPIDController(KP, KI, KD, self.qc_thres, per_state)

    def _step_qc_thres(self):
        self.qc_thres = self.decay_func(
            self.epoch) if self.epoch < self.decay_epoch else self.qc_end
        self.epoch += 1

    def _qc_training_setup(self, qc):
        qc_targ = deepcopy(qc)
        self.qc, self.qc_targ = to_device([qc, qc_targ], get_device_name())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.qc_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for safety critic
        self.qc_optimizer = Adam(self.qc.parameters(), lr=self.critic_lr)

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        self._update_critic(data)
        self._update_qc(data)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False
        for p in self.qc.parameters():
            p.requires_grad = False

        self._update_actor(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True
        for p in self.qc.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self._polyak_update_target(self.critic, self.critic_targ)
        self._polyak_update_target(self.qc, self.qc_targ)

    def post_epoch_process(self):
        '''
        Update the cost limit.
        '''
        if self.use_cost_decay:
            self._step_qc_thres()

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        def policy_loss():
            obs = data['obs']
            act, logp_pi = self.actor_forward(obs, False, True)
            q_pi, q_list = self.critic_forward(self.critic, obs, act)
            qc_pi, _ = self.critic_forward(self.qc, obs, act)

            # detach is very important here!
            # Otherwise the gradient will backprop through the multiplier.
            with torch.no_grad():
                multiplier = self.controller.control(qc_pi).detach()

            qc_penalty = ((qc_pi - self.qc_thres) * multiplier).mean()

            # Entropy-regularized policy loss
            loss_actor = (self.alpha * logp_pi - q_pi).mean()

            loss_pi = loss_actor + qc_penalty

            # Useful info for logging
            pi_info = dict(LogPi=to_ndarray(logp_pi),
                           Lagrangian=to_ndarray(multiplier),
                           LossActor=to_ndarray(loss_actor),
                           QcPenalty=to_ndarray(qc_penalty))

            return loss_pi, pi_info

        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = policy_loss()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Log actor update info
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

    def _update_qc(self, data):
        '''
        Update the qc network
        '''
        def critic_loss():
            obs, act, reward, obs_next, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(
                    data['obs2']), to_tensor(data['done'])

            _, q_list = self.critic_forward(self.qc, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                act_next, logp_a_next = self.actor_forward(obs_next,
                                                           deterministic=False,
                                                           with_logprob=True)
                # Target Q-values
                q_pi_targ, _ = self.critic_forward(self.qc_targ, obs_next, act_next)
                backup = reward + self.gamma * (q_pi_targ - self.alpha * logp_a_next)
            # MSE loss against Bellman backup
            loss_q = self.qc.loss(backup, q_list)
            # Useful info for logging
            q_info = dict()
            for i, q in enumerate(q_list):
                q_info["QCVals" + str(i)] = to_ndarray(q)
            return loss_q, q_info

        # First run one gradient descent step for Q1 and Q2
        self.qc_optimizer.zero_grad()
        loss_qc, loss_qc_info = critic_loss()
        loss_qc.backward()
        self.qc_optimizer.step()

        # Log critic update info
        # Record things
        self.logger.store(LossQC=loss_qc.item(), **loss_qc_info, QcThres=self.qc_thres)
