from typing import Optional, List

from utils.color import cprint
from models import Policy
from models import Value2
from models import Value
import ctypes

from torch.distributions import Normal
from qpsolvers import solve_qp
from collections import deque
from scipy.stats import norm
from scipy import optimize
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


class Agent:
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'

        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device, dtype=torch.float32)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device, dtype=torch.float32)
        self.n_envs = args.n_envs
        self.n_past_steps = args.n_steps if args.n_past_steps <= 0 else args.n_past_steps
        self.n_past_steps_per_env = int(self.n_past_steps/self.n_envs)
        self.n_update_steps = args.n_update_steps
        self.n_update_steps_per_env = int(self.n_update_steps/self.n_envs)

        # for RL
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff

        # for replay buffer
        self.replay_buffer_per_env = int(args.len_replay_buffer/args.n_envs)
        self.replay_buffer = [deque(maxlen=self.replay_buffer_per_env) for _ in range(args.n_envs)]

        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl

        # for constraint
        self.num_costs = args.num_costs
        self.cost_alphas = []
        self.sigma_units = []
        self.limit_values = []
        for cost_idx in range(self.num_costs):
            limit_value = args.limit_values.split(',')[cost_idx].strip()
            limit_value = float(limit_value)/(1.0 - self.discount_factor)
            cost_alpha = float(args.cost_alphas.split(',')[cost_idx].strip())
            self.limit_values.append(limit_value)
            self.cost_alphas.append(cost_alpha)
            self.sigma_units.append(norm.pdf(norm.ppf(cost_alpha))/cost_alpha)
        self.zeta = min(self.limit_values)

        # for solver
        self.bounds = optimize.Bounds(np.zeros(self.num_costs + 1), np.ones(self.num_costs + 1)*np.inf)
        def dual(x, g_H_inv_g, r_vector, S_mat, c_vector, max_kl):
            lam_vector = x[:-1]
            nu_scalar = x[-1]
            objective = (g_H_inv_g - 2.0*np.dot(r_vector, lam_vector) + np.dot(lam_vector, S_mat@lam_vector))/(2.0*nu_scalar + EPS) \
                            - np.dot(lam_vector, c_vector) + nu_scalar*max_kl
            return objective
        def dualJac(x, g_H_inv_g, r_vector, S_mat, c_vector, max_kl):
            lam_vector = x[:-1]
            nu_scalar = x[-1]
            jacobian = np.zeros_like(x)
            jacobian[:-1] = (S_mat@lam_vector - r_vector)/(nu_scalar + EPS) - c_vector
            jacobian[-1] = max_kl - (g_H_inv_g - 2.0*np.dot(r_vector, lam_vector) + np.dot(lam_vector, S_mat@lam_vector))/(2.0*(nu_scalar**2) + EPS)
            return jacobian
        self.dual = dual
        self.dualJac = dualJac

        # declare networks
        self.policy = Policy(args).to(args.device)
        self.reward_value = Value(args).to(args.device)
        self.cost_values = []
        self.cost_std_values = []
        for cost_idx in range(self.num_costs):
            self.cost_values.append(Value(args).to(args.device))
            self.cost_std_values.append(Value2(args).to(args.device))

        # optimizers
        self.reward_value_optimizer = torch.optim.Adam(self.reward_value.parameters(), lr=self.lr)
        self.cost_value_optimizers = []
        self.cost_std_value_optimizers = []
        for cost_idx in range(self.num_costs):
            self.cost_value_optimizers.append(torch.optim.Adam(self.cost_values[cost_idx].parameters(), lr=self.lr))
            self.cost_std_value_optimizers.append(torch.optim.Adam(self.cost_std_values[cost_idx].parameters(), lr=self.lr))

        # load
        self._load()

    """ public functions
    """
    def getAction(self, state, is_train):
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self._unnormalizeAction(mean + noise*std)
        else:
            action = self._unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action, mean, std

    def addTransition(self, env_idx, state, action, mu_mean, mu_std, reward, done, fail, next_state, *cost_list):
        self.replay_buffer[env_idx].append([state, action, mu_mean, mu_std, reward, done, fail, next_state, *cost_list])

    def train(self):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_lists = [[] for _ in range(self.num_costs)]
        cost_var_targets_lists = [[] for _ in range(self.num_costs)]
        reward_gaes_list = []
        cost_gaes_lists = [[] for _ in range(self.num_costs)]
        cost_var_gaes_lists = [[] for _ in range(self.num_costs)]
        mu_means_list = []
        mu_stds_list = []
        cost_means = None
        cost_var_means = None

        # latest trajectory
        temp_states_list, temp_actions_list, temp_reward_targets_list, temp_cost_targets_lists, temp_cost_var_targets_lists, \
            temp_reward_gaes_list, temp_cost_gaes_lists, temp_cost_var_gaes_lists, \
            temp_mu_means_list, temp_mu_stds_list, cost_means, cost_var_means = self._getTrainBatches(is_latest=True)
        for cost_idx in range(self.num_costs):
            cost_targets_lists[cost_idx] += temp_cost_targets_lists[cost_idx]
            cost_var_targets_lists[cost_idx] += temp_cost_var_targets_lists[cost_idx]
            cost_gaes_lists[cost_idx] += temp_cost_gaes_lists[cost_idx]
            cost_var_gaes_lists[cost_idx] += temp_cost_var_gaes_lists[cost_idx]
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        reward_gaes_list += temp_reward_gaes_list
        mu_means_list += temp_mu_means_list
        mu_stds_list += temp_mu_stds_list

        # random trajectory
        temp_states_list, temp_actions_list, temp_reward_targets_list, temp_cost_targets_lists, temp_cost_var_targets_lists, \
            temp_reward_gaes_list, temp_cost_gaes_lists, temp_cost_var_gaes_lists, \
            temp_mu_means_list, temp_mu_stds_list, _, _ = self._getTrainBatches(is_latest=False)
        for cost_idx in range(self.num_costs):
            cost_targets_lists[cost_idx] += temp_cost_targets_lists[cost_idx]
            cost_var_targets_lists[cost_idx] += temp_cost_var_targets_lists[cost_idx]
            cost_gaes_lists[cost_idx] += temp_cost_gaes_lists[cost_idx]
            cost_var_gaes_lists[cost_idx] += temp_cost_var_gaes_lists[cost_idx]
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        reward_gaes_list += temp_reward_gaes_list
        mu_means_list += temp_mu_means_list
        mu_stds_list += temp_mu_stds_list

        # convert to tensor
        with torch.no_grad():
            states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
            actions_tensor = self._normalizeAction(torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32))
            reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
            reward_gaes_tensor = torch.tensor(np.concatenate(reward_gaes_list, axis=0), device=self.device, dtype=torch.float32)
            mu_means_tensor = torch.tensor(np.concatenate(mu_means_list, axis=0), device=self.device, dtype=torch.float32)
            mu_stds_tensor = torch.tensor(np.concatenate(mu_stds_list, axis=0), device=self.device, dtype=torch.float32)
            cost_targets_tensors = []
            cost_std_targets_tensors = []
            cost_gaes_tensors = []
            cost_var_gaes_tensors = []
            for cost_idx in range(self.num_costs):
                cost_targets_tensors.append(torch.tensor(np.concatenate(cost_targets_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32))
                cost_std_targets_tensors.append(torch.tensor(np.sqrt(np.concatenate(cost_var_targets_lists[cost_idx], axis=0)), device=self.device, dtype=torch.float32))
                cost_gaes_tensors.append(torch.tensor(np.concatenate(cost_gaes_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32))
                cost_var_gaes_tensors.append(torch.tensor(np.concatenate(cost_var_gaes_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32))

        # ================== Value Update ================== #
        for _ in range(self.n_epochs):
            reward_value_loss = torch.mean(0.5*torch.square(self.reward_value(states_tensor) - reward_targets_tensor))
            self.reward_value_optimizer.zero_grad()
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_value.parameters(), self.max_grad_norm)
            self.reward_value_optimizer.step()

            cost_value_losses = []
            cost_var_value_losses = []
            for cost_idx in range(self.num_costs):
                cost_value_loss = torch.mean(0.5*torch.square(self.cost_values[cost_idx](states_tensor) - cost_targets_tensors[cost_idx]))
                self.cost_value_optimizers[cost_idx].zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_values[cost_idx].parameters(), self.max_grad_norm)
                self.cost_value_optimizers[cost_idx].step()
                cost_value_losses.append(cost_value_loss.item())

                cost_var_value_loss = torch.mean(0.5*torch.square(self.cost_std_values[cost_idx](states_tensor) - cost_std_targets_tensors[cost_idx]))
                self.cost_std_value_optimizers[cost_idx].zero_grad()
                cost_var_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_std_values[cost_idx].parameters(), self.max_grad_norm)
                self.cost_std_value_optimizers[cost_idx].step()
                cost_var_value_losses.append(cost_var_value_loss.item())
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        means, _, stds = self.policy(states_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()
        cur_dists = Normal(means, stds)
        old_dists = Normal(old_means, old_stds)
        mu_dists = Normal(mu_means_tensor, mu_stds_tensor)
        old_log_probs = torch.sum(old_dists.log_prob(actions_tensor), dim=1)
        mu_log_probs = torch.sum(mu_dists.log_prob(actions_tensor), dim=1)
        old_prob_ratios = torch.clamp(torch.exp(old_log_probs - mu_log_probs), 0.0, 1.0)
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
        kl_bonus = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(mu_dists, old_dists), dim=1))
        kl_bonus = torch.sqrt(kl_bonus*(self.max_kl + 0.25*kl_bonus)) - 0.5*kl_bonus
        max_kl = torch.clamp(self.max_kl - kl_bonus, 0.0, np.inf).item()

        # get objective
        objective, entropy = self._getObjective(cur_dists, old_dists, actions_tensor, reward_gaes_tensor, old_prob_ratios)
        g_tensor = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(kl, g_tensor)
        approx_g_tensor = self._Hx(kl, H_inv_g_tensor)
        g_H_inv_g_tensor = torch.dot(approx_g_tensor, H_inv_g_tensor)

        # for cost surrogate
        b_tensors = []
        H_inv_b_tensors = []
        c_scalars = []
        cost_surrogates = []
        safety_idx = -1
        for cost_idx in range(self.num_costs):
            cost_surrogate = self._getCostSurrogate(cur_dists, old_dists, actions_tensor, cost_gaes_tensors[cost_idx], \
                cost_var_gaes_tensors[cost_idx], old_prob_ratios, cost_means[cost_idx], cost_var_means[cost_idx], self.sigma_units[cost_idx])
            cost_scalar = cost_surrogate.item()
            b_tensor = flatGrad(cost_surrogate, self.policy.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(kl, b_tensor)
            approx_b_tensor = self._Hx(kl, H_inv_b_tensor)
            c_scalar = cost_scalar - self.limit_values[cost_idx]
            if cost_scalar > self.limit_values[cost_idx] and safety_idx == -1: safety_idx = cost_idx
            cost_surrogates.append(cost_surrogate)
            b_tensors.append(approx_b_tensor)
            H_inv_b_tensors.append(H_inv_b_tensor)
            c_scalars.append(c_scalar)
        B_tensor = torch.stack(b_tensors).T
        H_inv_B_tensor = torch.stack(H_inv_b_tensors).T
        S_tensor = B_tensor.T@H_inv_B_tensor
        r_tensor = approx_g_tensor@H_inv_B_tensor

        with torch.no_grad():
            # to numpy
            S_mat = S_tensor.detach().cpu().numpy()
            r_vector = r_tensor.detach().cpu().numpy()
            g_H_inv_g_scalar = g_H_inv_g_tensor.detach().cpu().numpy()
            c_vector = np.array(c_scalars)

            # find scaling factor
            const_lam_vector = solve_qp(P=(S_mat + np.eye(self.num_costs)*EPS), q=-c_vector, lb=np.zeros(self.num_costs))
            approx_kl = 0.5*np.dot(const_lam_vector, S_mat@const_lam_vector)

            # find search direction
            if approx_kl/max_kl - 1.0 > -0.001:
                optim_case = 0
                delta_theta = -H_inv_b_tensors[safety_idx]*torch.sqrt(2*max_kl/(torch.dot(H_inv_b_tensors[safety_idx], b_tensors[safety_idx]) + EPS))
            else:
                optim_case = 1 if safety_idx != -1 else 2
                x0 = np.ones(self.num_costs + 1)
                res = optimize.minimize(\
                    self.dual, x0, method='trust-constr', jac=self.dualJac,
                    args=(g_H_inv_g_scalar, r_vector, S_mat, c_vector, max_kl), 
                    bounds=self.bounds, options={'disp': True, 'maxiter': 200}
                )
                if res.success:
                    lam_vector, nu_scalar = res.x[:-1], res.x[-1]
                    lam_tensor = torch.tensor(lam_vector, device=self.device, dtype=torch.float32)
                    delta_theta = (H_inv_g_tensor - H_inv_B_tensor@lam_tensor)/(nu_scalar + EPS)
                else:
                    # there is no solution -> only minimize costs.
                    optim_case = 0
                    delta_theta = -torch.zeros_like(H_inv_B_tensor)

            # line search
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
            init_objective = objective.item()
            init_cost_surrogates = [cost_surr.item() for cost_surr in cost_surrogates]
            while True:
                theta = beta*delta_theta + init_theta
                beta *= self.line_decay
                self._applyParams(theta)
                means, _, stds = self.policy(states_tensor)
                cur_dists = Normal(means, stds)
                kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
                if kl > max_kl:
                    continue
                objective, entropy = self._getObjective(cur_dists, old_dists, actions_tensor, reward_gaes_tensor, old_prob_ratios)
                if optim_case == 2 and objective < init_objective:
                    continue
                if optim_case >= 1:
                    is_feasible = True
                    for cost_idx in range(self.num_costs):
                        cost_surrogate = self._getCostSurrogate(cur_dists, old_dists, actions_tensor, cost_gaes_tensors[cost_idx], \
                            cost_var_gaes_tensors[cost_idx], old_prob_ratios, cost_means[cost_idx], cost_var_means[cost_idx], self.sigma_units[cost_idx])
                        if cost_surrogate - init_cost_surrogates[cost_idx] > max(-c_vector[cost_idx], 0.0):
                            is_feasible = False
                            break
                    if is_feasible:
                        break
                else:
                    cost_surrogate = self._getCostSurrogate(cur_dists, old_dists, actions_tensor, cost_gaes_tensors[safety_idx], \
                        cost_var_gaes_tensors[safety_idx], old_prob_ratios, cost_means[safety_idx], cost_var_means[safety_idx], self.sigma_units[safety_idx])
                    if cost_surrogate - init_cost_surrogates[safety_idx] <= max(-c_vector[safety_idx], 0.0):
                        break
        # ================================================= #

        return objective.item(), init_cost_surrogates, reward_value_loss.item(), cost_value_losses, \
            cost_var_value_losses, entropy.item(), kl.item(), optim_case

    def save(self):
        save_dict = {
            'policy': self.policy.state_dict(),
            'reward_value': self.reward_value.state_dict(),
            'reward_value_optimizer': self.reward_value_optimizer.state_dict(),
        }
        for i in range(self.num_costs):
            save_dict[f'cost_value_{i}'] = self.cost_values[i].state_dict()
            save_dict[f'cost_std_value_{i}'] = self.cost_std_values[i].state_dict()
            save_dict[f'cost_value_optimizer_{i}'] = self.cost_value_optimizers[i].state_dict()
            save_dict[f'cost_std_value_optimizer_{i}'] = self.cost_std_value_optimizers[i].state_dict()
        torch.save(save_dict, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    """ private functions
    """
    def _normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def _unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def _getObjective(self, cur_dists, old_dists, actions, reward_gaes, old_prob_ratios):
        entropy = torch.mean(torch.sum(cur_dists.entropy(), dim=1))
        cur_log_probs = torch.sum(cur_dists.log_prob(actions), dim=1)
        old_log_probs = torch.sum(old_dists.log_prob(actions), dim=1)
        prob_ratios = torch.exp(cur_log_probs - old_log_probs)
        reward_gaes_mean = torch.mean(reward_gaes*old_prob_ratios)
        reward_gaes_std = torch.std(reward_gaes*old_prob_ratios)
        objective = torch.mean(prob_ratios*(reward_gaes*old_prob_ratios - reward_gaes_mean)/(reward_gaes_std + EPS))
        return objective, entropy

    def _getCostSurrogate(self, cur_dists, old_dists, actions, cost_gaes, cost_var_gaes, old_prob_ratios, cost_mean, cost_var_mean, sigma_unit):
        cur_log_probs = torch.sum(cur_dists.log_prob(actions), dim=1)
        old_log_probs = torch.sum(old_dists.log_prob(actions), dim=1)
        prob_ratios = torch.exp(cur_log_probs - old_log_probs)
        cost_gaes_mean = torch.mean(cost_gaes*old_prob_ratios)
        cost_var_gaes_mean = torch.mean(cost_var_gaes*old_prob_ratios)
        approx_cost_mean = cost_mean + (1.0/(1.0 - self.discount_factor))*torch.mean(prob_ratios*(cost_gaes*old_prob_ratios - cost_gaes_mean))
        approx_cost_var = cost_var_mean + (1.0/(1.0 - self.discount_factor**2))*torch.mean(prob_ratios*(cost_var_gaes*old_prob_ratios - cost_var_gaes_mean))
        cost_surrogate = approx_cost_mean + sigma_unit*torch.sqrt(torch.clamp(approx_cost_var, EPS, np.inf))
        return cost_surrogate

    def _getGaesTargets(self, rewards, values, dones, fails, next_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def _getVarGaesTargets(self, rewards, values, var_values, dones, fails, next_values, next_var_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = np.square(rewards[t] + (1.0 - fails[t])*self.discount_factor*next_values[t]) - np.square(values[t]) + \
                            (1.0 - fails[t])*(self.discount_factor**2)*next_var_values[t] + \
                            (1.0 - dones[t])*(self.discount_factor**2)*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - var_values[t])
        gaes = targets - var_values
        targets = np.clip(targets, 0.0, np.inf)
        return gaes, targets

    def _getTrainBatches(self, is_latest=False):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_lists = [[] for _ in range(self.num_costs)]
        cost_var_targets_lists = [[] for _ in range(self.num_costs)]
        reward_gaes_list = []
        cost_gaes_lists = [[] for _ in range(self.num_costs)]
        cost_var_gaes_lists = [[] for _ in range(self.num_costs)]
        cost_mean_lists = [[] for _ in range(self.num_costs)]
        cost_var_mean_lists = [[] for _ in range(self.num_costs)]
        mu_means_list = []
        mu_stds_list = []

        with torch.no_grad():
            for env_idx in range(self.n_envs):
                n_latest_steps = min(len(self.replay_buffer[env_idx]), self.n_past_steps_per_env)
                if is_latest:
                    start_idx = len(self.replay_buffer[env_idx]) - n_latest_steps
                    end_idx = start_idx + n_latest_steps
                    env_trajs = list(self.replay_buffer[env_idx])[start_idx:end_idx]
                else:
                    n_update_steps = min(len(self.replay_buffer[env_idx]), self.n_update_steps_per_env)
                    if n_update_steps <= n_latest_steps:
                        continue
                    start_idx = np.random.randint(0, len(self.replay_buffer[env_idx]) - n_update_steps + 1)
                    end_idx = start_idx + (n_update_steps - n_latest_steps)
                    env_trajs = list(self.replay_buffer[env_idx])[start_idx:end_idx]

                states = np.array([traj[0] for traj in env_trajs])
                actions = np.array([traj[1] for traj in env_trajs])
                mu_means = np.array([traj[2] for traj in env_trajs])
                mu_stds = np.array([traj[3] for traj in env_trajs])
                rewards = np.array([traj[4] for traj in env_trajs])
                dones = np.array([traj[5] for traj in env_trajs])
                fails = np.array([traj[6] for traj in env_trajs])
                next_states = np.array([traj[7] for traj in env_trajs])

                # convert to tensor
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                actions_tensor = self._normalizeAction(torch.tensor(actions, device=self.device, dtype=torch.float32))
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                mu_means_tensor = torch.tensor(mu_means, device=self.device, dtype=torch.float32)
                mu_stds_tensor = torch.tensor(mu_stds, device=self.device, dtype=torch.float32)

                # for rho
                means_tensor, _, stds_tensor = self.policy(states_tensor)
                old_dists = torch.distributions.Normal(means_tensor, stds_tensor)
                mu_dists = torch.distributions.Normal(mu_means_tensor, mu_stds_tensor)
                old_log_probs_tensor = torch.sum(old_dists.log_prob(actions_tensor), dim=1)
                mu_log_probs_tensor = torch.sum(mu_dists.log_prob(actions_tensor), dim=1)
                rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0)
                rhos = rhos_tensor.detach().cpu().numpy()

                # get GAEs and Tagets
                # for reward
                reward_values_tensor = self.reward_value(states_tensor)
                next_reward_values_tensor = self.reward_value(next_states_tensor)
                reward_values = reward_values_tensor.detach().cpu().numpy()
                next_reward_values = next_reward_values_tensor.detach().cpu().numpy()
                reward_gaes, reward_targets = self._getGaesTargets(rewards, reward_values, dones, fails, next_reward_values, rhos)

                # for cost
                for cost_idx in range(self.num_costs):
                    costs = np.array([traj[8 + cost_idx] for traj in env_trajs])
                    cost_values_tensor = self.cost_values[cost_idx](states_tensor)
                    next_cost_values_tensor = self.cost_values[cost_idx](next_states_tensor)
                    cost_values = cost_values_tensor.detach().cpu().numpy()
                    next_cost_values = next_cost_values_tensor.detach().cpu().numpy()
                    cost_gaes, cost_targets = self._getGaesTargets(costs, cost_values, dones, fails, next_cost_values, rhos)
                    # for cost var
                    cost_var_values_tensor = torch.square(self.cost_std_values[cost_idx](states_tensor))
                    next_cost_var_values_tensor = torch.square(self.cost_std_values[cost_idx](next_states_tensor))
                    cost_var_values = cost_var_values_tensor.detach().cpu().numpy()
                    next_cost_var_values = next_cost_var_values_tensor.detach().cpu().numpy()
                    cost_var_gaes, cost_var_targets = self._getVarGaesTargets(costs, cost_values, cost_var_values, dones, fails, next_cost_values, next_cost_var_values, rhos)

                    # save
                    cost_targets_lists[cost_idx].append(cost_targets)
                    cost_var_targets_lists[cost_idx].append(cost_var_targets)
                    cost_gaes_lists[cost_idx].append(cost_gaes)
                    cost_var_gaes_lists[cost_idx].append(cost_var_gaes)

                    # add cost mean & cost variance mean
                    cost_mean_lists[cost_idx].append(np.mean(costs)/(1.0 - self.discount_factor))
                    cost_var_mean_lists[cost_idx].append(np.mean(cost_var_targets))

                # save
                states_list.append(states)
                actions_list.append(actions)
                reward_gaes_list.append(reward_gaes)
                reward_targets_list.append(reward_targets)
                mu_means_list.append(mu_means)
                mu_stds_list.append(mu_stds)

        # get cost mean & cost variance mean
        cost_means = np.mean(cost_mean_lists, axis=1)
        cost_var_means = np.mean(cost_var_mean_lists, axis=1)

        return states_list, actions_list, reward_targets_list, cost_targets_lists, cost_var_targets_lists, \
            reward_gaes_list, cost_gaes_lists, cost_var_gaes_lists, mu_means_list, mu_stds_list, cost_means, cost_var_means

    def _applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def _Hx(self, kl:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def _conjugateGradient(self, kl:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self._Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/(rs_old + EPS))*p
            rs_old = rs_new
        return x

    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.reward_value.load_state_dict(checkpoint['reward_value'])
            self.reward_value_optimizer.load_state_dict(checkpoint['reward_value_optimizer'])
            for i in range(self.num_costs):
                self.cost_values[i].load_state_dict(checkpoint[f'cost_value_{i}'])
                self.cost_std_values[i].load_state_dict(checkpoint[f'cost_std_value_{i}'])
                self.cost_value_optimizers[i].load_state_dict(checkpoint[f'cost_value_optimizer_{i}'])
                self.cost_std_value_optimizers[i].load_state_dict(checkpoint[f'cost_std_value_optimizer_{i}'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            self.reward_value.initialize()
            for i in range(self.num_costs):
                self.cost_values[i].initialize()
                self.cost_std_values[i].initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
