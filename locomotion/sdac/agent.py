from distutils.dir_util import copy_tree
from typing import Optional, List
import glob

from utils.color import cprint
from models import Policy
from models import Value
import ctypes

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
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
    temp_a = 2.0/(maximum - minimum)
    temp_b = (maximum + minimum)/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
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

def ctype_arr_convert(arr):
    arr = np.ravel(arr)
    return (ctypes.c_double * len(arr))(*arr)


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
        self.n_steps = args.n_steps
        self.n_steps_per_env = int(self.n_steps/self.n_envs)
        self.n_update_steps = args.n_update_steps
        self.n_update_steps_per_env = int(self.n_update_steps/self.n_envs)

        # for RL
        self.lr = args.lr
        self.entropy_d = args.entropy_d
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

        # for distributional critic
        self.n_critics = args.n_critics
        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

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
        for cost_idx in range(self.num_costs):
            self.cost_values.append(Value(args).to(args.device))

        # for alpha
        self.adaptive_alpha = args.adaptive_alpha
        self.alpha_lr = args.alpha_lr
        self.log_alpha = torch.tensor(np.log(args.alpha_init + EPS), requires_grad=True, device=args.device)

        # optimizers
        self.reward_value_optimizer = torch.optim.Adam(self.reward_value.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.cost_value_optimizers = []
        for cost_idx in range(self.num_costs):
            self.cost_value_optimizers.append(torch.optim.Adam(self.cost_values[cost_idx].parameters(), lr=self.lr))

        # projection operator
        self._lib = ctypes.cdll.LoadLibrary(f'{os.path.dirname(os.path.abspath(__file__))}/cpp_modules/main.so')
        self._lib.projection.restype = None

        # load
        self._load()


    def getRandomAction(self):
        mean = torch.zeros(self.n_envs, self.action_dim, device=self.device)
        std = torch.ones(self.n_envs, self.action_dim, device=self.device)
        norm_action = Normal(mean, std).sample()
        squashed_action = torch.tanh(norm_action)
        log_prob = self.policy.logProb(norm_action, mean, torch.log(std), std)
        action = self._unnormalizeAction(squashed_action)
        return action, norm_action, log_prob

    def getAction(self, state:torch.Tensor, is_train:bool) -> List[torch.Tensor]:
        '''
        input:
            states:     Tensor(state_dim,)
            is_train:   boolean
        output:
            action:         Tensor(action_dim,)
            cliped_action:  Tensor(action_dim,)
        '''
        mean, log_std, std = self.policy(state)
        normal = Normal(mean, std)
        norm_action = normal.rsample()
        mu = torch.tanh(mean)
        pi = torch.tanh(norm_action)
        log_prob = self.policy.logProb(norm_action, mean, log_std, std)
        if is_train:
            action = self._unnormalizeAction(pi)
        else:
            action = self._unnormalizeAction(mu)
        return action, norm_action, log_prob

    def addTransition(self, env_idx, state, action, norm_action, log_prob, reward, done, fail, next_state, *cost_list):
        self.replay_buffer[env_idx].append([state, action, norm_action, log_prob, reward, done, fail, next_state, *cost_list])

    def train(self):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_lists = [[] for _ in range(self.num_costs)]

        # latest trajectory
        temp_states_list, temp_actions_list, \
            temp_reward_targets_list, temp_cost_targets_lists = self._getTrainBatches(is_latest=True)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        for cost_idx in range(self.num_costs):
            cost_targets_lists[cost_idx] += temp_cost_targets_lists[cost_idx]

        # random trajectory
        temp_states_list, temp_actions_list, \
            temp_reward_targets_list, temp_cost_targets_lists = self._getTrainBatches(is_latest=False)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        for cost_idx in range(self.num_costs):
            cost_targets_lists[cost_idx] += temp_cost_targets_lists[cost_idx]

        # convert to tensor
        with torch.no_grad():
            states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
            actions_tensor = self._normalizeAction(torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32))
            reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
            reward_targets_tensor.unsqueeze_(dim=1) # B x 1 x kN
            cost_targets_tensors = []
            for cost_idx in range(self.num_costs):
                cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_lists[cost_idx], axis=0), device=self.device, dtype=torch.float32)
                cost_targets_tensor.unsqueeze_(dim=1) # B x 1 x kN
                cost_targets_tensors.append(cost_targets_tensor)

        # ================== Value Update ================== #
        # calculate cdf
        with torch.no_grad():
            cum_prob = (torch.arange(self.n_quantiles, device=self.device, dtype=torch.float32) + 0.5)/self.n_quantiles
            cum_prob = cum_prob.view(1, 1, -1, 1) # 1 x 1 x M x 1

        for _ in range(self.n_epochs):
            # calculate quantile regression loss for reward
            current_reward_quantiles = self.reward_value(states_tensor, actions_tensor) # B x N x M
            # B x 1 x 1 x kN - B x N x M x 1 => B x N x M x kN
            pairwise_reward_delta = reward_targets_tensor.unsqueeze(-2) - current_reward_quantiles.unsqueeze(-1)
            reward_value_loss = torch.mean(pairwise_reward_delta*(cum_prob - (pairwise_reward_delta.detach() < 0).float()))
            self.reward_value_optimizer.zero_grad()
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_value.parameters(), self.max_grad_norm)
            self.reward_value_optimizer.step()

            cost_value_losses = []
            for cost_idx in range(self.num_costs):
                # calculate quantile regression loss for cost
                current_cost_quantiles = self.cost_values[cost_idx](states_tensor, actions_tensor)
                pairwise_cost_delta = cost_targets_tensors[cost_idx].unsqueeze(-2) - current_cost_quantiles.unsqueeze(-1)
                cost_value_loss = torch.mean(pairwise_cost_delta*(cum_prob - (pairwise_cost_delta.detach() < 0).float()))
                self.cost_value_optimizers[cost_idx].zero_grad()
                cost_value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cost_values[cost_idx].parameters(), self.max_grad_norm)
                self.cost_value_optimizers[cost_idx].step()
                cost_value_losses.append(cost_value_loss.item())
        # ================================================== #

        # ================= Policy Update ================= #
        # sample noise & get alpha
        with torch.no_grad():
            epsilons = torch.normal(mean=torch.zeros_like(actions_tensor), std=torch.ones_like(actions_tensor))
            alpha = self._getAlpha()

        # backup old policy
        means, _, stds = self.policy(states_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()
        cur_dists = TransformedDistribution(Normal(means, stds), TanhTransform())
        old_dists = TransformedDistribution(Normal(old_means, old_stds), TanhTransform())
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
        max_kl = self.max_kl

        # get objective
        objective, entropy = self._getObjective(states_tensor, alpha, epsilons, old_means, old_stds)
        g_tensor = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(kl, g_tensor)
        approx_g_tensor = self._Hx(kl, H_inv_g_tensor)
        g_H_inv_g_tensor = torch.dot(approx_g_tensor, H_inv_g_tensor)

        # for cost surrogate
        b_tensors = []
        H_inv_b_tensors = []
        c_scalars = []
        max_c_scalars = []
        cost_surrogates = []
        safety_mode = False
        for cost_idx in range(self.num_costs):
            cost_surrogate = self._getCostSurrogate(cost_idx, states_tensor, epsilons, old_means, old_stds)
            cost_scalar = cost_surrogate.item()
            b_tensor = flatGrad(cost_surrogate, self.policy.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(kl, b_tensor)
            approx_b_tensor = self._Hx(kl, H_inv_b_tensor)
            b_H_inv_b_tensor = torch.dot(approx_b_tensor, H_inv_b_tensor)
            max_c_scalar = np.sqrt(np.clip(2.0*max_kl*b_H_inv_b_tensor.item(), 0.0, np.inf))
            c_scalar = min(max_c_scalar, cost_scalar - self.limit_values[cost_idx])
            if cost_scalar >= self.limit_values[cost_idx]: safety_mode = True
            cost_surrogates.append(cost_surrogate)
            b_tensors.append(approx_b_tensor)
            H_inv_b_tensors.append(H_inv_b_tensor)
            c_scalars.append(c_scalar)
            max_c_scalars.append(max_c_scalar)
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
            scaling = 1.0 if approx_kl <= max_kl else np.sqrt(max_kl/approx_kl)

            # find search direction
            if approx_kl/max_kl - 1.0 > -0.001:
                for c_idx in range(len(c_vector)):
                    c_vector[c_idx] = min(max_c_scalars[c_idx], cost_surrogates[c_idx].item() - self.limit_values[c_idx] + self.zeta)
                const_lam_vector = solve_qp(P=(S_mat + np.eye(self.num_costs)*EPS), q=-c_vector, lb=np.zeros(self.num_costs))
                approx_kl = 0.5*np.dot(const_lam_vector, S_mat@const_lam_vector)
                scaling = 1.0 if approx_kl <= max_kl else np.sqrt(max_kl/approx_kl)
                lam_tensor = torch.tensor(const_lam_vector, device=self.device, dtype=torch.float32)
                delta_theta = -scaling*H_inv_B_tensor@lam_tensor
            else:
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
                    lam_tensor = torch.tensor(const_lam_vector, device=self.device, dtype=torch.float32)
                    delta_theta = -scaling*H_inv_B_tensor@lam_tensor

            # line search
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
            init_objective = objective.clone().detach()
            init_cost_surrogates = [cost_surr.clone().detach() for cost_surr in cost_surrogates]
            while True:
                theta = beta*delta_theta + init_theta
                beta *= self.line_decay
                self._applyParams(theta)
                means, _, stds = self.policy(states_tensor)
                cur_dists = TransformedDistribution(Normal(means, stds), TanhTransform())
                kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
                if kl > max_kl*1.5:
                    continue
                objective, entropy = self._getObjective(states_tensor, alpha, epsilons, old_means, old_stds)
                if not safety_mode and objective < init_objective:
                    continue
                is_feasible = True
                for cost_idx in range(self.num_costs):
                    cost_surrogate = self._getCostSurrogate(cost_idx, states_tensor, epsilons, old_means, old_stds)
                    if cost_surrogate - init_cost_surrogates[cost_idx] > max(-c_vector[cost_idx], 0.0):
                        is_feasible = False
                        break
                if is_feasible:
                    break
        # ================================================= #

        # Alpha Update
        if self.adaptive_alpha:
            entropy_constraint = self.entropy_d*self.action_dim
            self.damp_alpha = entropy_constraint - entropy
            alpha_loss = -self.log_alpha*(entropy_constraint - entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return objective.item(), [c.item() for c in init_cost_surrogates], reward_value_loss.item(), cost_value_losses, entropy.item(), kl.item()

    def save(self):
        save_dict = {
            'reward_value': self.reward_value.state_dict(),
            'policy': self.policy.state_dict(),
            'alpha': self.log_alpha,
            'reward_value_optimizer': self.reward_value_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }
        for i in range(self.num_costs):
            save_dict[f'cost_value_{i}'] = self.cost_values[i].state_dict()
            save_dict[f'cost_value_optimizer_{i}'] = self.cost_value_optimizers[i].state_dict()
        torch.save(save_dict, f"{self.checkpoint_dir}/model.pt")
        check_idx = len(glob.glob(f"{self.checkpoint_dir}*"))
        copy_tree(self.checkpoint_dir, self.checkpoint_dir + f'{check_idx}')
        cprint(f'[{self.name}] save success.', bold=True, color="blue")


    def _normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def _unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def _getObjective(self, states, alpha, epsilons, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        entropy = self._getEntropy(epsilons, means, log_stds, stds)
        pi = torch.tanh(means + epsilons*stds)
        with torch.no_grad():
            pi_old = torch.tanh(old_means + epsilons*old_stds)
            old_objective = torch.mean(self.reward_value(states, pi_old))
        objective = torch.mean(self.reward_value(states, pi)) + alpha*entropy
        objective -= self.discount_factor*old_objective
        objective /= (1.0 - self.discount_factor)
        return objective, entropy

    def _getCostSurrogate(self, cost_idx, states, epsilons, old_means, old_stds):
        means, _, stds = self.policy(states)
        pi = torch.tanh(means + epsilons*stds)
        cost_mean = torch.mean(self.cost_values[cost_idx](states, pi))
        cost_square_mean = torch.mean(torch.square(self.cost_values[cost_idx](states, pi)))
        cost_surrogate = cost_mean + self.sigma_units[cost_idx]*torch.sqrt(torch.clamp(cost_square_mean - cost_mean**2, EPS, np.inf))

        with torch.no_grad():
            pi_old = torch.tanh(old_means + epsilons*old_stds)
            old_cost_mean = torch.mean(self.cost_values[cost_idx](states, pi_old))
            old_cost_square_mean = torch.mean(torch.square(self.cost_values[cost_idx](states, pi_old)))
            old_cost_surrogate = old_cost_mean + self.sigma_units[cost_idx]*torch.sqrt(torch.clamp(old_cost_square_mean - old_cost_mean**2, EPS, np.inf))

        cost_surrogate -= self.discount_factor*old_cost_surrogate
        cost_surrogate /= (1.0 - self.discount_factor)
        return cost_surrogate

    def _getCostSurrogate2(self, cost_idx, states, epsilons, old_means, old_stds):
        means, _, stds = self.policy(states)
        pi = torch.tanh(means + epsilons*stds)
        with torch.no_grad():
            pi_old = torch.tanh(old_means + epsilons*old_stds)
            old_cost_mean = torch.mean(self.cost_values[cost_idx](states, pi_old))
            old_cost_square_mean = torch.mean(torch.square(self.cost_values[cost_idx](states, pi_old)))
        cost_mean = torch.mean(self.cost_values[cost_idx](states, pi))
        cost_mean -= self.discount_factor*old_cost_mean
        cost_mean /= (1.0 - self.discount_factor)
        cost_square_mean = torch.mean(torch.square(self.cost_values[cost_idx](states, pi)))
        cost_square_mean -= (self.discount_factor**2)*old_cost_square_mean
        cost_square_mean /= (1.0 - self.discount_factor**2)
        cost_surrogate = cost_mean + self.sigma_units[cost_idx]*torch.sqrt(torch.clamp(cost_square_mean - torch.square(cost_mean), EPS, np.inf))
        return cost_surrogate

    def _getEntropy(self, epsilon, mean, log_std, std):
        action = mean + epsilon*std
        entropy = -torch.mean(self.policy.logProb(action, mean, log_std, std))
        return entropy

    def _projection(self, quantiles1, weight1, quantiles2, weight2):
        n_quantiles1 = len(quantiles1)
        n_quantiles2 = len(quantiles2)
        n_quantiles3 = self.n_target_quantiles
        assert n_quantiles1 == self.n_quantiles*self.n_critics

        new_quantiles = np.zeros(n_quantiles3)
        cpp_quantiles1 = ctype_arr_convert(quantiles1)
        cpp_quantiles2 = ctype_arr_convert(quantiles2)
        cpp_new_quantiles = ctype_arr_convert(new_quantiles)

        self._lib.projection.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double*n_quantiles1), ctypes.c_int, ctypes.c_double, 
            ctypes.POINTER(ctypes.c_double*n_quantiles2), ctypes.c_int, ctypes.POINTER(ctypes.c_double*n_quantiles3)
        ]
        self._lib.projection(n_quantiles1, weight1, cpp_quantiles1, n_quantiles2, weight2, cpp_quantiles2, n_quantiles3, cpp_new_quantiles)
        new_quantiles = np.array(cpp_new_quantiles)
        return new_quantiles

    def _getQuantileTargets(self, rewards, dones, fails, rhos, next_quantiles):
        target_quantiles = np.zeros((next_quantiles.shape[0], self.n_target_quantiles))
        gae_target = rewards[-1] + self.discount_factor*(1.0 - fails[-1])*next_quantiles[-1]
        gae_weight = self.gae_coeff
        for t in reversed(range(len(target_quantiles))):
            target = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_quantiles[t]
            target = self._projection(target, 1.0 - self.gae_coeff, gae_target, gae_weight)
            target_quantiles[t, :] = target[:]
            if t != 0:
                if self.gae_coeff != 1.0:
                    gae_weight = self.gae_coeff*rhos[t]*(1.0 - dones[t-1])*(1.0 - self.gae_coeff + gae_weight)
                gae_target = rewards[t-1] + self.discount_factor*(1.0 - fails[t-1])*target
        return target_quantiles

    def _getTrainBatches(self, is_latest=False):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_lists = [[] for _ in range(self.num_costs)]

        with torch.no_grad():
            for env_idx in range(self.n_envs):
                n_latest_steps = min(len(self.replay_buffer[env_idx]), self.n_steps_per_env)
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
                norm_actions = np.array([traj[2] for traj in env_trajs])
                log_probs = np.array([traj[3] for traj in env_trajs])
                rewards = np.array([traj[4] for traj in env_trajs])
                dones = np.array([traj[5] for traj in env_trajs])
                fails = np.array([traj[6] for traj in env_trajs])
                next_states = np.array([traj[7] for traj in env_trajs])

                # convert to tensor
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                norm_actions_tensor = torch.tensor(norm_actions, device=self.device, dtype=torch.float32)
                mu_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)

                # for rho
                means_tensor, log_stds_tensor, stds_tensor = self.policy(states_tensor)
                old_log_probs_tensor = self.policy.logProb(norm_actions_tensor, means_tensor, log_stds_tensor, stds_tensor)
                rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 100.0)
                rhos = rhos_tensor.detach().cpu().numpy()

                # get GAEs and Tagets
                _, next_pi, _ = self.policy.sample(next_states_tensor)
                next_reward_quantiles_tensor = self.reward_value(next_states_tensor, next_pi).reshape(len(states), -1) # B x NM
                next_reward_quantiles = torch.sort(next_reward_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                reward_targets = self._getQuantileTargets(rewards, dones, fails, rhos, next_reward_quantiles)
                reward_targets_list.append(reward_targets)
                for cost_idx in range(self.num_costs):
                    costs = np.array([traj[8 + cost_idx] for traj in env_trajs])
                    next_cost_quantiles_tensor = self.cost_values[cost_idx](next_states_tensor, next_pi).reshape(len(states), -1) # B x NM
                    next_cost_quantiles = torch.sort(next_cost_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                    cost_targets = self._getQuantileTargets(costs, dones, fails, rhos, next_cost_quantiles)
                    cost_targets_lists[cost_idx].append(cost_targets)

                # save
                states_list.append(states)
                actions_list.append(actions)

        return states_list, actions_list, reward_targets_list, cost_targets_lists

    def _getAlpha(self):
        alpha = torch.exp(self.log_alpha)
        return alpha

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
            self.reward_value.load_state_dict(checkpoint['reward_value'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.log_alpha = checkpoint['alpha']
            self.reward_value_optimizer.load_state_dict(checkpoint['reward_value_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            for i in range(self.num_costs):
                self.cost_values[i].load_state_dict(checkpoint[f'cost_value_{i}'])
                self.cost_value_optimizers[i].load_state_dict(checkpoint[f'cost_value_optimizer_{i}'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            self.reward_value.initialize()
            for i in range(self.num_costs):
                self.cost_values[i].initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")


class EvalAgent(Agent):
    def __init__(self, args):
        super().__init__(args)

    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
