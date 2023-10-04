from typing import Optional, List

from utils.color import cprint
from models import Policy
from models import Value
import ctypes

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions import Normal
from collections import deque
from scipy.stats import norm
import numpy as np
import torch
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
        self.cost_alpha = args.cost_alpha
        self.sigma_unit = norm.pdf(norm.ppf(self.cost_alpha))/self.cost_alpha
        self.cost_d = args.cost_d

        # declare networks
        self.policy = Policy(args).to(args.device)
        self.reward_value = Value(args).to(args.device)
        self.cost_value = Value(args).to(args.device)

        # for alpha
        self.adaptive_alpha = args.adaptive_alpha
        self.alpha_lr = args.alpha_lr
        self.log_alpha = torch.tensor(np.log(args.alpha_init + EPS), requires_grad=True, device=args.device)

        # optimizers
        self.reward_value_optimizer = torch.optim.Adam(self.reward_value.parameters(), lr=self.lr)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

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

    def addTransition(self, env_idx, state, action, norm_action, log_prob, reward, cost, done, fail, next_state):
        self.replay_buffer[env_idx].append([state, action, norm_action, log_prob, reward, cost, done, fail, next_state])

    def train(self):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_list = []

        # latest trajectory
        temp_states_list, temp_actions_list, \
            temp_reward_targets_list, temp_cost_targets_list = self._getTrainBatches(is_latest=True)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        cost_targets_list += temp_cost_targets_list

        # random trajectory
        temp_states_list, temp_actions_list, \
            temp_reward_targets_list, temp_cost_targets_list = self._getTrainBatches(is_latest=False)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        cost_targets_list += temp_cost_targets_list

        # convert to tensor
        with torch.no_grad():
            states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
            actions_tensor = self._normalizeAction(torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32))
            reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
            cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_list, axis=0), device=self.device, dtype=torch.float32)
            reward_targets_tensor.unsqueeze_(dim=1) # B x 1 x kN
            cost_targets_tensor.unsqueeze_(dim=1) # B x 1 x kN

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

            # calculate quantile regression loss for cost
            current_cost_quantiles = self.cost_value(states_tensor, actions_tensor) # B x N x M
            # B x 1 x 1 x kN - B x N x M x 1 => B x N x M x kN
            pairwise_cost_delta = cost_targets_tensor.unsqueeze(-2) - current_cost_quantiles.unsqueeze(-1)
            cost_value_loss = torch.mean(pairwise_cost_delta*(cum_prob - (pairwise_cost_delta.detach() < 0).float()))
            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_value.parameters(), self.max_grad_norm)
            self.cost_value_optimizer.step()
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

        # get objective & cost surrogate
        objective, entropy = self._getObjective(states_tensor, alpha, epsilons, old_means, old_stds)
        cost_surrogate = self._getCostSurrogate(states_tensor, epsilons, old_means, old_stds)

        # get gradient
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        grad_b = flatGrad(-cost_surrogate, self.policy.parameters(), retain_graph=True)
        H_inv_g = self._conjugateGradient(kl, grad_g)
        approx_g = self._Hx(kl, H_inv_g)
        c_value = cost_surrogate.item() - self.cost_d/(1.0 - self.discount_factor)

        # ======== solve Lagrangian problem ======== #
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, H_inv_g)
            optim_case = 4
        else:
            H_inv_b = self._conjugateGradient(kl, grad_b)
            approx_b = self._Hx(kl, H_inv_b)

            scalar_q = torch.dot(approx_g, H_inv_g)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2/scalar_s
            B_value = 2*max_kl - c_value**2/scalar_s
            if c_value < 0 and B_value <= 0:
                optim_case = 3
            elif c_value < 0 and B_value > 0:
                optim_case = 2
            elif c_value >= 0 and B_value > 0:
                optim_case = 1
            else:
                optim_case = 0
        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q/(2*max_kl))
            nu = 0
        elif optim_case in [1, 2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value/B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q/(2*max_kl)), LB)
            f_a = lambda lam : -0.5*(A_value/(lam + EPS) + B_value*lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5*(scalar_q/(lam + EPS) + 2*max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam*c_value - scalar_r)/(scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2*max_kl/(scalar_s + EPS))
        # ========================================== #

        # line search
        with torch.no_grad():
            delta_theta = (1./(lam + EPS))*(H_inv_g + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
            init_objective = objective.clone().detach()
            init_cost_surrogate = cost_surrogate.clone().detach()
            while True:
                theta = beta*delta_theta + init_theta
                self._applyParams(theta)
                means, _, stds = self.policy(states_tensor)
                cur_dists = TransformedDistribution(Normal(means, stds), TanhTransform())
                kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
                objective, entropy = self._getObjective(states_tensor, alpha, epsilons, old_means, old_stds)
                cost_surrogate = self._getCostSurrogate(states_tensor, epsilons, old_means, old_stds)
                if kl <= max_kl*1.5 and (objective >= init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                    break
                beta *= self.line_decay
        # ================================================= #

        # Alpha Update
        if self.adaptive_alpha:
            entropy_constraint = self.entropy_d*self.action_dim
            alpha_loss = -self.log_alpha*(entropy_constraint - entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return objective.item(), cost_surrogate.item(), reward_value_loss.item(), cost_value_loss.item(), entropy.item(), kl.item(), alpha.item()

    def save(self):
        torch.save({
            'cost_value': self.cost_value.state_dict(),
            'reward_value': self.reward_value.state_dict(),
            'policy': self.policy.state_dict(),
            'alpha': self.log_alpha,
            'cost_value_optimizer': self.cost_value_optimizer.state_dict(),
            'reward_value_optimizer': self.reward_value_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def _softUpdate(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))

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

    def _getCostSurrogate(self, states, epsilons, old_means, old_stds):
        means, _, stds = self.policy(states)
        pi = torch.tanh(means + epsilons*stds)
        cost_mean = torch.mean(self.cost_value(states, pi))
        cost_square_mean = torch.mean(torch.square(self.cost_value(states, pi)))
        cost_surrogate = cost_mean + self.sigma_unit*torch.sqrt(torch.clamp(cost_square_mean - cost_mean**2, EPS, np.inf))
        with torch.no_grad():
            pi_old = torch.tanh(old_means + epsilons*old_stds)
            old_cost_mean = torch.mean(self.cost_value(states, pi_old))
            old_cost_square_mean = torch.mean(torch.square(self.cost_value(states, pi_old)))
            old_cost_surrogate = old_cost_mean + self.sigma_unit*torch.sqrt(torch.clamp(old_cost_square_mean - old_cost_mean**2, EPS, np.inf))
        cost_surrogate -= self.discount_factor*old_cost_surrogate
        cost_surrogate /= (1.0 - self.discount_factor)
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
        cost_targets_list = []

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
                costs = np.array([traj[5] for traj in env_trajs])
                dones = np.array([traj[6] for traj in env_trajs])
                fails = np.array([traj[7] for traj in env_trajs])
                next_states = np.array([traj[8] for traj in env_trajs])

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
                next_cost_quantiles_tensor = self.cost_value(next_states_tensor, next_pi).reshape(len(states), -1) # B x NM
                next_reward_quantiles = torch.sort(next_reward_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                next_cost_quantiles = torch.sort(next_cost_quantiles_tensor, dim=-1)[0].detach().cpu().numpy()
                reward_targets = self._getQuantileTargets(rewards, dones, fails, rhos, next_reward_quantiles)
                cost_targets = self._getQuantileTargets(costs, dones, fails, rhos, next_cost_quantiles)
                reward_targets_list.append(reward_targets)
                cost_targets_list.append(cost_targets)

                # save
                states_list.append(states)
                actions_list.append(actions)

        return states_list, actions_list, reward_targets_list, cost_targets_list

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
            checkpoint = torch.load(checkpoint_file)
            self.reward_value.load_state_dict(checkpoint['reward_value'])
            self.cost_value.load_state_dict(checkpoint['cost_value'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.log_alpha = checkpoint['alpha']
            self.reward_value_optimizer.load_state_dict(checkpoint['reward_value_optimizer'])
            self.cost_value_optimizer.load_state_dict(checkpoint['cost_value_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            self.reward_value.initialize()
            self.cost_value.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
