from storage import RolloutBuffer
from critic import Critic
from actor import Actor

from typing import Tuple
import numpy as np
import torch
import os

EPS = 1e-8

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
        self.reward_dim = args.__dict__.get('reward_dim', 1)
        self.cost_dim = args.__dict__.get('cost_dim', 1)
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps

        # for RL
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.n_actor_iters = args.n_actor_iters
        self.n_critic_iters = args.n_critic_iters
        self.max_grad_norm = args.max_grad_norm
        self.max_kl = args.max_kl
        self.clip_ratio = args.clip_ratio
        self.gae_coeff = args.gae_coeff
        self.model_cfg = args.model

        # for model
        self.actor = Actor(
            self.device, self.obs_dim, self.action_dim, self.action_bound_min, 
            self.action_bound_max, self.model_cfg['actor']).to(self.device)
        self.reward_critic = Critic(self.device, self.obs_dim, self.reward_dim, self.model_cfg['reward_critic']).to(self.device)
        self.cost_critic = Critic(self.device, self.obs_dim, self.cost_dim, self.model_cfg['cost_critic']).to(self.device)

        # for constraint
        self.con_thresholds = np.array(args.con_thresholds)/(1.0 - self.discount_factor)
        self.con_thresholds = torch.tensor(self.con_thresholds, device=self.device, dtype=torch.float32)
        assert self.con_thresholds.shape[0] == self.cost_dim
        self.log_con_lambdas = torch.tensor(np.zeros(self.cost_dim), requires_grad=True, device=self.device)
        self.getConLambdas = lambda: torch.exp(self.log_con_lambdas)
        self.con_lambdas_lr = args.con_lambdas_lr

        # for buffer
        self.rollout_buffer = RolloutBuffer(
            self.device, self.obs_dim, self.action_dim, self.reward_dim, self.cost_dim,
            self.discount_factor, self.gae_coeff, self.n_envs, self.n_steps)

        # for optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=self.critic_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.critic_lr)
        self.con_lambdas_optimizer = torch.optim.Adam([self.log_con_lambdas], lr=self.con_lambdas_lr)


    """ public functions
    """
    def getAction(self, state:torch.Tensor, deterministic:bool) -> torch.Tensor:
        action_shape = state.shape[:-1] + (self.action_dim,)
        ε = torch.randn(action_shape, device=self.device)
        self.actor.updateActionDist(state, ε)

        norm_action, unnorm_action = self.actor.sample(deterministic)

        self.state = state.detach().cpu().numpy()
        self.action = norm_action.detach().cpu().numpy()
        return unnorm_action

    def step(self, rewards, costs, dones, fails, next_states):
        rewards = np.array(rewards).reshape(-1, self.reward_dim)
        costs = np.array(costs).reshape(-1, self.cost_dim)
        self.rollout_buffer.addTransition(self.state, self.action, rewards, costs, dones, fails, next_states)

    def train(self):
        # get batches
        states_tensor, actions_tensor, reward_targets_tensor, cost_targets_tensor, \
            reward_gaes_tensor, cost_gaes_tensor = self.rollout_buffer.getBatches(self.reward_critic, self.cost_critic)

        # ================== Critic Update ================== #
        for _ in range(self.n_critic_iters):
            reward_critic_loss = self.reward_critic.getLoss(states_tensor, reward_targets_tensor)
            self.reward_critic_optimizer.zero_grad()
            reward_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_critic.parameters(), self.max_grad_norm)
            self.reward_critic_optimizer.step()

            cost_critic_loss = self.cost_critic.getLoss(states_tensor, cost_targets_tensor)
            self.cost_critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optimizer.step()
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        with torch.no_grad():
            con_lambdas = self.getConLambdas()
            reduced_gaes_tensor = reward_gaes_tensor.squeeze() - torch.sum(con_lambdas.unsqueeze(0)*cost_gaes_tensor, dim=-1)
            reduced_gaes_tensor = (reduced_gaes_tensor - reduced_gaes_tensor.mean())/(reduced_gaes_tensor.std() + EPS)
            ε = torch.normal(mean=torch.zeros_like(actions_tensor), std=torch.ones_like(actions_tensor))
            self.actor.updateActionDist(states_tensor, ε)
            old_action_dists = self.actor.getDist()
            old_log_probs_tensor = torch.sum(old_action_dists.log_prob(actions_tensor), dim=-1)

        for _ in range(self.n_actor_iters):
            self.actor.updateActionDist(states_tensor, ε)
            cur_action_dists = self.actor.getDist()
            kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
            if kl > 1.5*self.max_kl: break
            cur_log_probs_tensor = torch.sum(cur_action_dists.log_prob(actions_tensor), dim=-1)
            prob_ratios_tensor = torch.exp(cur_log_probs_tensor - old_log_probs_tensor)
            clipped_ratios_tensor = torch.clamp(prob_ratios_tensor, min=1.0-self.clip_ratio, max=1.0+self.clip_ratio)
            actor_loss = -torch.mean(torch.minimum(reduced_gaes_tensor*prob_ratios_tensor, reduced_gaes_tensor*clipped_ratios_tensor))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
        
        self.actor.updateActionDist(states_tensor, ε)
        cur_action_dists = self.actor.getDist()
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_action_dists, cur_action_dists), dim=-1))
        entropy = self.actor.getEntropy()
        # ================================================= #

        # con beta update
        con_lambdas_tensor = self.getConLambdas()
        con_lambdas_loss = torch.mean(con_lambdas_tensor*(self.con_thresholds - cost_targets_tensor.mean(dim=0)).detach())
        self.con_lambdas_optimizer.zero_grad()
        con_lambdas_loss.backward()
        self.con_lambdas_optimizer.step()
        self.log_con_lambdas.data = torch.clamp(self.log_con_lambdas.data, min=-5.0, max=5.0)

        return actor_loss.item(), reward_critic_loss.item(), cost_critic_loss.item(), entropy.item(), kl.item(), con_lambdas_tensor.detach().cpu().numpy()

    def save(self, model_num):
        save_dict = {
            'actor': self.actor.state_dict(),
            'reward_critic': self.reward_critic.state_dict(),
            'cost_critic': self.cost_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),            
            'reward_critic_optimizer': self.reward_critic_optimizer.state_dict(),            
            'cost_critic_optimizer': self.cost_critic_optimizer.state_dict(),            
            'log_con_lambdas': self.log_con_lambdas.data,
            'con_lambdas_optimizer': self.con_lambdas_optimizer.state_dict(),
        }
        torch.save(save_dict, f"{self.checkpoint_dir}/model_{model_num}.pt")
        print(f'[{self.name}] save success.')

    def load(self, model_num):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model_{model_num}.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.actor.load_state_dict(checkpoint['actor'])
            self.reward_critic.load_state_dict(checkpoint['reward_critic'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.reward_critic_optimizer.load_state_dict(checkpoint['reward_critic_optimizer'])
            self.cost_critic_optimizer.load_state_dict(checkpoint['cost_critic_optimizer'])
            self.log_con_lambdas.data = checkpoint['log_con_lambdas']
            self.con_lambdas_optimizer.load_state_dict(checkpoint['con_lambdas_optimizer'])
            print(f'[{self.name}] load success.')
            return int(model_num)
        else:
            self.actor.initialize()
            self.reward_critic.initialize()
            self.cost_critic.initialize()
            print(f'[{self.name}] load fail.')
            return 0
