from network_base import (
    MLP, initWeights, normalize, unnormalize
)

from typing import Tuple
import numpy as np
import torch
import gym


class Actor(torch.nn.Module):
    def __init__(self, device:torch.device, state_dim:int, action_dim:int, \
                action_bound_min:np.ndarray, action_bound_max:np.ndarray, actor_cfg:dict, \
                log_std_min:float=-4.0, log_std_max:float=2.0) -> None:

        torch.nn.Module.__init__(self)

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_min = torch.tensor(
            action_bound_min, device=device, dtype=torch.float32
        )
        self.action_bound_max = torch.tensor(
            action_bound_max, device=device, dtype=torch.float32
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # for model
        activation_name = actor_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.log_std_init = actor_cfg['log_std_init']
        self.add_module('model', MLP(
            input_size=self.state_dim, output_size=actor_cfg['mlp']['shape'][-1], \
            shape=actor_cfg['mlp']['shape'][:-1], activation=self.activation,
        ))
        self.add_module("mean_decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(actor_cfg['mlp']['shape'][-1], self.action_dim),
        ))
        self.add_module("std_decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(actor_cfg['mlp']['shape'][-1], self.action_dim),
        ))
        if 'out_activation' in actor_cfg.keys():
            self.out_activation = eval(f'torch.nn.functional.{actor_cfg["out_activation"]}')
        else:
            self.out_activation = lambda x: x
        
    def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        output: (mean, log_std, std)
        '''
        x = self.model(state)
        mean = self.out_activation(self.mean_decoder(x))
        log_std = self.std_decoder(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, log_std, std

    def updateActionDist(self, state:torch.Tensor, epsilon:torch.Tensor) -> None:
        self.action_mean, self.action_log_std, self.action_std = \
            self.forward(state)
        self.normal_action = self.action_mean + epsilon*self.action_std
        self.action_dist = torch.distributions.Normal(self.action_mean, self.action_std)

    def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            norm_action = self.action_mean
        else:
            norm_action = self.normal_action
        unnorm_action = unnormalize(norm_action, self.action_bound_min, self.action_bound_max)
        return norm_action, unnorm_action

    def getMeanStd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action_mean, self.action_std
    
    def getDist(self) -> torch.distributions.Distribution:
        return self.action_dist
        
    def getEntropy(self) -> torch.Tensor:
        '''
        return entropy of action distribution given state.
        '''
        entropy = torch.mean(torch.sum(self.action_dist.entropy(), dim=-1))
        return entropy
    
    def getLogProb(self) -> torch.Tensor:
        '''
        return log probability of action given state.
        '''
        log_prob = torch.sum(self.action_dist.log_prob(self.normal_action), dim=-1)
        return log_prob

    def initialize(self) -> None:
        for name, module in self.named_children():
            if name == 'std_decoder':
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)
