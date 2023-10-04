from network_base import (
    MLP, initWeights, normalize, unnormalize
)

from typing import Tuple
import numpy as np
import torch

class Critic(torch.nn.Module):
    def __init__(self, device:torch.device, state_dim:int, reward_dim:int, critic_cfg:dict) -> None:

        torch.nn.Module.__init__(self)

        self.device = device
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.critic_cfg = critic_cfg
        self.build()

    def build(self):
        activation_name = self.critic_cfg['mlp']['activation']
        self.activation = eval(f'torch.nn.{activation_name}')
        self.add_module('model', MLP(
            input_size=self.state_dim, output_size=self.reward_dim, \
            shape=self.critic_cfg['mlp']['shape'], activation=self.activation,
        ))
        if 'out_activation' in self.critic_cfg.keys():
            self.out_activation = eval(f'torch.nn.functional.{self.critic_cfg["out_activation"]}')
        else:
            self.out_activation = lambda x: x
        for item_idx in range(len(self.critic_cfg['clip_range'])):
            item = self.critic_cfg['clip_range'][item_idx]
            if type(item) == str:
                self.critic_cfg['clip_range'][item_idx] = eval(item)
        self.clip_range = self.critic_cfg['clip_range']

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        x = self.model(state)
        x = self.out_activation(x)
        x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        return x

    def getLoss(self, state:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self.forward(state), target)
    
    def initialize(self) -> None:
        self.apply(initWeights)
