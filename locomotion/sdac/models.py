import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m, init_bias=0.0):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_bias, 0.01)


class Value(nn.Module):
    def __init__(self, args, positive=False):
        super(Value, self).__init__()

        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation
        self.n_critics = args.n_critics
        self.n_quantiles = args.n_quantiles

        # Q architecture
        self.critics = []
        for critic_idx in range(self.n_critics):
            modules = [
                nn.Linear(self.state_dim + self.action_dim, self.hidden1_units),
                eval(f'nn.{self.activation}()'), 
                nn.Linear(self.hidden1_units, self.hidden2_units), 
                eval(f'nn.{self.activation}()'), 
                nn.Linear(self.hidden2_units, self.n_quantiles), 
            ]
            if positive: modules.append(nn.Softplus())
            critic = nn.Sequential(*modules)
            self.add_module(f"critic_{critic_idx}", critic)
            self.critics.append(critic)


    def forward(self, state, action):
        '''
        outputs: batch_size x n_critics x n_quantiles or n_critics x n_quantiles
        '''
        sa = torch.cat([state, action], -1)
        quantiles = []
        for critic_idx in range(self.n_critics):
            x = self.critics[critic_idx](sa)
            quantiles.append(x)
        x = torch.stack(quantiles, dim=-2)
        return x
    
    def initialize(self):
        self.apply(initWeights)


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation
        self.log_std_init = args.log_std_init
        
        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = eval(f'F.{self.activation.lower()}')
        
        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.fc_log_std = nn.Linear(self.hidden2_units, self.action_dim)


    def forward(self, state):
        x = self.act_fn(self.fc1(state))
        x = self.act_fn(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def sample(self, state):
        mean, log_std, std = self.forward(state)
        normal = Normal(mean, std)
        noise_action = normal.rsample()

        mu = torch.tanh(mean)
        pi = torch.tanh(noise_action)
        logp_pi = self.logProb(noise_action, mean, log_std, std)
        return mu, pi, logp_pi

    @torch.jit.export
    def logProb(self, noise_action, mean, log_std, std):
        normal = Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(noise_action), dim=-1)
        log_prob -= torch.sum(2.0*(np.log(2.0) - noise_action - torch.nn.Softplus()(-2.0*noise_action)), dim=-1)
        return log_prob

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx == 3:
                print("init bias:", self.log_std_init)
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)
