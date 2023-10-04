# ===== add python path ===== #
import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

from utils.vectorize import CustomSubprocVecEnv3
from utils.vectorize import CustomSubprocVecEnv2
from utils.vectorize import CustomSubprocVecEnv
from utils.normalize import RunningMeanStd
from utils.slackbot import Slackbot
from utils.logger import Logger
from utils.color import cprint
from utils.env import Env
from agent import Agent
import utils.register

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from collections import deque
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import time
import gym

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--name', type=str, default='RL', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(1e7), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    # for env
    parser.add_argument('--env_name', type=str, default='MITCheetah-v0', help='gym environment name.')
    parser.add_argument('--max_episode_steps', type=int, default=500, help='maximum steps of each episode.')
    parser.add_argument('--n_envs', type=int, default=5, help='# of environments.')
    parser.add_argument('--n_steps', type=int, default=1000, help='update after collecting n_steps.')
    parser.add_argument('--n_past_steps', type=int, default=0, help='# of past steps for cost mean & cost var mean.')
    parser.add_argument('--n_update_steps', type=int, default=10000, help='update steps.')
    parser.add_argument('--len_replay_buffer', type=int, default=100000, help='length of replay buffer.')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-3.0, help='log of initial std.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum of grad_nrom.')
    parser.add_argument('--lr', type=float, default=3e-4, help='value learning rate.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--n_epochs', type=int, default=100, help='# of updates.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='GAE coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.001, help='maximum kl divergence.')
    # for constraint
    parser.add_argument('--cost_alphas', type=str, default="1.0, 1.0, 1.0", help='cost alpha of CVaR.')
    parser.add_argument('--limit_values', type=str, default="0.025, 0.025, 0.8", help='cost limit value.')
    return parser

def train(args):
    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define Environment
    env_id = lambda: Env(args.env_name, args.seed, args.max_episode_steps)
    vec_env = make_vec_env(
        env_id=env_id, n_envs=args.n_envs, seed=args.seed,
        vec_env_cls=CustomSubprocVecEnv3,
        vec_env_kwargs={'start_method':'spawn', 'args':args},
    )

    # set args value for env
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high
    args.num_costs = vec_env.num_costs

    # define agent
    agent = Agent(args)

    # wandb
    if args.wandb:
        project_name = '[SDAC] cheetah'
        wandb.init(
            project=project_name, 
            config=args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # logger
    objective_logger = Logger(args.save_dir, 'objective')
    reward_value_loss_logger = Logger(args.save_dir, 'reward_value_loss')
    entropy_logger = Logger(args.save_dir, 'entropy')
    kl_logger = Logger(args.save_dir, 'kl')
    score_logger = Logger(args.save_dir, 'score')
    eplen_logger = Logger(args.save_dir, 'eplen')
    cv_logger = Logger(args.save_dir, 'cv')
    cost_surrogate_loggers = []
    cost_value_loss_loggers = []
    cost_var_value_loss_loggers = []
    cost_loggers = []
    for cost_idx in range(args.num_costs):
        cost_surrogate_loggers.append(Logger(args.save_dir, f'cost_surrogate_{cost_idx}'))
        cost_value_loss_loggers.append(Logger(args.save_dir, f'cost_value_loss_{cost_idx}'))
        cost_var_value_loss_loggers.append(Logger(args.save_dir, f'cost_var_value_loss_{cost_idx}'))
        cost_loggers.append(Logger(args.save_dir, f'cost_{cost_idx}'))

    # train
    observations = vec_env.reset()
    reward_history = [[] for _ in range(args.n_envs)]
    cv_history = [[] for _ in range(args.n_envs)]
    cost_histories = [[[] for _ in range(args.n_envs)] for _ in range(args.num_costs)]
    env_cnts = np.zeros(args.n_envs)
    total_step = 0
    slack_step = 0
    save_step = 0
    while total_step < args.total_steps:
        # ======= collect trajectories ======= #
        step = 0
        while step < args.n_steps:
            env_cnts += 1
            step += args.n_envs
            total_step += args.n_envs

            with torch.no_grad():
                obs_tensor = torch.tensor(observations, device=args.device, dtype=torch.float32)
                action_tensor, clipped_action_tensor, mean_tensor, std_tensor = agent.getAction(obs_tensor, True)
                actions = action_tensor.detach().cpu().numpy()
                clipped_actions = clipped_action_tensor.detach().cpu().numpy()
                means = mean_tensor.detach().cpu().numpy()
                stds = std_tensor.detach().cpu().numpy()
            next_observations, rewards, dones, infos = vec_env.step(clipped_actions)

            for env_idx in range(args.n_envs):
                reward_history[env_idx].append(rewards[env_idx])
                cv_history[env_idx].append(infos[env_idx]['num_cv'])
                for cost_idx in range(args.num_costs):
                    cost_histories[cost_idx][env_idx].append(infos[env_idx]['costs'][cost_idx])

                fail = env_cnts[env_idx] < args.max_episode_steps if dones[env_idx] else False
                done = True if env_cnts[env_idx] >= args.max_episode_steps else dones[env_idx]
                next_observation = infos[env_idx]['terminal_observation'] if dones[env_idx] else next_observations[env_idx]
                agent.addTransition(
                    env_idx, observations[env_idx], actions[env_idx], means[env_idx], stds[env_idx],
                    rewards[env_idx], float(done), float(fail), next_observation, *infos[env_idx]['costs'],
                )

                if dones[env_idx]:
                    ep_len = len(reward_history[env_idx])
                    eplen_logger.write([ep_len, ep_len])
                    score_logger.write([ep_len, np.sum(reward_history[env_idx])])
                    cv_logger.write([ep_len, np.sum(cv_history[env_idx])])
                    for cost_idx in range(args.num_costs):
                        cost_loggers[cost_idx].write([ep_len, np.sum(cost_histories[cost_idx][env_idx])])
                        cost_histories[cost_idx][env_idx] = []
                    reward_history[env_idx] = []
                    cv_history[env_idx] = []
                    env_cnts[env_idx] = 0

            observations = next_observations
        # ==================================== #

        objective, cost_surrogates, reward_value_loss, cost_value_losses, \
            cost_var_value_losses, entropy, kl, optim_case = agent.train()
        reward_value_loss_logger.write([step, reward_value_loss])
        objective_logger.write([step, objective])
        entropy_logger.write([step, entropy])
        kl_logger.write([step, kl])
        for cost_idx in range(args.num_costs):
            cost_surrogate_loggers[cost_idx].write([step, cost_surrogates[cost_idx]])
            cost_value_loss_loggers[cost_idx].write([step, cost_value_losses[cost_idx]])
            cost_var_value_loss_loggers[cost_idx].write([step, cost_var_value_losses[cost_idx]])

        print_len = max(int(args.n_steps/args.max_episode_steps), args.n_envs)
        log_data = {
            "rollout/step": total_step, 
            "rollout/score": score_logger.get_avg(print_len), 
            "rollout/ep_len": eplen_logger.get_avg(print_len),
            "rollout/ep_cv": cv_logger.get_avg(print_len),
            "metric/entropy":entropy_logger.get_avg(), 
            "metric/kl":kl_logger.get_avg(), 
            "metric/objective":objective_logger.get_avg(), 
            "metric/optim_case":optim_case, 
            "train/reward_value_loss":reward_value_loss_logger.get_avg(), 
        }
        for cost_idx in range(args.num_costs):
            log_data[f'rollout/cost_{cost_idx}'] = cost_loggers[cost_idx].get_avg(print_len)
            log_data[f"metric/cost_surrogate_{cost_idx}"] = cost_surrogate_loggers[cost_idx].get_avg()
            log_data[f'train/cost_value_loss_{cost_idx}'] = cost_value_loss_loggers[cost_idx].get_avg()
            log_data[f"train/cost_var_value_loss_{cost_idx}"] = cost_var_value_loss_loggers[cost_idx].get_avg(), 
        if args.wandb:
            wandb.log(log_data)
        print(log_data)

        if total_step - slack_step >= args.slack_freq and args.slack:
            slackbot.sendMsg(f"{project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
            slack_step += args.slack_freq

        if total_step - save_step >= args.save_freq:
            save_step += args.save_freq
            agent.save()
            reward_value_loss_logger.save()
            objective_logger.save()
            entropy_logger.save()
            kl_logger.save()
            score_logger.save()
            eplen_logger.save()
            cv_logger.save()
            for cost_idx in range(args.num_costs):
                cost_surrogate_loggers[cost_idx].save()
                cost_value_loss_loggers[cost_idx].save()
                cost_var_value_loss_loggers[cost_idx].save()
                cost_loggers[cost_idx].save()


def test(args):
    # define Environment
    env = Env(args.env_name, args.seed, args.max_episode_steps)
    obs_rms = RunningMeanStd(args.save_dir, env.observation_space.shape[0])
    episodes = int(10)

    # set args value for env
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high

    # define agent
    agent = Agent(args)

    for episode in range(episodes):
        obs = env.reset()
        obs = obs_rms.normalize(obs)
        done = False
        score = 0.0
        cost = 0.0
        cv = 0
        step = 0
        while True:
            step += 1
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=args.device, dtype=torch.float32)
                action_tensor, _, _, _ = agent.getAction(obs_tensor, True)
                action = action_tensor.detach().cpu().numpy()
            obs, reward, done, info = env.step(action)
            obs = obs_rms.normalize(obs)
            env.render()
            score += reward
            cv += info['num_cv']
            cost += info['cost']
            if done: break
            time.sleep(0.01)
        print(f"score : {score:.3f}, cv : {cv}, cost: {cost}")

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
    # set gpu
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device(f'cuda:{args.gpu_idx}')
        cprint('[torch] cuda is used.', bold=True, color='cyan')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.', bold=True, color='cyan')
    args.device = device
    # ========================= #

    if args.test:
        test(args)
    else:
        train(args)
