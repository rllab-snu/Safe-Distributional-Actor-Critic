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

from utils.vectorize import CustomSubprocVecEnv2
from utils.normalize import RunningMeanStd
from utils.slackbot import Slackbot
from utils.logger import Logger
from utils.color import cprint
from utils.env import Env
from agent import Agent
import utils.register

from stable_baselines3.common.env_util import make_vec_env
from ruamel.yaml import YAML
import numpy as np
import argparse
import random
import torch
import wandb
import time

def getPaser():
    parser = argparse.ArgumentParser(description='safe_rl')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(5e6), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--env_name', type=str, default='Safexp-PointGoal1-v0', help='gym environment name.')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='# of maximum episode steps.')
    parser.add_argument('--n_envs', type=int, default=5, help='gym environment name.')
    parser.add_argument('--cfg_path', type=str, default="safetygym.yaml")
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    return parser

def train(args):
    # wandb
    if args.wandb:
        project_name = '[Constrained RL] SafetyGym'
        wandb.init(
            project=project_name, 
            config=args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    # slackbot
    if args.slack:
        slackbot = Slackbot()

    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # define Environment
    env_id = lambda:Env(args.env_name, args.seed, args.max_episode_steps)
    vec_env = make_vec_env(
        env_id=env_id, n_envs=args.n_envs, seed=args.seed,
        vec_env_cls=CustomSubprocVecEnv2,
        vec_env_kwargs={'args':args, 'start_method':'spawn'},
    )

    # set args value for env
    args.obs_dim = vec_env.observation_space.shape[0]
    args.action_dim = vec_env.action_space.shape[0]
    args.action_bound_min = vec_env.action_space.low
    args.action_bound_max = vec_env.action_space.high
    args.reward_dim = 1
    args.cost_dim = 1

    # define agent
    agent = Agent(args)

    # logger
    actor_loss_logger = Logger(args.save_dir, 'actor_loss')
    reward_critic_loss_logger = Logger(args.save_dir, 'reward_critic_loss')
    cost_critic_loss_logger = Logger(args.save_dir, 'cost_critic_loss')
    kl_logger = Logger(args.save_dir, 'kl')
    entropy_logger = Logger(args.save_dir, 'entropy')
    score_logger = Logger(args.save_dir, 'score')
    eplen_logger = Logger(args.save_dir, 'eplen')
    cost_logger = Logger(args.save_dir, 'cost')
    cv_logger = Logger(args.save_dir, 'cv')

    # train
    observations = vec_env.reset()
    reward_history = [[] for _ in range(args.n_envs)]
    cost_history = [[] for _ in range(args.n_envs)]
    cv_history = [[] for _ in range(args.n_envs)]
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
                action_tensor = agent.getAction(obs_tensor, False)
                actions = action_tensor.cpu().numpy()

            clipped_actions = np.clip(actions, args.action_bound_min, args.action_bound_max)
            next_observations, rewards, dones, infos = vec_env.step(clipped_actions)

            temp_next_observations = np.zeros((args.n_envs, args.obs_dim))
            temp_rewards = np.zeros((args.n_envs, args.reward_dim))
            temp_costs = np.zeros((args.n_envs, args.cost_dim))
            temp_dones = np.zeros(args.n_envs)
            temp_fails = np.zeros(args.n_envs)
            for env_idx in range(args.n_envs):
                reward_history[env_idx].append(rewards[env_idx])
                if 'num_cv' in infos[env_idx].keys():
                    cv_history[env_idx].append(infos[env_idx]['num_cv'])
                else:
                    cv_history[env_idx].append(1 if infos[env_idx]['cost'] >= 0.5 else 0)
                cost_history[env_idx].append(infos[env_idx]['cost'])

                temp_next_observations[env_idx, :] = infos[env_idx]['terminal_observation'] if dones[env_idx] else next_observations[env_idx]
                temp_rewards[env_idx, 0] = rewards[env_idx]
                temp_costs[env_idx, 0] = infos[env_idx]['cost']
                temp_fails[env_idx] = env_cnts[env_idx] < args.max_episode_steps if dones[env_idx] else False
                temp_dones[env_idx] = True if env_cnts[env_idx] >= args.max_episode_steps else dones[env_idx]

                if temp_dones[env_idx]:
                    ep_len = len(reward_history[env_idx])
                    score = np.sum(reward_history[env_idx])
                    ep_cv = np.sum(cv_history[env_idx])
                    cost_sum = np.sum(cost_history[env_idx])

                    score_logger.write([ep_len, score])
                    eplen_logger.write([ep_len, ep_len])
                    cost_logger.write([ep_len, cost_sum])
                    cv_logger.write([ep_len, ep_cv])

                    reward_history[env_idx] = []
                    cost_history[env_idx] = []
                    cv_history[env_idx] = []
                    env_cnts[env_idx] = 0

            agent.step(temp_rewards, temp_costs, temp_dones, temp_fails, temp_next_observations)
            observations = next_observations
        # ==================================== #

        actor_loss, reward_critic_loss, cost_critic_loss, entropy, kl, con_lmabdas = agent.train()

        actor_loss_logger.write([step, actor_loss])
        reward_critic_loss_logger.write([step, reward_critic_loss])
        cost_critic_loss_logger.write([step, cost_critic_loss])
        kl_logger.write([step, kl])
        entropy_logger.write([step, entropy])

        print_len = max(int(args.n_steps/args.max_episode_steps), args.n_envs)
        log_data = {
            "rollout/step": total_step, 
            "rollout/score": score_logger.get_avg(print_len), 
            "rollout/cost": cost_logger.get_avg(print_len),
            "rollout/ep_len": eplen_logger.get_avg(print_len),
            "rollout/ep_cv": cv_logger.get_avg(print_len),
            "train/actor_loss":actor_loss_logger.get_avg(),
            "train/reward_critic_loss":reward_critic_loss_logger.get_avg(),
            "train/cost_critic_loss":cost_critic_loss_logger.get_avg(),
            "metric/kl":kl_logger.get_avg(), 
            "metric/entropy":entropy_logger.get_avg(),
            "metric/con_lambda":con_lmabdas[0],
        }
        if args.wandb:
            wandb.log(log_data)
        print(log_data)

        if total_step - slack_step >= args.slack_freq and args.slack:
            slackbot.sendMsg(f"{project_name}\nname: {wandb.run.name}\nsteps: {total_step}\nlog: {log_data}")
            slack_step += args.slack_freq

        if total_step - save_step >= args.save_freq:
            save_step += args.save_freq
            agent.save()
            actor_loss_logger.save()
            reward_critic_loss_logger.save()
            cost_critic_loss_logger.save()
            entropy_logger.save()
            kl_logger.save()
            score_logger.save()
            eplen_logger.save()
            cost_logger.save()
            cv_logger.save()


def test(args):
    pass


if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        algo_cfg = YAML().load(f)
    for key in algo_cfg.keys():
        args.__dict__[key] = algo_cfg[key]

    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
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
