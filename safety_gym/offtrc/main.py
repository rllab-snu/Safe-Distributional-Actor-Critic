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
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import random
import wandb
import time

def getPaser():
    parser = argparse.ArgumentParser(description='safe_rl')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--name', type=str, default='RL', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(1e6), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e6), help='# of time steps for slack message.')
    parser.add_argument('--total_steps', type=int, default=int(5e6), help='total training steps.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    # for env
    parser.add_argument('--env_name', type=str, default='Safexp-PointGoal1-v0', help='gym environment name.')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='# of maximum episode steps.')
    parser.add_argument('--n_envs', type=int, default=5, help='gym environment name.')
    parser.add_argument('--n_steps', type=int, default=1000, help='update after collecting n_steps.')
    parser.add_argument('--n_update_steps', type=int, default=10000, help='# of steps for update.')
    parser.add_argument('--len_replay_buffer', type=int, default=100000, help='length of replay buffer.')
    # for networks
    parser.add_argument('--activation', type=str, default='relu', help='activation function. relu, tanh, sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=-1.0, help='log of initial std.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate.')
    parser.add_argument('--n_epochs', type=int, default=20, help='# of value updates.')
    parser.add_argument('--gae_coeff', type=float, default=0.97, help='gae coefficient.')
    # trust region
    parser.add_argument('--damping_coeff', type=float, default=0.01, help='damping coefficient.')
    parser.add_argument('--num_conjugate', type=int, default=10, help='# of maximum conjugate step.')
    parser.add_argument('--line_decay', type=float, default=0.8, help='line decay.')
    parser.add_argument('--max_kl', type=float, default=0.001, help='maximum kl divergence.')
    # constraint
    parser.add_argument('--cost_alpha', type=float, default=1.0, help='CVaR\'s alpha.')
    parser.add_argument('--cost_d', type=float, default=0.025, help='constraint limit value.')
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
    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    tf.disable_eager_execution()

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

    # define agent
    agent = Agent(args)

    # logger
    objective_logger = Logger(args.save_dir, 'objective')
    cost_surrogate_logger = Logger(args.save_dir, 'cost_surrogate')
    v_loss_logger = Logger(args.save_dir, 'v_loss')
    cost_v_loss_logger = Logger(args.save_dir, 'cost_v_loss')
    cost_var_v_loss_logger = Logger(args.save_dir, 'cost_var_v_loss')
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
        trajectories = [[] for _ in range(args.n_envs)]
        step = 0
        while step < args.n_steps:
            env_cnts += 1
            step += args.n_envs
            total_step += args.n_envs

            actions, clipped_actions, means, stds = agent.getActions(observations, True)
            next_observations, rewards, dones, infos = vec_env.step(clipped_actions)

            for env_idx in range(args.n_envs):
                reward_history[env_idx].append(rewards[env_idx])
                if 'num_cv' in infos[env_idx].keys():
                    cv_history[env_idx].append(infos[env_idx]['num_cv'])
                else:
                    cv_history[env_idx].append(1 if infos[env_idx]['cost'] >= 0.5 else 0)
                cost_history[env_idx].append(infos[env_idx]['cost'])

                fail = env_cnts[env_idx] < args.max_episode_steps if dones[env_idx] else False
                dones[env_idx] = True if env_cnts[env_idx] >= args.max_episode_steps else dones[env_idx]
                next_observation = infos[env_idx]['terminal_observation'] if dones[env_idx] else next_observations[env_idx]
                trajectories[env_idx].append([observations[env_idx], actions[env_idx], rewards[env_idx], infos[env_idx]['cost'], dones[env_idx], next_observation, fail, means[env_idx], stds[env_idx]])

                if dones[env_idx]:
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

            observations = next_observations
        # ==================================== #

        v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy = agent.train(trajectories)

        objective_logger.write([step, objective])
        cost_surrogate_logger.write([step, cost_surrogate])
        v_loss_logger.write([step, v_loss])
        cost_v_loss_logger.write([step, cost_v_loss])
        cost_var_v_loss_logger.write([step, cost_var_v_loss])
        kl_logger.write([step, kl])
        entropy_logger.write([step, entropy])

        print_len = max(int(args.n_steps/args.max_episode_steps), args.n_envs)
        log_data = {
            "rollout/step": total_step, 
            "rollout/score": score_logger.get_avg(print_len), 
            "rollout/cost": cost_logger.get_avg(print_len),
            "rollout/ep_len": eplen_logger.get_avg(print_len),
            "rollout/ep_cv": cv_logger.get_avg(print_len),
            "train/value_loss":v_loss_logger.get_avg(), 
            "train/cost_value_loss":cost_v_loss_logger.get_avg(), 
            "train/cost_var_value_loss":cost_var_v_loss_logger.get_avg(), 
            "metric/objective":objective_logger.get_avg(), 
            "metric/cost_surrogate":cost_surrogate_logger.get_avg(), 
            "metric/kl":kl_logger.get_avg(), 
            "metric/entropy":entropy_logger.get_avg(),
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
            objective_logger.save()
            cost_surrogate_logger.save()
            v_loss_logger.save()
            cost_v_loss_logger.save()
            cost_var_v_loss_logger.save()
            entropy_logger.save()
            kl_logger.save()
            score_logger.save()
            eplen_logger.save()
            cv_logger.save()
            cost_logger.save()


def test(args):
    # for random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    random.seed(args.seed)
    tf.disable_eager_execution()

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
            action, clipped_action, mean, std = agent.getAction(obs, False)
            obs, reward, done, info = env.step(clipped_action)
            obs = obs_rms.normalize(obs)
            env.render()
            score += reward
            if 'num_cv' in info.keys():
                num_cv = info['num_cv']
            else:
                num_cv = 1 if info['cost'] >= 0.5 else 0
            cv += num_cv
            cost += info['cost']
            if done: break
            time.sleep(0.001)
        print(f"score : {score:.3f}, cv : {cv}, cost: {cost}")



if __name__ == "__main__":
    parser = getPaser()
    args = parser.parse_args()
    # ==== processing args ==== #
    # save_dir
    args.save_dir = f"results/{args.name}_s{args.seed}"
    # ========================= #


    if args.test:
        test(args)
    else:
        train(args)
