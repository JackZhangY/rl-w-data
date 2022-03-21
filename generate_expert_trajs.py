#! /home/zy/anaconda3/envs/il/bin/python

# import sys
import torch
import random
from il_offline_rl.model import Q_agent, AC_agent
from il_offline_rl.arguments import get_args
import argparse
import glob
import importlib
import os
import sys
import gym


import numpy as np
import yaml
from stable_baselines3.common.utils import set_random_seed

import torch.nn as nn
from  il_offline_rl.utils import get_saved_hyperparams, load_from_zip_file
from il_offline_rl.envs import make_vec_envs

def main(seed):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser(description='arguments for sampling expert trajectories')
    parser.add_argument("--env_name", type=str, default='BreakoutNoFrameskip-v4', help='RL task, default (Ant-v3)')
    parser.add_argument("--expert_algo", type=str, default='dqn', help='till now, choices: ppo|dqn|a2c(bug)')
    parser.add_argument("--num_trajs", type=int, default=1, help='number of expert trajectories')
    parser.add_argument("--num_processes", type=int, default=4, help='number of parallel envs')
    parser.add_argument("--demo_data_dir", type=str, default='/home/zy/zz/all_logs/rl-w-data/expert_trajs', help='path to save expert trajs')
    parser.add_argument("--rl_baseline_zoo_dir", type=str, default='/home/zy/zz/rl-baselines3-zoo/rl-trained-agents', help='model zoo path')
    parser.add_argument("--gpu_idx", type=int, default=0, help='gpu index')
    parser.add_argument("--cuda", action='store_true', default=False, help='whether to use gpu')
    parser.add_argument("--deterministic", action='store_true', default=False, help='whether to use deterministic action')
    parser.add_argument("--seed", type=int, default=seed, help='seed number')
    parser.add_argument("--gamma", type=float, default=0.99, help='use for vecnormalize the return')


    args = parser.parse_args()

    os.system(f'mkdir -p {args.demo_data_dir}/{args.expert_algo}')
    os.system(f'mkdir -p {args.demo_data_dir}/{args.expert_algo}/{args.env_name}')

    # use gpu or cpu
    torch.cuda.set_device(args.gpu_idx)
    device = torch.device('cuda:{}'.format(args.gpu_idx) if args.cuda else 'cpu')
    print(f'device: {device}')

    # set seed
    seed = args.seed
    print('setting seed: {}'.format(seed))
    set_random_seed(seed, using_cuda=True)

    # initial the env and expert model
    print('[Setting environment: {} hyperparams variables]'.format(args.env_name))
    log_path = os.path.join(args.rl_baseline_zoo_dir, args.expert_algo, f"{args.env_name}_1") # default suffix '_1'
    stats_path = os.path.join(log_path, args.env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    if stats_path is None:
        raise ValueError('Wrong stats path!')

    # setting hyperparameter with taking the config.yaml into account
    time = False
    if hyperparams.get('env_wrapper', None) is not None:
        if 'TimeFeatureWrapper' in hyperparams['env_wrapper']:
            time = True

    # print('hyperparams:{}'.format(hyperparams))
    # build envs
    is_atari = True if 'NoFrameskip' in args.env_name else False
    # deterministic = False if is_atari else True
    deterministic = args.deterministic
    env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, device, True, stats_path=stats_path,
                  hyperparams=hyperparams, time=time, normalize_obs=False, max_episode_steps=10000)

    input_dim = env.observation_space.shape[0]
    if env.action_space.__class__.__name__ == "Discrete":
        acs_dim = env.action_space.n
    elif env.action_space.__class__.__name__ == "Box":
        acs_dim = env.action_space.shape[0]
    else:
        raise ValueError('not support this type of action_space')

    # build expert policy
    policy_kwargs = {}
    policy_kwargs['hidden_size'] = 512 if is_atari else 64
    policy_kwargs['acti_fn'] = nn.ReLU() if is_atari else nn.Tanh()
    policy_kwargs['policy_dist'] = 'Cat' if is_atari else 'DNorm'
    policy_kwargs['is_mlp_base'] = False if is_atari else True
    policy_kwargs['is_V_critic'] = False if args.expert_algo in ['dqn', 'sac'] else True

    if hyperparams.get('policy_kwargs', None) is not None:
        expert_policy_kwargs = eval(hyperparams['policy_kwargs'])
        # print(expert_policy_kwargs)
        if expert_policy_kwargs.get('net_arch', None) is not None:
            policy_kwargs['hidden_size'] = expert_policy_kwargs['net_arch'][0]['pi'][0]
        if expert_policy_kwargs.get('activation_fn', None) is not None:
            policy_kwargs['acti_fn'] = expert_policy_kwargs['activation_fn']()


    policy = Q_agent if is_atari and args.expert_algo == 'dqn' else AC_agent

    model = policy(input_dim, acs_dim, **policy_kwargs).to(device)

    # print(list(model.named_parameters()))
    # print('model parameters')
    # for name, para in list(model.named_parameters()):
    #     print(name)
    #     print(para.shape)


    model_path = os.path.join(log_path, f"{args.env_name}.zip")
    if not os.path.isfile(model_path):
        raise ValueError(f"No model found for {args.expert_algo} on {args.env_name}, path:{model_path}")
    data, params, pytorch_variable = load_from_zip_file(model_path)
    # print('expert parameters')
    # for name, para in params['policy'].items():
    #     print(name)
    #     print(para.shape)
    # print(params['policy'])
    model.load_expert_model(params['policy'])
    model.set_agent_mode(training_mode=False)

    # sample expert trajs
    obs = env.reset()
    save_file_path = f'{args.demo_data_dir}/{args.expert_algo}/{args.env_name}'
    save_file_name = '{}_obs_acs_traj={}.npz'


    rtn_obs, rtn_acs, rtn_lens, ep_rewards = [], [], [], []
    save = True

    print(f'[start sampling expert({args.expert_algo}) trajs...]')
    steps = 0

    while True:
        with torch.no_grad():
            action = model.act(obs, deterministic=deterministic)
        if isinstance(env.action_space, gym.spaces.Box):
            clip_action = np.clip(action.cpu(), env.action_space.low, env.action_space.high)
        else:
            clip_action = action

        # todo: subsample
        try: # if vecnormalized mujoco
            ori_obs = env.venv.get_original_obs() # return np array
            if time: # if TimeFeatureWrapper
                ori_obs = ori_obs[:, :-1]
            rtn_obs.append(ori_obs)
        except: # if atari
            rtn_obs.append(obs.cpu().numpy().copy())

        rtn_acs.append(clip_action.cpu().numpy().copy())

        obs, reward, done, infos = env.step(clip_action)

        steps+=1

        for info in infos or done:
            if 'episode' in info.keys():
                print('---- the total statistic of No.{} trajectory: -----'.format(len(ep_rewards)))
                print(f"reward: {info['episode']['r']}")
                ep_rewards.append(info['episode']['r'])
                save = True
                rtn_obs.append(info['stack_terminal_observation'].unsqueeze(dim=0).cpu().numpy().copy())
                # note: no need to reset() when an episode is over, because the vec env has wrapped it at each step
                steps = 0

        if (len(ep_rewards) in list(range(1, args.num_trajs+1))) and save:
            rtn_obs_ = np.concatenate(rtn_obs, axis=0).astype(np.float32)
            rtn_acs_ = np.concatenate(rtn_acs, axis=0).astype(np.float32)


            print("trajectory length: {}".format(rtn_obs_.shape[0]))

            assert rtn_obs_.shape[0]-1 == rtn_acs_.shape[0], 'length error'
            data_dict = {'states': rtn_obs_, 'actions': rtn_acs_}

            save_path = os.path.join(save_file_path, save_file_name.format(args.env_name, args.seed * args.num_trajs + len(ep_rewards)))
            # np.savez_compressed(save_path, **data_dict)

            save = False

            rtn_obs.clear()
            rtn_acs.clear()


        if len(ep_rewards) == args.num_trajs:
            break

    env.close()
    return ep_rewards, args.env_name, args.deterministic

if __name__ == '__main__':
    total_ep_rewards = []
    deter = None
    for i in range(25):
        ep_rewards, env_name, deter = main(i)
        total_ep_rewards += ep_rewards

    if deter:
        title = 'deterministic'
    else:
        title = 'stochastic'

    print('########################')
    print(f'{title} expert on {env_name} : {np.mean(total_ep_rewards)}+-{np.std(total_ep_rewards)}')
    print('########################')












