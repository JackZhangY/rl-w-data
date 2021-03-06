import logging
import torch
import numpy as np
import os
import gym
from il_offline_rl.utils import make_dir
from torch.utils.tensorboard import SummaryWriter



class Logger(object):
    def __init__(self, log_dir, file_txt=None, filemode='w'):
        if file_txt is not None and file_txt.endswith('.txt'):
            make_dir(f'{log_dir}')

        self.log_dir = log_dir
        self.file_txt = file_txt
        self.writer = SummaryWriter(self.log_dir)

    def save_config(self, config_info):
        with open(f'{self.log_dir}/{self.file_txt}', 'w', encoding='utf-8') as f:
            f.write(config_info)

    # usage: self.add_scalar(args1, args2, args3)
    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(self, tag, values, global_step=None):
        self.writer.add_histogram(tag, values, global_step)



class BaseAlgo(Logger):
    """
    the Base Algorithm mainly include a Logger used to record the intermediate results,
    and the module used for online sampling with env and evaluating the imitation agent with eval env
    """

    def __init__(
            self, log_dir, eval_env, eval_log_interval, deterministic_eval=True, file_txt='log.txt',
            eval_num_trajs=10, env=None, online_replaybuffer=None):

        super(BaseAlgo, self).__init__(log_dir, file_txt)
        self.eval_env = eval_env
        self.eval_num_trajs = eval_num_trajs
        self.eval_times = 1
        self.obs_shape = eval_env.observation_space.shape
        self.action_space = eval_env.action_space
        self.eval_log_interval = eval_log_interval
        self.deterministic_eval = deterministic_eval

        # for discrete-action envs (Atari, LunarLander, CartPole), we use Q_agent, otherwise AC_agent
        self.is_discrete = True if eval_env.action_space.__class__.__name__ == 'Discrete' else False

        if env is not None: # used for online setting
            self.env = env
            self.online_rb = online_replaybuffer
            # todo: multiprocess online envs
            assert env.num_envs == 1, 'online sampling not support multiprocess envs now'
            self.total_timesteps = 0
            self.is_offline_rl = False
        else:
            self.is_offline_rl = True

        # to be overridden
        self.agent = None

    def evaluate(self):
        print('implementing No.{} policy evaluation...'.format(self.eval_times))
        eval_episode_rewards = []
        obs = self.eval_env.reset()

        end_flag = False
        while True:
            with torch.no_grad():
                action = self.agent.act(obs, deterministic=self.deterministic_eval)
                action = torch.clamp(action, -1, 1)
                # np.clip can only process the tensor stored in cpu
                # Though action has been transformed into [-1, 1] by TanhTransform and normalize action wrapper,
                # DiagGaussian is a gaussian policy without TanhTransfrom

            obs, reward, done, infos = self.eval_env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])
                    if len(eval_episode_rewards) >= self.eval_num_trajs:
                        end_flag = True
                        break
            if end_flag:
                break

        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
        self.eval_times += 1

        return np.mean(eval_episode_rewards)

    def train(self):
        """
        func include the whole training process
        :return:
        """
        raise NotImplementedError

    def update(self):
        """
        func used to update the training agent
        :return:
        """
        raise NotImplementedError

