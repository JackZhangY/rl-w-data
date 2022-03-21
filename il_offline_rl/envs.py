import os
import pickle
import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces.box import Box
from gym.wrappers.clip_action import ClipAction
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecEnvWrapper)
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from stable_baselines3.common.running_mean_std import RunningMeanStd
from il_offline_rl.wrappers import TimeLimit, TimeFeatureWrapper

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets, time=False, max_episode_steps=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dmc2gym.make(domain_name=domain, task_name=task)
            env = ClipAction(env)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            # note: addition about TimeLimit wrapper for Atari
            if max_episode_steps is not None:
                env = TimeLimit(env, max_episode_steps=max_episode_steps)


        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            assert max_episode_steps == env._max_episode_steps, 'mismatch of max_episode_steps in Mujoco'
            print('max episode steps of current env:{}'.format(env._max_episode_steps))
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        else:
            env = Monitor(env, allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = EpisodicLifeEnv(env)
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = WarpFrame(env, width=84, height=84)
                env = ClipRewardEnv(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        #note: addition about TimeFeatureWrapper, which add the current timestep into the observation
        if time:
            env = TimeFeatureWrapper(env)

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  allow_early_resets,
                  log_dir=None,
                  stats_path=None,
                  hyperparams=None,
                  time=False,
                  absorbing_state=False,
                  normalize_obs=True,
                  fixed_obs_rms=None,
                  max_episode_steps=None,
                  num_frame_stack=None):
    """

    :param env_name:
    :param seed:
    :param num_processes:
    :param gamma: training envs when using VecNormalize, used for normalizing reward.
    :param log_dir:
    :param device:
    :param allow_early_resets:
    :param stats_path: whether to load expert vecnormalize.pkl and direct to expert model config, not useful for atari
    :param hyperparams: hyperparams for normalize setting.
    :param time: whether to use TimeFeatureWrapper
    :param absorbing_state: whether to use absorbing state wrapper
    :param normalize_obs: whether to normalize obs when training and evaluating agent, not for expert trajs
    :param fixed_obs_rms: if None, keep a dynamical update of obs_rms, if not, should be a sequence that includes fixed
                          mean and var of obs_rms (mean, var).
    :param max_episode_steps:
    :param num_frame_stack:
    :return:
    """
    assert not (time and absorbing_state), 'TimeFeatureWrapper and VecAbsorbingState are incompatible for now!'

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, time, max_episode_steps)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # load existing model, mainly for mujoco and pybullet envs, but only mujoco can be used now,
    # because ppo model of pybullet has wrong vecnormalize.pkl
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print("with params: {}".format(hyperparams['normalize_kwargs']))
            path_ = os.path.join(stats_path, 'vecnormalize.pkl')
            if os.path.exists(path_):
                # envs = VecNormalizeBullet(envs, training=False, **hyperparams['normalize_kwargs'])
                # envs.load_running_average(path_)
                envs = VecNormalize.load(path_, envs)
                envs.training = False
                envs.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")
    else: # for training/evaluating env on pybullet and mujoco
        if len(envs.observation_space.shape) == 1: # exclude atari
            if gamma is None:
                # eval env, return the original reward
                envs = VecNormalize(envs, norm_obs=normalize_obs, norm_reward=False)
                # todo: key setting
                if fixed_obs_rms is not None:
                    # envs.eval() # not dynamically update the obs_rms and ret_rms
                    envs.obs_rms.mean, envs.obs_rms.var = fixed_obs_rms# setting the fixed mean and var for obs_rms
            else:
                # train env, return the normalized reward
                envs = VecNormalize(envs, norm_obs=normalize_obs, gamma=gamma)
                # todo: key setting
                if fixed_obs_rms is not None:
                    # envs.eval()
                    envs.obs_rms.mean, envs.obs_rms.var =  fixed_obs_rms

            envs.eval() # not dynamically update the obs_rms and ret_rms

    envs = VecPyTorch(envs, device)
    if absorbing_state: # only for mujoco
        envs = VecAbsorbingState(envs)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecAbsorbingState(VecEnvWrapper):
    """
    only used for Mujoco env with obs shape: (dim, ), and incompatible with TimeFeatureWrapper

    and should only wrap the VecPyTorch object
    """
    def __init__(self, venv):
        super(VecAbsorbingState, self).__init__(venv)
        self.device = venv.device

    def reset(self):
        obs = self.venv.reset() # (env_num, dim)
        obs = F.pad(obs, (0, 1), 'constant', 0)
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return F.pad(obs, (0, 1), 'constant', 0), reward, done, info

    def get_absorbing_state(self):
        # observation_space still keep the original size
        obs = torch.zeros([self.observation_space.shape[0]+1,]).to(self.device)
        obs[-1].copy_(torch.tensor(1.))
        # obs = torch.zeros([self.observation_space.shape[0],])
        # obs = torch.cat([obs, torch.tensor([1.])], dim=0).to(self.device)
        return obs

# todo: normalize action wrapper

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device) # .float() change np.float64 to torch.float32
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    # actually not used, used in step_wait() of stable_baseline2 version
    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) /
                          np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                # line 341-343: add for true n_stack terminal observation
                terminal_obs = self.stacked_obs[i].clone()# (n_stack, w, d)
                terminal_obs[-self.shape_dim0:] = torch.from_numpy(infos[i]['terminal_observation'].astype(np.float32)) # (n_stack, w, d)
                infos[i]['stack_terminal_observation'] = terminal_obs
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
