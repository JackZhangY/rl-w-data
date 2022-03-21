import numpy as np
import torch
import torch.nn as nn
from copy import copy
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy

from il_offline_rl.distributions import Bernoulli, Categorical, DiagGaussian, TransformedGaussian, TransformedGaussian2
from il_offline_rl.utils import init



# DQN fro Atari
Q_CNNbaseList = [
    ('critic.critic_base.base.0.weight', 'q_net.features_extractor.cnn.0.weight'),
    ('critic.critic_base.base.0.bias', 'q_net.features_extractor.cnn.0.bias'),
    ('critic.critic_base.base.2.weight', 'q_net.features_extractor.cnn.2.weight'),
    ('critic.critic_base.base.2.bias', 'q_net.features_extractor.cnn.2.bias'),
    ('critic.critic_base.base.4.weight', 'q_net.features_extractor.cnn.4.weight'),
    ('critic.critic_base.base.4.bias', 'q_net.features_extractor.cnn.4.bias'),
    ('critic.critic_base.base.7.weight', 'q_net.features_extractor.linear.0.weight'),
    ('critic.critic_base.base.7.bias', 'q_net.features_extractor.linear.0.bias'),
    ('critic.critic_output.linear.weight', 'q_net.q_net.0.weight'),
    ('critic.critic_output.linear.bias', 'q_net.q_net.0.bias')
]
# a2c/ppo for Atari
AC_CNNbaseList = [
    ('critic.critic_base.base.0.weight', 'features_extractor.cnn.0.weight'),
    ('critic.critic_base.base.0.bias', 'features_extractor.cnn.0.bias'),
    ('critic.critic_base.base.2.weight', 'features_extractor.cnn.2.weight'),
    ('critic.critic_base.base.2.bias', 'features_extractor.cnn.2.bias'),
    ('critic.critic_base.base.4.weight', 'features_extractor.cnn.4.weight'),
    ('critic.critic_base.base.4.bias', 'features_extractor.cnn.4.bias'),
    ('critic.critic_base.base.7.weight', 'features_extractor.linear.0.weight'),
    ('critic.critic_base.base.7.bias', 'features_extractor.linear.0.bias'),
    ('critic.critic_output.weight', 'value_net.weight'),
    ('critic.critic_output.bias', 'value_net.bias'),
    ('actor.actor_dist.linear.weight', 'action_net.weight'),
    ('actor.actor_dist.linear.bias', 'action_net.bias')
]
# a2c/ppo for mujoco
AC_MLPbaseList = [
    ('base.actor.0.weight', 'mlp_extractor.policy_net.0.weight'),
    ('base.actor.0.bias', 'mlp_extractor.policy_net.0.bias'),
    ('base.actor.2.weight', 'mlp_extractor.policy_net.2.weight'),
    ('base.actor.2.bias', 'mlp_extractor.policy_net.2.bias'),
    ('base.critic.0.weight', 'mlp_extractor.value_net.0.weight'),
    ('base.critic.0.bias', 'mlp_extractor.value_net.0.bias'),
    ('base.critic.2.weight', 'mlp_extractor.value_net.2.weight'),
    ('base.critic.2.bias', 'mlp_extractor.value_net.2.bias'),
    ('base.critic_linear.weight', 'value_net.weight'),
    ('base.critic_linear.bias', 'value_net.bias'),
    ('dist.fc_mean.weight', 'action_net.weight'),
    ('dist.fc_mean.bias', 'action_net.bias'),
    ('dist.logstd._bias', 'log_std')
]

PI_DIST = {'TNorm' : TransformedGaussian,
           'TNorm2': TransformedGaussian2,
           'DNorm' : DiagGaussian,
           'Cat': Categorical,
           'Ber': Bernoulli}


def orthogonal_init(bias_init=0., gain=1):
    return lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, bias_init), gain)

class MLPbase(nn.Module):
    def __init__(self, input_dim, hidden_size=64, init_fn=None, acti_fn=nn.Tanh()):
        super(MLPbase, self).__init__()

        if init_fn is None:
            init_fn = orthogonal_init()

        self.base = nn.Sequential(
            init_fn(nn.Linear(input_dim, hidden_size)), acti_fn,
            init_fn(nn.Linear(hidden_size, hidden_size)), acti_fn
        )

    def forward(self, inputs):
        return self.base(inputs)

class CNNbase(nn.Module):
    def __init__(self, n_stack, hidden_size=512, init_fn=None, acti_fn=nn.ReLU()):
        super(CNNbase, self).__init__()

        if init_fn is None:
            init_fn = orthogonal_init(0., nn.init.calculate_gain('relu'))

        self.base = nn.Sequential(
            init_fn(nn.Conv2d(n_stack, 32, 8, stride=4)), acti_fn,
            init_fn(nn.Conv2d(32, 64, 4, stride=2)), acti_fn,
            init_fn(nn.Conv2d(64, 64, 3, stride=1)), acti_fn,
            Flatten(),
            init_fn(nn.Linear(64 * 7 * 7, hidden_size)), acti_fn
        )

    def forward(self, inputs):
        x = inputs / 255.0
        return self.base(x)

class Critic(nn.Module):
    def __init__(self, input_dim, is_mlp_base, hidden_size, init_fn, acti_fn, acs_dim=0, policy_dist=None):
        """
        input_dim is obs_dim(+acs_dim) if mlp_base, or n_stack if cnn_base
        """
        super(Critic, self).__init__()
        self.base = MLPbase if is_mlp_base else CNNbase
        self.critic_base = self.base(input_dim, hidden_size=hidden_size, acti_fn=acti_fn)
        if PI_DIST is not None and acs_dim != 0:
            self.critic_output = PI_DIST[policy_dist](hidden_size, acs_dim)
        else:
            self.critic_output = init_fn(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        critic_hidden = self.critic_base(inputs)
        value = self.critic_output(critic_hidden)

        return (value,)

class DoubleCritic(Critic):
    def __init__(self, input_dim, is_mlp_base, hidden_size, init_fn, acti_fn):
        super(DoubleCritic, self).__init__(input_dim, is_mlp_base, hidden_size, init_fn, acti_fn)

        self.critic_base_ano = self.base(input_dim, hidden_size=hidden_size, acti_fn=acti_fn)
        self.critic_output_ano = init_fn(nn.Linear(hidden_size, 1))

    def forward(self, inputs):
        critic_hidden = self.critic_base(inputs)
        value = self.critic_output(critic_hidden)

        critic_hidden_ano = self.critic_base_ano(inputs)
        value_ano = self.critic_output_ano(critic_hidden_ano)

        return value, value_ano

class Actor(nn.Module):
    def __init__(self, acs_dim, policy_dist, hidden_size, acti_fn=nn.ReLU(), input_dim=None, shared_actor_base=False):
        """
        if given actor_base, acti_fn and input_dim are unnecessary
        """
        super(Actor, self).__init__()

        self.shared_actor_base = shared_actor_base
        if not shared_actor_base:
            self.actor_base = MLPbase(input_dim, hidden_size, acti_fn=acti_fn)
        self.actor_dist = PI_DIST[policy_dist](hidden_size, acs_dim)

    def forward(self, actor_hidden):
        # if not self.shared_actor_base:
        #     actor_hidden = self.actor_base(inputs)
        # else:
        #     actor_hidden = self.critic_base(inputs)
        dist = self.actor_dist(actor_hidden)
        return dist


class Q_agent(nn.Module):
    def __init__(
            self, input_dim, acs_dim, hidden_size, policy_dist, init_fn=None, acti_fn=nn.ReLU(), is_mlp_base=False,
            is_V_critic=False):

        """
        Q agent for Atari, LunarLander, CartPole, Acrobot (only for discrete-action envs)

        :param input_dim: obs dim (e.g. LunarLander, CartPole), or n_stack (for Atari envs)
        :param acs_dim:
        :param hidden_size:
        :param policy_dist:
        :param init_fn: only for critic output layer of V network
        :param acti_fn:
        :param is_mlp_base:
        :param is_V_critic:
        """
        super(Q_agent, self).__init__()

        self.is_mlp_base = is_mlp_base
        self.is_V_critic = is_V_critic

        if init_fn is None:
            init_fn = orthogonal_init()

        if self.is_mlp_base and self.is_V_critic:
            self.critic = Critic(input_dim, is_mlp_base, hidden_size, init_fn, acti_fn) # V network, cannot return action
        else:
            self.critic = Critic(input_dim, is_mlp_base, hidden_size, init_fn, acti_fn, acs_dim, policy_dist)

        self.add_params_property()
        self.set_agent_mode(training_mode=True)

    def set_agent_mode(self, training_mode=False):
        if training_mode:
            self.critic.train()
        else:
            self.critic.eval()

    def add_params_property(self):
        self.critic_params = list(self.critic.parameters())
        self.named_critic_params = list(self.critic.named_parameters())

    def load_expert_model(self, expert_params):
        """ load expert model """

        print('--- load expert model ---')
        if not self.is_mlp_base: # Q agent for Atari
            params_name_list = Q_CNNbaseList
            self.copy_trained_params(params_name_list, expert_params)
        else:
            #  todo: for Lunarlander, CartPole, MountainCar
            pass

    def copy_trained_params(self, params_name_list, target_params):
        expert_params_dict = OrderedDict()
        for (source_k, target_k) in params_name_list:
            expert_params_dict[source_k] = target_params[target_k]
        self.load_state_dict(expert_params_dict)

    def get_critic_params(self, named=False):
        if named:
            return self.named_critic_params
        else:
            return self.critic_params

    def act(self, inputs, deterministic=True):
        """ note: invalid when is_V_critic """

        act_hidden = self.critic.critic_base(inputs)
        dist = self.critic.critic_output(act_hidden)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action

    def get_value(self, inputs):
        pass

    def forward(self):
        pass


class AC_base(nn.Module):
    def __init__(self, input_dim, acs_dim, is_mlp_base=True, is_V_critic=False):
        super(AC_base, self).__init__()
        self.is_mlp_base = is_mlp_base
        self.is_V_critic = is_V_critic

        self.critic_input_dim = input_dim + acs_dim if self.is_mlp_base and not self.is_V_critic else input_dim
        self.shared_actor_base = False if is_mlp_base else True
        # override to init actor and critic
        self.critic, self.actor, self.actor_base = None, None, None

    def load_expert_model(self, expert_params):
        pass

    def set_agent_mode(self, training_mode=False):
        if training_mode:
            self.critic.train()
            self.actor.train()
        else:
            self.critic.eval()
            self.actor.eval()

    def add_params_property(self):
        self.critic_params = list(self.critic.parameters())
        self.named_critic_params = list(self.critic.named_parameters())
        self.actor_params = list(self.actor.parameters())
        self.named_actor_params = list(self.actor.named_parameters())

    def get_critic_params(self, named=False):
        if named:
            return self.named_critic_params
        else:
            return self.critic_params

    def get_actor_params(self, named=False):
        if named:
            return self.named_actor_params
        else:
            return self.actor_params

    def act(self, inputs, deterministic=True):
        """ used for evaluation """
        # if self.shared_actor_base: actor_hidden = self.critic.critic_base(inputs)
        # else:
        #     actor_hidden = self.actor.actor_base(inputs)

        actor_hidden = self.actor_base(inputs)

        dist = self.actor(actor_hidden)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action

    def get_action(self, inputs):
        """ inputs to actor net"""
        # if self.shared_actor_base:
        #     actor_hidden = self.critic.critic_base(inputs)
        # else:
        #     actor_hidden = self.actor.actor_base(inputs)
        actor_hidden = self.actor_base(inputs)

        dist = self.actor(actor_hidden)
        actions = dist.rsample()
        log_probs = dist.log_probs(actions)

        return dist.mode(), actions, log_probs

    def get_value(self, inputs):
        """ inputs to critic net """
        value = self.critic(inputs)

        return value # return a tuple includes tensor

    def get_V(self, inputs):
        # V(s) = Q(s, \pi(s))
        _, action, log_probs = self.get_action(inputs)
        obs_act_tensor = torch.cat((inputs, action), dim=1)
        current_Q = self.get_value(obs_act_tensor)[0]

        return current_Q



class AC_agent(AC_base):
    def __init__(
            self, input_dim, acs_dim, hidden_size, policy_dist, init_fn=None, acti_fn=nn.Tanh(), is_mlp_base=True,
            is_V_critic=False):
        """
        AC agent only for mujoco, Atari(when sampling expert data using PPO), if being used in Atari tasks, note that the
        usage of shared critic base.

        :param input_dim: obs dim when mlp (mujoco), or n_stack when cnn (Atari)
        :param acs_dim:
        :param hidden_size: an int number
        :param policy_dist: type of policy distribution (options: 'TranGaussian', 'DiagGaussian', 'Categorical', 'Bernoulli')
        :param init_fn: only for cirtic_output (last Linear layer), other block (e.g. critic_base, actor_base, actor_dist)
                        use its default init_fn
        :param acti_fn:
        :param is_mlp_base: whether to use mlp or cnn as base network
        :param is_V_cirtic: whether the critic network is V or Q value network
        """
        super(AC_agent, self).__init__(input_dim, acs_dim, is_mlp_base, is_V_critic)

        if init_fn is None:
            init_fn = orthogonal_init()
        # init critic and actor
        self.critic = Critic(self.critic_input_dim, self.is_mlp_base, hidden_size, init_fn, acti_fn)
        # shared_actor_base = False if self.is_mlp_base else True
        self.actor = Actor(acs_dim, policy_dist, hidden_size, acti_fn, input_dim, self.shared_actor_base)
        self.actor_base = self.critic.critic_base if self.shared_actor_base else self.actor.actor_base
        self.add_params_property()
        self.set_agent_mode(training_mode=True)

    def load_expert_model(self, expert_params):
        """ load expert model """

        print('--- load expert model ---')
        if not self.is_mlp_base: # AC for Atari
            params_name_list = AC_CNNbaseList
            self.copy_trained_params(params_name_list, expert_params)
        else:
            params_name_list = AC_MLPbaseList # only for mujoco, todo: for Lunarlander, CartPole, MountainCar
            expert_params['log_std'] = expert_params['log_std'].unsqueeze(dim=1)
            self.copy_trained_params(params_name_list,expert_params)

    def copy_trained_params(self, params_name_list, target_params):
        expert_params_dict = OrderedDict()
        for (source_k, target_k) in params_name_list:
            expert_params_dict[source_k] = target_params[target_k]
        self.load_state_dict(expert_params_dict)

    def forward(self):
        pass

class SAC_agent(AC_base):
    """ only for mujoco """
    def __init__(
            self, input_dim, acs_dim, hidden_size, policy_dist, init_fn=None, acti_fn=nn.Tanh(), is_mlp_base=True,
            is_V_critic=False, is_double_Q=False):
        super(SAC_agent, self).__init__(input_dim, acs_dim, is_mlp_base, is_V_critic)

        self.is_double_Q = is_double_Q
        if init_fn is None:
            init_fn = orthogonal_init()

        # init double critic and target
        _Critic= DoubleCritic if self.is_double_Q else Critic
        self.critic = _Critic(self.critic_input_dim, self.is_mlp_base, hidden_size, init_fn, acti_fn)
        self.actor = Actor(acs_dim, policy_dist, hidden_size, acti_fn, input_dim, shared_actor_base=False)
        self.actor_base = self.actor.actor_base # SAC agent only for mujoco tasks, so no shared critic base
        self.add_params_property()
        self.set_agent_mode(training_mode=True)

        # init target critic
        self.target_critic = _Critic(self.critic_input_dim, self.is_mlp_base, hidden_size, init_fn, acti_fn)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

    def get_value(self, inputs, both=False):
        value = self.critic(inputs)

        if self.is_double_Q and not both:
            value = (torch.min(*value), )

        return value # a tuple includes one or two tensor

    def get_target_value(self, inputs, both=False):
        value = self.target_critic(inputs)

        if self.is_double_Q and not both:
            value = (torch.min(*value), )

        return value # a tuple includes one or two tensor

    def get_V(self, inputs, alpha):
        # V(s) = Q(s,\pi(s)) - \log\pi(s)
        _, action, log_probs = self.get_action(inputs)
        obs_act_tensor = torch.cat((inputs, action), dim=1)
        current_Q = self.get_value(obs_act_tensor)[0]
        current_V = current_Q - alpha.detach() * log_probs

        return current_V

    def get_target_V(self, inputs, alpha):
        _, action, log_probs = self.get_action(inputs)
        obs_act_tensor = torch.cat((inputs, action), dim=1)
        target_Q = self.get_target_value(obs_act_tensor)[0]
        target_V = target_Q - alpha.detach() * log_probs

        return target_V

    def soft_update(self, tau):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

# class Policy(nn.Module):
#     def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
#         super(Policy, self).__init__()
#         if base_kwargs is None:
#             base_kwargs = {}
#         if base is None:
#             if len(obs_shape) == 3:
#                 base = CNNBase
#             elif len(obs_shape) == 1:
#                 base = MLPBase
#             else:
#                 raise NotImplementedError
#
#         self.obs_shape = obs_shape
#         # note: whether the obs_shape is (w, d, c) or (c, w, d), make_vec_env has TranposeImage wrapper which return the
#         # obs_shape with (c, w, d)
#         self.base = base(self.obs_shape[0],  **base_kwargs)
#
#         # atari (AC/Value-based)
#         if action_space.__class__.__name__ == "Discrete":
#             num_outputs = action_space.n
#             self.dist = Categorical(self.base.output_size, num_outputs)
#         # mujoco (AC-based)
#         elif action_space.__class__.__name__ == "Box":
#             num_outputs = action_space.shape[0]
#             self.dist = DiagGaussian(self.base.output_size, num_outputs)
#         elif action_space.__class__.__name__ == "MultiBinary":
#             num_outputs = action_space.shape[0]
#             self.dist = Bernoulli(self.base.output_size, num_outputs)
#         else:
#             raise NotImplementedError
#
#     @property
#     def is_recurrent(self):
#         return self.base.is_recurrent
#
#     @property
#     def recurrent_hidden_state_size(self):
#         """Size of rnn_hx."""
#         return self.base.recurrent_hidden_state_size
#
#     def forward(self, inputs, rnn_hxs, masks):
#         raise NotImplementedError
#
#     def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
#         value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
#         dist = self.dist(actor_features)
#
#         if deterministic:
#             action = dist.mode()
#         else:
#             action = dist.sample()
#
#         action_log_probs = dist.log_probs(action)
#         dist_entropy = dist.entropy().mean()
#
#         return value, action, action_log_probs, rnn_hxs
#
#     def get_value(self, inputs, rnn_hxs, masks):
#         value, _, _ = self.base(inputs, rnn_hxs, masks)
#         return value
#
#     def evaluate_actions(self, inputs, rnn_hxs, masks, action):
#         value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
#         dist = self.dist(actor_features)
#
#         action_log_probs = dist.log_probs(action)
#         dist_entropy = dist.entropy().mean()
#
#         return value, action_log_probs, dist_entropy, rnn_hxs
#
#     def load_expert_model(self, expert_params, expert_algo):
#         """
#
#         :param expert_params:  OrderedDict(trained params)
#         :return:
#         """
#         print('--- load expert model ---')
#         if expert_algo == 'dqn':
#             params_name_list = Q_CNNbaseList
#             self.copy_trained_params(params_name_list, expert_params)
#         elif expert_algo in ['a2c', 'ppo']:
#             if len(self.obs_shape)==3: # for Atari
#                 params_name_list = AC_CNNbaseList
#                 self.copy_trained_params(params_name_list, expert_params)
#             elif len(self.obs_shape)==1: # for mujoco
#                 params_name_list = AC_MLPbaseList
#
#                 expert_params['log_std'] = expert_params['log_std'].unsqueeze(dim=1)
#                 self.copy_trained_params(params_name_list,expert_params)
#             else:
#                 raise NotImplementedError
#         else:
#             raise ValueError('current net structure not support {} expert'.format(expert_algo))
#
#     def copy_trained_params(self, params_name_list, target_params):
#         expert_params_dict = OrderedDict()
#         for (source_k, target_k) in params_name_list:
#             expert_params_dict[source_k] = target_params[target_k]
#         self.load_state_dict(expert_params_dict)
#
#
# class NNBase(nn.Module):
#     def __init__(self, recurrent, recurrent_input_size, hidden_size):
#         super(NNBase, self).__init__()
#
#         self._hidden_size = hidden_size
#         self._recurrent = recurrent
#
#         if recurrent:
#             self.gru = nn.GRU(recurrent_input_size, hidden_size)
#             for name, param in self.gru.named_parameters():
#                 if 'bias' in name:
#                     nn.init.constant_(param, 0)
#                 elif 'weight' in name:
#                     nn.init.orthogonal_(param)
#
#     @property
#     def is_recurrent(self):
#         return self._recurrent
#
#     @property
#     def recurrent_hidden_state_size(self):
#         if self._recurrent:
#             return self._hidden_size
#         return 1
#
#     @property
#     def output_size(self):
#         return self._hidden_size
#
#     def _forward_gru(self, x, hxs, masks):
#         if x.size(0) == hxs.size(0):
#             x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
#             x = x.squeeze(0)
#             hxs = hxs.squeeze(0)
#         else:
#             # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
#             N = hxs.size(0)
#             T = int(x.size(0) / N)
#
#             # unflatten
#             x = x.view(T, N, x.size(1))
#
#             # Same deal with masks
#             masks = masks.view(T, N)
#
#             # Let's figure out which steps in the sequence have a zero for any agent
#             # We will always assume t=0 has a zero in it as that makes the logic cleaner
#             has_zeros = ((masks[1:] == 0.0) \
#                             .any(dim=-1)
#                             .nonzero()
#                             .squeeze()
#                             .cpu())
#
#             # +1 to correct the masks[1:]
#             if has_zeros.dim() == 0:
#                 # Deal with scalar
#                 has_zeros = [has_zeros.item() + 1]
#             else:
#                 has_zeros = (has_zeros + 1).numpy().tolist()
#
#             # add t=0 and t=T to the list
#             has_zeros = [0] + has_zeros + [T]
#
#             hxs = hxs.unsqueeze(0)
#             outputs = []
#             for i in range(len(has_zeros) - 1):
#                 # We can now process steps that don't have any zeros in masks together!
#                 # This is much faster
#                 start_idx = has_zeros[i]
#                 end_idx = has_zeros[i + 1]
#
#                 rnn_scores, hxs = self.gru(
#                     x[start_idx:end_idx],
#                     hxs * masks[start_idx].view(1, -1, 1))
#
#                 outputs.append(rnn_scores)
#
#             # assert len(outputs) == T
#             # x is a (T, N, -1) tensor
#             x = torch.cat(outputs, dim=0)
#             # flatten
#             x = x.view(T * N, -1)
#             hxs = hxs.squeeze(0)
#
#         return x, hxs


# all the net size matches with the model given by stable_baselines3 (dqn and ppo, a2c)
# class CNNBase(NNBase):
#     def __init__(self, num_inputs, action_dim=0, recurrent=False, hidden_size=512, acti_fn=nn.ReLU(), algo=None):
#         super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), nn.init.calculate_gain('relu'))
#
#         self.main = nn.Sequential(
#             init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), acti_fn,
#             init_(nn.Conv2d(32, 64, 4, stride=2)), acti_fn,
#             init_(nn.Conv2d(64, 64, 3, stride=1)), acti_fn, Flatten(),
#             init_(nn.Linear(64 * 7 * 7, hidden_size)), acti_fn)
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0))
#
#         # note: this critic linear layer only for AC algos (a2c, ppo), not value-based algo (dqn)
#         self.has_critic = False
#         if algo in ['ppo', 'a2c']:
#             self.has_critic = True
#             self.critic_linear = init_(nn.Linear(hidden_size, 1))
#
#         self.train()
#
#     def forward(self, inputs, rnn_hxs, masks):
#         x = self.main(inputs / 255.0)
#
#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#         if self.has_critic:
#             value = self.critic_linear(x)
#         else:
#             value = None
#
#         return value, x, rnn_hxs


# only for some AC-based algos (a2c, ppo)
# todo: SAC
# class MLPBase(NNBase):
#     def __init__(self, num_inputs, action_dim=0, recurrent=False, hidden_size=64, acti_fn=nn.Tanh(), algo=None):
#         super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
#
#         if recurrent:
#             num_inputs = hidden_size
#
#
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                                constant_(x, 0), np.sqrt(2))
#
#         self.actor = nn.Sequential(
#             init_(nn.Linear(num_inputs, hidden_size)), acti_fn,
#             init_(nn.Linear(hidden_size, hidden_size)), acti_fn)
#         self.critic = nn.Sequential(
#             init_(nn.Linear(num_inputs+action_dim, hidden_size)), acti_fn,
#             init_(nn.Linear(hidden_size, hidden_size)), acti_fn)
#         self.critic_linear = init_(nn.Linear(hidden_size, 1))
#
#         self.train()
#
#         self.critic_params = list(self.critic.parameters()) + list(self.critic_linear.parameters())
#         self.actor_base_params = list(self.actor.parameters())
#
#         self.named_critic_params = list(self.critic.named_parameters()) + list(self.critic_linear.named_parameters())
#         self.named_actor_base_params = list(self.actor.named_parameters())
#
#
#
#     def forward(self, inputs, rnn_hxs, masks, value_ret=True, actor_feat_ret=True):
#         # note: only used when critic net is V value net
#         value, hidden_actor = None, None
#         x = inputs
#
#         if self.is_recurrent:
#             x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
#
#         if value_ret:
#             hidden_critic = self.critic(x)
#             value = self.critic_linear(hidden_critic)
#         if actor_feat_ret:
#             hidden_actor = self.actor(x)
#
#         return value, hidden_actor, rnn_hxs
