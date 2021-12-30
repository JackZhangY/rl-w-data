import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, TransformedGaussian
from a2c_ppo_acktr.utils import init



# DQN fro Atari
Value_CNNbaseList = [
    ('base.main.0.weight', 'q_net.features_extractor.cnn.0.weight'),
    ('base.main.0.bias', 'q_net.features_extractor.cnn.0.bias'),
    ('base.main.2.weight', 'q_net.features_extractor.cnn.2.weight'),
    ('base.main.2.bias', 'q_net.features_extractor.cnn.2.bias'),
    ('base.main.4.weight', 'q_net.features_extractor.cnn.4.weight'),
    ('base.main.4.bias', 'q_net.features_extractor.cnn.4.bias'),
    ('base.main.7.weight', 'q_net.features_extractor.linear.0.weight'),
    ('base.main.7.bias', 'q_net.features_extractor.linear.0.bias'),
    ('dist.linear.weight', 'q_net.q_net.0.weight'),
    ('dist.linear.bias', 'q_net.q_net.0.bias')
]
# a2c/ppo for Atari
AC_CNNbaseList = [
    ('base.main.0.weight', 'features_extractor.cnn.0.weight'),
    ('base.main.0.bias', 'features_extractor.cnn.0.bias'),
    ('base.main.2.weight', 'features_extractor.cnn.2.weight'),
    ('base.main.2.bias', 'features_extractor.cnn.2.bias'),
    ('base.main.4.weight', 'features_extractor.cnn.4.weight'),
    ('base.main.4.bias', 'features_extractor.cnn.4.bias'),
    ('base.main.7.weight', 'features_extractor.linear.0.weight'),
    ('base.main.7.bias', 'features_extractor.linear.0.bias'),
    ('base.critic_linear.weight', 'value_net.weight'),
    ('base.critic_linear.bias', 'value_net.bias'),
    ('dist.linear.weight', 'action_net.weight'),
    ('dist.linear.bias', 'action_net.bias')
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


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)



class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.obs_shape = obs_shape
        # note: whether the obs_shape is (w, d, c) or (c, w, d), make_vec_env has TranposeImage wrapper which return the
        # obs_shape with (c, w, d)
        self.base = base(self.obs_shape[0],  **base_kwargs)

        # atari (AC/Value-based)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        # mujoco (AC-based)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def load_expert_model(self, expert_params, expert_algo):
        """

        :param expert_params:  OrderedDict(trained params)
        :return:
        """
        print('--- load expert model ---')
        if expert_algo == 'dqn':
            params_name_list = Value_CNNbaseList
            self.copy_trained_params(params_name_list, expert_params)
        elif expert_algo in ['a2c', 'ppo']:
            if len(self.obs_shape)==3: # for Atari
                params_name_list = AC_CNNbaseList
                self.copy_trained_params(params_name_list, expert_params)
            elif len(self.obs_shape)==1: # for mujoco
                params_name_list = AC_MLPbaseList

                expert_params['log_std'] = expert_params['log_std'].unsqueeze(dim=1)
                self.copy_trained_params(params_name_list,expert_params)
            else:
                raise NotImplementedError
        else:
            raise ValueError('current net structure not support {} expert'.format(expert_algo))

    def copy_trained_params(self, params_name_list, target_params):
        expert_params_dict = OrderedDict()
        for (source_k, target_k) in params_name_list:
            expert_params_dict[source_k] = target_params[target_k]
        self.load_state_dict(expert_params_dict)


class Separate_AC_Net(Policy):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Separate_AC_Net, self).__init__(obs_shape, action_space, base, base_kwargs)

        # atari (AC/Value-based)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            # todo: for Atari envs
            # self.dist = Categorical(self.base.output_size, num_outputs)
        # mujoco (AC-based)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = TransformedGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            # todo: for other envs
            # self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.critic_params = self.base.critic_params
        self.actor_params = self.base.actor_base_params+list(self.dist.parameters())

        self.named_critic_params = self.base.named_critic_params
        self.named_actor_params = self.base.named_actor_base_params+list(self.dist.named_parameters())

    def to_device(self, device):
        self.base.to(device)
        self.dist.to(device)

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

    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        # action can't propagate gradient, only used for action when evaluating
        # value, action_feat, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        _, action_feat, _ = self.base(inputs, rnn_hxs, masks, value_ret=False)
        dist, mode = self.dist(action_feat)

        if deterministic:
            action = mode
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)

        return action

    def get_value(self, inputs, rnn_hxs=None, masks=None):
        value, _, _ = self.base(inputs, rnn_hxs, masks, actor_feat_ret=False)
        return value

    def forward(self, inputs, rnn_hxs=None, masks=None):
        _, action_feat, _ = self.base(inputs, rnn_hxs, masks, value_ret=False)
        dist, mode = self.dist(action_feat)

        # .rsample() returns the differentiable action
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)

        return mode, actions, log_probs



class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


# all the net size matches with the model given by stable_baselines3 (dqn and ppo, a2c)
class CNNBase(NNBase):
    def __init__(self, num_inputs, action_dim=0, recurrent=False, hidden_size=512, acti_fn=nn.ReLU(), algo=None):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), acti_fn,
            init_(nn.Conv2d(32, 64, 4, stride=2)), acti_fn,
            init_(nn.Conv2d(64, 64, 3, stride=1)), acti_fn, Flatten(),
            init_(nn.Linear(64 * 7 * 7, hidden_size)), acti_fn)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        # note: this critic linear layer only for AC algos (a2c, ppo), not value-based algo (dqn)
        self.has_critic = False
        if algo in ['ppo', 'a2c']:
            self.has_critic = True
            self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        if self.has_critic:
            value = self.critic_linear(x)
        else:
            value = None

        return value, x, rnn_hxs
        # return self.critic_linear(x), x, rnn_hxs


# only for some AC-based algos (a2c, ppo)
# todo: SAC
class MLPBase(NNBase):
    def __init__(self, num_inputs, action_dim=0, recurrent=False, hidden_size=64, acti_fn=nn.Tanh(), algo=None):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), acti_fn,
            init_(nn.Linear(hidden_size, hidden_size)), acti_fn)
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs+action_dim, hidden_size)), acti_fn,
            init_(nn.Linear(hidden_size, hidden_size)), acti_fn)
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

        self.critic_params = list(self.critic.parameters()) + list(self.critic_linear.parameters())
        self.actor_base_params = list(self.actor.parameters())

        self.named_critic_params = list(self.critic.named_parameters()) + list(self.critic_linear.named_parameters())
        self.named_actor_base_params = list(self.actor.named_parameters())



    def forward(self, inputs, rnn_hxs, masks, value_ret=True, actor_feat_ret=True):
        # note: only used when critic net is V value net
        value, hidden_actor = None, None
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        if value_ret:
            hidden_critic = self.critic(x)
            value = self.critic_linear(hidden_critic)
        if actor_feat_ret:
            hidden_actor = self.actor(x)

        return value, hidden_actor, rnn_hxs
