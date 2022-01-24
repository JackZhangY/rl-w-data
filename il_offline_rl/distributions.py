import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import TanhTransform, Transform
from torch.distributions import TransformedDistribution, Normal

from il_offline_rl.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

LOG_STD_MIN = -5
LOG_STD_MAX = 2

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class FixedTransformedNormal(torch.distributions.transformed_distribution.TransformedDistribution):
    def log_probs(selfself, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.tanh(self.base_dist.loc)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_fn=None):
        super(Categorical, self).__init__()

        if init_fn is None:

            init_fn = lambda m: init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01)

        self.linear = init_fn(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class TransformedGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_fn=None):
        """

        :param num_inputs: hidden size
        :param num_outputs: action_dim
        """
        super(TransformedGaussian, self).__init__()

        if init_fn is None:
            init_fn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))

        self.fc_mean_log_std = init_fn(nn.Linear(num_inputs, 2*num_outputs))
        self.train()

    def forward(self, actor_feat):
        out = self.fc_mean_log_std(actor_feat)
        mean, log_std = torch.chunk(out, 2, dim=1)
        # mode = torch.tanh(mu)

        log_std = torch.tanh(log_std)
        assert LOG_STD_MAX > LOG_STD_MIN
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()

        # dist = TransformedDistribution(Normal(mu, std), TanhTransform().with_cache(cache_size=1)) # with_cache for backward

        return FixedTransformedNormal(Normal(mean, std), TanhTransform().with_cache(cache_size=1))

class TransformedGaussian2(nn.Module):
    """
    reference from SAC implementation, main difference with TransformedGaussian:
    split mean and log_std linear, LOG_STD_MIN(MAX),

    a little difference with original SAC implementation: init_fn
    """
    def __init__(self, num_inputs, num_outputs, init_fn=None):
        super(TransformedGaussian2, self).__init__()

        if init_fn is None:
            init_fn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                     constant_(x, 0))

        self.mean_linear = init_fn(nn.Linear(num_inputs, num_outputs))
        self.log_std_linear = init_fn(nn.Linear(num_inputs, num_outputs))

        self.train()

    def forward(self, actor_feat):
        mean = self.mean_linear(actor_feat)
        log_std = self.log_std_linear(actor_feat)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()

        return FixedTransformedNormal(Normal(mean, std), TanhTransform().with_cache(cache_size=1))

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_fn=None):
        super(DiagGaussian, self).__init__()

        if init_fn is None:
            init_fn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))

        self.fc_mean = init_fn(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_fn=None):
        super(Bernoulli, self).__init__()

        if init_fn:
            init_fn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))

        self.linear = init_fn(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
