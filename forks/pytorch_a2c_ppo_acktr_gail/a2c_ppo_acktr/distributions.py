import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = (
    lambda self, actions: log_prob_cat(self, actions.squeeze(-1))
    .view(actions.size(0), -1)
    .sum(-1)
    .unsqueeze(-1)
)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)
FixedCategorical.get_probs = lambda self: self.probs

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(
    -1, keepdim=True
)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean
FixedNormal.get_probs = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = (
    lambda self, actions: log_prob_bernoulli(self, actions)
    .view(actions.size(0), -1)
    .sum(-1)
    .unsqueeze(-1)
)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()
FixedBernoulli.get_probs = lambda self: self.probs


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
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
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class DistributionGeneratorTuple(nn.Module):
    def __init__(self, distributions):
        super(DistributionGeneratorTuple, self).__init__()
        self.distributions = nn.ModuleList(distributions)

    def forward(self, x):
        return DistributionTuple([d.forward(x) for d in self.distributions])

    def __call__(self, x):
        return DistributionTuple([d(x) for d in self.distributions])


class DistributionTuple:
    def __init__(self, distributions):
        self.distributions = distributions

    def mode(self):
        # print(f"mean={self.distributions[1].mean}, sd={self.distributions[1].stddev}")
        return torch.cat([d.mode().float() for d in self.distributions], 1)

    def sample(self):
        # print(f"mean={self.distributions[1].mean}, sd={self.distributions[1].stddev}")
        return torch.cat([d.sample().float() for d in self.distributions], 1)

    def log_probs(self, action):
        # lp = [d.log_probs(a) for a, d in zip(action, self.distributions)]
        min_index = 0
        max_index = 0
        log_prob = 0
        for d in self.distributions:
            if len(d.batch_shape) == 1:
                max_index += 1
            else:
                max_index += d.batch_shape[1]
            log_prob += d.log_probs(action[:, min_index:max_index])
            min_index = max_index
        return log_prob

    def entropy(self):
        return sum([d.entropy() for d in self.distributions])

    def get_probs(self):
        probs = torch.cat([d.get_probs() for d in self.distributions], dim=1)
        # print(probs)
        return probs

