"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from forks.rlkit.rlkit.policies.base import Policy
from forks.rlkit.rlkit.torch import pytorch_util as ptu
from forks.rlkit.rlkit.torch.core import eval_np
from forks.rlkit.rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from forks.rlkit.rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(nn.Module):
    """
    A class for defining a fully connected NN using pytorch.

    :param hidden_sizes: sizes of each hidden layer
    :param output_size: size of output layer
    :param input_size: size of input layer
    :param init_w: maximum absolute value of initial weights in output layer
    :param hidden_activation: Activation function for hidden layers (default relu)
    :param output_activation: Activation function for output layer (default none)
    :param hidden_init: initialisation function for hidden layer weights
    :param b_init_value: initial bias for hidden layer
    :param layer_norm: flag to specify if hidden layers are normalised (default false)
    :param layer_norm_kwargs: ???
    :returns: initialised pytorch NN object
    """

    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            # hidden_init(fc.weight)
            # fc.bias.data.fill_(b_init_value)

            fc.weight.data.uniform_(-init_w, init_w)
            fc.bias.data.uniform_(b_init_value, b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(self, *args, obs_normalizer: TorchFixedNormalizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
