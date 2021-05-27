"""
Torch argmax policy
"""
import numpy as np
from torch import nn
import torch

import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        if isinstance(obs, tuple):
            action_obs = _flatten_tuple(obs)
            action_obs = ptu.from_numpy(action_obs).float().unsqueeze(0)
            q_values = self.qf(action_obs).squeeze(0)

        else:
            obs = np.expand_dims(obs, axis=0)
            obs = ptu.from_numpy(obs).float()
            q_values = self.qf(obs).squeeze(0)

        q_values_np = ptu.get_numpy(q_values)
        return (q_values_np.argmax(), False), {}


def _flatten_tuple(observation):
    """Assumes observation is a tuple of tensors. converts ((n,c, h, w),(n, x)) -> (n,c*h*w+x)"""
    image, fc = observation
    flat = image.flatten()
    return np.concatenate([flat, fc])
