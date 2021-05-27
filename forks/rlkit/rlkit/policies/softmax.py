"""
Torch argmax policy
"""
import numpy as np
from scipy.special import softmax
from torch import nn

import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.policies.base import Policy


class SoftmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf, temperature=1):
        super().__init__()
        self.qf = qf
        self.temperature = temperature

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        probabilities = softmax(q_values_np / self.temperature)
        if abs(sum(probabilities) - 1) > 0.0001:
            print("probabilities:", probabilities)
            print("sum:", sum(probabilities))
            print("q_values:", q_values)
            print("obs:", obs)
        action = np.random.choice(np.arange(0, len(probabilities)), p=probabilities)
        # print(f"chose action {action} with probability {probabilities[action]}")
        return ((action, False), {})


def dense_sensitivity(obs):
    original_obs = obs.clone().detach()
    mean = {}
    std = {}
    q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
    mean["original"] = q_values.mean()
    std["original"] = q_values.std()

    for pos in range(36, 47):
        for s in [-1, 0, 1, 2, 3, 4, 5]:
            run_id = f"pos: {pos} val:{s}"
            obs = original_obs.clone().detach()
            obs[0, pos] = s
            q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
            mean[run_id] = q_values.mean()
            std[run_id] = q_values.std()
    return (mean, std)


def position_sensitivity(obs):
    original_obs = obs.clone().detach()
    mean = {}
    std = {}
    q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
    mean["original"] = q_values.mean()
    std["original"] = q_values.std()

    for pos in range(9):
        run_id = f"pos: ({pos%3},{pos//3}) "
        obs = original_obs.clone().detach()
        obs[0, 27:36] = 0
        obs[0, 27 + pos] = 1
        q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
        mean[run_id] = q_values.mean()
        std[run_id] = q_values.std()
    return (mean, std)


def position_sensitivity_detailed(obs):
    original_obs = obs.clone().detach()
    mean = {}
    std = {}
    q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
    mean["original"] = q_values.mean()
    std["original"] = q_values.std()

    for pos in range(9):
        for x in [0.2, 0.5, 0.8]:
            for y in [0.2, 0.5, 0.8]:
                run_id = f"room: ({pos//3},{pos%3}), pos: ({x},{y}) "
                obs = original_obs.clone().detach()
                obs[0, 27:36] = 0
                obs[0, 27 + pos] = 1
                obs[0, 36] = x
                obs[0, 37] = y
                q_values = ptu.get_numpy(self.qf(obs).squeeze(0))
                mean[run_id] = q_values.mean()
                std[run_id] = q_values.std()
    return (mean, std)
