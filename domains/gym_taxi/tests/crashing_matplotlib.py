# from matplotlib import pyplot as plt

# import numpy as np

# fig = plt.figure(figsize=(8, 8))
# fig.add_subplot(1, 4, 1)
# plt.imshow(np.zeros((4, 4)))
# plt.show()


import pickle

import numpy as np
from matplotlib import pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F


from gym.spaces import Tuple

# from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.distributions import (
#     Bernoulli,
#     Categorical,
#     DiagGaussian,
#     DistributionGeneratorTuple,
# )
# import math

# from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import AddBias, init

import glob
import os

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import VecNormalize

import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

# from forks.baselines.baselines import bench
# from forks.baselines.baselines.common.atari_wrappers import make_atari, wrap_deepmind, wrap_pysc2
# from forks.baselines.baselines.common.vec_env import VecEnvWrapper
# from forks.baselines.baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from forks.baselines.baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
# from forks.baselines.baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass


# try:
#     import roboschool
# except ImportError:
#     pass

# try:
#     import pybullet_envs
# except ImportError:
#     pass


# from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import _unflatten_tuple


# def view_rollouts(rollouts):
#     shape = ((2, 84, 84), (6,))
#     rollouts.rewards = rollouts.rewards.cpu()
#     rollouts.actions = rollouts.actions.cpu()
#     rollouts.obs = rollouts.obs.cpu()

#     for actor in range(rollouts.rewards.shape[1]):
#         for i in range(rollouts.rewards.shape[0]):
#             obs = _unflatten_tuple(shape, rollouts.obs[i])
#             next_obs = _unflatten_tuple(shape, rollouts.obs[i + 1])
#             action = rollouts.actions[i][actor]
#             reward = rollouts.rewards[i][actor]

#             obs = np.zeros((4, 4))
#             next_obs = np.ones((4, 4))
#             render_tuple(actor, obs, next_obs, action, reward)


# def mixed_convert_to_human(actor, obs):
#     img, dense = obs
#     img = img[actor].numpy()
#     dense = dense[actor]
#     # passenger

#     output = np.zeros(shape=(3, 84, 84), dtype=np.uint8)
#     output[0:3] = np.abs(1 - img[0]) * 255
#     output[0] = output[0] + img[1] * 255
#     # output[1] = output[1] + img[2] * 255
#     # output[2] = output[2] + img[3] * 255

#     return (np.transpose(output, (1, 2, 0)),)


def render_tuple(actor, obs, next_obs, action, reward):

    # img = mixed_convert_to_human(actor, obs)
    # img2 = mixed_convert_to_human(actor, next_obs)

    # plt.ion()

    fig = plt.figure(figsize=(8, 8))

    fig.add_subplot(1, 4, 1)
    plt.imshow(obs)
    fig.add_subplot(1, 4, 2)
    plt.imshow(next_obs)
    plt.show()


obs = np.zeros((4, 4))
next_obs = np.ones((4, 4))
render_tuple(0, obs, next_obs, "a", "r")


# if __name__ == "__main__":
#     a = pickle.load(open("../../symbolic-goal-generation/rollouts.pkl", "rb"))
#     view_rollouts(a)

