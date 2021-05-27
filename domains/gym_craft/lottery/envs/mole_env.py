"""
.. module:: taxi_env
   :synopsis: Provides gym environment wrappers for the underlying taxi simulator.
"""

import random
import math
import sys
from contextlib import closing
from six import StringIO
import numpy as np
import gym
import json
from gym import error, spaces, utils
from gym.utils import seeding

import networkx as nx
from matplotlib import pyplot as plt

from domains.gym_taxi.utils.representations import resize_image
from domains.gym_taxi.utils.spaces import Json
from domains.gym_taxi.simulator.taxi_world import Taxi, Passenger

CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84
LOCATIONS = [(0, 0), (4, 0), (0, 4), (3, 4)]

ACTION_COUNT = 9
MAX_EPISODE_LENGTH = 1000
RESOURCES = [
    (5, 5, 5),
    (5, 5, 5),
    (50, 0, 5),
    (50, 0, 5),
    (0, 50, 5),
    (0, 50, 5),
    (0, 0, 20),
    (0, 0, 20),
]

MOVE_FAIL_PROBABILITY = 0.15

COIN_RATIOS = [1, 0.9, 0.7, 0.5, 0.2, 0]
COIN_PROBABILITIES = [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]

INCEDENTAL_RATIOS = [0.7, 0.5, 0.3, 0.2, 0.1, 0]
INCEDENTAL_PROBABILITIES = [0.05, 0.05, 0.1, 0.2, 0.2, 0.4]


def incidental_fraction():
    return random.choices(INCEDENTAL_RATIOS, weights=INCEDENTAL_PROBABILITIES)[0]


class WhackAMoleEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, scaling="log"):
        self.steps = 0
        self.lastaction = None
        self.seed()
        self._init_simulator()

        self.action_space = spaces.Discrete(ACTION_COUNT)

        self.observation_space = spaces.Box(-3, 10, (9,), dtype=np.float32)
        self.scaling = scaling

    def _init_simulator(self):

        # set up room layouts
        coins = [c for (_, _, c) in RESOURCES]
        random.shuffle(coins)
        self.resources = [5] + coins

        self.room = (0, 0)

    def get_raw_obs(self):
        return self.resources

    def get_obs(self):
        obs = self.scale_obs(self.get_raw_obs())
        return obs

    def scale_obs(self, obs):
        if self.scaling == "log":
            return self.log_scale_obs(obs)
        if self.scaling == "linear":
            return self.linear_scale_obs(obs)

    def log_scale_obs(self, obs):
        contents = [math.log(v + 1) for v in obs]
        contents = [c - 1.5 for c in contents]

        return contents

    def linear_scale_obs(self, obs):
        contents = obs

        contents = [c / 10 - 1 for c in contents]

        return contents

    def reset(self):
        self._init_simulator()
        self.steps = 0
        self.lastaction = None
        return self.get_obs()

    def step(self, action):

        # self.steps += 1 #may be multiple steps per action, leave to act function
        obs, reward, done, info = self.act(action)

        if self.steps >= MAX_EPISODE_LENGTH:
            done = True
            info["bad_transition"] = True
        return obs, reward, done, info

    def act(self, action):

        initial = self.get_raw_obs()
        done = False
        try:
            reward = self.resources[action]
            self.resources[action] = 0
        except:
            reward = self.resources[action[0]]
            self.resources[action[0]] = 0

        self.steps += 50
        if sum(self.resources) == 0:
            done = True

        return (self.get_obs(), reward, done, {})

    def close(self):
        return

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    craft = WhackAMoleEnv()
    craft.reset()
    done = False
    while not done:
        a = random.randint(0, 8)
        print(a)
        obs, reward, done, info = craft.step(a)

    print(reward)
