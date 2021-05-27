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

ACTION_COUNT = 16
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


class CraftLotteryEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        intermediate=True,
        deterministic=False,
        simple=False,
        scaling="log",
        experience_interval=1,
    ):
        self.steps = 0
        self.lastaction = None
        self.seed()
        self._init_simulator()

        if deterministic:
            self.move_fail_probability = 0
            self.coin_probabilities = [1, 0, 0, 0, 0, 0]
        else:
            self.move_fail_probability = MOVE_FAIL_PROBABILITY
            self.coin_probabilities = COIN_PROBABILITIES

        self.action_space = spaces.Discrete(ACTION_COUNT)
        if simple:
            self.observation_space = spaces.Box(-1, 10, (18,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-1, 10, (47,), dtype=np.float32)
        self.intermediate = intermediate
        self.simple = simple
        self.scaling = scaling
        self.experience_interval = experience_interval

    def _init_simulator(self):  # TODO: figure out how to make abstract
        # self.taxi = tuple([int(n) for n in np.random.randint(5, size=2)])
        # self.passenger = random.choice(LOCATIONS)
        # self.destination = random.choice(LOCATIONS)
        # self.carrying = 0

        # self.obs = json.dumps(
        #     [self.taxi, self.passenger, self.destination, self.carrying]
        # )

        # set up room layouts
        episode_resources = RESOURCES.copy()
        random.shuffle(episode_resources)
        self.resources = [5, 5, 5] + [v for t in episode_resources for v in t]

        # set up map connectivity
        self.map = nx.grid_graph([3, 3])
        edges = list(self.map.edges())
        for u, v in edges:
            if random.random() < 0.1:
                self.map.remove_edge(u, v)

        self.position = (random.uniform(0, 0.3), random.uniform(0, 0.3))
        self.bearing = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.inventory = [0] * 6
        self.room = (0, 0)

    def get_raw_obs(self):
        dense = list(self.position) + list(self.bearing) + [0] + self.inventory
        rooms = [0] * 9
        rooms[self.room[0] + 3 * self.room[1]] = 1

        return self.resources + rooms + dense

    def get_obs(self):
        obs = self.scale_obs(self.get_raw_obs())
        if self.simple:
            return obs[2:27:3] + obs[27:36]
        else:
            return obs

    def scale_obs(self, obs):
        if self.scaling == "log":
            return self.log_scale_obs(obs)
        if self.scaling == "linear":
            return self.linear_scale_obs(obs)

    def log_scale_obs(self, obs):
        contents = [math.log(v + 1) for v in obs[0:27]]
        contents = [c - 1.5 for c in contents]
        for i in range(len(contents)):
            if i % 3 < 2:
                contents[i] = contents[i] / 4

        items = [math.log(v + 1) for v in obs[41:]]
        return contents + obs[27:41] + items

    def linear_scale_obs(self, obs):
        contents = obs[0:27]
        for i in range(len(contents)):
            if i % 3 < 2:
                contents[i] = contents[i] / 2

        contents = [c / 10 - 1 for c in contents]

        items = [math.log(v + 1) for v in obs[41:]]
        return contents + obs[27:41] + items

    def reset(self):
        self._init_simulator()
        self.steps = 0
        self.lastaction = None
        return self.get_obs()

    def step(self, action):
        # self.steps += 1 #may be multiple steps per action, leave to act function
        obs, reward, done, info = self.act(action)
        if not self.intermediate:
            info.pop("intermediate_experience")

        if self.steps >= MAX_EPISODE_LENGTH:
            done = True
            info["bad_transition"] = True
        return obs, reward, done, info

    def act(self, action):
        initial = self.get_raw_obs()
        self.bearing = (random.uniform(-1, 1), random.uniform(-1, 1))
        done = False
        reward = 0
        intermediate_experience = []

        # move somewhere
        if action < 9:

            destination = (action % 3, action // 3)
            try:
                rooms = nx.shortest_path(self.map, source=self.room, target=destination)
            except nx.NetworkXNoPath as e:
                self.steps += 1
                return (
                    self.get_obs(),
                    reward,
                    done,
                    {"intermediate_experience": intermediate_experience},
                )

            if len(rooms) < 2:
                self.steps += 1
                # TODO: randomly jiggle position
            else:
                for room in rooms[1:]:
                    final_step, reward_step, done_step, info_step = self.move_room(room)
                    reward += reward_step
                    intermediate_experience = [
                        (o, r + reward_step) for (o, r) in intermediate_experience
                    ]
                    intermediate_experience.extend(info_step["intermediate_experience"])
                    if done_step:

                        return (
                            self.get_obs(),
                            reward,
                            done,
                            {"intermediate_experience": intermediate_experience},
                        )
                    elif self.intermediate:
                        intermediate_experience.append((final_step, 0))
                if len(intermediate_experience) > 0:
                    intermediate_experience.pop()

        # collect coins
        elif action == 15:
            # if there are no coins in the current room
            if self.resources[self.coin_index] == 0:
                self.steps += 1
                # TODO: randomly jiggle position
            else:
                # figure out how many resources are collected
                percentage_collected = random.choices(
                    COIN_RATIOS, weights=self.coin_probabilities
                )[0]
                coin_count = self.reduce_resource(self.coin_index, percentage_collected)
                reward = coin_count

                tree_count = self.reduce_resource(
                    self.tree_index, incidental_fraction()
                )
                self.inventory[0] += tree_count
                rock_count = self.reduce_resource(
                    self.rock_index, incidental_fraction()
                )
                self.inventory[1] += rock_count

                # increment steps
                if percentage_collected < 1:
                    steps = 100
                else:
                    min_steps = (coin_count + tree_count + rock_count) * 3
                    steps = random.randint(min_steps, 99)

                final = self.get_raw_obs()
                intermediate_experience = self.interpolate(
                    tree_count, coin_count, rock_count, steps, initial, reward
                )

                self.steps += steps
        # get items
        else:
            item_index = action - 9
            if self.inventory[item_index] > 0:
                self.steps += 1
                # TODO: randomly jiggle position
            else:
                self.inventory[item_index] += 1
                steps = random.randint(1, 20)

                final = self.get_raw_obs()
                intermediate_experience = self.interpolate(0, 0, 0, steps, initial, 0)
                if item_index == 0:
                    item = "wood"
                elif item_index == 1:
                    item = "stone"
                elif item_index == 2:
                    item = "plank"
                elif item_index == 3:
                    item = "stick"
                elif item_index == 4:
                    item = "wood_pick"
                elif item_index == 5:
                    item = "stone_pick"
                self.steps += steps

        return (
            self.get_obs(),
            reward,
            done,
            {"intermediate_experience": intermediate_experience},
        )

    def move_room(self, room):
        initial = self.get_raw_obs()

        # randomly remove some things
        coin_count = self.reduce_resource(self.coin_index, incidental_fraction())
        reward = coin_count
        tree_count = self.reduce_resource(self.tree_index, incidental_fraction())
        self.inventory[0] += tree_count
        rock_count = self.reduce_resource(self.rock_index, incidental_fraction())
        self.inventory[1] += rock_count

        done = False
        positions = [0.15, 0.33, 0.5, 0.66, 0.85]

        # check for completion
        if random.random() < self.move_fail_probability:
            steps = 100
            done = True
            x, y = self.room

            self.position = (
                random.uniform(positions[x] - 0.1, positions[x] + 0.1),
                random.uniform(positions[y] - 0.1, positions[y] + 0.1),
            )
        else:
            min_steps = (coin_count + tree_count + rock_count) * 3
            steps = random.randint(min_steps, 99)

            x = self.room[0] + room[0]
            y = self.room[1] + room[1]

            self.position = (
                random.uniform(positions[x] - 0.03, positions[x] + 0.03),
                random.uniform(positions[y] - 0.03, positions[y] + 0.03),
            )

        self.steps += steps
        if self.steps >= MAX_EPISODE_LENGTH:
            done = True

        intermediate_experience = self.interpolate(
            tree_count, coin_count, rock_count, steps, initial, reward
        )
        self.room = room

        return (
            self.get_raw_obs(),
            reward,
            done,
            {"intermediate_experience": intermediate_experience},
        )

    def interpolate(self, tree_count, coin_count, rock_count, steps, initial, reward):
        if not self.intermediate:
            return []
        obs_list = []
        obs = initial.copy()
        turn_count = (steps - tree_count - coin_count - rock_count) // 2
        move_count = steps - tree_count - coin_count - rock_count - turn_count
        actions = (
            ["tree"] * tree_count
            + ["rock"] * rock_count
            + ["coin"] * coin_count
            + ["move"] * move_count
            + ["turn"] * turn_count
        )
        random.shuffle(actions)
        for a in actions:
            if a == "tree":
                obs[self.tree_index] -= 1
                obs[40] += 1
            if a == "rock":
                obs[self.rock_index] -= 1
                obs[41] += 1
            if a == "coin":
                obs[self.coin_index] -= 1
                reward -= 1
            if a == "move":
                x, y = obs[36:38]
                obs[36] = random.uniform(max(0, x - 0.05), min(1, x + 0.05))
                obs[37] = random.uniform(max(0, y - 0.05), min(1, y + 0.05))
            if a == "turn":
                sin, cos = obs[38:40]
                obs[38] = random.uniform(max(-1, sin - 0.25), min(1, sin + 0.25))
                obs[39] = random.uniform(max(-1, cos - 0.25), min(1, cos + 0.25))

            obs_list.append((self.scale_obs(obs.copy()), reward))
        return obs_list[:: self.experience_interval]

    def reduce_resource(self, index, fraction):
        n = min(round(self.resources[index] * fraction), 25)
        self.resources[index] -= n
        return n

    def close(self):
        return

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def tree_index(self):
        return (self.room[0] + 3 * self.room[1]) * 3

    @property
    def rock_index(self):
        return self.tree_index + 1

    @property
    def coin_index(self):
        return self.tree_index + 2


if __name__ == "__main__":
    craft = CraftLotteryEnv(experience_interval=5)
    craft.reset()
    done = False
    while not done:
        a = random.randint(0, 15)
        print(a)
        obs, reward, done, info = craft.step(a)

    print(reward)
