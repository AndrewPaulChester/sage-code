"""
.. module:: taxi_env
   :synopsis: Provides gym environment wrappers for the underlying taxi simulator.
"""

import sys
from contextlib import closing
from six import StringIO
import numpy as np
import gym
import json
from gym import error, spaces, utils
from gym.utils import seeding
from domains.gym_craft.simulator.craft_world import CraftWorldSimulator, ACTIONS
from domains.gym_taxi.utils.spaces import Json
from domains.gym_craft.utils.representations import (
    json_to_image,
    resize_image,
    json_to_screen,
    json_to_mixed,
    json_to_symbolic,
    json_to_pddl,
    json_to_dense,
    json_to_train,
    json_to_masked,
    json_to_zoomed,
    json_to_centered,
    json_to_zoomed_binary,
    json_to_centered_binary,
    json_to_binary,
    json_to_abstract,
    json_to_both,
)
from domains.gym_craft.utils.config import (
    DIRECTIONS,
    OFFSETS,
    MAX_EPISODE_LENGTH,
    ORIGINAL,
    RANDOM_RESOURCES,
    ROOM_FREE,
    SMALL_STATIC,
    SMALL_RANDOM,
    SPARSE,
    COIN,
    SINGLE_COIN,
    LARGE_COIN,
    ROOMS,
    COIN_ROOMS,
    EASY_ROOMS,
    POOR_ROOMS,
)

from domains.gym_craft.utils.utils import facing_block
from matplotlib import pyplot as plt

# from simulator
ACTION_COUNT = len(ACTIONS)


# gym environment specific
CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84

# 11 is the number of variables in the dense input (position, inventory contents, etc)
# 12 is the dimension of the uvfa embedding: 23 = dense + ufva
# 23 is only used when the uvfa embedding is coming directly from the environment, not when it is coming from the learner
OBS_SPACES = {
    ("screen", "original"): (3, None),
    ("mixed", "original"): (3, 11),
    ("mixed", "random_resources"): (3, 11),
    ("mixed", "room_free"): (3, 11),
    ("mixed", "small_static"): (3, 11),
    ("mixed", "small_random"): (3, 11),
    ("mixed", "sparse"): (3, 11),
    ("mixed", "coin"): (3, 11),
    ("mixed", "single_coin"): (3, 11),
    ("mixed", "large_coin"): (3, 11),
    ("mixed", "rooms"): (3, 11),
    ("train", "rooms"): (3, 23),
    ("train", "coin_rooms"): (3, 23),
    ("masked", "rooms"): (3, 23),
    ("zoomed", "rooms"): (3, 23),
    ("zoomed_binary", "rooms"): (5, 23),
    ("centered_binary", "rooms"): (5, 23),
    ("zoomed_binary", "coin_rooms"): (5, 23),
    ("zoomed_binary", "poor_rooms"): (5, 12),
    ("zoomed_binary", "easy_rooms"): (5, 23),
    ("zoomed", "coin_rooms"): (3, 23),
    ("centered", "coin_rooms"): (3, 23),
    ("centered_binary", "coin_rooms"): (5, 23),
    ("centered_binary", "easy_rooms"): (5, 23),
    ("binary", "coin_rooms"): (5, 23),
    ("binary", "easy_rooms"): (5, 23),
    ("zoomed", "easy_rooms"): (3, 23),
    ("dense", "coin"): (None, 11),
    ("dense", "single_coin"): (None, 13),
    ("dense", "sparse"): (None, 11),
    ("abstract", "rooms"): (None, 47),
    ("both", "rooms"): (5, 47),
    ("abstract", "random_resources"): (None, 32),
}

SCENARIOS = {
    "original": ORIGINAL,
    "random_resources": RANDOM_RESOURCES,
    "room_free": ROOM_FREE,
    "small_static": SMALL_STATIC,
    "small_random": SMALL_RANDOM,
    "sparse": SPARSE,
    "coin": COIN,
    "single_coin": SINGLE_COIN,
    "large_coin": LARGE_COIN,
    "rooms": ROOMS,
    "coin_rooms": ROOMS,
    "coin_rooms": COIN_ROOMS,
    "easy_rooms": EASY_ROOMS,
    "poor_rooms": POOR_ROOMS,
}

CONVERTERS = {
    "screen": json_to_screen,
    "mixed": json_to_mixed,
    "dense": json_to_dense,
    "abstract": json_to_abstract,
    "train": json_to_train,
    "masked": json_to_masked,
    "zoomed": json_to_zoomed,
    "centered": json_to_centered,
    "zoomed_binary": json_to_zoomed_binary,
    "centered_binary": json_to_centered_binary,
    "binary": json_to_binary,
    "both": json_to_both,
}

ACTION_SIZE = {
    "full": spaces.Box(-np.inf, np.inf, (7,)),
    "move-only": spaces.Box(-np.inf, np.inf, (2,)),
    "move-continuous": spaces.Box(-np.inf, np.inf, (3,)),
    "move-uniform": spaces.Box(-np.inf, np.inf, (3,)),
    "rooms": spaces.Box(-np.inf, np.inf, (5,)),
}
# ACTION_CONVERTER = {"full": json_to_screen, "move-only": json_to_mixed, "move-continuous": json_to_mixed}


def _construct_image(representation, scenario):
    channels, length = OBS_SPACES[(representation, scenario)]
    if channels is None:  # purely MLP input
        return spaces.Box(0, 4, (length,), dtype=np.float32)
    if length is None:  # purely image based input
        return spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
    else:  # mixed input
        screen = spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
        dense = spaces.Box(0, 4, (length,), dtype=np.float32)
        return spaces.Tuple((screen, dense))


class BaseCraftEnv(gym.Env):
    """
    Base class for all gym craft environments
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed=0, frameskip=4):
        self.steps = 0
        self.lastaction = None
        self.seed(seed)
        self.sim = self._init_simulator()
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.score = 0
        self.frameskip = frameskip

    def _step(self, action):
        translated_action = ACTIONS(action + 1)
        self.steps += 1
        self.lastaction = translated_action.name
        if action < 4:
            repeat_action = translated_action
        else:
            repeat_action = ACTIONS["noop"]

        obs, reward, done, info = self.sim.act(translated_action)
        for _ in range(self.frameskip - 1):
            if done:
                continue
            obs, r, done, info = self.sim.act(repeat_action)
            reward += r

        self.score += reward
        info["score"] = self.score
        if done:
            print(f"completed, score of: {self.score}")
            self.score = 0

        if self.steps == self.sim.timeout:
            done = True
            info["bad_transition"] = True
            print(f"timed out, score of: {self.score}")
            self.score = 0

        return obs, reward, done, info

    def reset(self):
        self.sim = self._init_simulator()
        self.steps = 0
        self.lastaction = None
        return self.sim._get_state_json()

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class JsonCraftEnv(BaseCraftEnv):
    def __init__(
        self, representation, scenario, actions, rewards=None, seed=0, frameskip=4
    ):
        self.rewards = rewards
        self.scenario_name = scenario
        self.scenario = SCENARIOS[scenario]
        super().__init__(seed, frameskip)
        image = _construct_image(representation, scenario)
        self.observation_space = Json(
            self.scenario["size"], image=image, converter=CONVERTERS[representation]
        )
        self.observation_space.controller_converter = (
            json_to_zoomed_binary
        )  # TODO: remove hardcoding of controller converter - should be part of env specification.
        self.first = True
        self.action_space = spaces.Discrete(ACTION_COUNT - 4)
        if scenario == "sparse":
            self.action_space = spaces.Discrete(ACTION_COUNT - 4)
        self.symbolic_action_space = ACTION_SIZE[actions]
        self.actions = actions

    def _init_simulator(self):
        return CraftWorldSimulator(
            self.np_random, self.scenario_name, **self.scenario, rewards=self.rewards
        )

    def step(self, action):
        return self._step(action)

    def reset(self):
        return super().reset()

    def convert_to_human(self, js):
        img = np.transpose(json_to_image(js), (1, 2, 0))

        env = json.loads(js)

        position = env["player"]["position"]
        bearing = env["player"]["bearing"]
        inventory = env["player"]["inventory"]
        # output = img[0:3].copy()
        # output[0:3] = np.abs(1 - output[0]) * 255
        # output[0] = output[0] + img[1] * 255
        # output[1] = output[1] + img[2] * 255
        # output[2] = output[2] + img[3] * 255
        # if img.shape[0] > 4:
        #     output[0] = output[0] + img[4] * 25
        #     output[1] = output[1] + img[4] * 25
        #     output[2] = output[2] + img[4] * 25
        # return (
        #     np.transpose(output, (1, 2, 0)),
        #     env["taxi"]["fuel"],
        #     env["taxi"]["money"],
        # )
        return img, position, bearing, inventory

    def render(self, mode="human"):

        img, position, bearing, inventory = self.convert_to_human(
            self.sim._get_state_json()
        )
        if self.first:
            plt.ion()
            self.first = False
            fig, ax = plt.subplots()
            self.ax = ax
            self.im = ax.imshow(img)
        # fig = plt.figure(figsize=(8, 8))
        # for i in range(4):
        #     fig.add_subplot(1, 4, i + 1)
        #     plt.imshow(img[i])
        self.ax.set_title(
            f"position: {position}, bearing: {bearing}\n inventory: {inventory}"
        )
        self.im.set_data(img)

        plt.pause(0.001)
        plt.draw()

    def random_symbolic_action(self):
        pass

    def convert_to_subgoal(self, action):
        pass

    def expand_symbolic_plan(self, plan):
        return plan

    def get_symbolic_observation(self, obs):
        """ Provides the current symbolic representation of the environment. """
        return json_to_symbolic(obs)

    def project_symbolic_state(self, obs, action):
        """ Calculates a projected state from the current observation and symbolic action.
            Ideally this should hook into the pddl definitions, but for now we duplicate.
        """
        symbolic = self.get_symbolic_observation(obs)

        command, argument = action
        projected = {}
        if command == "move":
            projected["position"] = argument
        elif command == "clear":
            projected["clear"] = {argument: True}
        elif command == "face":
            projected["facing"] = {argument: True}
        elif command == "mine":
            item = self.sim.gamedata.tiles[argument]["mineable"]["item"]
            projected["inventory"] = {item: symbolic["inventory"][item] + 1}
        elif command == "craft":
            quantity = self.sim.gamedata.recipes[argument]["quantity"]
            projected["inventory"] = {
                argument: symbolic["inventory"][argument] + quantity
            }
        return projected

    def check_projected_state_met(self, obs, projected):
        """ Checks if observation is compatibile with the projected partial state. """
        if projected is None:
            return False

        symbolic = self.get_symbolic_observation(obs)

        for category, value in projected.items():
            if category == "position":
                if symbolic[category] != value:
                    return False
            else:
                for k, v in projected[category].items():
                    if symbolic[category][k] != v:
                        return False
        return True

    def convert_to_action(self, subgoal, obs):
        # print("inside _convert_to_action")
        # print(obs, subgoal)
        if obs is not None:
            symbolic = self.get_symbolic_observation(obs)
        else:
            symbolic = {"position": (5, 5)}  # for logging only
        action = {"have": None, "move": None, "clear": None}

        if self.actions == "full":
            have, move, item, quantity, move_dir = subgoal.int().cpu().tolist()
            if move:
                position = facing_block(symbolic["position"], DIRECTIONS(move_dir + 1))
                action["move"] = position
            elif have:  # changed to elif to make mutually exclusive for now.
                action["have"] = (self.sim.gamedata.items[int(item)], quantity)

            # if clear:
            #     action["clear"] = DIRECTIONS(clear_dir + 1)

        elif self.actions == "move-only":
            move, move_dir = subgoal.int().cpu().tolist()
            if move:
                position = facing_block(symbolic["position"], DIRECTIONS(move_dir + 1))
                action["move"] = position

        elif self.actions == "move-continuous":
            move, move_x, move_y = subgoal.cpu().tolist()
            if int(move):
                x, y = symbolic["position"]
                action["move"] = tuple(
                    np.clip([int(x + move_x), int(y + move_y)], 1, self.sim.size - 2)
                )

        elif self.actions == "move-uniform":
            move, move_x, move_y = subgoal.int().cpu().tolist()
            if int(move):
                x, y = symbolic["position"]
                action["move"] = tuple(
                    np.clip(
                        [int(x + move_x - 5), int(y + move_y - 5)], 1, self.sim.size - 2
                    )
                )
        return action

    def generate_pddl(self, ob, subgoal):
        pddl = json_to_pddl(ob)
        pddl = pddl.replace("$goal$", self._action_to_pddl(subgoal))
        return pddl

    def _action_to_pddl(self, action):
        goal = ""

        if action["have"] is not None:
            item, quantity = action["have"]
            if item.endswith("pickaxe"):
                goal += f"({item} p)\n"
            else:
                goal += f"(>= (have p {item}) {quantity})\n"
        if action["move"] is not None:
            goal += f"(moved p c{action['move'][0]} c{action['move'][1]})\n"
        if action["clear"] is not None:
            goal += f"(cleared p {action['clear'].name})\n"

        return goal

    def expand_actions(self, obs, actions):
        new_actions = []
        symbolic = self.get_symbolic_observation(obs)
        for (command, arg) in actions:
            if command in ("move", "clear"):
                # position = facing_block(symbolic["position"], DIRECTIONS[arg])
                new_actions.append((command, arg))
            else:
                new_actions.append((command, arg))

        return new_actions
