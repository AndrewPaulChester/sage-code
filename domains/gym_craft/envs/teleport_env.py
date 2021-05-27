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
    LARGE_COIN
)

from domains.gym_craft.utils.utils import facing_block
from matplotlib import pyplot as plt

# from simulator
ACTION_COUNT = len(ACTIONS)


# gym environment specific
CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84


class TeleportEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.steps = 0
        self.lastaction = None
        self._seed()
        self._init_simulator()

        screen = spaces.Box(
            0, 1, (3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
        dense = spaces.Box(0, 4, (11,), dtype=np.float32)
        self.observation_space = Json(
            42, image=spaces.Tuple((screen, dense)), converter=json_to_mixed
        )
        self.action_space = spaces.Box(-np.inf, np.inf, (2,))
        # self.symbolic_action_space = spaces.Box(-np.inf, np.inf, (3,))
        # self.actions = "move-continuous"
        #    "move-uniform": spaces.Box(-np.inf, np.inf, (3,)),

    def _init_simulator(self):  # TODO: figure out how to make abstract
        self.sim = CraftWorldSimulator(
            self.np_random, 'large_coin', **LARGE_COIN, rewards={"base": 0, "failed-action": 0, "drop-off": 1}
        )
        self.coin = np.unravel_index(
            self.sim.terrain.terrain[:, :, self.sim.gamedata.tiles["coin"]["index"]].argmax(), self.sim.terrain.terrain.shape[0:2]
        )
        self.position,_ = self.sim.player.discrete_coords()
        self.obs = self.sim._get_state_json()

    def reset(self):
        self._init_simulator()
        self.steps = 0
        self.lastaction = None
        return self.obs

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.act(action)

        if self.steps == MAX_EPISODE_LENGTH:
            done = True
            info["bad_transition"] = True
        return obs, reward, done, info

    def act(self,action):
        action[0]+=self.position[0]
        action[1]+=self.position[1]
        x,y = tuple(np.clip(action,1,40))
        action = (int(x),int(y))
        reward = 0
        done = False

        
        if action == self.coin:
            reward = 1
            done = True
        else:
            startx = self.position[0] if self.position[0] < action[0] else action[0]
            stopx = self.position[0] if self.position[0] > action[0] else action[0]
            starty =  self.position[1] if self.position[1] < action[1] else action[1]
            stopy = self.position[1] if self.position[1] > action[1] else action[1]
            if self.coin in zip(range(startx,stopx+1),range(starty,stopy+1)) and self.np_random.rand() < 0.5:
                reward = 1
                done = True
        self.position = action
        self.sim.player.position = action
        return self.sim._get_state_json(),reward,done,{}

    def close(self):
        return

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]





    # def random_symbolic_action(self):
    #     pass

    # def convert_to_subgoal(self, action):
    #     pass

    # def expand_symbolic_plan(self, plan):
    #     return plan

    # def get_symbolic_observation(self, obs):
    #     """ Provides the current symbolic representation of the environment. """
    #     return json_to_symbolic(obs)

    # def project_symbolic_state(self, obs, action):
    #     """ Calculates a projected state from the current observation and symbolic action.
    #         Ideally this should hook into the pddl definitions, but for now we duplicate.
    #     """
    #     symbolic = self.get_symbolic_observation(obs)

    #     command, argument = action
    #     projected = {}
    #     if command == "move":
    #         projected["position"] = argument
    #     elif command == "clear":
    #         projected["clear"] = {argument: True}
    #     elif command == "face":
    #         projected["facing"] = {argument: True}
    #     elif command == "mine":
    #         item = self.sim.gamedata.tiles[argument]["mineable"]["item"]
    #         projected["inventory"] = {item: symbolic["inventory"][item] + 1}
    #     elif command == "craft":
    #         quantity = self.sim.gamedata.recipes[argument]["quantity"]
    #         projected["inventory"] = {
    #             argument: symbolic["inventory"][argument] + quantity
    #         }
    #     return projected

    # def check_projected_state_met(self, obs, projected):
    #     """ Checks if observation is compatibile with the projected partial state. """
    #     if projected is None:
    #         return False

    #     symbolic = self.get_symbolic_observation(obs)

    #     for category, value in projected.items():
    #         if category == "position":
    #             if symbolic[category] != value:
    #                 return False
    #         else:
    #             for k, v in projected[category].items():
    #                 if symbolic[category][k] != v:
    #                     return False
    #     return True

    # def convert_to_action(self, subgoal, obs):
    #     # print("inside _convert_to_action")
    #     # print(obs, subgoal)
    #     if obs is not None:
    #         symbolic = self.get_symbolic_observation(obs)
    #     else:
    #         symbolic = {"position": (5, 5)}  # for logging only
    #     action = {"have": None, "move": None, "clear": None}

    #     if self.actions == "full":
    #         have, move, item, quantity, move_dir = subgoal.int().cpu().tolist()
    #         if move:
    #             position = facing_block(symbolic["position"], DIRECTIONS(move_dir + 1))
    #             action["move"] = position
    #         elif have:  # changed to elif to make mutually exclusive for now.
    #             action["have"] = (self.sim.gamedata.items[int(item)], quantity)

    #         # if clear:
    #         #     action["clear"] = DIRECTIONS(clear_dir + 1)

    #     elif self.actions == "move-only":
    #         move, move_dir = subgoal.int().cpu().tolist()
    #         if move:
    #             position = facing_block(symbolic["position"], DIRECTIONS(move_dir + 1))
    #             action["move"] = position

    #     elif self.actions == "move-continuous":
    #         move, move_x, move_y = subgoal.cpu().tolist()
    #         if int(move):
    #             x, y = symbolic["position"]
    #             action["move"] = tuple(
    #                 np.clip([int(x + move_x), int(y + move_y)], 1, self.sim.size - 2)
    #             )

    #     elif self.actions == "move-uniform":
    #         move, move_x, move_y = subgoal.int().cpu().tolist()
    #         if int(move):
    #             x, y = symbolic["position"]
    #             action["move"] = tuple(np.clip([int(x + move_x - 5), int(y + move_y - 5)], 1, self.sim.size - 2))
    #     return action

    # def generate_pddl(self, ob, subgoal):
    #     pddl = json_to_pddl(ob)
    #     pddl = pddl.replace("$goal$", self._action_to_pddl(subgoal))
    #     return pddl

    # def _action_to_pddl(self, action):
    #     goal = ""

    #     if action["have"] is not None:
    #         item, quantity = action["have"]
    #         if item.endswith("pickaxe"):
    #             goal += f"({item} p)\n"
    #         else:
    #             goal += f"(>= (have p {item}) {quantity})\n"
    #     if action["move"] is not None:
    #         goal += f"(moved p c{action['move'][0]} c{action['move'][1]})\n"
    #     if action["clear"] is not None:
    #         goal += f"(cleared p {action['clear'].name})\n"

    #     return goal

    # def expand_actions(self, obs, actions):
    #     new_actions = []
    #     symbolic = self.get_symbolic_observation(obs)
    #     for (command, arg) in actions:
    #         if command in ("move", "clear"):
    #             # position = facing_block(symbolic["position"], DIRECTIONS[arg])
    #             new_actions.append((command, arg))
    #         else:
    #             new_actions.append((command, arg))

    #     return new_actions
