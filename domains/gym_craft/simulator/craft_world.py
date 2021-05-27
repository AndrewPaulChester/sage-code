"""
.. module:: taxi_world
   :synopsis: Simulates the taxi world environment based on actions passed in.
"""

from enum import Enum
import math
from itertools import count
from collections import defaultdict
import argparse
import numpy as np
from domains.gym_craft.utils.representations import env_to_json

from domains.gym_craft.utils.config import DOORS, DOOR_PROBABILITIES, ROOM_TYPES

# from domains.gym_taxi.utils.utils import generate_random_walls
from domains.gym_craft.utils.utils import (
    discrete_direction,
    facing_block,
    discrete_position,
    blocking_terrain,
)
from domains.gym_craft.simulator.game_data import GameData

# class Tiles(AutoNumber):("Tiles", "tree rock wall coin")
# TILES = Tiles()

ACTIONS = Enum(
    "Actions",
    "forward backward left right noop mine plank stick wooden_pickaxe stone_pickaxe fl fr bl br",
)


# TILES = Enum("Tiles", zip(["tree", "rock", "wall", "coin"],count()))

DEFAULT_REWARDS = {"base": -1, "failed-action": -10, "drop-off": 20}
MAX_EPISODE_LENGTH = 500

MAX_SPEED = 0.4
ACCELERATION = 0.15
DRAG = 0.3
TURN_SPEED = 7.5


# Adjacent room selection stuff
OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


class CraftWorldSimulator(object):
    def __init__(
        self,
        random,
        scenario,
        size,
        timeout=MAX_EPISODE_LENGTH,
        walls="One room",
        random_resources=None,
        random_rooms=None,
        rewards=None,
        starting_zone=None,
        mapping="uvfa",
    ):
        """
        Houses the game state and transition dynamics for the taxi world.

        :param size: size of gridworld
        :returns: this is a description of what is returned
        :raises keyError: raises an exception
        """
        self.random = random
        self.scenario = scenario
        self.size = size
        self.starting_zone = starting_zone
        self.rewards = rewards if rewards is not None else DEFAULT_REWARDS
        self.gamedata = GameData()
        self.timeout = timeout

        self.terrain = Terrain(
            random, size, self.gamedata.tiles, walls, random_resources, random_rooms
        )
        self.player = Player(
            random, size, self.terrain, self.gamedata.items, self.starting_zone
        )

        self.goal = None
        uvfa_mapping = {
            ("no-op", None): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("collect", "coins"): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", (-1, 0)): [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", (1, 0)): [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", (0, 1)): [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", (0, -1)): [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("face", "tree"): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ("face", "rock"): [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ("mine", "tree"): [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ("mine", "rock"): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ("craft", "plank"): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ("craft", "stick"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ("craft", "wooden_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ("craft", "stone_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        multihead_mapping = {
            ("no-op", None): 0,
            ("face", "tree"): 1,
            ("face", "rock"): 2,
            ("move", (0, -1)): 3,
            ("move", (0, 1)): 4,
            ("move", (1, 0)): 5,
            ("move", (-1, 0)): 6,
            ("mine", "tree"): 7,
            ("mine", "rock"): 8,
            ("craft", "plank"): 9,
            ("craft", "stick"): 10,
            ("craft", "wooden_pickaxe"): 11,
            ("craft", "stone_pickaxe"): 12,
            ("collect", "coins"): 13,
        }
        self.mapping = multihead_mapping if mapping == "multihead" else uvfa_mapping

    def _get_state_json(self):
        return env_to_json(self)

    def act(self, action):
        """
        Advances the game state by one step
        :param action: action provided by the agent
        :returns: observation of the next state
        :raises assertionError: raises an exception if action is invalid
        """
        # action, param = action
        # print(action)
        param = 0
        assert action in ACTIONS
        if action == ACTIONS.noop:
            reward = self.rewards["base"]
        elif action == ACTIONS.mine:
            reward = self.attempt_mine()
        elif action in (
            ACTIONS.plank,
            ACTIONS.stick,
            ACTIONS.wooden_pickaxe,
            ACTIONS.stone_pickaxe,
        ):
            reward = self.attempt_craft(action.name)
        else:
            blocking_terrain = self.terrain.get_blocking_terrain()
            self.player.move(action, blocking_terrain)
            reward = self.check_coins()

        done = not np.any(
            self.terrain.terrain[:, :, self.gamedata.tiles["coin"]["index"]]
        )
        env_info = {}
        if self.player.changed_room:
            env_info["changed_room"] = True
            self.player.changed_room = False
        return self._get_state_json(), reward, done, env_info

    def check_coins(self):
        tiles = self.gamedata.tiles
        (x, y), _ = self.player.discrete_coords()
        if self.terrain.terrain[x, y, tiles["coin"]["index"]]:
            self.terrain.terrain[x, y, tiles["coin"]["index"]] = False
            return self.rewards["drop-off"]
        return self.rewards["base"]

    def attempt_craft(self, item):
        recipe = self.gamedata.recipes[item]
        if (
            recipe["catalyst"] != "None"
            and self.player.inventory[recipe["catalyst"]] == 0
        ):
            # print(f"need catalyst {recipe['catalyst']} to create {item}")
            return self.rewards["failed-action"]
        else:
            has_ingredients = True
            for ingredient, quantity in recipe["ingredients"].items():
                if self.player.inventory[ingredient] < quantity:
                    # print(
                    #     f"need {quantity} {ingredient} to create {item}, have {self.player.inventory[ingredient]}"
                    # )
                    return self.rewards["failed-action"]

            for ingredient, quantity in recipe["ingredients"].items():
                self.player.inventory[ingredient] -= quantity

            self.player.inventory[item] += recipe["quantity"]

            return self.rewards["base"]

    def attempt_mine(self):
        facing = self.facing_block()

        if not np.any(self.terrain.terrain[facing]):
            # print("nothing to mine")
            return self.rewards["failed-action"]

        tile_index = np.argmax(self.terrain.terrain[facing])
        tile = self.gamedata.get_tile(tile_index)
        mineable = self.gamedata.tiles[tile]["mineable"]
        if mineable["required"] == "N/A":
            # print("unmineable block")
            return self.rewards["failed-action"]
        elif (
            mineable["required"] != "None"
            and self.player.inventory[mineable["required"]] == 0
        ):
            # print(f"need {mineable['required']} to mine {tile}")
            return self.rewards["failed-action"]
        else:
            self.player.inventory[mineable["item"]] += 1
            self.terrain.terrain[facing][tile_index] = False
            return self.rewards["base"]

    def facing_block(self):
        """
        Returns the block the player is currently facing.
        :return: block player is currently facing
        """
        location, direction = self.player.discrete_coords()
        return facing_block(location, direction)

    def setup_training(self, step):
        # randomise position, inventory contents, and tile presence?
        materials = self.random.gamma(
            1, 5, len(self.gamedata.items) - 2
        )  # TODO: Magic number
        for i in range(len(materials)):
            self.player.inventory[self.gamedata.items[i]] = int(materials[i])
        self.player.inventory["wooden_pickaxe"] = self.random.choice(
            [0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1]
        )
        self.player.inventory["stone_pickaxe"] = self.random.choice(
            [0, 1, 2], p=[0.7, 0.2, 0.1]
        )

        self.terrain.random_clear(self.gamedata.tiles)
        self.player.bearing = self.random.randint(12) * 30

        if step == "craft":
            self.player.random_location(self.terrain)
            # add ingredients to inventory
            item = self.random.choice([k for k in self.gamedata.recipes])
            for k, v in self.gamedata.recipes[item]["ingredients"].items():
                if self.player.inventory[k] < v:
                    self.player.inventory[k] = v
            self.goal = self.mapping[(step, item)]
            return item
        elif step == "face":
            # ensure room has the right tile
            tile = self.random.choice(
                [
                    k
                    for k, v in self.gamedata.tiles.items()
                    if v["mineable"]["item"] != "N/A"
                ]
            )
            self.player.random_location(
                self.terrain, self.gamedata.tiles[tile]["index"]
            )
            self.goal = self.mapping[(step, tile)]
            return tile
        elif step == "mine":
            tile = self.random.choice(
                [
                    k
                    for k, v in self.gamedata.tiles.items()
                    if v["mineable"]["item"] != "N/A"
                ]
            )
            item = self.gamedata.tiles[tile]["mineable"]["required"]
            if item != "None":
                self.player.inventory[item] += 1

            self.player.random_location(
                self.terrain, self.gamedata.tiles[tile]["index"]
            )

            # change facing tile to right tile
            facing = self.facing_block()
            while self.terrain.terrain[facing][self.gamedata.tiles["wall"]["index"]]:
                self.player.bearing = (self.player.bearing + 90) % 360
                facing = self.facing_block()

            self.terrain.terrain[facing] = False
            self.terrain.terrain[facing][self.gamedata.tiles[tile]["index"]] = True
            self.goal = self.mapping[(step, tile)]
            return tile

        elif step == "collect":
            # ensure room has coins
            self.player.random_location(
                self.terrain, self.gamedata.tiles["coin"]["index"]
            )
            self.goal = self.mapping[(step, "coins")]
            return None
        elif step == "move":
            self.player.random_location(self.terrain)
            # add pick to inventory if doors are stone
            self.player.inventory["wooden_pickaxe"] += 1

            # this is a horrendously complicated way of selecting an adjacent room
            position = self.player.position
            x = int(position[0] / 10)
            y = int(position[1] / 10)
            n = int(self.size / 10)

            self.random.shuffle(OFFSETS)

            for (xo, yo) in OFFSETS:
                nx = x + xo
                ny = y + yo
                if (
                    nx >= 0 and nx < n and ny >= 0 and ny < n
                ):  # is the adjacent room in the bounds of the grid
                    door_x = (nx + x) * 5 + 5
                    door_y = (ny + y) * 5 + 5
                    if not self.terrain.terrain[(door_x, door_y)][
                        self.gamedata.tiles["wall"]["index"]
                    ]:  # is the doorway to the adjacent room open.
                        break

            self.goal = self.mapping[(step, (xo, yo))]
            return nx, ny


class Terrain(object):
    """
    Represents the road network for the current simulation
    """

    def __init__(
        self, random, grid_size, tiles, walls, random_resources=None, rooms=None
    ):
        self.random = random
        self.terrain = np.zeros((grid_size, grid_size, len(tiles)), dtype=np.bool)
        # exterior walls
        self.terrain[0, :, tiles["wall"]["index"]] = True
        self.terrain[-1, :, tiles["wall"]["index"]] = True
        self.terrain[:, 0, tiles["wall"]["index"]] = True
        self.terrain[:, -1, tiles["wall"]["index"]] = True

        if walls == "rooms":
            for i in range(10, grid_size - 1, 10):
                # walls
                self.terrain[:, i, tiles["wall"]["index"]] = True
                self.terrain[i, :, tiles["wall"]["index"]] = True

                # "doors"
                for j in range(5, grid_size, 10):
                    self.terrain[j, i, tiles["wall"]["index"]] = False
                    self.terrain[i, j, tiles["wall"]["index"]] = False
                    d1, d2 = random.choice(DOORS, 2, p=DOOR_PROBABILITIES)

                    if d1 is not None:
                        self.terrain[j, i, tiles[d1]["index"]] = True

                    if d2 is not None:
                        self.terrain[i, j, tiles[d2]["index"]] = True

        if random_resources is not None:
            for ((x1, y1), (x2, y2), resources) in random_resources:
                self._populate_area(x1, y1, x2, y2, resources, tiles)
        elif rooms is not None:
            room_list = []
            for room, c in rooms.items():
                room_list.extend([room] * c)
            self.random.shuffle(room_list)

            # if there is a mixed room, ensure it is in top left
            if "mixed" in room_list:
                room_list.remove("mixed")
                room_list += ["mixed"]

            for x1 in range(1, grid_size, 10):
                for y1 in range(1, grid_size, 10):
                    room_type = room_list.pop()
                    self._populate_area(
                        x1, y1, x1 + 8, y1 + 8, ROOM_TYPES[room_type], tiles
                    )

        else:
            self.terrain[2, 2, tiles["rock"]["index"]] = True
            self.terrain[3:5, 4:7, tiles["tree"]["index"]] = True

            self.terrain[17, 2, tiles["tree"]["index"]] = True
            self.terrain[13:18, 7, tiles["tree"]["index"]] = True

            self.terrain[12:14, 4:6, tiles["rock"]["index"]] = True

            self.terrain[4, 11, tiles["rock"]["index"]] = True
            self.terrain[6, 18, tiles["rock"]["index"]] = True
            self.terrain[7, 12, tiles["rock"]["index"]] = True
            self.terrain[8, 14, tiles["rock"]["index"]] = True

            self.terrain[2, 3, tiles["coin"]["index"]] = True
            self.terrain[3, 3, tiles["coin"]["index"]] = True
            self.terrain[7, 4, tiles["coin"]["index"]] = True

            self.terrain[6, 14, tiles["coin"]["index"]] = True
            self.terrain[8, 11, tiles["coin"]["index"]] = True
            self.terrain[7, 13, tiles["coin"]["index"]] = True

            self.terrain[11, 9, tiles["coin"]["index"]] = True
            self.terrain[13, 2, tiles["coin"]["index"]] = True

            self.terrain[13, 15, tiles["coin"]["index"]] = True
            self.terrain[15, 13, tiles["coin"]["index"]] = True
            self.terrain[18, 12, tiles["coin"]["index"]] = True
            self.terrain[16, 14, tiles["coin"]["index"]] = True
            self.terrain[12, 12, tiles["coin"]["index"]] = True

        self.block_mask = [False] * len(tiles)
        for tile in tiles.values():
            if tile["blocking"]:
                self.block_mask[tile["index"]] = True

    def get_blocking_terrain(self):
        return blocking_terrain(self.terrain, self.block_mask)

    def _populate_area(self, x1, y1, x2, y2, resources, tiles):
        cells = [(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)]
        total = sum([x for x in resources.values()])
        self.random.shuffle(cells)
        for tile, quantity in resources.items():
            for _ in range(quantity):
                coords = cells.pop()
                self.terrain[coords][tiles[tile]["index"]] = True

    def random_clear(self, tiles):
        wall_index = tiles["wall"]["index"]
        grid_size = self.terrain.shape[0]
        for x1 in range(1, grid_size, 10):
            for y1 in range(1, grid_size, 10):
                if self.random.rand() < 0.3:
                    self._depopulate_area(x1, y1, x1 + 10, y1 + 10, wall_index)

    def _depopulate_area(self, x1, y1, x2, y2, wall_index):
        for x in range(x1, x2):
            for y in range(y1, y2):
                if not self.terrain[x, y, wall_index] and self.random.rand() < 0.7:
                    self.terrain[x, y, :] = False


class Player(object):
    """
    Contains information about the position and state of the player
    """

    def __init__(self, random, grid_size, terrain, items, starting_zone=None):
        self.random = random
        if terrain:
            invalid = terrain.get_blocking_terrain()
            chosen = False
            while not chosen:
                if starting_zone:
                    self.position = self._sample_position(starting_zone)
                else:
                    self.position = tuple(random.randint(10, size=2))
                if not invalid[self.position]:
                    chosen = True
        else:
            if starting_zone:
                self.position = self._sample_position(starting_zone)
            else:
                self.position = tuple(random.randint(grid_size, size=2))

        self.inventory = {k: 0 for k in items}

        self.room = (0, 0)  # TODO: make this work if not always starting in top left.
        self.bearing = 0
        self.speed = 0
        self.changed_room = False

    def _sample_position(self, starting_zone):
        (x1, y1), (x2, y2) = starting_zone
        cells = [(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)]
        self.random.shuffle(cells)
        return cells[0]

    def move(self, action, terrain):
        if action in [ACTIONS.forward, ACTIONS.fr, ACTIONS.fl]:
            acceleration = 1
        elif action in [ACTIONS.backward, ACTIONS.br, ACTIONS.bl]:
            acceleration = -1
        else:
            acceleration = 0

        if action in [ACTIONS.right, ACTIONS.fr, ACTIONS.br]:
            turning = 1
        elif action in [ACTIONS.left, ACTIONS.fl, ACTIONS.bl]:
            turning = -1
        else:
            turning = 0

        self.update_speed(acceleration)
        self.update_bearing(turning)
        self.update_coords(terrain)
        return self.discrete_coords()

    def discrete_coords(self):
        position = discrete_position(self.position)
        direction = discrete_direction(self.bearing)
        return position, direction

    def update_coords(self, terrain):
        (x_pos, y_pos) = self.position
        new_x_pos = x_pos + self.speed * math.sin(self.bearing * 2 * math.pi / 360)
        new_y_pos = y_pos - self.speed * math.cos(
            self.bearing * 2 * math.pi / 360
        )  # minus because y-axis flipped (0 is top)

        if abs(new_x_pos - x_pos) > abs(new_y_pos - y_pos):
            new_x_pos, _ = self.collision_detection(
                terrain, x_pos, y_pos, new_x_pos, y_pos
            )
            _, new_y_pos = self.collision_detection(
                terrain, new_x_pos, y_pos, new_x_pos, new_y_pos
            )
        else:
            _, new_y_pos = self.collision_detection(
                terrain, x_pos, y_pos, x_pos, new_y_pos
            )
            new_x_pos, _ = self.collision_detection(
                terrain, x_pos, new_y_pos, new_x_pos, new_y_pos
            )

        self.position = (new_x_pos, new_y_pos)

        if int(new_x_pos) % 10 != 0 and int(new_y_pos) % 10 != 0:  # not in a doorway
            old = self.room
            self.room = int(new_x_pos / 10), int(new_y_pos / 10)
            if old != self.room:
                self.changed_room = True

            # print(self.room)

    def collision_detection(self, terrain, x_pos, y_pos, new_x_pos, new_y_pos):

        # check collisions
        # current logic allows moving through diagonal walls - maybe should be fixed
        if terrain[int(new_x_pos), int(new_y_pos)]:
            if int(new_x_pos) > int(x_pos):  # or x_pos == int(new_x_pos):
                new_x_pos = int(new_x_pos) - 0.01
            elif int(new_x_pos) < int(x_pos):
                new_x_pos = int(x_pos) + 0.01

            if int(new_y_pos) > int(y_pos):  # or y_pos == int(new_y_pos):
                new_y_pos = int(new_y_pos) - 0.01
            elif int(new_y_pos) < int(y_pos):
                new_y_pos = int(y_pos) + 0.01
        return (new_x_pos, new_y_pos)

    def update_bearing(self, turning):
        self.bearing = (self.bearing + TURN_SPEED * turning) % 360

    def update_speed(self, acceleration):
        self.speed *= DRAG
        self.speed += acceleration * ACCELERATION
        self.speed = (
            min(self.speed, MAX_SPEED)
            if self.speed > 0
            else max(self.speed, -MAX_SPEED)
        )

    def random_location(self, terrain, tile=None):
        invalid = terrain.get_blocking_terrain()
        grid_size = terrain.terrain.shape[0]
        chosen = False
        while not chosen:
            self.position = tuple(self.random.randint(grid_size, size=2))
            if not invalid[self.position]:
                x = self.position[0] - self.position[0] % 10
                y = self.position[1] - self.position[1] % 10
                if tile is None or np.any(
                    terrain.terrain[x : x + 10, y : y + 10, tile]
                ):
                    self.room = (int(self.position[0] / 10), int(self.position[1] / 10))
                    chosen = True
                    self.changed_room = False
