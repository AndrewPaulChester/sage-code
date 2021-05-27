"""
.. module:: representations
   :synopsis: Contains functions to convert between different representations of the taxi world.
   Functions are of the form x_to_y where x and y are represenation formats, from the list below:
   env - the actual TaxiWorld simulator class itself, holds more than just the current state.
   json - a general purpose JSON serialisation of the current state. All functions should convert to/from this as a common ground.
   image - a four channel image based encoding, designed for input to CNN
   discrete - a discretised encoding only valid for a 5x5 grid, designed for standard Q-agents and compatibility with gym
   pddl - planning domain file designed for input to planner, but lacks a goal.

"""


import math
from collections import defaultdict
import json
import cv2
import numpy as np
from domains.gym_craft.utils.utils import (
    json_default,
    discrete_direction,
    facing_block,
    discrete_position,
    blocking_terrain,
)
from domains.gym_craft.utils.config import DIRECTIONS

RED = np.zeros(3, dtype=np.uint8) + [255, 0, 0]


def env_to_json(env):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    player = env.player.__dict__.copy()
    del player["random"]
    return json.dumps(
        {
            "scenario": env.scenario,
            "n": env.size,
            "terrain": env.terrain.terrain,
            "block_mask": env.terrain.block_mask,
            "tiles": env.gamedata.tiles,
            "player": player,
            "goal": env.goal,
        },
        default=json_default,
    )


def flatten_terrain(terrain):
    return np.argmax(terrain, 2)


def colour_array(tiles):
    return np.array([tile["colour"] for tile in tiles.values()], dtype=np.uint8)


def json_to_image(js):
    """
    Converts craft world state from json to image representation

    :param js: craft world state in json format
    :return: craft world state in image format
    """
    env = json.loads(js)
    return create_image(env)
    # return create_image(env, representation="zoomed") #For debugging zoomed view


def create_image(env, representation="full", planes="colour"):
    if planes == "colour":
        img = create_colour_image(env)
    else:
        img = create_binary_image(env)

    if representation == "full":
        return img
    else:
        position = discrete_position(env["player"]["position"])
        x = env["player"]["room"][0] * 10
        y = env["player"]["room"][1] * 10
        if representation == "zoomed":
            return img[:, y : y + 11, x : x + 11]
        elif representation == "masked":
            masked = np.zeros_like(img)
            masked[:, y : y + 11, x : x + 11] = img[:, y : y + 11, x : x + 11]
            return masked
        elif representation == "centered":
            centered = np.zeros_like(img[:, 0:21, 0:21])

            x1, xo, x2 = _calculate_offsets(position[0], 10)
            y1, yo, y2 = _calculate_offsets(position[1], 10)

            _, y, x = img[:, y1:y2, x1:x2].shape
            centered[:, yo : yo + y, xo : xo + x] = img[:, y1:y2, x1:x2]
            return centered


def _calculate_offsets(x, n):
    if x < n:
        x1 = 0
        xo = n - x
    else:
        x1 = x - n
        xo = 0
    return x1, xo, x + n + 1


def create_binary_image(env):
    terrain = np.array(env["terrain"], dtype=np.uint8) * 255
    x, y, channels = terrain.shape
    player = np.zeros((x, y, 1), dtype=np.uint8)
    position = discrete_position(env["player"]["position"])
    player[position] = 255
    image = np.concatenate((terrain, player), axis=2)

    return np.transpose(image, (2, 1, 0))  # swapping x&y


def create_colour_image(env):
    channels = 3
    image_size = env["n"]
    image = np.ones((image_size, image_size, channels), dtype=np.uint8) * 255
    mask = np.any(env["terrain"], 2)
    image[mask] = colour_array(env["tiles"])[flatten_terrain(env["terrain"])][mask]

    position = discrete_position(env["player"]["position"])
    direction = discrete_direction(env["player"]["bearing"])
    facing = facing_block(position, direction)
    image[position] = RED
    # if sum(image[facing]) == 765:
    #     image[facing] -= np.ones(3, dtype=np.uint8) * 50
    # else:
    #     image[facing] = np.clip(image[facing] + np.ones(3, dtype=np.uint8) * 50, 0, 255)

    return np.transpose(image, (2, 1, 0))  # swapping x&y


def json_to_mixed(js, train=False, representation="full", planes="colour"):
    """
    Converts craft world state from json to mixed representation, which is part image, part dense input

    :param js: craft world state in json format
    :return: craft world state in mixed format
    """
    env = json.loads(js)

    image = create_image(env, representation, planes)

    dense = get_dense(env, train)
    return (resize_image(image, 84), np.array(dense, dtype=np.float32))


def json_to_train(js):
    return json_to_mixed(js, train=True)


def json_to_zoomed(js):
    return json_to_mixed(js, train=True, representation="zoomed")


def json_to_masked(js):
    return json_to_mixed(js, train=True, representation="masked")


def json_to_centered(js):
    return json_to_mixed(js, train=True, representation="centered")


def json_to_centered_binary(js):
    return json_to_mixed(js, train=True, representation="centered", planes="binary")


def json_to_zoomed_binary(js):
    return json_to_mixed(js, train=True, representation="zoomed", planes="binary")


def json_to_binary(js):
    return json_to_mixed(js, train=True, representation="full", planes="binary")


def json_to_dense(js):
    """
    Converts craft world state from json to dense representation, which is all dense input

    :param js: craft world state in json format
    :return: craft world state in dense format
    """
    env = json.loads(js)

    dense = get_dense(env)
    if env["scenario"] == "single_coin":
        terrain = np.array(env["terrain"])
        x, y = np.unravel_index(
            terrain[:, :, env["tiles"]["coin"]["index"]].argmax(), terrain.shape[0:2]
        )
        dense += [x / env["n"], y / env["n"]]

    return np.array(dense, dtype=np.float32)


def get_dense(env, train=False):
    n = env["n"]
    position = env["player"]["position"]
    bearing = env["player"]["bearing"]
    speed = env["player"]["speed"]

    sin = math.sin(bearing * 2 * math.pi / 360)
    cos = math.cos(bearing * 2 * math.pi / 360)
    dense = [position[0] / n, position[1] / n, sin, cos, speed]
    dense += [math.log(v + 1) for v in env["player"]["inventory"].values()]
    # TODO: Zeroing out speed for now because it seems like speed signal is causing issues - think about why
    dense[4] = 0

    # env["player"]
    if train and env["goal"] is not None:
        try:
            dense.extend(env["goal"])
        except TypeError:
            dense.append(env["goal"])

    return [d for d in dense]


def json_to_abstract(js):
    symbolic = json_to_symbolic(js, representation="rooms")
    env = json.loads(js)
    dense = get_dense(env)
    contents = [math.log(v + 1) for d in symbolic["rooms"].values() for v in d.values()]
    position = [0] * 9
    x, y = symbolic["room"]
    position[y * 3 + x] = 1

    encoding = {
        "tree": [1, 0, 0],
        "rock": [0, 1, 0],
        "wall": [0, 0, 1],
        "none": [0, 0, 0],
    }
    # Not using doors yet, but could be added
    doors = [b for v in symbolic["doors"].values() for b in encoding[v]]
    # TODO: implement better scaling....
    contents = [c - 1.5 for c in contents]
    for i in range(len(contents)):
        if i % 3 < 2:
            contents[i] = contents[i] / 4

    return contents + position + dense


def json_to_symbolic(js, representation="simple"):
    """
    Converts craft world state from json to structured symbolic representation. Consists of:
        - inventory: defaultdict of current items
        - facing: boolean dict of mineable tiles currently faced
        - clear: boolean dict of directions that are currently clear (i.e. not blocked)
        - position: discrete coordinates of player

    if representation="rooms", also contains information about positions and contents of rooms.

    :param js: craft world state in json format
    :return: craft world state in symbolic format
    """
    env = json.loads(js)
    state = {}
    state["inventory"] = defaultdict(int, env["player"]["inventory"])

    position = discrete_position(env["player"]["position"])

    state["position"] = position
    direction = discrete_direction(env["player"]["bearing"])
    facing_position = facing_block(position, direction, env["n"])
    terrain = np.array(env["terrain"])
    state["facing"] = {}
    for tile, info in env["tiles"].items():
        if info["mineable"]["item"] != "N/A":
            state["facing"][tile] = terrain[facing_position][info["index"]]

    state["clear"] = {}
    blocked = blocking_terrain(terrain, env["block_mask"])
    for direction in DIRECTIONS:
        facing_position = facing_block(position, direction)
        state["clear"][direction.name] = not blocked[facing_position]

    if representation == "rooms":
        rooms = {}
        doors = {}
        size = env["n"]
        for i, x1 in enumerate(range(1, size, 10)):
            for j, y1 in enumerate(range(1, size, 10)):
                rooms[(i, j)] = _summarise_area(
                    env["tiles"], terrain, x1, y1, x1 + 9, y1 + 9
                )

        flat = flatten_terrain(terrain)
        for i, x in enumerate(range(10, size - 1, 10)):
            for j, y in enumerate(range(5, size, 10)):
                tile_index = np.argmax(terrain[x, y])
                doors[("h", i, j)] = list(env["tiles"].keys())[tile_index]
                tile_index = np.argmax(terrain[y, x])
                doors[("v", j, i)] = list(env["tiles"].keys())[tile_index]

        state["room"] = tuple(env["player"]["room"])
        state["door"] = (
            position[0] % 10 == 0 or position[1] % 10 == 0
        )  # flag if agent is in a doorway

        state["rooms"] = rooms
        state["doors"] = doors

    return state


def _summarise_area(tiles, terrain, x1, y1, x2, y2):
    area = {}
    for tile in tiles:
        if tile == "wall":
            continue
        index = tiles[tile]["index"]
        area[tile] = np.count_nonzero(terrain[x1:x2, y1:y2, index])
    return area


def json_to_both(js):
    image, _ = json_to_mixed(js, train=True, representation="zoomed", planes="binary")
    abstract = json_to_abstract(js)
    return image, abstract


def json_to_pddl(js, representation="simple"):
    """
    Converts craft world state from json to pddl representation

    :param js: craft world state in json format
    :return: craft world state in pddl format
    """

    symbolic = json_to_symbolic(js, representation)
    items = []
    for item in symbolic["inventory"]:
        if item in ("stone_pickaxe", "wooden_pickaxe", "furnace"):
            if symbolic["inventory"][item] > 0:
                items.append(f"({item} p)")
            else:
                items.append(f"(not ({item} p))")
        else:
            items.append(f"(= (have p {item}) {symbolic['inventory'][item]})")

    cleared = []
    for direction in symbolic["clear"]:
        if symbolic["clear"][direction]:
            cleared.append(f"(cleared p {direction})")
        else:
            cleared.append(f"(not (cleared p {direction}))")

    faced = []
    for tile in symbolic["facing"]:
        if symbolic["facing"][tile]:
            faced.append(f"(facing p {tile})")
        else:
            faced.append(f"(not (facing p {tile}))")

    if representation == "rooms":

        # player room location
        location = f"(in p r{symbolic['room'][0]}{symbolic['room'][1]})"

        # room list
        rooms = [f"r{x}{y}" for x, y in symbolic["rooms"]]

        # room contents
        contents = [
            f"(= (contains r{x}{y} {tile}) {quantity})"
            for ((x, y), d) in symbolic["rooms"].items()
            for (tile, quantity) in d.items()
        ]

        doors = []
        adjacent = []
        # door list
        # room adjacency list
        for (d, i, j), tile in symbolic["doors"].items():
            if d == "h":
                doors.append(f"(door r{i}{j} r{i+1}{j} {tile})")
                doors.append(f"(door r{i+1}{j} r{i}{j} {tile})")
                adjacent.append(f"(adjacent r{i}{j} r{i+1}{j} east)")
                adjacent.append(f"(adjacent r{i+1}{j} r{i}{j} west)")
            elif d == "v":
                doors.append(f"(door r{i}{j} r{i}{j+1} {tile})")
                doors.append(f"(door r{i}{j+1} r{i}{j} {tile})")
                adjacent.append(f"(adjacent r{i}{j} r{i}{j+1} south)")
                adjacent.append(f"(adjacent r{i}{j+1} r{i}{j} north)")

        # [f"(door r{i}{j} r{i}{j+1} {tile})"d,i,j,tile for (d,i,j),tile in doors.items() where d = 'h']
        # [f"(door r{i}{j} r{i+1}{j} {tile})"d,i,j,tile for (d,i,j),tile in doors.items() where d = 'v']

        # n = math.sqrt(len(symbolic["rooms"]))
        # for i in range(n):
        #     for j in range(n):
        #         if i < n - 1:
        #             adjacent.append(f"(adjacent r{i}{j} r{i+1}{j} east)")
        #             adjacent.append(f"(adjacent r{i+1}{j} r{i}{j} west)")
        #         if j < n - 1:
        #             adjacent.append(f"(adjacent r{i}{j} r{i}{j+1} south)")
        #             adjacent.append(f"(adjacent r{i}{j+1} r{i}{j} north)")

        return _compose_pddl(
            representation,
            "\n".join(items),
            "\n".join(cleared),
            "\n".join(faced),
            location,
            " ".join(rooms),
            "\n".join(adjacent),
            "\n".join(contents),
            "\n".join(doors),
        )

    return _compose_pddl(
        representation, "\n".join(items), "\n".join(cleared), "\n".join(faced)
    )


def json_to_screen(js):
    return resize_image(json_to_image(js), 84)  # TODO: magic number


def resize_image(img, size):
    """
    Modifies image dimensions . 
    :param img: taxi world state in image format
    :param size: size for converted 
    :return: taxi world state in image format of specified size
    """
    resized = cv2.resize(
        np.transpose(img, (1, 2, 0)), (size, size), interpolation=cv2.INTER_AREA
    )

    if len(resized.shape) == 2:
        return np.expand_dims(resized, axis=0)

    return np.transpose(resized, (2, 0, 1))


def _get_coord_from_np_array(array):
    return tuple((np.transpose((array == 1).nonzero())[0] / 2).astype(int))


def _compose_pddl(
    representation,
    items,
    cleared,
    faced,
    location=None,
    rooms=None,
    adjacent=None,
    contents=None,
    doors=None,
):
    if representation == "rooms":
        return _compose_rooms_pddl(
            items, cleared, faced, location, rooms, adjacent, contents, doors
        )
    elif representation == "simple":
        return _compose_craft_pddl(items, cleared, faced)


def _compose_rooms_pddl(
    items, cleared, faced, location, rooms, adjacent, contents, doors
):
    return f"""
        (define (problem craft0) (:domain craftrooms)
        (:objects 
            p - player
            {rooms} - room
        )

        (:init
            ;todo: put the initial state's facts and numeric values here
        
            (= (cost) 0)
            {items}
            {cleared}
            {location}

            {adjacent}

            {contents}

            {doors}

            {faced}
        )

        (:goal (and
                $goal$
            )
        )

        ;un-comment the following line if metric is needed
        (:metric minimize (cost))
        )
        """


def _compose_craft_pddl(items, cleared, faced):
    return f"""
    (define (problem craft0) (:domain craft)
    (:objects 
        p - player
    )

    (:init
        ;todo: put the initial state's facts and numeric values here
       
        (= (cost) 0)
       
  (not (moved p c0 c0))
(not (moved p c0 c1))
(not (moved p c0 c2))
(not (moved p c0 c3))
(not (moved p c0 c4))
(not (moved p c0 c5))
(not (moved p c0 c6))
(not (moved p c0 c7))
(not (moved p c0 c8))
(not (moved p c0 c9))
(not (moved p c0 c10))
(not (moved p c0 c11))
(not (moved p c0 c12))
(not (moved p c0 c13))
(not (moved p c0 c14))
(not (moved p c0 c15))
(not (moved p c0 c16))
(not (moved p c0 c17))
(not (moved p c0 c18))
(not (moved p c0 c19))
(not (moved p c1 c0))
(not (moved p c1 c1))
(not (moved p c1 c2))
(not (moved p c1 c3))
(not (moved p c1 c4))
(not (moved p c1 c5))
(not (moved p c1 c6))
(not (moved p c1 c7))
(not (moved p c1 c8))
(not (moved p c1 c9))
(not (moved p c1 c10))
(not (moved p c1 c11))
(not (moved p c1 c12))
(not (moved p c1 c13))
(not (moved p c1 c14))
(not (moved p c1 c15))
(not (moved p c1 c16))
(not (moved p c1 c17))
(not (moved p c1 c18))
(not (moved p c1 c19))
(not (moved p c2 c0))
(not (moved p c2 c1))
(not (moved p c2 c2))
(not (moved p c2 c3))
(not (moved p c2 c4))
(not (moved p c2 c5))
(not (moved p c2 c6))
(not (moved p c2 c7))
(not (moved p c2 c8))
(not (moved p c2 c9))
(not (moved p c2 c10))
(not (moved p c2 c11))
(not (moved p c2 c12))
(not (moved p c2 c13))
(not (moved p c2 c14))
(not (moved p c2 c15))
(not (moved p c2 c16))
(not (moved p c2 c17))
(not (moved p c2 c18))
(not (moved p c2 c19))
(not (moved p c3 c0))
(not (moved p c3 c1))
(not (moved p c3 c2))
(not (moved p c3 c3))
(not (moved p c3 c4))
(not (moved p c3 c5))
(not (moved p c3 c6))
(not (moved p c3 c7))
(not (moved p c3 c8))
(not (moved p c3 c9))
(not (moved p c3 c10))
(not (moved p c3 c11))
(not (moved p c3 c12))
(not (moved p c3 c13))
(not (moved p c3 c14))
(not (moved p c3 c15))
(not (moved p c3 c16))
(not (moved p c3 c17))
(not (moved p c3 c18))
(not (moved p c3 c19))
(not (moved p c4 c0))
(not (moved p c4 c1))
(not (moved p c4 c2))
(not (moved p c4 c3))
(not (moved p c4 c4))
(not (moved p c4 c5))
(not (moved p c4 c6))
(not (moved p c4 c7))
(not (moved p c4 c8))
(not (moved p c4 c9))
(not (moved p c4 c10))
(not (moved p c4 c11))
(not (moved p c4 c12))
(not (moved p c4 c13))
(not (moved p c4 c14))
(not (moved p c4 c15))
(not (moved p c4 c16))
(not (moved p c4 c17))
(not (moved p c4 c18))
(not (moved p c4 c19))
(not (moved p c5 c0))
(not (moved p c5 c1))
(not (moved p c5 c2))
(not (moved p c5 c3))
(not (moved p c5 c4))
(not (moved p c5 c5))
(not (moved p c5 c6))
(not (moved p c5 c7))
(not (moved p c5 c8))
(not (moved p c5 c9))
(not (moved p c5 c10))
(not (moved p c5 c11))
(not (moved p c5 c12))
(not (moved p c5 c13))
(not (moved p c5 c14))
(not (moved p c5 c15))
(not (moved p c5 c16))
(not (moved p c5 c17))
(not (moved p c5 c18))
(not (moved p c5 c19))
(not (moved p c6 c0))
(not (moved p c6 c1))
(not (moved p c6 c2))
(not (moved p c6 c3))
(not (moved p c6 c4))
(not (moved p c6 c5))
(not (moved p c6 c6))
(not (moved p c6 c7))
(not (moved p c6 c8))
(not (moved p c6 c9))
(not (moved p c6 c10))
(not (moved p c6 c11))
(not (moved p c6 c12))
(not (moved p c6 c13))
(not (moved p c6 c14))
(not (moved p c6 c15))
(not (moved p c6 c16))
(not (moved p c6 c17))
(not (moved p c6 c18))
(not (moved p c6 c19))
(not (moved p c7 c0))
(not (moved p c7 c1))
(not (moved p c7 c2))
(not (moved p c7 c3))
(not (moved p c7 c4))
(not (moved p c7 c5))
(not (moved p c7 c6))
(not (moved p c7 c7))
(not (moved p c7 c8))
(not (moved p c7 c9))
(not (moved p c7 c10))
(not (moved p c7 c11))
(not (moved p c7 c12))
(not (moved p c7 c13))
(not (moved p c7 c14))
(not (moved p c7 c15))
(not (moved p c7 c16))
(not (moved p c7 c17))
(not (moved p c7 c18))
(not (moved p c7 c19))
(not (moved p c8 c0))
(not (moved p c8 c1))
(not (moved p c8 c2))
(not (moved p c8 c3))
(not (moved p c8 c4))
(not (moved p c8 c5))
(not (moved p c8 c6))
(not (moved p c8 c7))
(not (moved p c8 c8))
(not (moved p c8 c9))
(not (moved p c8 c10))
(not (moved p c8 c11))
(not (moved p c8 c12))
(not (moved p c8 c13))
(not (moved p c8 c14))
(not (moved p c8 c15))
(not (moved p c8 c16))
(not (moved p c8 c17))
(not (moved p c8 c18))
(not (moved p c8 c19))
(not (moved p c9 c0))
(not (moved p c9 c1))
(not (moved p c9 c2))
(not (moved p c9 c3))
(not (moved p c9 c4))
(not (moved p c9 c5))
(not (moved p c9 c6))
(not (moved p c9 c7))
(not (moved p c9 c8))
(not (moved p c9 c9))
(not (moved p c9 c10))
(not (moved p c9 c11))
(not (moved p c9 c12))
(not (moved p c9 c13))
(not (moved p c9 c14))
(not (moved p c9 c15))
(not (moved p c9 c16))
(not (moved p c9 c17))
(not (moved p c9 c18))
(not (moved p c9 c19))
(not (moved p c10 c0))
(not (moved p c10 c1))
(not (moved p c10 c2))
(not (moved p c10 c3))
(not (moved p c10 c4))
(not (moved p c10 c5))
(not (moved p c10 c6))
(not (moved p c10 c7))
(not (moved p c10 c8))
(not (moved p c10 c9))
(not (moved p c10 c10))
(not (moved p c10 c11))
(not (moved p c10 c12))
(not (moved p c10 c13))
(not (moved p c10 c14))
(not (moved p c10 c15))
(not (moved p c10 c16))
(not (moved p c10 c17))
(not (moved p c10 c18))
(not (moved p c10 c19))
(not (moved p c11 c0))
(not (moved p c11 c1))
(not (moved p c11 c2))
(not (moved p c11 c3))
(not (moved p c11 c4))
(not (moved p c11 c5))
(not (moved p c11 c6))
(not (moved p c11 c7))
(not (moved p c11 c8))
(not (moved p c11 c9))
(not (moved p c11 c10))
(not (moved p c11 c11))
(not (moved p c11 c12))
(not (moved p c11 c13))
(not (moved p c11 c14))
(not (moved p c11 c15))
(not (moved p c11 c16))
(not (moved p c11 c17))
(not (moved p c11 c18))
(not (moved p c11 c19))
(not (moved p c12 c0))
(not (moved p c12 c1))
(not (moved p c12 c2))
(not (moved p c12 c3))
(not (moved p c12 c4))
(not (moved p c12 c5))
(not (moved p c12 c6))
(not (moved p c12 c7))
(not (moved p c12 c8))
(not (moved p c12 c9))
(not (moved p c12 c10))
(not (moved p c12 c11))
(not (moved p c12 c12))
(not (moved p c12 c13))
(not (moved p c12 c14))
(not (moved p c12 c15))
(not (moved p c12 c16))
(not (moved p c12 c17))
(not (moved p c12 c18))
(not (moved p c12 c19))
(not (moved p c13 c0))
(not (moved p c13 c1))
(not (moved p c13 c2))
(not (moved p c13 c3))
(not (moved p c13 c4))
(not (moved p c13 c5))
(not (moved p c13 c6))
(not (moved p c13 c7))
(not (moved p c13 c8))
(not (moved p c13 c9))
(not (moved p c13 c10))
(not (moved p c13 c11))
(not (moved p c13 c12))
(not (moved p c13 c13))
(not (moved p c13 c14))
(not (moved p c13 c15))
(not (moved p c13 c16))
(not (moved p c13 c17))
(not (moved p c13 c18))
(not (moved p c13 c19))
(not (moved p c14 c0))
(not (moved p c14 c1))
(not (moved p c14 c2))
(not (moved p c14 c3))
(not (moved p c14 c4))
(not (moved p c14 c5))
(not (moved p c14 c6))
(not (moved p c14 c7))
(not (moved p c14 c8))
(not (moved p c14 c9))
(not (moved p c14 c10))
(not (moved p c14 c11))
(not (moved p c14 c12))
(not (moved p c14 c13))
(not (moved p c14 c14))
(not (moved p c14 c15))
(not (moved p c14 c16))
(not (moved p c14 c17))
(not (moved p c14 c18))
(not (moved p c14 c19))
(not (moved p c15 c0))
(not (moved p c15 c1))
(not (moved p c15 c2))
(not (moved p c15 c3))
(not (moved p c15 c4))
(not (moved p c15 c5))
(not (moved p c15 c6))
(not (moved p c15 c7))
(not (moved p c15 c8))
(not (moved p c15 c9))
(not (moved p c15 c10))
(not (moved p c15 c11))
(not (moved p c15 c12))
(not (moved p c15 c13))
(not (moved p c15 c14))
(not (moved p c15 c15))
(not (moved p c15 c16))
(not (moved p c15 c17))
(not (moved p c15 c18))
(not (moved p c15 c19))
(not (moved p c16 c0))
(not (moved p c16 c1))
(not (moved p c16 c2))
(not (moved p c16 c3))
(not (moved p c16 c4))
(not (moved p c16 c5))
(not (moved p c16 c6))
(not (moved p c16 c7))
(not (moved p c16 c8))
(not (moved p c16 c9))
(not (moved p c16 c10))
(not (moved p c16 c11))
(not (moved p c16 c12))
(not (moved p c16 c13))
(not (moved p c16 c14))
(not (moved p c16 c15))
(not (moved p c16 c16))
(not (moved p c16 c17))
(not (moved p c16 c18))
(not (moved p c16 c19))
(not (moved p c17 c0))
(not (moved p c17 c1))
(not (moved p c17 c2))
(not (moved p c17 c3))
(not (moved p c17 c4))
(not (moved p c17 c5))
(not (moved p c17 c6))
(not (moved p c17 c7))
(not (moved p c17 c8))
(not (moved p c17 c9))
(not (moved p c17 c10))
(not (moved p c17 c11))
(not (moved p c17 c12))
(not (moved p c17 c13))
(not (moved p c17 c14))
(not (moved p c17 c15))
(not (moved p c17 c16))
(not (moved p c17 c17))
(not (moved p c17 c18))
(not (moved p c17 c19))
(not (moved p c18 c0))
(not (moved p c18 c1))
(not (moved p c18 c2))
(not (moved p c18 c3))
(not (moved p c18 c4))
(not (moved p c18 c5))
(not (moved p c18 c6))
(not (moved p c18 c7))
(not (moved p c18 c8))
(not (moved p c18 c9))
(not (moved p c18 c10))
(not (moved p c18 c11))
(not (moved p c18 c12))
(not (moved p c18 c13))
(not (moved p c18 c14))
(not (moved p c18 c15))
(not (moved p c18 c16))
(not (moved p c18 c17))
(not (moved p c18 c18))
(not (moved p c18 c19))
(not (moved p c19 c0))
(not (moved p c19 c1))
(not (moved p c19 c2))
(not (moved p c19 c3))
(not (moved p c19 c4))
(not (moved p c19 c5))
(not (moved p c19 c6))
(not (moved p c19 c7))
(not (moved p c19 c8))
(not (moved p c19 c9))
(not (moved p c19 c10))
(not (moved p c19 c11))
(not (moved p c19 c12))
(not (moved p c19 c13))
(not (moved p c19 c14))
(not (moved p c19 c15))
(not (moved p c19 c16))
(not (moved p c19 c17))
(not (moved p c19 c18))
(not (moved p c19 c19))



        {items}

        {cleared}

        {faced}
    )

    (:goal (and
            $goal$
        )
    )

    ;un-comment the following line if metric is needed
    (:metric minimize (cost))
    )
    """
