

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


from math import floor, ceil
import json
import cv2
import numpy as np
from domains.gym_taxi.utils.utils import json_default
from domains.gym_taxi.utils.config import LOCS, PREDICTABLE5


def env_to_json(env):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return json.dumps(
        {
            "n": env.size,
            "network": {
                "nodes": sorted(list(env.road_network.network.nodes)),
                "edges": sorted(list(env.road_network.network.edges)),
            },
            "taxi": env.taxi.__dict__,
            "passenger": [p.__dict__ for p in env.passengers],
            "fuel_stations": [f.__dict__ for f in env.fuel_stations],
            "concurrent_passengers": env.concurrent_passengers,
        },
        default=json_default,
    )


def json_to_discrete(js):
    """
    Converts taxi world state from json to discrete representation

    :param js: taxi world state in json format
    :return: taxi world state in discrete format
    """
    env = json.loads(js)
    taxi_col, taxi_row = env["taxi"]["location"]
    if len(env["passenger"])==0:
        return 500+taxi_row*5+taxi_col
    else:
        pass_loc = (
            4
            if env["passenger"][0]["in_taxi"]
            else LOCS.index(tuple(env["passenger"][0]["location"]))
        )
        dest_idx = LOCS.index(tuple(env["passenger"][0]["destination"]))
        return _encode(taxi_row, taxi_col, pass_loc, dest_idx)

def json_to_discrete_predictable(js):
    """
    Converts taxi world state from json to discrete representation

    :param js: taxi world state in json format
    :return: taxi world state in discrete format
    """
    env = json.loads(js)
    taxi_col, taxi_row = env["taxi"]["location"]
    if len(env["passenger"])==0:
        return 500+taxi_row*5+taxi_col
    else:
        pass_loc = (
            4
            if env["passenger"][0]["in_taxi"]
            else PREDICTABLE5['passenger_locations'].index(tuple(env["passenger"][0]["location"]))
        )
        dest_idx = PREDICTABLE5['passenger_destinations'].index(tuple(env["passenger"][0]["destination"]))
        return _encode(taxi_row, taxi_col, pass_loc, dest_idx)

def json_to_image(js):
    """
    Converts taxi world state from json to image representation

    :param js: taxi world state in json format
    :return: taxi world state in image format
    """
    env = json.loads(js)
    channels = 5 if len(env["fuel_stations"]) > 0 else 4
    image_size = 2 * env["n"] - 1
    image = np.zeros((channels, image_size, image_size), dtype=np.uint8)
    fill_map(env, image[0], image_size)

    image[1][tuple(2 * i for i in env["taxi"]["location"])] = 1
    for p in env["passenger"]:
        if not p["in_taxi"]:
            image[2][tuple(2 * i for i in p["location"])] = 1
        image[3][tuple(2 * i for i in p["destination"])] = 1
    for f in env["fuel_stations"]:
        image[4][tuple(2 * i for i in f["location"])] = f["price"]

    return np.transpose(image, (0, 2, 1))  # swapping x&y


def fill_map(env, image, image_size):
    nodes = [(2 * x, 2 * y) for (x, y) in env["network"]["nodes"]]
    edges = [(x1 + x2, y1 + y2) for ((x1, y1), (x2, y2)) in env["network"]["edges"]]
    for n in nodes:
        image[n] = 1
    for n in edges:
        image[n] = 1
    # for the odd,odd coordinates, need to interpolate values.
    # Will be passable if at least 3 of it's neighbours are passable
    for x in range(1, image_size, 2):
        for y in range(1, image_size, 2):
            passable_neighbours = (
                image[x - 1][y] + image[x + 1][y] + image[x][y - 1] + image[x][y + 1]
            )
            image[x][y] = 1 if passable_neighbours > 2 else 0


def json_to_mixed(js):
    """
    Converts taxi world state from json to mixed representation, which is part image, part dense input

    :param js: taxi world state in json format
    :return: taxi world state in mixed format
    """
    env = json.loads(js)
    # channels = 5 if len(env["fuel_stations"]) > 0 else 4
    n = env["n"]
    image_size = 2 * n - 1
    image = np.zeros((2, image_size, image_size), dtype=np.uint8)
    fill_map(env, image[0], image_size)

    image[1][tuple(2 * i for i in env["taxi"]["location"])] = 1

    # assert (
    #     len(env["passenger"]) < 2
    # ), "json_to_mixed does not currently support multiple passengers"
    dense = get_dense(env, n)[2:] #removing taxi location from dense input
    return (
        resize_image(np.transpose(image, (0, 2, 1)), 84),
        np.array(dense, dtype=np.float32),
    )


def json_to_one_hot(js):
    """
    Converts taxi world state from json to onehot representation, which is part image, part onehot dense input

    :param js: taxi world state in json format
    :return: taxi world state in onehot format
    """
    image = json_to_image(js)

    evens = np.arange(0, image.shape[1], 2)
    return (
        resize_image(np.expand_dims(image[0], 0), 84),
        image[np.ix_(np.arange(1, 4), evens, evens)].flatten(),
    )


def get_dense(env, n):

    dense = [i / n for i in env["taxi"]["location"]]

    for f in env["fuel_stations"]:
        dense += [i / n for i in f["location"]]
        dense += [f["price"]]

    for p in env["passenger"]:
        if p["in_taxi"]:
            in_taxi = 1
            location = [i / n for i in env["taxi"]["location"]]
        else:
            in_taxi = 0
            location = [i / n for i in p["location"]]

        destination = [i / n for i in p["destination"]]
        passenger = [1, in_taxi] + location + destination
        dense += passenger

    dense += [0] * 6 * (env["concurrent_passengers"] - len(env["passenger"]))
    return [
        d * 4 for d in dense
    ]  # this scales it to ~[0-4] to be more in line with the magnitudes of the output of the CNN, fairly hackity hack!


def json_to_mlp(js):
    """
    Converts taxi world state from json to mlp representation, which is just dense input

    :param js: taxi world state in json format
    :return: taxi world state in mlp format
    """
    env = json.loads(js)
    # channels = 5 if len(env["fuel_stations"]) > 0 else 4
    n = env["n"]
    return get_dense(env, n)


def json_to_both(js):
    """
    Converts taxi world state from json to both representation, which is all image, and dense input

    :param js: taxi world state in json format
    :return: taxi world state in both format
    """

    return (json_to_screen(js), json_to_mlp(js))


def json_to_pddl(js, representation="simple"):
    """
    Converts taxi world state from json to pddl representation

    :param js: taxi world state in json format
    :return: taxi world state in pddl format
    """
    env = json.loads(js)
    if representation == "simple":
        nodes = "$nodes$"
    else:
        nodes = " ".join(sorted([f"sx{x}y{y}" for (x, y) in env["network"]["nodes"]]))

    if representation == "simple":
        edges = ""
    else:
        edges = "\n".join(
            sorted(
                [
                    _edge_direction(x1, x2, y1, y2)
                    for ((x1, y1), (x2, y2)) in env["network"]["edges"]
                ]
            )
        )

    if env["taxi"]["passenger"] is None:
        empty = "(empty t)"
    else:
        empty = f"(carrying-passenger t p{env['taxi']['passenger']})"

    x, y = env["taxi"]["location"]
    taxi_location = f"(in t sx{x}y{y})"

    passengers = " ".join([f"p{p['pid']}" for p in env["passenger"]])

    passenger_location = "\n".join(
        [
            f"(in p{p['pid']} sx{p['location'][0]}y{p['location'][1]})"
            for p in env["passenger"]
            if not p["in_taxi"]
        ]
    )
    passenger_destination = "\n".join(
        [
            f"(destination p{p['pid']} sx{p['destination'][0]}y{p['destination'][1]})"
            for p in env["passenger"]
        ]
    )

    fuel_stations = " ".join([f"f{f['fid']}" for f in env["fuel_stations"]])
    fuel_location = "\n".join(
        [
            f"(in f{f['fid']} sx{f['location'][0]}y{f['location'][1]}) "
            for f in env["fuel_stations"]
        ]
    )

    # fuel = (
    #     "(full-tank t)"
    #     if env["taxi"]["fuel"] == env["taxi"]["max_fuel"]
    #     else "(not (full-tank t))"
    # )
    # money = "(has-money t)" if env["taxi"]["money"] > 0 else "(not (has-money t))"
    # price = ""
    # numeric variables
    fuel = f"(= (fuel t) {env['taxi']['fuel']})"
    money = f"(= (money t) {env['taxi']['money']})"
    price = "\n".join(
        [f"(= (price f{f['fid']}) {f['price']} ) " for f in env["fuel_stations"]]
    )

    return _compose_pddl(
        env["n"],
        nodes,
        edges,
        empty,
        taxi_location,
        passengers,
        passenger_location,
        passenger_destination,
        fuel,
        money,
        fuel_stations,
        fuel_location,
        price,
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


def _encode(taxi_row, taxi_col, pass_loc, dest_idx):
    """
    from https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    """
    # (5) 5, 5, 4
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i


def _edge_direction(x1, x2, y1, y2):
    # if nodes are the wrong way round, swap them
    if x1 == x2 and y1 == y2 + 1:
        y2, y1 = y1, y2  # as we don't have a below predicate
    elif y1 == y2 and x1 == x2 + 1:
        x2, x1 = x1, x2  # as we don't have a right predicate

    # now the nodes should be in the right order
    if x1 == x2 and y1 == y2 - 1:
        return f"(above sx{x1}y{y1} sx{x2}y{y2})"
    elif y1 == y2 and x1 == x2 - 1:
        return f"(left sx{x1}y{y1} sx{x2}y{y2})"
    else:
        raise ValueError("coordinates not adjacent")


def _get_coord_from_np_array(array):
    return tuple((np.transpose((array == 1).nonzero())[0] / 2).astype(int))


def _compose_pddl(
    n,
    nodes,
    edges,
    empty,
    taxi_location,
    passengers,
    passenger_location,
    passenger_destination,
    fuel,
    money,
    fuel_stations,
    fuel_location,
    price,
):
    return f"""
    (define (problem problem{n}p) (:domain navigation)
    (:objects 
        {nodes} - space
        t - taxi
        {passengers} - passenger
        {fuel_stations} - fuelstation
    )

    (:init
        ;todo: put the initial state's facts and numeric values here
        {edges}
        {empty}
        {taxi_location}
        {fuel}
        {money}
        {passenger_location}
        {passenger_destination}
        {fuel_location}
        {price}
        ;(= (capacity t) 100)
    )

    (:goal (and
            $goal$
        )
    )

    ;un-comment the following line if metric is needed
    ;(:metric minimize (???))
    )
    """


def image_to_json(image):
    """
    Converts taxi world state from image to json representation
    DEPRECATED - Should rely on json that is returned from the environment directly

    :param image: taxi world state in image format
    :return: taxi world state in json format
    """
    image = np.transpose(image, (0, 2, 1))  # swap x & y
    node_list = []
    edge_list = []
    for ((x, y), v) in np.ndenumerate(image[0]):
        if v == 1:
            if x % 2 == 0 and y % 2 == 0:
                node_list.append((int(x / 2), int(y / 2)))
            elif x % 2 == 0 or y % 2 == 0:
                edge_list.append(
                    (
                        (int(floor(x / 2)), int(floor(y / 2))),
                        (int(ceil(x / 2)), int(ceil(y / 2))),
                    )
                )
    network = {"nodes": sorted(node_list), "edges": sorted(edge_list)}

    taxi = {}
    passenger = {}

    taxi["location"] = _get_coord_from_np_array(image[1])

    passenger["destination"] = _get_coord_from_np_array(image[3])

    if len(np.transpose((image[2] == 1).nonzero())) > 0:  # passenger exists
        passenger["location"] = _get_coord_from_np_array(image[2])
        taxi["has_passenger"] = 0
        passenger["in_taxi"] = False
    else:
        passenger["in_taxi"] = True
        taxi["has_passenger"] = 1

    n = ceil(len(image[0]) / 2)

    return json.dumps(
        {"n": n, "network": network, "taxi": taxi, "passenger": passenger},
        default=json_default,
    )


def image_to_pddl(image):
    """
    Converts taxi world state from image to pddl representation. 
    DEPRECATED - should use image to json and json to pddl

    :param image: taxi world state in image format
    :return: taxi world state in pddl format
    """
    image = np.transpose(image, (0, 2, 1))  # swap x & y
    node_list = []
    edge_list = []
    for ((x, y), v) in np.ndenumerate(image[0]):
        if v == 1:
            if x % 2 == 0 and y % 2 == 0:
                node_list.append((int(x / 2), int(y / 2)))
            else:
                edge_list.append(
                    (
                        (int(floor(x / 2)), int(floor(y / 2))),
                        (int(ceil(x / 2)), int(ceil(y / 2))),
                    )
                )

    nodes = " ".join(sorted([f"sx{x}y{y}" for (x, y) in node_list]))

    edges = "\n".join(
        sorted([_edge_direction(x1, x2, y1, y2) for ((x1, y1), (x2, y2)) in edge_list])
    )

    empty = "(carrying-passenger t p)"
    passenger_location = ""

    x, y = _get_coord_from_np_array(image[1])
    taxi_location = f"(in t sx{x}y{y})"

    if len(np.transpose((image[2] == 1).nonzero())) > 0:  # passenger exists
        x, y = _get_coord_from_np_array(image[2])
        passenger_location = f"(in p sx{x}y{y})"
        empty = f"(empty t)"

    x, y = _get_coord_from_np_array(image[3])
    passenger_destination = f"(destination p sx{x}y{y})"

    n = ceil(len(image[0]) / 2)

    return _compose_pddl(
        n, nodes, edges, empty, taxi_location, passenger_location, passenger_destination
    )


def env_to_pddl(env):
    """
    Converts taxi world state from env to pddl representation. 
    DEPRECATED - should use env to json and json to pddl

    :param env: taxi world state in env format
    :return: taxi world state in pddl format
    """
    road_network = env.road_network.network
    nodes = " ".join(sorted([f"sx{x}y{y}" for (x, y) in road_network.nodes]))

    edges = "\n".join(
        sorted(
            [
                _edge_direction(x1, x2, y1, y2)
                for ((x1, y1), (x2, y2)) in road_network.edges
            ]
        )
    )
    if env.taxi.has_passenger == 0:
        empty = "(empty t)"
        x, y = env.passenger.location
        passenger_location = f"(in p sx{x}y{y})"
    else:
        empty = "(carrying-passenger t p)"
        passenger_location = ""

    x, y = env.taxi.location
    taxi_location = f"(in t sx{x}y{y})"

    x, y = env.passenger.destination
    passenger_destination = f"(destination p sx{x}y{y})"

    return _compose_pddl(
        env.size,
        nodes,
        edges,
        empty,
        taxi_location,
        passenger_location,
        passenger_destination,
    )

