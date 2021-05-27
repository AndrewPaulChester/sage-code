"""
.. module:: config
   :synopsis: Contains config parameters for the craft world.
"""
from enum import Enum


# TODO: consider if agent should be able to face diagonals
DIRECTIONS = Enum("Directions", "north south east west")

OFFSETS = {
    DIRECTIONS.north: (0, -1),
    DIRECTIONS.south: (0, 1),
    DIRECTIONS.west: (-1, 0),
    DIRECTIONS.east: (1, 0),
}


MAX_EPISODE_LENGTH = 1000
LARGE_GRID_SIZE = 42
MEDIUM_GRID_SIZE = 31
FIXED_GRID_SIZE = 21
SMALL_GRID_SIZE = 11


ROOM_TYPES = {
    "empty": {},
    "treasure": {"coin": 20},
    "change": {"coin": 2},
    "coins": {"coin": 9},
    "forest": {"tree": 10},
    "mountain": {"rock": 10},
    "mixed": {"tree": 5, "rock": 5, "coin": 5},
    "jungle": {"tree": 50, "coin": 5},
    "poor-mixed": {"tree": 5, "rock": 5, "coin": 1},
    "poor-jungle": {"tree": 50, "coin": 2},
    "castle": {"rock": 50, "coin": 5},
}

DOORS = [None, "tree", "rock", "wall"]
DOOR_PROBABILITIES = [0.1, 0.4, 0.4, 0.1]

ROOMS = {
    "size": MEDIUM_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "rooms",
    "random_rooms": {
        "empty": 0,
        "treasure": 2,
        "forest": 0,
        "mountain": 0,
        "mixed": 3,
        "jungle": 2,
        "castle": 2,
    },
    "starting_zone": ((1, 1), (9, 9)),
}

POOR_ROOMS = {
    "size": MEDIUM_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "rooms",
    "random_rooms": {
        "empty": 1,
        "treasure": 1,
        "coins": 1,
        "change": 1,
        "forest": 0,
        "mountain": 0,
        "mixed": 1,
        "poor-mixed": 2,
        "jungle": 1,
        "castle": 1,
    },
    "starting_zone": ((1, 1), (9, 9)),
}
COIN_ROOMS = {
    "size": MEDIUM_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "rooms",
    "random_rooms": {"coins": 9},
    "starting_zone": ((1, 1), (9, 9)),
}

EASY_ROOMS = {
    "size": MEDIUM_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "rooms",
    "random_rooms": {"coins": 6, "mixed": 3},
    "starting_zone": ((1, 1), (9, 9)),
}


ORIGINAL = {"size": FIXED_GRID_SIZE, "timeout": MAX_EPISODE_LENGTH, "walls": "rooms"}

ROOM_FREE = {
    "size": FIXED_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [((1, 1), (19, 19), {"rock": 20, "tree": 24, "coin": 2})],
}

RANDOM_RESOURCES = {
    "size": FIXED_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "rooms",
    "random_resources": [
        ((1, 1), (9, 9), {"rock": 3, "tree": 7, "coin": 2}),
        ((1, 11), (9, 19), {"rock": 6, "tree": 3, "coin": 4}),
        ((11, 1), (19, 9), {"rock": 2, "tree": 5, "coin": 6}),
        ((11, 11), (19, 19), {"rock": 1, "tree": 2, "coin": 8}),
    ],
}

SMALL_RANDOM = {
    "size": SMALL_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [
        ((4, 1), (4, 9), {"tree": 9}),
        ((7, 1), (7, 9), {"rock": 9}),
        ((1, 1), (3, 9), {"rock": 2, "tree": 3, "coin": 2}),
        ((5, 1), (6, 9), {"rock": 1, "tree": 2, "coin": 4}),
        ((8, 1), (9, 9), {"coin": 6}),
    ],
}

SMALL_STATIC = {
    "size": SMALL_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [
        ((4, 1), (4, 9), {"tree": 9}),
        ((7, 1), (7, 9), {"rock": 9}),
        ((1, 6), (2, 6), {"tree": 2}),
        ((2, 4), (2, 4), {"tree": 1}),
        ((3, 8), (3, 8), {"coin": 1}),
        ((1, 4), (1, 4), {"coin": 1}),
        ((2, 2), (2, 2), {"rock": 1}),
        ((3, 6), (3, 6), {"rock": 1}),
        ((6, 2), (6, 3), {"tree": 2}),
        ((5, 1), (5, 1), {"coin": 1}),
        ((5, 8), (5, 8), {"coin": 1}),
        ((6, 6), (6, 7), {"coin": 2}),
        ((5, 2), (5, 2), {"rock": 1}),
        ((8, 4), (9, 6), {"coin": 6}),
    ],
}

SPARSE = {
    "size": SMALL_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [((0, 0), (0, 0), {"coin": 1})],
}

COIN = {
    "size": SMALL_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [((1, 1), (9, 9), {"coin": 9})],
}

SINGLE_COIN = {
    "size": SMALL_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [((1, 1), (9, 9), {"coin": 1})],
}
LARGE_COIN = {
    "size": LARGE_GRID_SIZE,
    "timeout": MAX_EPISODE_LENGTH,
    "walls": "one room",
    "random_resources": [((1, 1), (40, 40), {"coin": 1})],
    "starting_zone": ((1, 1), (40, 40)),
}
