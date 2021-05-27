"""
.. module:: taxi_world
   :synopsis: Simulates the taxi world environment based on actions passed in.
"""

from enum import Enum
import argparse
import numpy as np
import networkx as nx
from domains.gym_taxi.utils.representations import env_to_json
from domains.gym_taxi.utils.config import MAX_EPISODE_LENGTH
from domains.gym_taxi.utils.utils import generate_random_walls


ACTIONS = Enum("Actions", "north south east west pickup dropoff refuel noop")

DIRECTIONS = {
    ACTIONS.north: (0, -1),
    ACTIONS.south: (0, 1),
    ACTIONS.west: (-1, 0),
    ACTIONS.east: (1, 0),
}

DEFAULT_REWARDS = {"base": -1, "failed-action": -10, "drop-off": 20}


def _get_destination(location, direction):
    """
    Returns the intended destination for the movement, does not check if road exists.
    :param location: current location of taxi
    :param direction: direction to move as ACTIONS enum
    :return: intended location after movement
    """
    return tuple([sum(x) for x in zip(location, DIRECTIONS[direction])])


class TaxiWorldSimulator(object):
    def __init__(
        self,
        random,
        size,
        passenger_locations=[],
        passenger_destinations=[],
        wall_locations=None,
        fuel_station_locations=[],
        delivery_limit=1,
        concurrent_passengers=1,
        timeout=MAX_EPISODE_LENGTH,
        passenger_creation_probability=1,
        fuel_station_creation_probability=0,
        fuel_use=0,
        random_walls=True,
        taxi_locations=None,
        rewards=None,
    ):
        """
        Houses the game state and transition dynamics for the taxi world.

        :param size: size of gridworld
        :returns: this is a description of what is returned
        :raises keyError: raises an exception
        """
        self.random = random
        self.seed_id = hash(self.random)
        self.time = 0
        self.size = size
        self.passenger_locations = passenger_locations
        self.passenger_destinations = passenger_destinations
        self.fuel_station_locations = fuel_station_locations
        self.delivery_limit = delivery_limit
        self.concurrent_passengers = concurrent_passengers
        self.timeout = timeout
        self.passenger_creation_probability = passenger_creation_probability
        self.fuel_station_creation_probability = fuel_station_creation_probability
        self.rewards = rewards if rewards is not None else DEFAULT_REWARDS
        self.done = False

        self.road_network = RoadNetwork(self.random, size, wall_locations, random_walls)
        self.taxi = Taxi(self.random, size, taxi_locations)
        self.passengers = []
        self.passengers.append(
            Passenger(self.random, 0, size, passenger_locations, passenger_destinations)
        )
        self.passenger_id = 1
        self.fuel_use = fuel_use
        self.fuel_stations = []
        if fuel_use > 0 or fuel_station_creation_probability > 0:
            self.fuel_stations.append(
                FuelStation(self.random, 0, size, [(size // 2, size // 2)])
            )
        self.fuel_station_id = 1

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
        param = 0
        assert action in ACTIONS
        if action == ACTIONS.noop:
            reward = self.rewards["base"]
        elif action == ACTIONS.pickup:
            reward = self.attempt_pickup()
        elif action == ACTIONS.dropoff:
            reward = self.attempt_dropoff()
        elif action == ACTIONS.refuel:
            reward = self.attempt_refuel(param)
        else:
            start = self.taxi.location
            destination = _get_destination(start, action)
            if self.road_network.network.has_edge(start, destination):
                self.taxi.location = destination
            reward = self.rewards["base"]
            self.taxi.fuel -= self.fuel_use
        self.try_spawn_passenger()
        # self.try_spawn_fuel_station()
        self.update_fuel_prices()
        if self.delivery_limit == 0:
            self.done = True
        if self.taxi.fuel == 0:
            reward = -100
            self.done = True
        self.time += 1
        return self._get_state_json(), reward, self.done, {}

    def attempt_dropoff(self):
        pid = self.taxi.passenger
        if pid is None:
            return self.rewards["failed-action"]
        passenger = next((p for p in self.passengers if p.pid == pid))
        if self.taxi.location == passenger.destination:
            self.taxi.passenger = None
            self.taxi.money += 200
            self.passengers.remove(passenger)
            self.delivery_limit -= 1
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Delivered passenger.")
            return self.rewards["drop-off"]
        else:
            return self.rewards["failed-action"]

    def attempt_pickup(self):
        if self.taxi.passenger is not None:
            return self.rewards["failed-action"]
        for passenger in self.passengers:
            if self.taxi.location == passenger.location:
                self.taxi.passenger = passenger.pid
                passenger.in_taxi = True
                return self.rewards["base"]
        return self.rewards["failed-action"]

    def attempt_refuel(self, quantity):
        for fuel_station in self.fuel_stations:
            if self.taxi.location == fuel_station.location:
                refuel = min(quantity, self.taxi.max_fuel - self.taxi.fuel)
                if self.taxi.money >= refuel * fuel_station.price:
                    self.taxi.fuel += refuel
                    self.taxi.money -= refuel * fuel_station.price
                    return self.rewards["base"]
        return self.rewards["failed-action"]

    def try_spawn_passenger(self):
        """
        Spawns a passenger
        :param a: 
        :returns: 
        """
        if (
            len(self.passengers) < self.concurrent_passengers
            and self.random.uniform() < self.passenger_creation_probability
        ):
            self.passengers.append(
                Passenger(
                    self.random,
                    self.passenger_id,
                    self.size,
                    self.passenger_locations,
                    self.passenger_destinations,
                )
            )
            self.passenger_id += 1
            # print(f"seed-id: {self.seed_id}. Time: {self.time} Spawned passenger.")

    def try_spawn_fuel_station(self):
        """
        Spawns a fuel station
        """
        if self.random.uniform() < self.fuel_station_creation_probability:
            self.fuel_stations.append(
                FuelStation(
                    self.random,
                    self.fuel_station_id,
                    self.size,
                    self.fuel_station_locations,
                )
            )
            self.fuel_station_id += 1

    def update_fuel_prices(self):
        for f in self.fuel_stations:
            f.randomise_price(self.random)


class RoadNetwork(object):
    """
    Represents the road network for the current simulation
    """

    def __init__(self, random, grid_size, wall_locations, random_walls):

        self.network = nx.grid_graph([grid_size, grid_size])
        if random_walls:
            # could add warning that wall locations are being overriden
            wall_locations = generate_random_walls(grid_size, random)
        if wall_locations:
            self.network.remove_edges_from(wall_locations)


class FuelStation(object):
    """
    Contains information about the position and price of a fuel station
    """

    def __init__(self, random, fid, grid_size, locations):
        if locations:
            self.location = random.choice(locations)
        else:
            self.location = tuple(random.randint(grid_size, size=2))
        self.price = random.randint(10)
        self.fid = fid
        self.price_change_probability = 0.02

    def randomise_price(self, random):
        if random.uniform() < self.price_change_probability:
            self.price = random.randint(10)


class Taxi(object):
    """
    Contains information about the position and state of the taxi
    """

    def __init__(self, random, grid_size, locations):
        if locations:
            self.location = locations[random.randint(len(locations))]
        else:
            self.location = tuple(random.randint(grid_size, size=2))
        self.passenger = None

        self.fuel = 100
        self.max_fuel = 100
        self.money = 0


class Passenger(object):
    """
    Contains all relevant information about the passengers
    """

    def __init__(self, random, pid, grid_size, locations, destinations=None):
        # if destinations are blank, try re-using locations
        # have to copy so that the original lists (even constants!) are not modified for next time.
        self.pid = pid
        destinations = destinations.copy() if destinations else locations.copy()

        if locations:
            self.location = locations[random.randint(len(locations))]
        else:
            self.location = tuple(random.randint(grid_size, size=2))

        try:
            destinations.remove(self.location)
        except ValueError:
            pass

        if destinations:
            self.destination = destinations[random.randint(len(destinations))]
        else:
            self.destination = tuple(random.randint(grid_size, size=2))

        self.in_taxi = False
