from domains.gym_craft.simulator.craft_world import CraftWorldSimulator
from domains.gym_craft.utils.representations import (
    json_to_symbolic,
    json_to_screen,
    json_to_mixed,
    json_to_pddl,
    json_to_centered,
)
from domains.gym_craft.utils.utils import blocking_terrain

from domains.gym_craft.envs.rooms_env import TrainRoomsEnv

env = TrainRoomsEnv("zoomed", "rooms", "full")

env.reset()
block_mask = env.sim.terrain.block_mask
terrain = env.sim.terrain.terrain
blocking = blocking_terrain(terrain, block_mask)

while True:
    env.reset()
    env.render()
    env.sim.terrain.random_clear(env.sim.gamedata.tiles)
    env.render()
    env.sim.terrain.random_clear(env.sim.gamedata.tiles)
    env.render()
    env.sim.terrain.random_clear(env.sim.gamedata.tiles)
    env.render()
    env.sim.terrain.random_clear(env.sim.gamedata.tiles)
    env.render()


# blocking[30, 26] = False

# env.sim.player.speed = 0.856788
# env.sim.player.position = (29.044, 25.405)
# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)
# env.sim.player.update_coords(blocking)

# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)
# env.sim.player.update_coords(blocking)

# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)

# env.sim.player.update_coords(blocking)

# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)

# env.sim.player.update_coords(blocking)

# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)

# env.sim.player.update_coords(blocking)

# print(env.sim.player.position)
# print(env.sim.player.speed)
# print(env.sim.player.bearing)

