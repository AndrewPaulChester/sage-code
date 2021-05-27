
from domains.gym_craft.simulator.craft_world import CraftWorldSimulator
from domains.gym_craft.utils.representations import json_to_symbolic,json_to_screen,json_to_mixed
from domains.gym_craft.envs.teleport_env import TeleportEnv

env = TeleportEnv()

env.reset()
env.act([5,8])
env.act([-3,4])