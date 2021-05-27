# from domains.gym_craft.simulator.craft_world import Terrain
# import numpy as np
# terrain = Terrain(None,8,None,None)
# print(terrain.get_blocking_terrain())
# a = 3
# TILES(np.argmax(terrain.terrain[4,6]))

# terrain[4,6]


# from domains.gym_craft.simulator.game_data import GameData

# gamedata = GameData()

# gamedata.tiles["wall"]["index"]
# gamedata.get_tile(2)


# import yaml


# with open("gym_craft/utils/gamedata.yaml",'r') as f:
#     config = yaml.safe_load(f)


# print(config)


from domains.gym_craft.simulator.craft_world import CraftWorldSimulator
from domains.gym_craft.utils.representations import (
    json_to_symbolic,
    json_to_screen,
    json_to_image,
    json_to_mixed,
    json_to_pddl,
    json_to_centered,
    json_to_abstract,
    json_to_both,
)
from domains.gym_craft.envs.craft_env import JsonCraftEnv


import numpy as np
from matplotlib import pyplot as plt


def save_image(json):

    img = np.transpose(json_to_image(json), (1, 2, 0))
    plt.ion()

    fig, ax = plt.subplots()
    im = ax.imshow(img)
    # fig = plt.figure(figsize=(8, 8))
    # for i in range(4):
    #     fig.add_subplot(1, 4, i + 1)
    #     plt.imshow(img[i])

    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    im.set_data(img)

    plt.pause(0.001)
    plt.draw()

    plt.show(block=True)

    fig.savefig("craftworld.pdf", format="pdf", bbox_inches="tight")


env = JsonCraftEnv("both", "rooms", "full")
# env = JsonCraftEnv("mixed","original",'full')
# env = JsonCraftEnv("centered", "coin_rooms", "full")

env.reset()
# env.sim.setup_training("")
# print(json_to_pddl(env.reset(), "rooms"))
save_image(env.reset())

json_to_both(env.reset())
json_to_abstract(env.reset())
