import domains.gym_taxi
from domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
import domains.gym_taxi.utils.representations as convert
from domains.gym_taxi.utils.config import MULTI, ORIGINAL, FUEL, PREDICTABLE
import numpy as np

from matplotlib import pyplot as plt


def save_image(json, i):

    img = convert.json_to_image(json)
    output = img[0:3].copy()
    output[0:3] = output[0] * 255
    output[0] = output[0] - img[2] * 255 - img[3] * 255
    output[1] = output[1] - img[1] * 255 - img[3] * 255
    output[2] = output[2] - img[1] * 255 - img[2] * 255

    img = np.transpose(output, (1, 2, 0))
    plt.ion()

    fig, ax = plt.subplots()
    im = ax.imshow(img)
    # plt.axis("off")
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

    # plt.show(block=True)

    fig.savefig(f"taxiworld{i}.pdf", format="pdf", bbox_inches="tight")


for s in range(30):
    rng = np.random.RandomState()
    rng.seed(s)

    world = TaxiWorldSimulator(rng, **PREDICTABLE)

    save_image(world._get_state_json(), s)
# img = convert.json_to_image(js)
# screen = convert.json_to_screen(js)
# # one_hot = convert.json_to_one_hot(js)
# mixed = convert.json_to_mixed(js)
# # from_env = convert.env_to_pddl(world)
# from_json = convert.json_to_pddl(js)
