import domains.gym_taxi
from domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
import domains.gym_taxi.utils.representations as convert
from domains.gym_taxi.utils.config import MULTI, ORIGINAL, FUEL
import numpy as np

rng = np.random.RandomState()
rng.seed(0)

world = TaxiWorldSimulator(rng, **ORIGINAL)

js = world._get_state_json()
img = convert.json_to_image(js)
screen = convert.json_to_screen(js)
# one_hot = convert.json_to_one_hot(js)
mixed = convert.json_to_mixed(js)
# from_env = convert.env_to_pddl(world)
from_json = convert.json_to_pddl(js)
# from_image = convert.image_to_pddl(img)
# img_to_json = convert.image_to_json(img)


assert js == img_to_json, "json and img_2_json conflict"

assert from_env == from_json, "from_env and from_json conflict"
assert from_env == from_image, "from_env and from_img conflict"
assert from_json == from_image, "from_json and from_img conflict"


# world.taxi.has_passenger = 1
# world.passenger.in_taxi = True

# js = world._get_state_json()
# img = convert.json_to_image(js)

# from_env = convert.env_to_pddl(world)
# from_json = convert.json_to_pddl(js)
# from_image = convert.image_to_pddl(img)
# img_to_json = convert.image_to_json(img)

# from_json2 = convert.json_to_pddl(img_to_json)

# assert js == img_to_json, "json and img_2_json conflict"

# assert from_env == from_json, "from_env and from_json conflict"
# assert from_env == from_image, "from_env and from_img conflict"
# assert from_json == from_image, "from_json and from_img conflict"
# assert from_json == from_json2, "from_json and from_json2 conflict"
