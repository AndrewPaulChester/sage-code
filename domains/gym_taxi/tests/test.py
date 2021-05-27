import domains.gym_taxi
from domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
import domains.gym_taxi.utils.representations as convert
import cv2
from matplotlib import pyplot as plt
import numpy as np


# test small -> big -> small conversion fidelity
img2 = np.random.randint(0, 2, [4, 39, 39], dtype=np.uint8)
for i in range(1000):
    img = np.random.randint(0, 2, [4, 39, 39], dtype=np.uint8)
    big = np.transpose(
        cv2.resize(
            np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_AREA
        ),
        (2, 0, 1),
    )
    small = np.transpose(
        cv2.resize(
            np.transpose(big, (1, 2, 0)), (39, 39), interpolation=cv2.INTER_AREA
        ),
        (2, 0, 1),
    )
    assert (img == small).all()
    # assert (img == img2).all()  # should fail
    img2 = img
    print(i)


world = TaxiWorldSimulator(5)

js = world._get_state_json()
img = convert.json_to_image(js)

fig = plt.figure(figsize=(8, 8))
for i in range(4):
    fig.add_subplot(1, 4, i + 1)
    plt.imshow(img[i])
plt.show()

from_env = convert.env_to_pddl(world)
from_json = convert.json_to_pddl(js)
from_image = convert.image_to_pddl(img)
img_to_json = convert.image_to_json(img)


assert js == img_to_json, "json and img_2_json conflict"

assert from_env == from_json, "from_env and from_json conflict"
assert from_env == from_image, "from_env and from_img conflict"
assert from_json == from_image, "from_json and from_img conflict"


world.taxi.has_passenger = 1
world.passenger.in_taxi = True

js = world._get_state_json()
img = convert.json_to_image(js)

from_env = convert.env_to_pddl(world)
from_json = convert.json_to_pddl(js)
from_image = convert.image_to_pddl(img)
img_to_json = convert.image_to_json(img)

from_json2 = convert.json_to_pddl(img_to_json)

assert js == img_to_json, "json and img_2_json conflict"

assert from_env == from_json, "from_env and from_json conflict"
assert from_env == from_image, "from_env and from_img conflict"
assert from_json == from_image, "from_json and from_img conflict"
assert from_json == from_json2, "from_json and from_json2 conflict"

big1 = np.transpose(
    cv2.resize(np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_AREA),
    (2, 0, 1),
)  # this is good
big3 = np.transpose(
    cv2.resize(np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_NEAREST),
    (2, 0, 1),
)  # so is this
big2 = np.transpose(
    cv2.resize(np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_BITS),
    (2, 0, 1),
)
big4 = np.transpose(
    cv2.resize(np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_LINEAR),
    (2, 0, 1),
)
big5 = np.transpose(
    cv2.resize(
        np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_LINEAR_EXACT
    ),
    (2, 0, 1),
)
big6 = np.transpose(
    cv2.resize(np.transpose(img, (1, 2, 0)), (84, 84), interpolation=cv2.INTER_CUBIC),
    (2, 0, 1),
)


fig = plt.figure(figsize=(16, 16))
for i in range(4):
    fig.add_subplot(7, 4, i + 1)
    plt.imshow(img[i])
    fig.add_subplot(7, 4, i + 5)
    plt.imshow(big1[i])
    fig.add_subplot(7, 4, i + 9)
    plt.imshow(big2[i])
    fig.add_subplot(7, 4, i + 13)
    plt.imshow(big3[i])
    fig.add_subplot(7, 4, i + 17)
    plt.imshow(big4[i])
    fig.add_subplot(7, 4, i + 21)
    plt.imshow(big5[i])
    fig.add_subplot(7, 4, i + 25)
    plt.imshow(big6[i])
plt.show()

