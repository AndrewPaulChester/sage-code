from domains.gym_taxi.envs import taxi_env
import time

import numpy as np

# np.random.seed(1)
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))

# np.random.seed(1)
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))

# np.random.seed(2)
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))
# print(np.random.randint(6))


env = taxi_env.JsonTaxiEnv("mixed", "predictable")

# env.reset()

# env.render()
# env.render()
# time.sleep(2)
# env.reset()
env.seed(1)
env.reset()
env.render()
env.render()
time.sleep(20)

