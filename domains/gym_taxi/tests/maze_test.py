from domains.gym_taxi.utils.config import maze
import numpy as np

for i in range(1000):
    m = maze(19,19,0.1,0.1)
    for ((x, y), v) in np.ndenumerate(m):
            if v:
                if x % 2 != 0 and y % 2 != 0:
                    print(f"failed round {i}: ({x},{y})")
    print(f"finished round {i}")