import numpy as np

from matplotlib import pyplot as plt


terrain = np.array([[0,0,0],[0,1,1],[0,1,2]])


colours = np.array([[0,0,255],[255,0,0],[0,255,0]])


image = colours[terrain]


plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(image)
plt.draw()
plt.pause(10)
plt.draw()
