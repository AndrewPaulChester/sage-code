import numpy as np
from matplotlib import pyplot as plt
import math
MAX_SPEED = 2
ACCELERATION = 0.5
DRAG = 0.3
TURN_SPEED=5

IMAGE = np.array([
[0,0,0,1,0,0,0],
[0,0,1,1,1,0,0],
[0,1,1,1,1,1,0],
[1,1,1,1,1,1,1],
[0,1,1,1,1,1,0],
[0,0,1,1,1,0,0],
[0,0,0,1,0,0,0]])

def main():
    position=(42 ,42)
    speed=0
    bearing=0
    acc=0
    turn=0

    plt.ion()
    fig, ax = plt.subplots()
    img = np.zeros((420,420))
    img[207:214,207:214]=IMAGE
    im = ax.imshow(img)

    for i in range(1000):
        acc+=np.random.rand()-0.5
        turn+=np.random.rand()-0.5
        acc=np.clip(acc,-1,1)
        turn=np.clip(turn,-1,1)
        (position,bearing,speed) = update_coords(position,bearing,speed,acc,turn)
        print(acc,turn)
        print(position)
        render(ax,im,position,bearing,speed)

def update_coords(position,bearing,speed,acceleration,turning):
    (x_pos,y_pos) = position
    speed = update_speed(speed,acceleration)
    bearing = (bearing + TURN_SPEED*turning) % 360
    x_pos += speed * math.sin(bearing*2*math.pi/360)
    y_pos += speed * math.cos(bearing*2*math.pi/360)
    return ((x_pos,y_pos),bearing,speed)

def update_speed(speed,acceleration):
    speed *= DRAG
    speed += acceleration*ACCELERATION
    speed = min(speed,MAX_SPEED) if speed > 0 else max(speed,-MAX_SPEED)
    return speed


def render(ax,im,position,bearing,speed):

    x_pos,y_pos = position
    img = np.zeros((420,420))
    x = int(x_pos*5)
    y = int(y_pos*5)
    img[x:x+7,y:y+7]=IMAGE
    
    # plt.scatter(x,y)
    # plt.show()
    im.set_data(img)
    ax.set_title(f"bearing : {bearing}, speed: {speed}")

    plt.pause(0.001)
    plt.draw()



if __name__ == "__main__":
    main()