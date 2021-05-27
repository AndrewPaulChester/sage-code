from domains.gym_craft.envs.craft_env import JsonCraftEnv
render = True
#actions = ([5]*10 + [9])*100
actions = (([5]*10 + [9])*20 + [10] + [11] + [12])*5
env = JsonCraftEnv("screen","original")
obs = env.reset()
for a in actions:
    env.step(a)
    if render:
        env.render()


