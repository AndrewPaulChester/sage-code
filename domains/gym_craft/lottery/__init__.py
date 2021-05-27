from gym.envs.registration import register

REWARDS = {
    "v1": {"base": -1, "failed-action": -1, "drop-off": 20},
    "v2": {"base": 0, "failed-action": 0, "drop-off": 1},
    "v3": {"base": -0.05, "failed-action": -0.05, "drop-off": 1},
}

register(id="craft-lottery-v0", entry_point="gym_craft.lottery.envs:CraftLotteryEnv")
register(
    id="craft-lottery-v1",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"intermediate": False},
)
register(
    id="craft-lottery-deterministic-v1",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"intermediate": False, "deterministic": True},
)
register(
    id="craft-lottery-deterministic-simple-v1",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"intermediate": False, "deterministic": True, "simple": True},
)


register(id="craft-mole-v0", entry_point="gym_craft.lottery.envs:WhackAMoleEnv")

register(
    id="craft-lottery-linear-v0",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"scaling": "linear"},
)
register(
    id="craft-lottery-linear-v1",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"scaling": "linear", "intermediate": False},
)
register(
    id="craft-lottery-v2",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"experience_interval": 5},
)
register(
    id="craft-lottery-linear-v2",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"scaling": "linear", "experience_interval": 5},
)
register(
    id="craft-lottery-v3",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"experience_interval": 3},
)
register(
    id="craft-lottery-linear-v3",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"scaling": "linear", "experience_interval": 3},
)

register(
    id="craft-lottery-v4",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"experience_interval": 10},
)
register(
    id="craft-lottery-linear-v4",
    entry_point="gym_craft.lottery.envs:CraftLotteryEnv",
    kwargs={"scaling": "linear", "experience_interval": 10},
)
