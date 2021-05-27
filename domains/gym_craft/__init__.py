from gym.envs.registration import register

REWARDS = {
    "v1": {"base": -1, "failed-action": -1, "drop-off": 20},
    "v2": {"base": 0, "failed-action": 0, "drop-off": 1},
    "v3": {"base": -0.05, "failed-action": -0.05, "drop-off": 1},
}

ENVS = {
    "original-craft-": {
        "representation": "screen",
        "scenario": "original",
        "actions": "full",
    },
    "mixed-craft-": {
        "representation": "mixed",
        "scenario": "original",
        "actions": "full",
    },
    "room-free-craft-": {
        "representation": "mixed",
        "scenario": "room_free",
        "actions": "full",
    },
    "random-craft-": {
        "representation": "mixed",
        "scenario": "random_resources",
        "actions": "full",
    },
    "small-static-craft-": {
        "representation": "mixed",
        "scenario": "small_static",
        "actions": "full",
    },
    "small-random-craft-": {
        "representation": "mixed",
        "scenario": "small_random",
        "actions": "full",
    },
    "sparse-craft-": {
        "representation": "mixed",
        "scenario": "sparse",
        "actions": "move-only",
    },
    "coin-craft-": {"representation": "mixed", "scenario": "coin", "actions": "full"},
    "coin-move-craft-": {
        "representation": "mixed",
        "scenario": "coin",
        "actions": "move-only",
    },
    "coin-continuous-craft-": {
        "representation": "mixed",
        "scenario": "coin",
        "actions": "move-continuous",
    },
    "single-coin-craft-": {
        "representation": "mixed",
        "scenario": "single_coin",
        "actions": "full",
    },
    "dense-single-coin-craft-": {
        "representation": "dense",
        "scenario": "single_coin",
        "actions": "full",
    },
    "large-coin-craft-": {
        "representation": "mixed",
        "scenario": "large_coin",
        "actions": "full",
    },
    "uniform-large-coin-craft-": {
        "representation": "mixed",
        "scenario": "large_coin",
        "actions": "move-uniform",
    },
    "continuous-large-coin-craft-": {
        "representation": "mixed",
        "scenario": "large_coin",
        "actions": "move-continuous",
    },
    "continuous-sparse-craft-": {
        "representation": "mixed",
        "scenario": "sparse",
        "actions": "move-continuous",
    },
    "dense-continuous-sparse-craft-": {
        "representation": "dense",
        "scenario": "sparse",
        "actions": "move-continuous",
    },
    "uniform-sparse-craft-": {
        "representation": "mixed",
        "scenario": "sparse",
        "actions": "move-uniform",
    },
    "dense-uniform-sparse-craft-": {
        "representation": "dense",
        "scenario": "sparse",
        "actions": "move-uniform",
    },
}

ROOM_ENVS = {
    "rooms-craft-": {
        "representation": "mixed",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "abstract-rooms-craft-": {
        "representation": "abstract",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "both-rooms-craft-": {
        "representation": "both",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "abstract-fixed-craft-": {
        "representation": "abstract",
        "scenario": "random_resources",
        "actions": "rooms",
    },
}
TRAIN_ENVS = {
    "train-craft-": {
        "representation": "train",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "train-collect-craft-": {
        "representation": "train",
        "scenario": "rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "masked-collect-craft-": {
        "representation": "masked",
        "scenario": "rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-collect-craft-": {
        "representation": "zoomed",
        "scenario": "rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-collect-coin-craft-": {
        "representation": "zoomed",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-collect-easy-craft-": {
        "representation": "zoomed",
        "scenario": "easy_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "collect-coin-craft-": {
        "representation": "train",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "centered-collect-coin-craft-": {
        "representation": "centered",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "centered-binary-collect-coin-craft-": {
        "representation": "centered_binary",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-binary-collect-craft-": {
        "representation": "zoomed_binary",
        "scenario": "rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "centered-binary-collect-craft-": {
        "representation": "centered_binary",
        "scenario": "rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-binary-collect-coin-craft-": {
        "representation": "zoomed_binary",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "zoomed-binary-collect-easy-craft-": {
        "representation": "zoomed_binary",
        "scenario": "easy_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "centered-binary-collect-easy-craft-": {
        "representation": "centered_binary",
        "scenario": "easy_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "binary-collect-easy-craft-": {
        "representation": "binary",
        "scenario": "easy_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "binary-collect-coin-craft-": {
        "representation": "binary",
        "scenario": "coin_rooms",
        "actions": "rooms",
        "steps": ["collect"],
    },
    "centered-binary-train-craft-": {
        "representation": "centered_binary",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "zoomed-binary-train-craft-": {
        "representation": "zoomed_binary",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "zoomed-binary-train-poor-craft-": {
        "representation": "zoomed_binary",
        "scenario": "poor_rooms",
        "actions": "rooms",
    },
}

HIERARCHICAL_ENVS = {
    "hierarchical-rooms-craft-": {
        "representation": "mixed",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "hierarchical-abstract-rooms-craft-": {
        "representation": "abstract",
        "scenario": "rooms",
        "actions": "rooms",
    },
    "hierarchical-abstract-fixed-craft-": {
        "representation": "abstract",
        "scenario": "random_resources",
        "actions": "rooms",
    },
}


def multi_register_json(entry, envs, rewards):
    for k, v in envs.items():
        register(id=k + "v0", entry_point=f"domains.gym_craft.envs:{entry}", kwargs=v.copy())
        for i, r in rewards.items():
            v["rewards"] = r.copy()
            register(id=k + i, entry_point=f"domains.gym_craft.envs:{entry}", kwargs=v.copy())


multi_register_json("JsonCraftEnv", ENVS, REWARDS)
multi_register_json("JsonRoomsEnv", ROOM_ENVS, REWARDS)
multi_register_json("TrainRoomsEnv", TRAIN_ENVS, REWARDS)
multi_register_json("HierarchicalRoomsEnv", HIERARCHICAL_ENVS, REWARDS)


register(id="teleport-env-v0", entry_point="domains.gym_craft.envs:TeleportEnv")
