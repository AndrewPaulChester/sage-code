import yaml

# from itertools import enumerate


class GameData(object):
    def __init__(
        self,
        file="domains/gym_craft/utils/gamedata.yaml",
    ):
        with open(file, "r") as f:
            config = yaml.safe_load(f)
        self.items = [name for name in config["items"]]
        self.tiles = {k: v for k, v in config["tiles"].items()}
        self.recipes = {k: v for k, v in config["recipes"].items()}

        for i, k in enumerate(self.tiles):
            self.tiles[k]["index"] = i

    def get_tile(self, index):
        return list(self.tiles.keys())[index]
