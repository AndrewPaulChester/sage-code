import numpy as np
import torch
import forks.rlkit.rlkit.torch.pytorch_util as ptu
import domains.gym_taxi.simulator.taxi_world as taxi_world
import domains.gym_craft.simulator.craft_world as craft_world

ACTION_MAPPING = {
    "move-left": taxi_world.ACTIONS.west,
    "move-down": taxi_world.ACTIONS.south,
    "move-up": taxi_world.ACTIONS.north,
    "move-right": taxi_world.ACTIONS.east,
    "pick-up": taxi_world.ACTIONS.pickup,
    "drop-off": taxi_world.ACTIONS.dropoff,
    "no-op": taxi_world.ACTIONS.noop,
}


class BaseController(object):
    """Base controller class that performs learning and acting.
    
    Designed to be extended with full functionality.
    """

    def __init__(self):
        pass

    # def step(self, ob, reward, done):
    #     pass


class TaxiController(BaseController):
    """Controller class that performs action mapping for the taxi environment.
    """

    def step(self, action, ob):
        # print(action)
        if isinstance(action[0], str):
            return self._step_individual(action[0], ob)
        else:
            return [self._step_individual(a[0], ob) for a in action][
                0
            ]  # TODO: figure out if this is dodgy in parallel environments.

    def _step_individual(self, action, ob):
        # print(action)
        return (ACTION_MAPPING[action].value - 1, False), {}


class CraftController(BaseController):
    """Controller class that performs learning and acting.
    n is the grid size of the environment, used for rescaling things 
    
    """

    def __init__(self, policy, n=11, policy_type="uvfa"):
        self.policy = policy
        self.multihead_mapping = {
            ("no-op", None): 0,
            ("face", "tree"): 1,
            ("face", "rock"): 2,
            ("move", (0, -1)): 3,
            ("move", (0, 1)): 4,
            ("move", (1, 0)): 5,
            ("move", (-1, 0)): 6,
            ("mine", "tree"): 7,
            ("mine", "rock"): 8,
            ("craft", "plank"): 9,
            ("craft", "stick"): 10,
            ("craft", "wooden_pickaxe"): 11,
            ("craft", "stone_pickaxe"): 12,
            ("collect", "coins"): 13,
        }
        self.n = n
        self.policy_type = policy_type

        # self.uvfa_mapping = {
        #     ("no-op", None): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("face", "tree"): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #     ("face", "rock"): [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #     ("clear", "north"): [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("clear", "south"): [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("clear", "east"): [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("clear", "west"): [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("move", "north"): [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("move", "south"): [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("move", "east"): [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("move", "west"): [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     ("mine", "tree"): [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #     ("mine", "rock"): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #     ("craft", "plank"): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #     ("craft", "stick"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #     ("craft", "wooden_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #     ("craft", "stone_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        # }
        self.uvfa_mapping = {
            ("no-op", None): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("collect", "coins"): [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("clear", "north"): [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("clear", "south"): [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("clear", "east"): [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("clear", "west"): [0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", "north"): [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", "south"): [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", "east"): [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("move", "west"): [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ("face", "tree"): [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ("face", "rock"): [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ("mine", "tree"): [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ("mine", "rock"): [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            ("craft", "plank"): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ("craft", "stick"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ("craft", "wooden_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ("craft", "stone_pickaxe"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }

    def step(self, action, ob):
        """ This version of the function is designed for concatenated UVFA approach """
        try:
            conv, fc = ob
        except ValueError:
            fc = ob
            conv = None

        if self.policy_type == "uvfa":

            mapped = []
            for i in range(len(action)):
                command, arg = action[i]
                if command in ("move", "clear"):
                    # not actually north, coordinates overwritten next
                    mapped.append(self.uvfa_mapping[(command, "north")])

                    # Translating absolute room destination into relative direction
                    x = int(fc[i, 0] * self.n / 10)
                    y = int(fc[i, 1] * self.n / 10)
                    mapped[i][0] = arg[0] - x
                    mapped[i][1] = arg[1] - y

                    # Translating absolute destination into relative location for cells
                    # mapped[i][0:2] = (
                    #     torch.tensor(arg, dtype=torch.float32, device=ptu.device)
                    #     + 0.5  # to move target to centre of cell
                    # ) - fc[i, 0:2] * self.n
                    # # print("position " + str(fc[i, 0:2] * self.n))
                else:
                    mapped.append(self.uvfa_mapping[action[i]])
            mapped = ptu.tensor(mapped, dtype=torch.float32)
            fc = torch.cat((fc, mapped), axis=1)

            # if isinstance(ob[0], tuple):
            #     for m, (conv, fc) in zip(mapped, ob):
            #         fc = torch.concat(fc, m)
            if conv is not None:
                (a, e), ai = self.policy.get_action((conv, fc))
            else:
                (a, e), ai = self.policy.get_action(fc)

            ai["symbolic_action"] = mapped
            # print(f"symbolic = {action}, atomic = {craft_world.ACTIONS(int(a[0,0])+1)}")
            return (a, e), ai

        elif self.policy_type == "multihead":
            mapped = [self.multihead_mapping[a] for a in action]
            fc = torch.cat((fc, mapped), axis=1)
            (a, e), ai = self.policy.get_action((conv, fc))
            ai["symbolic_action"] = torch.tensor(
                mapped, dtype=torch.float32, device=ptu.device
            )
            print(f"symbolic = {action}, atomic = {craft_world.ACTIONS(int(a[0,0])+1)}")
            return (a, e), ai

    # def step(self, action, ob):
    #     """ This version of the function is designed for multi-policy head approach """
    #     mapped = [self.multihead_mapping[a] for a in action]
    #     return self.policy.get_action((mapped, ob))


class PretrainedController(BaseController):
    """This is a controller that uses individually pre-trained components. 
    Contains a dictionary that maps actions to sub-controllers. 
    
    """

    def __init__(self, controllers):
        self.controllers = controllers
        self.mapping = {
            "collect": 0,
            "no-op": 1,
            "clear": 1,
            "move": 1,
            "face": 1,
            "mine": 1,
            "craft": 1,
        }

    def step(self, action, ob):
        """ This currently assumes a single threaded approach, can try to vectorise in the future if need be."""
        command, arg = action[0]
        controller = self.mapping[command]
        return self.controllers[controller].step(action, ob)

        # This is a start at making a parallel implementation
        # indices = [self.mapping[c] for c, a in action]
        # output = []

        # for i in range(len(self.controllers)):
        #     obs = ob[indices == i]
        #     acts = action[indices == i]
        #     output.append(self.controllers[i].step(acts, obs))

        # return output

    @property
    def policy(self):
        return [c.policy for c in self.controllers]
