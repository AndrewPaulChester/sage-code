import numpy as np

from domains.gym_craft.envs.craft_env import JsonCraftEnv, ACTIONS
from domains.gym_craft.utils.representations import (
    json_to_image,
    resize_image,
    json_to_screen,
    json_to_mixed,
    json_to_symbolic,
    json_to_pddl,
    json_to_dense,
)
from domains.gym_craft.utils.utils import facing_block
from domains.gym_craft.utils.config import DIRECTIONS


class JsonRoomsEnv(JsonCraftEnv):
    def get_symbolic_observation(self, obs):
        """ Provides the current symbolic representation of the environment. """
        return json_to_symbolic(obs, representation="rooms")

    def project_symbolic_state(self, obs, action):
        """ Calculates a projected state from the current observation and symbolic action.
            Ideally this should hook into the pddl definitions, but for now we duplicate.
        """
        symbolic = self.get_symbolic_observation(obs)

        command, argument = action
        projected = {}
        if command == "move":
            projected["room"] = argument
            projected["door"] = False
        elif command == "face":
            projected["room"] = symbolic["room"]
            projected["facing"] = {argument: True}
        elif command == "mine":
            # projected["room"] = symbolic["room"]
            item = self.sim.gamedata.tiles[argument]["mineable"]["item"]
            projected["inventory"] = {item: symbolic["inventory"][item] + 1}
        elif command == "craft":
            # projected["room"] = symbolic["room"]
            quantity = self.sim.gamedata.recipes[argument]["quantity"]
            projected["inventory"] = {
                argument: symbolic["inventory"][argument] + quantity
            }
        elif command == "collect":
            projected["room"] = symbolic["room"]
            projected["rooms"] = {symbolic["room"]: {"coin": 0}}
        return projected

    def check_projected_state_met(self, obs, projected):
        """ Checks if observation is compatibile with the projected partial state. """
        if projected is None:
            return False

        symbolic = self.get_symbolic_observation(obs)

        for category, value in projected.items():
            if category in ("room", "door"):
                if symbolic[category] != value:
                    return False
            elif category == "rooms":
                for r, d in projected[category].items():
                    for k, v in d.items():
                        if symbolic[category][r][k] != v:
                            return False
            else:
                for k, v in projected[category].items():
                    if symbolic[category][k] != v:
                        return False
        return True

    def convert_to_action(self, subgoal, obs):
        # print("inside _convert_to_action")
        # print(obs, subgoal)
        if obs is not None:
            symbolic = self.get_symbolic_observation(obs)
        else:
            symbolic = {"room": (3, 3)}  # for logging only
        action = {"have": None, "move": None, "collect": None}

        if self.actions == "rooms":
            try:
                switch, move_x, move_y, item, quantity = subgoal.int().cpu().tolist()
            except AttributeError:
                if subgoal < 9:
                    action["move"] = (subgoal // 3, subgoal % 3)
                    switch = -1
                elif subgoal < 15:
                    switch = 1
                    item = subgoal - 9
                    quantity = 1
                elif subgoal == 15:
                    switch = 2
            if switch == 0:
                x, y = symbolic["room"]
                action["move"] = tuple(
                    np.clip(
                        [int(x + move_x - 3), int(y + move_y - 3)],
                        0,
                        int(self.sim.size / 10) - 1,
                    )
                )
            elif switch == 1:
                action["have"] = (self.sim.gamedata.items[int(item)], quantity)
            elif switch == 2:
                action["collect"] = symbolic["room"]
        else:
            raise ValueError("invalid action space for environment")
        # print(action)
        return action

    def generate_pddl(self, ob, subgoal):
        pddl = json_to_pddl(ob, representation="rooms")
        pddl = pddl.replace("$goal$", self._action_to_pddl(subgoal))
        return pddl

    def _action_to_pddl(self, action):
        goal = ""

        if action["have"] is not None:
            item, quantity = action["have"]
            if item.endswith("pickaxe"):
                goal += f"({item} p)\n"
            else:
                goal += f"(>= (have p {item}) {quantity})\n"
        if action["move"] is not None:
            goal += f"(in p r{action['move'][0]}{action['move'][1]})\n"
        if action["collect"] is not None:
            goal += f"(= (contains r{action['collect'][0]}{action['collect'][1]} coin)  0)\n"

        return goal

    def expand_actions(self, obs, actions):
        new_actions = []
        # symbolic = self.get_symbolic_observation(obs)
        for (command, arg) in actions:
            new_actions.append((command, arg))

        return new_actions


# STEPS = ["craft", "collect", "face", "mine", "move"]
STEPS = ["craft", "face", "mine", "move"]
# STEPS = ["move"]


class TrainRoomsEnv(JsonRoomsEnv):
    def __init__(self, *args, **kwargs):
        try:
            steps = kwargs.pop("steps")
        except KeyError as e:
            steps = STEPS

        super().__init__(*args, **kwargs)
        self.step_index = 0
        self.step_list = steps

    def _step(self, action):
        translated_action = ACTIONS(action + 1)
        self.steps += 1
        self.lastaction = translated_action.name
        if action < 4:
            repeat_action = translated_action
        else:
            repeat_action = ACTIONS["noop"]

        obs, reward, done, info = self.sim.act(translated_action)
        changed_room = info.get("changed_room", False)
        for _ in range(self.frameskip - 1):
            if done:
                continue
            obs, r, done, info = self.sim.act(repeat_action)
            changed_room = changed_room or info.get("changed_room", False)
            reward += r
        reward /= 100

        self.score += reward
        info["score"] = self.score

        done = self.check_projected_state_met(obs, self.projected_state)

        if done:
            reward += 1
            # print(f"completed, score of: {self.score}")
            self.score = 0

        if self.steps == self.sim.timeout:
            done = True
            info["bad_transition"] = True
            # print(f"timed out, score of: {self.score}")
            self.score = 0

        # if changed_room and not done:
        #     done = True
        #     reward -= 1
        #     self.score = 0

        return obs, reward, done, info

    def reset(self):
        self.sim = self._init_simulator()
        command = self.step_list[self.step_index]
        argument = self.sim.setup_training(command)
        print(command, argument)
        self.step_index = (self.step_index + 1) % len(self.step_list)
        obs = self.sim._get_state_json()
        self.projected_state = self.project_symbolic_state(obs, (command, argument))
        self.steps = 0
        self.lastaction = None
        return obs


class HierarchicalRoomsEnv(JsonRoomsEnv):
    def convert_to_action(self, subgoal, obs):
        return {"key": subgoal}  # A dictionary is expected

    def generate_pddl(self, ob, subgoal):
        return subgoal["key"]  # Unpacking the fake dictionary
