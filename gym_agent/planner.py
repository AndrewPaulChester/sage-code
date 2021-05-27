import sys
import os
import subprocess
import re


class BasePlanner(object):
    """Base planner class that performs planning.
    
    Designed to be extended with full functionality.
    """

    def __init__(self):
        pass

    def plan(self, problem):
        pass


class FDPlanner(BasePlanner):
    """Base learner class that performs learning and acting.
    
    Designed to be extended with full functionality.
    """

    def __init__(self):
        pass

    def plan(self, problem, i=0):

        planner_path = os.environ["PLANNER_PATH"] + "downward/fast-downward.py"
        domain = os.path.join(
            os.path.dirname(__file__), "../planning/taxi/simple-domain.pddl"
        )
        search_method = "astar(blind())"
        problem_file = f"./fd-test-problem{i}.pddl"
        sas_file = f"./output{i}.sas"

        with open(problem_file, "w") as pf:
            pf.write(problem)

        plan_file = f"./fd-test{i}.plan"
        with open(plan_file, "w") as pf:  # overwrite old plan file
            pf.write("")

        cmd = [
            planner_path,
            "--plan-file",
            plan_file,
            "--sas-file",
            sas_file,
            domain,
            problem_file,
            "--search",
            search_method,
        ]
        # print(cmd)
        sys.stdout.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        # print("in planner: ", i, stdout.decode("utf-8"), flush=True)
        # print("in planner: ", i, problem_file, plan_file, flush=True)
        if re.search("Solution found\.", stdout.decode("utf-8")):
            return self.parse_plan_file(plan_file)
        else:
            return None

    def parse_plan_file(self, plan_file):
        actions = []
        try:
            with open(plan_file) as pf:
                for action in pf.readlines():
                    if action[0] == "(":
                        a = action.split()[0][1:]
                        if a == "move":
                            actions.append(self.convert_move_action(action))
                        else:
                            actions.append(a)
            return actions
        except FileNotFoundError:
            return None

    def convert_move_action(self, action):
        parts = action[1:-2].replace(" t ", " ").split()
        start = self.convert_coordinate(parts[1])
        end = self.convert_coordinate(parts[2])
        return f"{parts[0]} {start} {end}"

    def convert_coordinate(self, coordinate):
        return coordinate.replace("sx", "(").replace("y", ",") + ")"


class ENHSPPlanner(BasePlanner):
    """Planner using ENHSP (for numeric domains)
    """

    def __init__(self):
        pass

    def plan(self, problem, i=0):

        planner_path = os.environ["PLANNER_PATH"] + "ENHSP/enhsp"
        domain = os.path.join(
            os.path.dirname(__file__), "../planning/craft/rooms-domain.pddl"
        )
        problem_file = os.path.join(os.path.dirname(__file__), "./problem{i}.pddl")
        problem_file = f"./fd-test-problem{i}.pddl"
        with open(problem_file, "w") as pf:
            pf.write(problem)

        cmd = [planner_path, "-o", domain, "-f", problem_file, "-planner", "sat"]
        # print(cmd)
        sys.stdout.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

        if re.search("Problem Solved", stdout.decode("utf-8")):
            return self.parse_plan_file(stdout.decode("utf-8"))
        else:
            return None

    def parse_plan_file(self, plan_file):
        actions = []
        for a in plan_file.split("\n"):
            if a[0:9] == "(0.00000)":
                action = a.split()[1:]
                if action[0] == "move":
                    # actions.append((action[0], action[5]))
                    actions.append(
                        # (action[0], (int(action[5][1:]), int(action[8][1:])))
                        (action[0], (int(action[11][1]), int(action[11][2])))
                    )
                elif action[0] == "clear":
                    actions.append((action[0], action[5]))
                elif action[0] == "face":
                    actions.append((action[0], action[5]))
                elif action[0].startswith("mine"):
                    words = action[0].split("-")
                    actions.append((words[0], words[1]))
                elif action[0].startswith("craft"):
                    words = action[0].split("-")
                    actions.append((words[0], "_".join(words[1:])))
                elif action[0].startswith("collect"):
                    words = action[0].split("-")
                    actions.append((words[0], "_".join(words[1:])))
                else:
                    raise ValueError(f"Unexpected action: {action[0]}")

        return actions


class DummyHierarchicalPlanner(object):
    """Base planner class that performs planning.
    
    Designed to be extended with full functionality.
    """

    def __init__(self):
        self.mapping = {
            0: ("no-op", None),
            1: ("face", "tree"),
            2: ("face", "rock"),
            3: ("move", (0, -1)),
            4: ("move", (0, 1)),
            5: ("move", (1, 0)),
            6: ("move", (-1, 0)),
            7: ("mine", "tree"),
            8: ("mine", "rock"),
            9: ("craft", "plank"),
            10: ("craft", "stick"),
            11: ("craft", "wooden_pickaxe"),
            12: ("craft", "stone_pickaxe"),
            13: ("collect", "coins"),
        }

    def plan(self, problem, i=0):
        # print(self.mapping[problem])
        return [self.mapping[problem]]


if __name__ == "__main__":
    # planner = ENHSPPlanner()
    planner = FDPlanner()
    actions = planner.plan(
        """        (define (problem problem5p) (:domain navigation)
    (:objects 
        sx0y0 sx4y3 - space
        t - taxi
        p0 - passenger
         - fuelstation
    )

    (:init
        ;todo: put the initial state's facts and numeric values here
        
        (empty t)
        (in t sx0y0)
        (= (fuel t) 100)
        (= (money t) 0)
        (in p0 sx0y0)
        (destination p0 sx4y3)
        
        
        ;(= (capacity t) 100)
    )

    (:goal (and
            (delivered p0)
(in t sx0y0)

        )
    )

    ;un-comment the following line if metric is needed
    ;(:metric minimize (???))
    )
    """
    )
    print(actions)
