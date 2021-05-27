"""
Symbolic goal generation policy
"""
import numpy as np
from torch import nn
import torch.multiprocessing as mp
import torch
import json


from forks.baselines.baselines.common.vec_env.vec_env import clear_mpi_env_vars

import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.policies.base import Policy
from forks.rlkit.rlkit.pythonplusplus import dict_of_list__to__list_of_dicts


# from gym_agent.controller import QController
from gym_agent.learner import _action_decode
from gym_agent.planner import FDPlanner

from domains.gym_taxi.utils.config import FIXED_GRID_SIZE, DISCRETE_ENVIRONMENT_STATES
import domains.gym_taxi.envs.taxi_env as taxi_env


def _convert_to_torch(raw_obs):

    if raw_obs.shape[1] == 2 and len(raw_obs.shape) == 2:
        action_obs = tuple(
            [
                torch.tensor(o).float().unsqueeze(0).to(torch.device("cuda:0"))
                for o in raw_obs[0]
            ]
        )
    else:
        action_obs = torch.tensor(raw_obs).float().to(torch.device("cuda:0"))

    return action_obs


def _convert_to_torch_parallel(raw_obs):

    if raw_obs.shape[1] == 2 and len(raw_obs.shape) == 2:
        image = []
        dense = []
        for obs in raw_obs:
            image.append(torch.tensor(obs[0]).float().to(torch.device("cuda:0")))
            dense.append(torch.tensor(obs[1]).float().to(torch.device("cuda:0")))
        action_obs = (torch.stack(image, dim=0), torch.stack(dense, dim=0))
    else:
        action_obs = torch.tensor(raw_obs).float().to(torch.device("cuda:0"))

    return action_obs
    # return torch.tensor(raw_obs).float().to(torch.device("cuda:0"))


class LearnPlanPolicy(Policy):
    def __init__(
        self,
        learner,
        planner,
        controller,
        num_processes=1,
        vectorised=False,
        env=None,
        step_limit=100,
        true_parallel=False,
    ):
        super().__init__()
        self.planner = planner
        self.controller = controller
        self.learner = learner
        self.num_processes = num_processes
        self.actions = [[]] * num_processes
        self.projected = [None] * num_processes
        self.steps = [0] * num_processes
        self.vectorised = vectorised
        self.true_parallel = true_parallel
        self.subgoals = []
        self.env = env
        self.json_to_screen = env.observation_space.converter
        self.json_to_controller_screen = env.observation_space.controller_converter
        self.create_workers()
        self.step_limit = step_limit

    def check_action_status(self, obs, valid_envs=None):
        # print(obs)
        # print(obs.shape)
        n = self.num_processes
        step_complete = [False] * n
        step_timeout = [False] * n
        plan_ended = [False] * n
        if valid_envs is None:
            valid_envs = [True] * n

        for i, o in enumerate(obs):
            self.steps[i] += 1
            # did we complete a symbolic action
            if self._symbolic_action_complete(o, i) and valid_envs[i]:
                action = self.actions[i].pop(0)

                if len(self.actions[i]) > 0:
                    self.projected[i] = self.env.project_symbolic_state(
                        o, self.actions[i][0]
                    )
                else:
                    self.projected[i] = None
                if action != ("no-op", None):
                    step_complete[i] = True
                self.steps[i] = 0

            if self.steps[i] == self.step_limit:
                self.actions[i] = []
                self.steps[i] = 0
                step_timeout[i] = True

            if len(self.actions[i]) == 0:
                plan_ended[i] = True

        return step_timeout, step_complete, plan_ended

    def get_action(self, obs, valid_envs=[True]):
        """
        Gets an set of actions using get_action_parallel if multiple threads exist. 
        For debugging purposes, if only a single thread exists calls maybe_get_action instead.
        """
        if self.vectorised:
            if self.true_parallel and len(valid_envs) > 1:
                # print(valid_envs)
                results = self._get_action_parallel(obs, valid_envs)
                # print(results)
            else:
                results = []
                for i, (o, v) in enumerate(zip(obs, valid_envs)):
                    results.append(self._maybe_get_action((i, o, v)))
                    # print(results)

            return results
        return self._get_action(obs, 0)

    def _maybe_get_action(self, args):
        """
        Checks to see if this environment is currently valid, and either gets the next action or returns a no-op accordingly
        """
        i, o, v = args
        if v:
            return self._get_action(o[0], i)
        else:
            print("INVALID ENVIRONMENT")
            return self.controller.step([("no-op", None)], None)
            # this is a complete noop

    def _get_action(self, obs, i):
        """
        Policy function for the agent - given an observation, returns an atomic action.
        Needs to co-ordinate a range of activities in the three tier case:
            1. Check to see if the current observation completes a symbolic action, and if so, pop from queue
            2. Check to see if a new plan needs to be generated, and generate subgoal and symbolic actions if so
            3. Given the current symbolic action, generate a new atomic action.

        Only used for single threaded execution, primarily during debugging.
        """
        explored = False
        failed = False
        subgoal = None
        agent_info = {}
        img = self.json_to_screen(obs)
        controller_img = self.json_to_controller_screen(obs)
        if self.vectorised:
            img = _convert_to_torch(np.array([img]))
            if not isinstance(
                controller_img, str
            ):  # don't convert if it's taxi and so irrelevant
                controller_img = _convert_to_torch(np.array([controller_img]))
        else:
            controller_img = _convert_to_torch(np.array([controller_img]))

        # if no plan exists
        if len(self.actions[i]) == 0:
            (subgoal, explored), agent_info = self._get_subgoal(obs, img, i)
            # print(f"Replanned: new subgoal is {subgoal}")
            # get actions from planner

            # if subgoal["move"] is not None:
            #     # bypass planner for move only actions for efficiency
            #     action_list = [("move", subgoal["move"])]
            # else:
            action_list, failed = self.get_plan((obs, subgoal, i))

            # movement only hack
            # if agent_info["subgoal"][0][4] == 0:
            #     action_list = [("move", "north")]
            # if agent_info["subgoal"][0][4] == 1:
            #     action_list = [("move", "south")]
            # if agent_info["subgoal"][0][4] == 2:
            #     action_list = [("move", "east")]
            # if agent_info["subgoal"][0][4] == 3:
            #     action_list = [("move", "west")]
            # action_list = self.env.expand_actions(obs, action_list)

            # print(f"Symbolic actions: {action_list}")
            self.actions[i] = action_list
            self.projected[i] = self.env.project_symbolic_state(obs, self.actions[i][0])
            agent_info["failed"] = failed
            agent_info["explored"] = explored

        # this currently signifies that the next observation is the end of the plan and should be stored -
        # this will need to be rethought as it won't be possible to tell in advance which action will end the plan
        if len(self.actions[i]) == 1:
            agent_info["empty"] = True

        # print(self.actions)
        # get atomic action from low-level controller for next action in sequence
        (action, e), aic = self._get_atomic_action(controller_img, i)
        aic["agent_info_learn"] = agent_info

        return (action, e), aic

    def _symbolic_action_complete(self, obs, i):
        return self.env.check_projected_state_met(obs, self.projected[i])

    def _get_subgoal(self, obs, img, i):
        agent_info = {}
        if self.learner is None:
            subgoal = self._get_random_action(obs)
            agent_info["subgoal"] = 1
        elif hasattr(self.learner, "json_requested"):
            (subgoal, explored), agent_info = self.learner.get_action(obs)
        else:
            # preprocess observation

            # print(f"taxi_loc: {json.loads(obs)['taxi']['location']}")

            # get subgoal from learner
            (subgoal, explored), agent_info = self.learner.get_action(img)
            # print(subgoal)
            agent_info["subgoal"] = subgoal
            try:
                agent_info["rnn_hxs"] = agent_info["rnn_hxs"][i].unsqueeze(0)
            except KeyError:
                pass  # DQN doesn't have rnn_hxs

            # transform subgoal
            if self.vectorised:
                subgoal = subgoal[0]
            subgoal = self.env.convert_to_action(subgoal, obs)
            # print(subgoal)
        return (subgoal, explored), agent_info

    def _get_atomic_action(self, img, i):
        return self.controller.step([self.actions[i][0]], img)

    def _get_action_parallel(self, obs, valid_envs):
        """
        Gets atomic actions for the environment, parallelising plan execution.
        """
        queued = np.array([len(a) == 0 for a in self.actions])
        need_plan = queued & valid_envs
        # print("need_plan: ", need_plan)
        # print(
        #     "obs: ",
        #     # [(json.loads(o[0])["taxi"], json.loads(o[0])["passenger"]) for o in obs],
        #     [json.loads(o[0])["passenger"] for o in obs],
        # )
        # print("pre actions: ", [len(a) for a in self.actions])

        # if no plan exists
        if need_plan.any():
            # preprocess observation
            # print(obs.shape)
            img = np.array([self.json_to_screen(o[0]) for o in obs[need_plan]])
            # print(f"img {img}")
            # print(img.shape)
            # get subgoal from learner
            (subgoal, explored), agent_info = self.learner.get_action(
                _convert_to_torch_parallel(img)
            )
            # print(f"subgoal {subgoal}")
            # print(f"explored {explored}")
            # print(f"agent_info {agent_info}")
            agent_info["subgoal"] = subgoal

            subg = []
            for o, s in zip(obs[need_plan], subgoal):
                x = self.env.convert_to_action(s, o[0])
                # print("x", x)
                subg.append(x)
            # print("subg", subg)
            subgoal = subg

            n = np.sum(need_plan)
            # print(n)
            args = [
                (o[0], s, i) for i, (o, s) in enumerate(zip(obs[need_plan], subgoal))
            ]
            self.plan_async(args)
            x = self.step_wait()
            # print(x)
            action_list = [a for a, _ in x]
            failed = [r for _, r in x]
            # action_list, failed = self.get_plan(obs, subgoal, i)

            for i in range(len(self.actions)):
                if need_plan[i]:
                    self.actions[i] = action_list.pop(0)
            agent_info["failed"] = failed
            # print(f"agent_info {agent_info}")

            agent_info = dict_of_list__to__list_of_dicts(agent_info, np.sum(need_plan))
            for ai in agent_info:
                for k, v in ai.items():
                    try:
                        ai[k] = v.unsqueeze(0)
                    except AttributeError:
                        pass
            # print(f"agent_info {agent_info}")

        # print(self.actions)
        e_list = []
        ai_list = []
        a_list = []
        for i in range(len(self.actions)):
            if need_plan[i]:
                e_list.append(explored[0].unsqueeze(0))
                explored = explored[1:]
                ai_list.append({"agent_info_learn": agent_info.pop(0)})
            else:
                e_list.append(False)
                ai_list.append({"agent_info_learn": {}})

            if valid_envs[i]:
                if len(self.actions[i]) == 1:
                    ai_list[i]["agent_info_learn"]["empty"] = True
                (a, _), _ = self.controller.step(self.actions[i][0], None)
                a_list.append(a)
            else:
                (a, _), _ = self.controller.step(("no-op", None), None)
                a_list.append(a)

        # get atomic action from low-level controller for next action in sequence
        # print("a_list", a_list)
        # print("e_list", e_list)
        # print("ai_list", ai_list)

        # print("post_actions:", [len(a) for a in self.actions])
        return list(zip(list(zip(a_list, e_list)), ai_list))

    def _break(self, subgoal):
        empty = subgoal["empty"]
        delivered = subgoal["delivered"]
        location = subgoal["location"]
        passenger = subgoal["passenger"]
        if (
            (
                delivered is not None and delivered != [0]
            )  # delivering non-existent passenger
            or (
                passenger is not None and passenger != [0]
            )  # carrying non-existent passenger
            or (passenger is not None and empty)  # carrying and empty
            or (
                delivered is not None
                and passenger is not None
                and delivered == passenger
            )  # delivered and carrying
            or (
                not empty
                and delivered is None
                and location is None
                and passenger is None
            )
        ):  # no goal
            return False
        return True

    def get_plan(self, args):
        obs, subgoal, i = args
        # print("in plan", i)
        # produce planning problem given subgoal
        if (
            "empty" in subgoal
            and subgoal["empty"] == False
            and subgoal["delivered"] is None
            and subgoal["location"] is None
            and subgoal["passenger"] is None
        ):
            # print("idling", i)
            return [("no-op", None)], False

        if all(v is None for v in subgoal.values()):
            # print("idling", i)
            return [("no-op", None)], False

        problem = self.env.generate_pddl(obs, subgoal)

        # get plan from planner
        actions = self.planner.plan(problem, i)
        if actions is None:
            # print("proposed invalid plan", i)
            return (
                [("no-op", None)],
                True,
            )  # TODO: figure out how to accomodate no-ops in both environments simultaneously
        actions = self.env.expand_actions(obs, actions)
        # (subgoal["location"] is not None and subgoal[delivered] is None and subgoal['passenger'] is None) or ()
        self.subgoals.append((subgoal, actions))
        if len(actions) > 0:
            # print(subgoal)
            # print(actions, i)
            # perform update step on learner.
            return actions, False
        # print("idling", i)
        return [("no-op", None)], False

    def reset_selected(self, masks):
        # print(f"resetting: {masks}")
        for i in range(self.num_processes):
            if masks[i]:
                self.reset(i)

    def reset(self, i=0):
        self.actions[i] = []
        self.steps[i] = 0
        self.projected[i] = None
        # self.learner.reset(i)

    def create_workers(self):
        ctx = mp.get_context("spawn")
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for _ in self.actions:
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=_subproc_worker, args=(child_pipe, parent_pipe)
                )
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def plan_async(self, actions):

        self.current_pipes = len(actions)
        # print(f"planning: {self.current_pipes}")
        assert len(actions) == len(self.parent_pipes[: self.current_pipes])
        for pipe, act in zip(self.parent_pipes[: self.current_pipes], actions):
            pipe.send(("plan", act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes[: self.current_pipes]]
        # print(f"waited: {self.current_pipes}")
        self.waiting_step = False
        return outs

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()


def _subproc_worker(pipe, parent_pipe):
    planner = FDPlanner()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            # if cmd == "reset":
            #     pipe.send(_write_obs(env.reset()))
            if cmd == "plan":
                actions = sub_get_plan(data, planner)
                # print("sending results in pipe")
                pipe.send(actions)
            # elif cmd == "render":
            #     pipe.send(env.render(mode="rgb_array"))
            elif cmd == "close":
                pipe.send(None)
                break
            else:
                raise RuntimeError("Got unrecognized cmd %s" % cmd)
    except KeyboardInterrupt:
        print("Plan worker: got KeyboardInterrupt")
    # finally:
    #     env.close()


# TODO: see if some of the logic of get plan and sub_get_plan can be factored out
def sub_get_plan(data, planner):
    obs, subgoal, i = data
    # print("subgoal: ", subgoal)
    problem = taxi_env.generate_pddl(obs, subgoal)
    # print(i, problem, flush=True)
    # env = json.loads(obs)
    # get plan from planner
    actions = planner.plan(problem, i)
    # print(i, actions, flush=True)
    if actions is None:
        # print("proposed invalid plan", i)
        return [("no-op", None)], True
    actions = taxi_env.expand_actions(obs, actions)

    # (subgoal["location"] is not None and subgoal[delivered] is None and subgoal['passenger'] is None) or ()

    if len(actions) > 0:
        # print("passenger: ", env["passenger"])
        # print(actions, i)
        # perform update step on learner.
        return actions, False
    # print("idling", i)
    return [("no-op", None)], False
