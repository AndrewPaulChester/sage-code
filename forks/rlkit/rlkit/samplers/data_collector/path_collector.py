from collections import deque, OrderedDict, Counter, defaultdict

from torch import Tensor
import numpy as np

import forks.rlkit.rlkit.torch.pytorch_util as ptu

from forks.rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from forks.rlkit.rlkit.samplers import rollout_functions
from forks.rlkit.rlkit.samplers.data_collector.base import PathCollector


class MdpPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        rollout=rollout_functions.rollout,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout = rollout

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._total_score = 0
        self._epoch_score = 0
        self._num_episodes = 0
        self._epoch_episodes = 0
        self._plan_lengths = defaultdict(list)

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths):
        """
        Collects num_steps worth of experience.

        :param max_path_length: maximum path length to collect - only needed for environments that don't terminate
        :param num_steps: number of steps to collect
        :param discard_incomplete_paths: flag to determine if paths that are terminated before environment completion are kept
        :returns: a list of dictionaries
        """
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected
            )
            path = self._rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path["actions"])
            # adding to handle intermediate experience
            if "intermediate_experience" in (path["env_infos"][0]):
                path = self.extend_path(path, path_len, max_path_length_this_loop)

            path_len = len(path["actions"])

            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def extend_path(self, path, path_len, max_path_length_this_loop):
        # adding to handle intermediate experience
        length = path_len
        path_score = sum(path["rewards"]).item()
        self._total_score += path_score
        self._epoch_score += path_score
        episodes = sum(path["terminals"]).item()
        self._num_episodes += episodes
        self._epoch_episodes += episodes

        path["plan_lengths"] = [1] * length

        for i in range(path_len):
            action = path["actions"][i]
            explored = path["explored"][i]
            agent_info = path["agent_infos"][i]
            next_obs = path["next_observations"][i]
            env_info = path["env_infos"][i]
            terminal = path["terminals"][i]

            plan_length = len(env_info["intermediate_experience"]) + 1
            self._plan_lengths[action.item()].append(plan_length)
            path["plan_lengths"][i] = plan_length

            for (obs, reward) in env_info.pop("intermediate_experience"):
                path["actions"] = np.concatenate(
                    (path["actions"], np.expand_dims(action, 0))
                )
                path["explored"] = np.concatenate(
                    (path["explored"], np.expand_dims(explored, 0))
                )
                path["agent_infos"] = np.concatenate(
                    (path["agent_infos"], np.expand_dims(agent_info, 0))
                )
                path["next_observations"] = np.concatenate(
                    (path["next_observations"], np.expand_dims(next_obs, 0))
                )
                path["terminals"] = np.concatenate(
                    (path["terminals"], np.expand_dims(terminal, 0))
                )
                path["observations"] = np.concatenate(
                    (path["observations"], np.expand_dims(obs, 0))
                )
                path["rewards"] = np.concatenate(
                    (path["rewards"], np.expand_dims(np.expand_dims(reward, 0), 0))
                )
                path["env_infos"] = np.concatenate(
                    (path["env_infos"], np.expand_dims(env_info, 0))
                )
                plan_length -= 1
                path["plan_lengths"].append(plan_length)
                length += 1
                if length == max_path_length_this_loop:
                    return path

        return path

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch, hard_reset=False):
        self._epoch_score = 0
        self._epoch_episodes = 0
        self._plan_lengths = defaultdict(list)

        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        if hard_reset:
            self._env.reset()
        try:
            self._policy.learner.es.anneal_epsilon()
        except AttributeError:
            try:
                self._policy.es.anneal_epsilon()
            except AttributeError:
                print("not using linear annealing, or problem somewhere")

    def get_diagnostics(self):
        average_score = (
            0 if self._num_episodes == 0 else self._total_score / self._num_episodes
        )
        epoch_score = (
            0 if self._epoch_episodes == 0 else self._epoch_score / self._epoch_episodes
        )

        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
                ("average score", average_score),
                ("epoch score", epoch_score),
            ]
        )

        if len(self._plan_lengths) > 0:

            action_lengths = [0] * 16  # TODO: fix magic number
            action_counts = [0] * 16
            for action, lengths in self._plan_lengths.items():
                action_lengths[action] = sum(lengths)
                action_counts[action] = len(lengths)

            a = {}
            for i in range(16):
                if action_counts[i] > 0:
                    action_lengths[i] = action_lengths[i] / action_counts[i]
                a[f"action {i} count"] = action_counts[i]
                a[f"action {i} length"] = action_lengths[i]
            stats.update(a)

        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True
            )
        )
        return stats

    def get_snapshot(self):
        return dict(env=self._env, policy=self._policy)


def pprint(path):
    for p in path:
        s = "["
        for i in p:
            s += f"{i:.2f},"
        s = s[:-1] + "]"
        print(s)


class IntermediatePathCollector(MdpPathCollector):
    """
    Responsible for generating experience for the trainer to use. 
    """

    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        rollout=rollout_functions.rollout,
        gamma=0.99,
        naive_discounting=False,
        experience_interval=1,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout = rollout
        self.gamma = gamma
        self.naive_discounting = naive_discounting
        self.experience_interval = experience_interval

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._num_episodes = 0
        self._total_score = 0
        self._epoch_episodes = 0
        self._epoch_score = 0

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths):
        """
        Collects num_steps worth of experience.

        :param max_path_length: maximum path length to collect - only needed for environments that don't terminate
        :param num_steps: number of steps to collect
        :param discard_incomplete_paths: flag to determine if paths that are terminated before environment completion are kept
        :returns: a list of dictionaries
        """
        paths = []
        num_steps_collected = 0
        done = True
        obs = None
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected
            )
            path, (done, obs) = self._rollout(
                self._env,
                self._policy,
                restart=done,
                starting_obs=obs,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                experience_interval=self.experience_interval,
            )
            path_len = len(path["actions"])
            # if (
            #     path_len != max_path_length
            #     and not path["terminals"][-1]
            #     and discard_incomplete_paths
            # ):
            #     break

            self._epoch_episodes += path["terminals"].sum()
            self._num_episodes += path["terminals"].sum()
            converted_path = self.extract_intermediate_experience(path, path_len)
            path_len = len(converted_path["actions"])
            num_steps_collected += path_len
            paths.append(converted_path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def extract_intermediate_experience(self, path, path_len):

        ai = path["agent_infos"][0]["agent_info_learn"]
        # for k, v in ai.items():
        #     if isinstance(v, Tensor):
        #         ai[k] = ptu.get_numpy(v)

        # e = path["explored"][0, 0].item()
        e = ai["explored"]
        action = ai["subgoal"]
        next_obs = path["next_observations"][-1]

        path["explored"].fill(e)
        path["actions"].fill(action)
        path["agent_infos"] = [{}] * path_len
        path["next_observations"][:] = next_obs
        path["plan_lengths"] = list(reversed(range(1, path_len + 1)))

        acc_reward = 0
        for i, r in enumerate(reversed(path["rewards"])):
            if not self.naive_discounting:
                acc_reward *= self.gamma
            acc_reward += r
            self._total_score += r.item()
            self._epoch_score += r.item()
            path["rewards"][path_len - i - 1] = acc_reward

        new_path = {}
        for k, v in path.items():
            new_path[k] = v[:: self.experience_interval]

        return new_path

    def get_snapshot(self):
        return dict(
            env=self._env,
            controller=self._policy.controller,
            learner=self._policy.learner,
        )

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        average_score = (
            0 if self._num_episodes == 0 else self._total_score / self._num_episodes
        )
        epoch_score = (
            0 if self._epoch_episodes == 0 else self._epoch_score / self._epoch_episodes
        )
        explored = [path["explored"][0] for path in self._epoch_paths]
        paths_explored = (
            0 if len(explored) == 0 else sum(explored).item() / len(explored)
        )

        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
                ("average score", average_score),
                ("epoch score", epoch_score),
                ("plans explored", paths_explored),
            ]
        )
        action_lengths = [
            (path["actions"][0].item(), len(path["actions"]))
            for path in self._epoch_paths
        ]
        action_lengths = [0] * 16  # TODO: fix magic number
        action_counts = [0] * 16
        for path in self._epoch_paths:
            action_lengths[path["actions"][0].item()] += len(path["actions"])
            action_counts[path["actions"][0].item()] += 1

        a = {}
        for i in range(16):
            if action_counts[i] > 0:
                action_lengths[i] = action_lengths[i] / action_counts[i]
            a[f"action {i} count"] = action_counts[i]
            a[f"action {i} length"] = action_lengths[i]
        stats.update(a)

        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True
            )
        )
        return stats

    def end_epoch(self, epoch, hard_reset=False):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._epoch_score = 0
        self._epoch_episodes = 0

        if hard_reset:
            self._env.reset()

        try:
            self._policy.learner.es.anneal_epsilon()
        except AttributeError:
            pass


class GoalConditionedPathCollector(PathCollector):
    def __init__(
        self,
        env,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        observation_key="observation",
        desired_goal_key="desired_goal",
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length, num_steps - num_steps_collected
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path["actions"])
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
            ]
        )
        stats.update(
            create_stats_ordered_dict(
                "path length", path_lens, always_show_all_stats=True
            )
        )
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
