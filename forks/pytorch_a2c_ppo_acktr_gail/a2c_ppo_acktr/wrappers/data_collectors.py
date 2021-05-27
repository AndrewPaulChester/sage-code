import numpy as np
from collections import defaultdict

from gym.spaces import Tuple, Box
import torch
import gtimer as gt

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.storage import RolloutStorage, AsyncRollouts

from forks.rlkit.rlkit.core.external_log import LogPathCollector
import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.util.multi_queue import MultiQueue
from forks.rlkit.rlkit.core.eval_util import create_stats_ordered_dict
import forks.rlkit.rlkit.pythonplusplus as ppp

from domains.gym_taxi.utils.spaces import Json
from domains.gym_craft.envs.craft_env import ACTIONS


def _flatten_tuple(observation):
    """Assumes observation is a tuple of tensors. converts ((n,c, h, w),(n, x)) -> (n,c*h*w+x)"""
    image, fc = observation
    flat = image.view(image.shape[0], -1)
    return torch.cat((flat, fc), dim=1)


class RolloutStepCollector(LogPathCollector):
    def __init__(
        self,
        env,
        policy,
        device,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        num_processes=1,
    ):
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs, num_processes
        )
        self.num_processes = num_processes

        self.device = device
        self.is_json = isinstance(env.observation_space, Json)
        self.is_tuple = False
        if self.is_json:
            self.json_to_screen = env.observation_space.converter
            self.is_tuple = isinstance(env.observation_space.image, Tuple)

        self.shape = (
            (
                (
                    env.observation_space.image[0].shape,
                    env.observation_space.image[1].shape,
                )
                if self.is_tuple
                else env.observation_space.image.shape
            )
            if self.is_json
            else env.observation_space.shape
        )
        self._rollouts = RolloutStorage(
            max_num_epoch_paths_saved,
            num_processes,
            self.shape,
            env.action_space,
            1,  # hardcoding reccurent hidden state off for now.
        )

        raw_obs = env.reset()
        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        self.obs = (
            raw_obs
            if isinstance(self, HierarchicalStepCollector)
            or isinstance(self, ThreeTierStepCollector)
            else action_obs
        )

        # print(raw_obs.shape)
        # print(action_obs.shape)
        # print(stored_obs.shape)

        self._rollouts.obs[0].copy_(stored_obs)
        self._rollouts.to(self.device)

    def _convert_to_torch(self, raw_obs):
        if self.is_json:
            list_of_observations = [self.json_to_screen(o[0]) for o in raw_obs]
            if self.is_tuple:
                tuple_of_observation_lists = zip(*list_of_observations)
                action_obs = tuple(
                    [
                        torch.tensor(list_of_observations).float().to(self.device)
                        for list_of_observations in tuple_of_observation_lists
                    ]
                )
            else:
                action_obs = torch.tensor(list_of_observations).float().to(self.device)
        else:
            action_obs = torch.tensor(raw_obs).float().to(self.device)
        return action_obs

    def get_rollouts(self):
        return self._rollouts

    def collect_one_step(self, step, step_total):
        with torch.no_grad():
            (action, explored), agent_info = self._policy.get_action(self.obs)
        # print(action)
        # print(explored)
        # print(agent_info)

        value = agent_info["value"]
        action_log_prob = agent_info["probs"]
        recurrent_hidden_states = agent_info["rnn_hxs"]

        # Observe reward and next obs
        raw_obs, reward, done, infos = self._env.step(ptu.get_numpy(action))
        if self._render:
            self._env.render(**self._render_kwargs)

        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        self.obs = action_obs

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )

        self._rollouts.insert(
            stored_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
        )

        flat_ai = ppp.dict_of_list__to__list_of_dicts(agent_info, len(action))

        # print(flat_ai)
        self.add_step(action, action_log_prob, reward, done, value, flat_ai)


class HierarchicalStepCollector(RolloutStepCollector):
    def __init__(
        self,
        env,
        policy,
        device,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        num_processes=1,
        gamma=1,
        no_plan_penalty=False,
        naive_discounting=False,
    ):
        super().__init__(
            env,
            policy,
            device,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
            num_processes,
        )
        self.action_queue = MultiQueue(num_processes)
        self.obs_queue = MultiQueue(num_processes)
        self.cumulative_reward = np.zeros(num_processes)
        self.discounts = np.ones(num_processes)
        self.naive_discounting = naive_discounting
        if self.naive_discounting:
            self.gamma = 1
            self.plan_length = np.ones(num_processes)
        else:
            self.gamma = gamma
            self.plan_length = np.zeros(num_processes)
        self.no_plan_penalty = no_plan_penalty

    def collect_one_step(self, step, step_total):
        """
        This needs to handle the fact that different environments can have different plan lengths, and so the macro steps are not in sync. 

        Issues: 
            cumulative reward
            multiple queued experiences for some environments
            identifying termination

        While there is an environment which hasn't completed one macro-step, it takes simultaneous steps in all environments.
        Keeps two queues, action and observation, which correspond to the start and end of the macro step.
        Each environment is checked to see if the current information should be added to the action or observation queues.
        When the observation queue (which is always 0-1 steps behind the action queue) has an action for all environments, these
        are retrieved and inserted in the rollout storage.

        To ensure that environments stay on policy (i.e. we don't queue up lots of old experience in the buffer), and we don't plan more 
        than required, each step we check to see if any environments have enough experience for this epoch, and if so we execute a no-op.
        """
        remaining = step_total - step

        step_count = 0
        while not self.obs_queue.check_layer():
            # print(remaining, step_total, step)
            valid_envs = [len(q) < remaining for q in self.obs_queue.queues]
            # print(valid_envs)
            with torch.no_grad():
                results = self._policy.get_action(self.obs, valid_envs=valid_envs)

            action = np.array([[a] for (a, _), _ in results])
            # print(f"actions: {action}")
            # Observe reward and next obs
            raw_obs, reward, done, infos = self._env.step(action)

            if self._render:
                self._env.render(**self._render_kwargs)
            self.obs = raw_obs
            self.discounts *= self.gamma
            if not self.naive_discounting:
                self.plan_length += 1
            self.cumulative_reward += reward * self.discounts
            # print("results now")
            # call this to update the actions (tells policy current plan step was completed)
            step_timeout, step_complete, plan_ended = self._policy.check_action_status(
                self.obs.squeeze(1), valid_envs
            )

            for i, ((a, e), ai) in enumerate(results):
                # print(f"results: {i}, {((a, e), ai)}")

                # unpack the learner agent info now that learn_plan_policy is three tier.
                ai = ai["agent_info_learn"]
                # print(f"results: {i}, {((a, e), ai)}")
                if (
                    ai.get("failed") and not self.no_plan_penalty
                ):  # add a penalty for failing to generate a plan
                    self.cumulative_reward[i] -= 0.5
                    # print("FAILED")
                if "subgoal" in ai:
                    # print("SUBGOAL")
                    self.action_queue.add_item(
                        (ai["rnn_hxs"], ai["subgoal"], ai["probs"], e, ai["value"], ai),
                        i,
                    )
                if (done[i] and valid_envs[i]) or "empty" in ai:
                    # print("EMPTY")
                    if done[i]:
                        # print("DONE")
                        self._policy.reset(i)
                    self.obs_queue.add_item(
                        (
                            self.obs[i],
                            self.cumulative_reward[i],
                            done[i],
                            infos[i],
                            self.plan_length[i],
                        ),
                        i,
                    )
                    self.cumulative_reward[i] = 0
                    self.discounts[i] = 1
                    self.plan_length[i] = 0
            step_count += 1
            # print("results done")
            # print(step_count)

        # [
        #     print(f"obs queue layer {i} length {len(q)}")
        #     for i, q in enumerate(self.obs_queue.queues)
        # ]
        # [
        #     print(f"action queue layer {i} length {len(q)}")
        #     for i, q in enumerate(self.action_queue.queues)
        # ]
        o_layer = self.obs_queue.pop_layer()
        a_layer = self.action_queue.pop_layer()
        layer = [o + a for o, a in zip(o_layer, a_layer)]
        obs, reward, done, infos, plan_length, recurrent_hidden_states, action, action_log_prob, explored, value, agent_info = [
            z for z in zip(*layer)
        ]

        raw_obs = np.array(obs)
        recurrent_hidden_states = torch.cat(recurrent_hidden_states)
        action = torch.cat(action)
        action_log_prob = torch.cat(action_log_prob)
        explored = np.array(explored)
        value = torch.cat(value)
        reward = np.array(reward)
        plan_length = np.array(plan_length)

        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        plan_length = torch.from_numpy(plan_length).unsqueeze(dim=1).float()

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )

        self._rollouts.insert(
            stored_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            reward,
            masks,
            bad_masks,
            plan_length,
        )
        self.add_step(action, action_log_prob, reward, done, value, agent_info)

    def get_snapshot(self):
        return dict(env=self._env, policy=self._policy.learner)


class ThreeTierStepCollector(RolloutStepCollector):
    def __init__(
        self,
        env,
        policy,
        device,
        ancillary_goal_size,
        symbolic_action_size,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        num_processes=1,
        gamma=1,
        no_plan_penalty=False,
        meta_num_epoch_paths=None,
    ):

        super().__init__(
            env,
            policy,
            device,
            max_num_epoch_paths_saved,
            render,
            render_kwargs,
            num_processes,
        )
        self.action_queue = MultiQueue(num_processes)
        self.obs_queue = MultiQueue(num_processes)
        self.cumulative_reward = np.zeros(num_processes)
        self.discounts = np.ones(num_processes)
        self.plan_length = np.zeros(num_processes)
        self.step_length = np.zeros(num_processes)
        self.gamma = gamma
        self.no_plan_penalty = no_plan_penalty

        self._learn_rollouts = AsyncRollouts(
            meta_num_epoch_paths,
            num_processes,
            self.shape,
            Box(-np.inf, np.inf, (ancillary_goal_size,)),
            1,  # hardcoding reccurent hidden state off for now.
        )

        self._learn_rollouts.obs[0].copy_(self._rollouts.obs[0])
        self._learn_rollouts.to(self.device)

        if env.action_space.__class__.__name__ == "Discrete":
            self.action_size = 1
        else:
            self.action_size = env.action_space.shape[0]

        self.ancillary_goal_size = ancillary_goal_size
        n, p, l = self._rollouts.obs.shape
        self._rollouts.obs = torch.zeros(n, p, l + symbolic_action_size).to(self.device)
        self.symbolic_actions = torch.zeros(p, symbolic_action_size).to(self.device)
        # logging

        self.epoch_stats = {
            "plan": defaultdict(self.init_goal),
            "step": defaultdict(self.init_goal),
            "action": defaultdict(lambda: defaultdict(int)),
        }
        self.plans = [None] * num_processes

    def get_rollouts(self):
        return self._rollouts, self._learn_rollouts

    def parallelise_results(self, results):
        action = torch.zeros((self.num_processes, self.action_size))
        explored = torch.zeros((self.num_processes, 1))
        value = torch.zeros((self.num_processes, 1))
        action_log_prob = torch.zeros((self.num_processes, 1))
        recurrent_hidden_states = torch.zeros((self.num_processes, 1))

        invalid = torch.zeros((self.num_processes, 1))
        goal = torch.zeros((self.num_processes, self.ancillary_goal_size))
        learn_value = torch.zeros((self.num_processes, 1))
        learn_action_log_prob = torch.zeros((self.num_processes, 1))
        learn_recurrent_hidden_states = torch.zeros((self.num_processes, 1))
        plan_mask = np.zeros((self.num_processes,), dtype=np.bool)

        agent_info_control = []

        for i, ((a, e), aic) in enumerate(results):
            ail = aic.pop("agent_info_learn")
            action[i] = a
            explored[i] = e
            value[i] = aic["value"]
            action_log_prob[i] = aic["probs"]
            # print("buffer", recurrent_hidden_states.shape)
            # print("rnnhxs", aic["rnn_hxs"])
            # print("probs", aic["probs"])
            recurrent_hidden_states = aic["rnn_hxs"]
            agent_info_control.append(aic)
            self.symbolic_actions[i] = aic["symbolic_action"]
            if "subgoal" in ail:
                plan_mask[i] = 1
                invalid[i] = ail["failed"]
                goal[i] = ail["subgoal"]
                learn_value[i] = ail["value"]
                learn_action_log_prob[i] = ail["probs"]
                learn_recurrent_hidden_states = ail["rnn_hxs"]

        return (
            action,
            explored,
            value,
            action_log_prob,
            recurrent_hidden_states,
            invalid,
            goal,
            learn_value,
            learn_action_log_prob,
            learn_recurrent_hidden_states,
            plan_mask,
            agent_info_control,
        )

    def collect_one_step(self, step, step_total):
        """
        This needs to be aware of both high and low level experience, as well as internal and external rewards. 
        High level experience is: S,G,S',R
        Low level experience is: s,a,s',r
        every frame always goes into low level experience, and rewards are calculated as external + completion bonus.

        Will always be a single pass, as low level actions are always generated. What about when there is no valid plan?

        What does the interface between this and learn_plan_policy need to be?
            - action selected
            - goal selected (if any) - if the goal is selected, then add last frames observation to the buffer. 
            - invalid goal selected - if so, penalise learner (potentially)
            - symbolic-step timeout (true/false) - add negative reward to learner
            - symbolic-step completed (true/false) - add positive reward to controller
            - agent info / explored for both high and low levels. 
        """

        with torch.no_grad():
            results = self._policy.get_action(
                self.obs, valid_envs=[True] * self.num_processes
            )

        (
            action,
            explored,
            value,
            action_log_prob,
            recurrent_hidden_states,
            invalid,
            goal,
            learn_value,
            learn_action_log_prob,
            learn_recurrent_hidden_states,
            plan_mask,
            agent_info_control,
        ) = self.parallelise_results(results)

        # value = agent_info_control["value"]
        # action_log_prob = agent_info_control["probs"]
        # recurrent_hidden_states = agent_info_control["rnn_hxs"]

        # agent_info_learn = agent_info_control.pop("agent_info_learn")

        # if there was any planning that step:

        if any(plan_mask):
            # handle invalid plans if desired
            self._learn_rollouts.action_insert(
                learn_recurrent_hidden_states,
                goal,
                learn_action_log_prob,
                learn_value,
                plan_mask,
            )

        # Observe reward and next obs
        raw_obs, reward, done, infos = self._env.step(ptu.get_numpy(action))
        if self._render:
            self._env.render(**self._render_kwargs)

        # perform observation conversions
        action_obs = self._convert_to_torch(raw_obs)
        stored_obs = _flatten_tuple(action_obs) if self.is_tuple else action_obs
        augmented_obs = torch.cat((stored_obs, self.symbolic_actions), axis=1)
        self.obs = raw_obs

        # check to see whether symbolic actions are complete
        step_timeout, step_complete, plan_ended = self._policy.check_action_status(
            self.obs.squeeze(1)
        )
        # print("plan ended", plan_ended)
        # print("reward shape", self.cumulative_reward.shape)

        self.log_step(
            action,
            goal,
            agent_info_control,
            reward,
            done,
            step_complete,
            step_timeout,
            plan_ended,
        )

        internal_reward = (reward / 10) + np.array(step_complete) * 1

        self.discounts *= self.gamma
        self.plan_length += 1
        self.step_length += 1
        # TODO: Revisit penalty for timed out plans
        self.cumulative_reward += (
            reward * self.discounts
        )  # - np.array(step_timeout) * 1

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
        )
        self._policy.reset_selected(done)

        internal_reward = torch.from_numpy(internal_reward).unsqueeze(dim=1).float()
        environment_reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        self._rollouts.insert(
            augmented_obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            internal_reward,
            masks,
            bad_masks,
        )
        observation_needed = np.logical_or(plan_ended, done)
        if np.any(observation_needed):
            self._learn_rollouts.observation_insert(
                stored_obs,
                torch.from_numpy(self.cumulative_reward).unsqueeze(dim=1).float(),
                masks,
                bad_masks,
                observation_needed,
                plan_length=torch.from_numpy(self.plan_length).unsqueeze(dim=1).float(),
            )

        # TODO: this is doing logging of stats for tb.... will figure it out later

        # flat_ai = ppp.dict_of_list__to__list_of_dicts(agent_info_control, len(action))

        self.add_step(
            action, action_log_prob, environment_reward, done, value, agent_info_control
        )

        # reset plans
        self.cumulative_reward[observation_needed] = 0
        self.discounts[observation_needed] = 1
        self.plan_length[observation_needed] = 0
        self.step_length[observation_needed] = 0
        self.step_length[np.logical_or(step_complete, step_timeout)] = 0

    def log_step(
        self,
        act,
        goal,
        agent_info_control,
        reward,
        done,
        step_complete,
        step_timeout,
        plan_ended,
    ):

        # clean representations
        for i in range(self.num_processes):

            plan = self.clean_goal(goal[i])
            if plan is not None:
                self.plans[i] = plan
            else:
                plan = self.plans[i]

            step = self.clean_step(
                agent_info_control[i]["symbolic_action"][0]
            )  # or should the zero be i
            action = self.clean_action(act[i])

            # log plan stats
            if plan is not None:
                self.epoch_stats["plan"][plan]["reward"] += reward[i]
                if plan_ended[i] or done[i]:
                    self.epoch_stats["plan"][plan]["count"] += 1
                    self.epoch_stats["plan"][plan]["length"].append(self.plan_length[i])
                    if step_complete[i]:
                        self.epoch_stats["plan"][plan]["success"] += 1

            # log step stats
            if step is not None:
                self.epoch_stats["step"][step]["reward"] += reward[i]
                if step_complete[i] or step_timeout[i] or plan_ended[i] or done[i]:
                    self.epoch_stats["step"][step]["count"] += 1
                    self.epoch_stats["step"][step]["length"].append(self.step_length[i])
                    if step_complete[i]:
                        self.epoch_stats["step"][step]["success"] += 1

                # log action stats
                self.epoch_stats["action"][step][action] += 1

    def get_snapshot(self):
        return dict(
            env=self._env,
            controller=self._policy.controller.policy,
            learner=self._policy.learner,
        )

    def get_diagnostics(self):
        stats = super().get_diagnostics()

        for plan, values in self.epoch_stats["plan"].items():
            stats[f"plan/{plan}/count"] = values["count"]
            stats[f"plan/{plan}/success"] = (
                0 if values["count"] == 0 else values["success"] / values["count"]
            )
            stats[f"plan/{plan}/reward"] = (
                0 if values["count"] == 0 else values["reward"] / values["count"]
            )
            stats.update(
                create_stats_ordered_dict(f"plan/{plan}/length", values["length"])
            )

        for step, values in self.epoch_stats["step"].items():
            stats[f"step/{step}/count"] = values["count"]
            stats[f"step/{step}/success"] = (
                0 if values["count"] == 0 else values["success"] / values["count"]
            )
            stats[f"step/{step}/reward"] = (
                0 if values["count"] == 0 else values["reward"] / values["count"]
            )
            stats.update(
                create_stats_ordered_dict(f"step/{step}/length", values["length"])
            )
            total = sum([c for c in self.epoch_stats["action"][step].values()])
            for action, count in self.epoch_stats["action"][step].items():
                stats[f"action/{step}/{action}/percentage"] = (
                    0 if total == 0 else count / total
                )

        return stats

    def init_goal(self):
        return {"count": 0, "success": 0, "reward": 0, "length": []}

    def end_epoch(self, epoch):

        self.epoch_stats = {
            "plan": defaultdict(self.init_goal),
            "step": defaultdict(self.init_goal),
            "action": defaultdict(lambda: defaultdict(int)),
        }
        super().end_epoch(epoch)

    def clean_goal(self, goal):
        action = self._policy.env.convert_to_action(goal, None)

        if action["have"] is not None and action["move"] is not None:
            return "mixed"
        if action["have"] is not None:
            item, quantity = action["have"]
            return "have" + item
        if action["move"] is not None:
            return f"move ({action['move'][0]},{action['move'][1]})"
        if action["collect"] is not None:
            return f"collect coins"
        return None  # "noop"

    def clean_step(self, step):
        if step.sum().item() == 0:
            return None
        if step.shape == torch.Size([]):
            names = [
                None,
                "face tree",
                "face rock",
                "move",
                "move",
                "move",
                "move",
                "mine tree",
                "mine rock",
                "craft plank",
                "craft stick",
                "craft wooden_pickaxe",
                "craft stone_pickaxe",
                "collect",
            ]
            return names[int(step.item())]
        names = [
            "move",
            "collect",
            "face tree",
            "face rock",
            "mine tree",
            "mine rock",
            "craft plank",
            "craft stick",
            "craft wooden pickaxe",
            "craft stone pickaxe",
        ]
        index = step[2:].argmax().item()
        return names[index]

    def clean_action(self, action):
        return ACTIONS(action.item() + 1).name
