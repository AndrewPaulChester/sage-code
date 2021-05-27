from gym.spaces import Discrete
from domains.gym_taxi.utils.spaces import Json

from forks.rlkit.rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from forks.rlkit.rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, "info_sizes"):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
        )

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        if isinstance(self._action_space, Discrete) and not isinstance(
            self._ob_space, Json
        ):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


class PlanReplayBuffer(EnvReplayBuffer):
    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        # print(path["rewards"].mean())
        for (
            i,
            (
                obs,
                action,
                explored,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info,
                plan_length,
            ),
        ) in enumerate(
            zip(
                path["observations"],
                path["actions"],
                path["explored"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
                path["plan_lengths"],
            )
        ):
            self.add_sample(
                observation=obs,
                action=action,
                explored=explored,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                plan_length=plan_length,
            )
        self.terminate_episode()

    def add_sample(
        self,
        observation,
        action,
        explored,
        reward,
        terminal,
        next_observation,
        env_info,
        plan_length,
        **kwargs
    ):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action

        self._observations[self._top] = observation.reshape(-1)
        self._actions[self._top] = new_action
        self._explored[self._top] = explored
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._plan_lengths[self._top] = plan_length
        self._next_obs[self._top] = next_observation.reshape(-1)

        # for key in self._env_info_keys:
        #     self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            explored=self._explored[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            plan_lengths=self._plan_lengths[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch
