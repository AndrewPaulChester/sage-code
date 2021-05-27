import numpy as np
from domains.gym_craft.utils.representations import json_to_screen

import torch

import forks.rlkit.rlkit.torch.pytorch_util as ptu


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(env, agent, max_path_length=np.inf, render=False, render_kwargs=None):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    explored = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    o = _flatten_tuple(_convert_to_torch(o, env))
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        (a, e), agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        next_o = _flatten_tuple(_convert_to_torch(next_o, env))
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        explored.append(e)
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        # ADDED THIS SECTION TO HANDLE INTERMEDIATE EXPERIENCE
        if "intermediate_experience" in env_info:
            path_length += len(env_info["intermediate_experience"])

        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        explored=np.array(explored).reshape(-1, 1),
        rewards=np.array(rewards, dtype=np.float32).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def _flatten_tuple(observation):
    """Assumes observation is a tuple of tensors. converts ((n,c, h, w),(n, x)) -> (n,c*h*w+x)"""
    image, fc = observation
    flat = image.flatten()
    return np.concatenate([flat, fc])


def _convert_to_torch(raw_obs, env):
    return env.observation_space.converter(raw_obs)


def intermediate_rollout(
    env,
    agent,
    restart=True,
    starting_obs=None,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    experience_interval=1,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    explored = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    if restart:
        o = env.reset()
        agent.reset()
    else:
        o = starting_obs
    next_o = None
    path_length = 0
    i = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        (a, e), agent_info = agent.get_action(o)
        try:
            a = a.item()
        except AttributeError:
            pass
        if isinstance(o, str):
            o = env.observation_space.converter(o)
        if isinstance(o, tuple):
            o = _flatten_tuple(o)
        next_o, r, d, env_info = env.step(a)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        explored.append(e)
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        if i % experience_interval == 0:
            path_length += 1
        i += 1
        step_timeout, step_complete, plan_ended = agent.check_action_status([next_o])
        if d or step_timeout[0] or plan_ended[0]:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    if isinstance(next_o, str):
        next_o_converted = env.observation_space.converter(next_o)
    if isinstance(next_o_converted, tuple):
        next_o_converted = _flatten_tuple(next_o_converted)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o_converted = np.array([next_o_converted])
    next_observations = np.vstack(
        (observations[1:, :], np.expand_dims(next_o_converted, 0))
    )
    return (
        dict(
            observations=observations,
            actions=actions,
            explored=np.array(explored).reshape(-1, 1),
            rewards=np.array(rewards, dtype=np.float32).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        ),
        (d, next_o),
    )


def hierarchical_rollout(
    env, agent, max_path_length=np.inf, render=False, render_kwargs=None
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    explored = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    cumulative_reward = 0
    first_time = True
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        (a, e), agent_info = agent.get_action(o, [0])
        next_o, r, d, env_info = env.step(a)
        if agent_info.get("subgoal") is not None:
            img = json_to_screen(o)
            observations.append(img)
            actions.append(agent_info["subgoal"])
            explored.append(e)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if not first_time:
                rewards.append(cumulative_reward)
                terminals.append(d)
            first_time = False
            cumulative_reward = 0

        cumulative_reward += r
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    rewards.append(cumulative_reward)
    terminals.append(d)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (observations[1:, :], np.expand_dims(json_to_screen(next_o), 0))
    )
    return dict(
        observations=observations,
        actions=actions,
        explored=np.array(explored).reshape(-1, 1),
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
