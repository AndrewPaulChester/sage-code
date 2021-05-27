import gym
from torch import nn as nn
import os
import pickle
import numpy as np


from forks.rlkit.rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from forks.rlkit.rlkit.exploration_strategies.epsilon_greedy import (
    EpsilonGreedy,
    AnnealedEpsilonGreedy,
)
from forks.rlkit.rlkit.policies.argmax import ArgmaxDiscretePolicy
from forks.rlkit.rlkit.torch.dqn.dqn import DQNTrainer
from forks.rlkit.rlkit.torch.conv_networks import CNN
import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from forks.rlkit.rlkit.launchers.launcher_util import setup_logger
from forks.rlkit.rlkit.launchers import common
from forks.rlkit.rlkit.samplers.data_collector import MdpStepCollector, MdpPathCollector

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import utils
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import TransposeImage, make_vec_envs
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import CNNBase, create_output_distribution

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.policies import WrappedPolicy, MultiPolicy
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.trainers import PPOTrainer, MultiTrainer, DummyTrainer
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.data_collectors import (
    RolloutStepCollector,
    HierarchicalStepCollector,
    ThreeTierStepCollector,
)
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.algorithms import TorchIkostrikovRLAlgorithm
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import distributions

from gym_agent.learn_plan_policy import LearnPlanPolicy
from gym_agent.controller import CraftController, PretrainedController
from gym_agent.planner import ENHSPPlanner


def experiment(variant):
    common.initialise(variant)

    expl_envs, eval_envs = common.create_environments(variant)

    (
        obs_shape,
        obs_space,
        action_space,
        n,
        mlp,
        channels,
        fc_input,
    ) = common.get_spaces(expl_envs)

    # # CHANGE TO ORDINAL ACTION SPACE
    # action_space = gym.spaces.Box(-np.inf, np.inf, (8,))
    # expl_envs.action_space = action_space
    # eval_envs.action_space = action_space
    ANCILLARY_GOAL_SIZE = variant[
        "ancillary_goal_size"
    ]  # This is the length of the action space for the learner
    SYMBOLIC_ACTION_SIZE = 12
    GRID_SIZE = 31

    base = common.create_networks(variant, n, mlp, channels, fc_input)
    control_base = common.create_networks(
        variant, n, mlp, channels, fc_input + SYMBOLIC_ACTION_SIZE
    )  # for uvfa goal representation

    dist = common.create_symbolic_action_distributions(
        variant["action_space"], base.output_size
    )

    control_dist = distributions.Categorical(base.output_size, action_space.n)

    eval_learner = WrappedPolicy(
        obs_shape,
        action_space,
        ptu.device,
        base=base,
        deterministic=True,
        dist=dist,
        num_processes=variant["num_processes"],
        obs_space=obs_space,
    )

    planner = ENHSPPlanner()

    # collect

    filepath = "skills/collect.pkl"  # collect
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                policies = pickle.load(openfile)
            except EOFError:
                break

    loaded_collect_policy = policies["exploration/policy"]
    loaded_collect_policy.rnn_hxs = loaded_collect_policy.rnn_hxs[0].unsqueeze(0)

    eval_collect = CraftController(loaded_collect_policy, n=GRID_SIZE)

    expl_collect = CraftController(loaded_collect_policy, n=GRID_SIZE)

    # other
    filepath = "skills/move.pkl"  # other
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                policies = pickle.load(openfile)
            except EOFError:
                break

    loaded_other_policy = policies["exploration/policy"]
    loaded_other_policy.rnn_hxs = loaded_other_policy.rnn_hxs[0].unsqueeze(0)

    eval_other = CraftController(loaded_other_policy, n=GRID_SIZE)
    expl_other = CraftController(loaded_other_policy, n=GRID_SIZE)

    eval_controller = PretrainedController([eval_collect, eval_other])
    expl_controller = PretrainedController([expl_collect, expl_other])

    function_env = gym.make(variant["env_name"])

    eval_policy = LearnPlanPolicy(
        eval_learner,
        planner,
        eval_controller,
        num_processes=variant["num_processes"],
        vectorised=True,
        env=function_env,
    )

    expl_learner = WrappedPolicy(
        obs_shape,
        action_space,
        ptu.device,
        base=base,
        deterministic=False,
        dist=dist,
        num_processes=variant["num_processes"],
        obs_space=obs_space,
    )

    expl_policy = LearnPlanPolicy(
        expl_learner,
        planner,
        expl_controller,
        num_processes=variant["num_processes"],
        vectorised=True,
        env=function_env,
    )

    eval_path_collector = ThreeTierStepCollector(
        eval_envs,
        eval_policy,
        ptu.device,
        ANCILLARY_GOAL_SIZE,
        SYMBOLIC_ACTION_SIZE,
        max_num_epoch_paths_saved=variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=1,
        no_plan_penalty=True,
        meta_num_epoch_paths=variant["meta_num_steps"],
    )
    expl_path_collector = ThreeTierStepCollector(
        expl_envs,
        expl_policy,
        ptu.device,
        ANCILLARY_GOAL_SIZE,
        SYMBOLIC_ACTION_SIZE,
        max_num_epoch_paths_saved=variant["num_steps"],
        num_processes=variant["num_processes"],
        render=variant["render"],
        gamma=variant["trainer_kwargs"]["gamma"],
        no_plan_penalty=variant.get("no_plan_penalty", False),
        meta_num_epoch_paths=variant["meta_num_steps"],
    )

    learn_trainer = PPOTrainer(
        actor_critic=expl_policy.learner, **variant["trainer_kwargs"]
    )
    control_trainer = DummyTrainer()
    trainer = MultiTrainer([control_trainer, learn_trainer])

    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_envs)

    algorithm = TorchIkostrikovRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_envs,
        evaluation_env=eval_envs,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
        # batch_size,
        # max_path_length,
        # num_epochs,
        # num_eval_steps_per_epoch,
        # num_expl_steps_per_train_loop,
        # num_trains_per_train_loop,
        # num_train_loops_per_epoch=1,
        # min_num_steps_before_training=0,
    )

    algorithm.to(ptu.device)

    algorithm.train()
