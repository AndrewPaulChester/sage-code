import gym
from torch import nn as nn
import os
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

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.policies import WrappedPolicy
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.trainers import PPOTrainer
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.data_collectors import (
    RolloutStepCollector,
    HierarchicalStepCollector,
)
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.algorithms import TorchIkostrikovRLAlgorithm
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import distributions

from gym_agent.learn_plan_policy import LearnPlanPolicy


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

    # CHANGE TO FACTORD ACTION SPACE
    action_space = gym.spaces.Box(-np.inf, np.inf, (10,))
    expl_envs.action_space = action_space
    eval_envs.action_space = action_space

    base = common.create_networks(variant, n, mlp, channels, fc_input)

    bernoulli_dist = distributions.Bernoulli(base.output_size, 4)
    continuous_dist = distributions.DiagGaussian(base.output_size, 6)
    dist = distributions.DistributionGeneratorTuple((bernoulli_dist, continuous_dist))

    eval_policy = LearnPlanPolicy(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=base,
            deterministic=True,
            dist=dist,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
        ),
        num_processes=variant["num_processes"],
        vectorised=True,
        json_to_screen=expl_envs.observation_space.converter,
    )
    expl_policy = LearnPlanPolicy(
        WrappedPolicy(
            obs_shape,
            action_space,
            ptu.device,
            base=base,
            deterministic=False,
            dist=dist,
            num_processes=variant["num_processes"],
            obs_space=obs_space,
        ),
        num_processes=variant["num_processes"],
        vectorised=True,
        json_to_screen=expl_envs.observation_space.converter,
    )

    eval_path_collector = HierarchicalStepCollector(
        eval_envs,
        eval_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["algorithm_kwargs"][
            "num_eval_steps_per_epoch"
        ],
        num_processes=variant["num_processes"],
        render=variant["render"],
    )
    expl_path_collector = HierarchicalStepCollector(
        expl_envs,
        expl_policy,
        ptu.device,
        max_num_epoch_paths_saved=variant["num_steps"],
        num_processes=variant["num_processes"],
        render=variant["render"],
    )
    # added: created rollout(5,1,(4,84,84),Discrete(6),1), reset env and added obs to rollout[step]

    trainer = PPOTrainer(actor_critic=expl_policy.learner, **variant["trainer_kwargs"])
    # missing: by this point, rollout back in sync.
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_envs)
    # added: replay buffer is new
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
    # missing: device back in sync
    algorithm.train()

