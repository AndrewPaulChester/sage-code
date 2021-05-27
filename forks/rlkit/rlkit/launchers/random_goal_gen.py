import gym
from torch import nn as nn

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
from forks.rlkit.rlkit.samplers.data_collector import MdpPathCollector
from forks.rlkit.rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from forks.rlkit.rlkit.samplers.rollout_functions import hierarchical_rollout

from domains.gym_taxi.utils.config import FIXED_GRID_SIZE, DISCRETE_ENVIRONMENT_STATES

SYMBOLIC_ACTION_COUNT = 4 * (FIXED_GRID_SIZE * FIXED_GRID_SIZE + 1)

from gym_agent.learn_plan_policy import LearnPlanPolicy


def experiment(variant):
    setup_logger("name-of-experiment", variant=variant)
    ptu.set_gpu_mode(True)

    expl_env = gym.make(variant["env_name"])
    eval_env = gym.make(variant["env_name"])
    obs_dim = expl_env.observation_space.image.shape[1]
    channels = expl_env.observation_space.image.shape[0]
    action_dim = SYMBOLIC_ACTION_COUNT
    symbolic_action_space = gym.spaces.Discrete(SYMBOLIC_ACTION_COUNT)
    symb_env = gym.make(variant["env_name"])
    symb_env.action_space = symbolic_action_space

    qf = CNN(
        input_width=obs_dim,
        input_height=obs_dim,
        input_channels=channels,
        output_size=action_dim,
        kernel_sizes=[8, 4],
        n_channels=[16, 32],
        strides=[4, 2],
        paddings=[0, 0],
        hidden_sizes=[256],
    )
    target_qf = CNN(
        input_width=obs_dim,
        input_height=obs_dim,
        input_channels=channels,
        output_size=action_dim,
        kernel_sizes=[8, 4],
        n_channels=[16, 32],
        strides=[4, 2],
        paddings=[0, 0],
        hidden_sizes=[256],
    )
    qf_criterion = nn.MSELoss()

    eval_policy = LearnPlanPolicy(None)
    expl_policy = LearnPlanPolicy(None)
    eval_path_collector = MdpPathCollector(
        eval_env, eval_policy, rollout=hierarchical_rollout, render=variant["render"]
    )
    expl_path_collector = MdpPathCollector(
        expl_env, expl_policy, rollout=hierarchical_rollout, render=variant["render"]
    )
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"]
    )
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], symb_env)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()
