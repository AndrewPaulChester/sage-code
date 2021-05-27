import math

import torch
import torch.nn as nn

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import init
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import Policy, create_output_distribution


class WrappedPolicy(Policy):
    def __init__(
        self,
        obs_shape,
        action_space,
        device,
        base=None,
        base_kwargs=None,
        deterministic=False,
        dist=None,
        num_processes=1,
        obs_space=None,
        symbolic_action_size=0,
    ):
        super(WrappedPolicy, self).__init__(
            obs_shape,
            action_space,
            base,
            base_kwargs,
            dist,
            obs_space,
            symbolic_action_size,
        )
        self.deterministic = deterministic
        self.rnn_hxs = torch.zeros(num_processes, 1)
        self.masks = torch.ones(num_processes, 1)
        self.device = device

    def get_action(self, inputs, rnn_hxs=None, masks=None, valid_envs=None):
        # print(inputs.shape)
        # inputs = torch.from_numpy(inputs).float().to(self.device)

        if rnn_hxs is None:
            rnn_hxs = self.rnn_hxs
        if masks is None:
            masks = self.masks

        value, action, action_log_probs, rnn_hxs, probs = self.act(
            inputs, rnn_hxs, masks, self.deterministic
        )  # Need to be careful about rnn and masks - won't work for recursive

        agent_info = {
            "value": value,
            "probs": action_log_probs,
            "rnn_hxs": rnn_hxs,
            "dist": probs,
        }
        explored = action_log_probs < math.log(0.5)
        # return value, action, action_log_probs, rnn_hxs
        return (action, explored), agent_info

    def reset(self):
        pass


class MultiPolicy(WrappedPolicy):
    def __init__(
        self,
        obs_shape,
        action_space,
        device,
        num_options,
        base=None,
        base_kwargs=None,
        deterministic=False,
        dist=None,
        num_processes=1,
        obs_space=None,
    ):
        super(MultiPolicy, self).__init__(
            obs_shape,
            action_space,
            device,
            base,
            base_kwargs,
            deterministic,
            dist,
            num_processes,
            obs_space,
        )
        self.num_options = num_options
        self.dist = None
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critics = nn.ModuleList(
            [init_(nn.Linear(base._hidden_size, 1)) for i in range(num_options)]
        )
        self.dists = nn.ModuleList(
            [
                create_output_distribution(action_space, self.base.output_size)
                for i in range(num_options)
            ]
        )
        self.dists.to(device)
        self.critics.to(device)
        self.action = torch.zeros(num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_processes, 1).to(device)
        self.probs = torch.zeros(num_processes, action_space.n).to(device)

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """Expects last number in the fc to be the option index"""
        conv = inputs[0]
        fc = inputs[1]
        options = fc[:, -1].int().tolist()
        fc[:, -1] = 0
        inputs = (conv, fc)
        # calls CNN or MLP base. Rnn HXS is unused if not recurrent
        # value is critic output, actor features are actor output.
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist is output distribution (varies depending on action space)
        for i, o in enumerate(options):
            value[i] = self.critics[o](actor_features[i])
            dist = self.dists[o](actor_features[i])

            if deterministic:
                self.action[i] = dist.mode()
            else:
                self.action[i] = dist.sample()

            # print(i)
            # print(self.action_log_probs.shape)
            # print(self.action.shape)
            self.action_log_probs[i] = dist.log_probs(self.action[i])
            self.probs[i] = dist.get_probs()

        # print(action)
        return value, self.action, self.action_log_probs, rnn_hxs, self.probs

    def get_value(self, inputs, rnn_hxs, masks):
        options = inputs[:, -1].int().tolist()
        inputs[:, -1] = 0

        inputs = self._try_convert(inputs)
        value, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        for i, o in enumerate(options):
            value[i] = self.critics[o](actor_features[i])
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # given an sample from the replay buffer, returns the state value,
        # the action log prob (for the chosen action) and the entropy of the distribution
        options = inputs[:, -1].int().tolist()
        inputs[:, -1] = 0

        inputs = self._try_convert(inputs)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist_entropy = torch.zeros_like(value)
        action_log_probs = torch.zeros_like(value)
        for i, o in enumerate(options):
            value[i] = self.critics[o](actor_features[i])
            dist = self.dists[o](actor_features[i])

            action_log_probs[i] = dist.log_probs(action[i])
            dist_entropy[i] = dist.entropy().mean()

        return value, action_log_probs, dist_entropy.mean(), rnn_hxs
