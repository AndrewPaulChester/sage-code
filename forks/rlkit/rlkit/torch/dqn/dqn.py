from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from forks.rlkit.rlkit.torch.torch_rl_algorithm import TorchTrainer


class DQNTrainer(TorchTrainer):
    def __init__(
        self,
        qf,
        target_qf,
        learning_rate=1e-3,
        soft_target_tau=1e-3,
        target_update_period=1,
        qf_criterion=None,
        discount=0.99,
        reward_scale=1.0,
        adam_eps=1e-5,
        naive_discounting=False,
        huber_loss=False,
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(), lr=self.learning_rate, eps=adam_eps
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.naive_discounting = naive_discounting
        self.huber_loss = huber_loss

    def train_from_torch(self, batch):
        rewards = batch["rewards"] * self.reward_scale
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        try:
            plan_lengths = batch["plan_lengths"]
            if self.naive_discounting:
                plan_lengths = torch.ones_like(plan_lengths)
        except KeyError as e:
            plan_lengths = torch.ones_like(rewards)

        """
        Compute loss
        """

        target_q_values = self.target_qf(next_obs).detach().max(1, keepdim=True)[0]
        y_target = (
            rewards
            + (1.0 - terminals)
            * torch.pow(self.discount, plan_lengths)
            * target_q_values
        )
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        # huber loss correction.
        if self.huber_loss:
            y_target = torch.max(y_target, y_pred.sub(1))
            y_target = torch.min(y_target, y_pred.add(1))
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        # for param in self.qf.parameters():  # introduced parameter clipping
        #     param.grad.data.clamp_(-1, 1)
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict("Y Predictions", ptu.get_numpy(y_pred))
            )

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.qf, self.target_qf]

    def get_snapshot(self):
        return dict(qf=self.qf, target_qf=self.target_qf)
