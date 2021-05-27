import numpy as np
import torch

import forks.rlkit.rlkit.torch.pytorch_util as ptu
from forks.rlkit.rlkit.core.eval_util import create_stats_ordered_dict
from forks.rlkit.rlkit.torch.dqn.dqn import DQNTrainer


class DoubleDQNTrainer(DQNTrainer):
    def train_from_torch(self, batch):
        rewards = batch["rewards"]
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

        best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs).detach()
        y_target = (
            rewards
            + (1.0 - terminals)
            * torch.pow(self.discount, plan_lengths)
            * target_q_values
        )
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        if self.huber_loss:
            y_target = torch.max(y_target, y_pred.sub(1))
            y_target = torch.min(y_target, y_pred.add(1))
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft target network updates
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
