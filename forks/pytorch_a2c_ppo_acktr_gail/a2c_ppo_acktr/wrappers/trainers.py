from collections import OrderedDict
import torch

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.algo.a2c_acktr import A2C_ACKTR
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.algo.ppo import PPO
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import utils
from forks.rlkit.rlkit.torch.torch_rl_algorithm import TorchTrainer
from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.storage import AsyncRollouts


class A2CTrainer(A2C_ACKTR, TorchTrainer):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        use_gae,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        lr=None,
        eps=None,
        alpha=None,
        max_grad_norm=None,
        acktr=False,
    ):
        super(A2CTrainer, self).__init__(
            actor_critic,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            alpha,
            max_grad_norm,
            acktr,
        )
        # unclear if these are actually used
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.initial_lr = lr

    def decay_lr(self, epoch, num_epochs):
        utils.update_linear_schedule(self.optimizer, epoch, num_epochs, self.initial_lr)

    def train(self, batch):
        self._num_train_steps += 1
        self.train_from_torch(batch)

    def train_from_torch(self, batch):

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                batch.obs[-1], batch.recurrent_hidden_states[-1], batch.masks[-1]
            ).detach()

        # compute returns
        batch.compute_returns(
            next_value,
            self.use_gae,
            self.gamma,
            self.gae_lambda,
            self.use_proper_time_limits,
        )
        # update agent - return values are only diagnostic
        value_loss, action_loss, dist_entropy = self.update(batch)
        # TODO: add loss + entropy to eval_statistics.

        # re-initialise experience buffer with current state
        batch.after_update()

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor_critic]

    def get_snapshot(self):
        return dict(actor_critic=self.actor_critic)


# TODO: merge A2C + PPO trainers
class PPOTrainer(PPO, TorchTrainer):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        use_gae,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        lr=None,
        eps=None,
        clip_param=None,
        ppo_epoch=None,
        num_mini_batch=None,
        max_grad_norm=None,
        acktr=False,
    ):
        super(PPOTrainer, self).__init__(
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
        )
        # unclear if these are actually used
        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits
        self.initial_lr = lr

    def decay_lr(self, epoch, num_epochs):
        utils.update_linear_schedule(self.optimizer, epoch, num_epochs, self.initial_lr)

    def train(self, batch):
        self._num_train_steps += 1
        self.train_from_torch(batch)

    def train_from_torch(self, batch):

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                batch.obs[-1], batch.recurrent_hidden_states[-1], batch.masks[-1]
            ).detach()

        # compute returns
        batch.compute_returns(
            next_value,
            self.use_gae,
            self.gamma,
            self.gae_lambda,
            self.use_proper_time_limits,
        )
        # update agent - return values are only diagnostic
        value_loss, action_loss, dist_entropy = self.update(batch)
        # TODO: add loss + entropy to eval_statistics.

        # re-initialise experience buffer with current state
        batch.after_update()

        """
        Save some statistics for eval
        """
        print(
            f"value loss {value_loss}, action loss {action_loss}, dist_entropy {dist_entropy}"
        )
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics["Value Loss"] = value_loss
            self.eval_statistics["Action Loss"] = action_loss
            self.eval_statistics["Distribution Entropy"] = dist_entropy

    def get_diagnostics(self):
        self.eval_statistics["num_train_steps"] = self._num_train_steps
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.actor_critic]

    def get_snapshot(self):
        return dict(actor_critic=self.actor_critic)


class MultiTrainer:
    def __init__(self, trainers):
        self.trainers = trainers

    def decay_lr(self, epoch, num_epochs):
        for trainer in self.trainers:
            trainer.decay_lr(epoch, num_epochs)

    def train(self, batches):
        for trainer, batch in zip(self.trainers, batches):
            if isinstance(batch, AsyncRollouts):
                if batch.max_step == batch.num_steps - 1:
                    trainer.train(batch)
            else:
                trainer.train(batch)

    def get_diagnostics(self):

        diagnostics = {}
        for k, v in self.trainers[0].get_diagnostics().items():
            diagnostics["controller/" + k] = v

        for k, v in self.trainers[1].get_diagnostics().items():
            diagnostics["learner/" + k] = v

        return diagnostics

        # return self.trainers[0].eval_statistics

    def end_epoch(self, epoch):
        for trainer in self.trainers:
            trainer.end_epoch(epoch)

    @property
    def networks(self):
        n = []
        for trainer in self.trainers:
            n.extend(trainer.networks)
        return n

    def get_snapshot(self):

        snapshot = {}
        for k, v in self.trainers[0].get_snapshot().items():
            snapshot["controller/" + k] = v

        for k, v in self.trainers[1].get_snapshot().items():
            snapshot["learner/" + k] = v

        return snapshot
        # return [
        #     trainer.get_snapshot() for trainer in self.trainers
        # ]  # this returns a list which may break things


class DummyTrainer(TorchTrainer):
    def __init__(self, *args, **kwargs):
        pass

    def train(self, batch):
        pass

    def train_from_torch(self, batch):
        pass

    def get_snapshot(self):
        return {}

    @property
    def networks(self):
        return []

    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def decay_lr(self, epoch, num_epochs):
        pass
