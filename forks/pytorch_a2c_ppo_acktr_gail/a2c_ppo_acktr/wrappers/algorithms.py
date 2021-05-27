import abc
import time
import gtimer as gt

from forks.pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.wrappers.data_collectors import RolloutStepCollector

from forks.rlkit.rlkit.core.rl_algorithm import BaseRLAlgorithm
from forks.rlkit.rlkit.data_management.replay_buffer import ReplayBuffer


class IkostrikovRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: RolloutStepCollector,
        evaluation_data_collector: RolloutStepCollector,
        replay_buffer: ReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        use_linear_lr_decay,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.use_linear_lr_decay = use_linear_lr_decay

        assert (
            self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop
        ), "Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop"

    def _train(self):
        self.training_mode(False)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True
        ):
            print(f"in train, with eval to go: {self.num_eval_steps_per_epoch}")
            for step in range(self.num_eval_steps_per_epoch):

                self.eval_data_collector.collect_one_step(
                    step, self.num_eval_steps_per_epoch
                )
            gt.stamp("evaluation sampling")
            print("done with eval")

            for _ in range(self.num_train_loops_per_epoch):
                # this if check could be moved inside the function
                if self.use_linear_lr_decay:
                    # decrease learning rate linearly
                    self.trainer.decay_lr(epoch, self.num_epochs)

                for step in range(self.num_expl_steps_per_train_loop):
                    self.expl_data_collector.collect_one_step(
                        step, self.num_expl_steps_per_train_loop
                    )
                    # time.sleep(1)

                gt.stamp("exploration sampling", unique=False)

                rollouts = self.expl_data_collector.get_rollouts()
                gt.stamp("data storing", unique=False)
                self.training_mode(True)
                self.trainer.train(rollouts)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

    def evaluate(self):
        self._start_epoch = 0
        self.training_mode(False)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs), save_itrs=True
        ):

            for step in range(self.num_eval_steps_per_epoch):
                self.eval_data_collector.collect_one_step(
                    step, self.num_eval_steps_per_epoch
                )
            gt.stamp("evaluation sampling")

            self._end_epoch(epoch)


class TorchIkostrikovRLAlgorithm(IkostrikovRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


# class ThreeTierRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
#     def __init__(
#         self,
#         high_trainer,
#         low_trainer,
#         exploration_env,
#         evaluation_env,
#         exploration_data_collector: RolloutStepCollector,
#         evaluation_data_collector: RolloutStepCollector,
#         replay_buffer: ReplayBuffer,
#         batch_size,
#         max_path_length,
#         num_epochs,
#         num_eval_steps_per_epoch,
#         num_expl_steps_per_train_loop,
#         num_trains_per_train_loop,
#         use_linear_lr_decay,
#         num_train_loops_per_epoch=1,
#         min_num_steps_before_training=0,
#     ):
#         #TODO: Need to be careful about the trainer call in logging, might either need to alter base methods or put two trainers in one?
#         super().__init__(
#             trainer,
#             exploration_env,
#             evaluation_env,
#             exploration_data_collector,
#             evaluation_data_collector,
#             replay_buffer,
#         )
#         self.batch_size = batch_size
#         self.max_path_length = max_path_length
#         self.num_epochs = num_epochs
#         self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
#         self.num_trains_per_train_loop = num_trains_per_train_loop
#         self.num_train_loops_per_epoch = num_train_loops_per_epoch
#         self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
#         self.min_num_steps_before_training = min_num_steps_before_training
#         self.use_linear_lr_decay = use_linear_lr_decay

#         assert (
#             self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop
#         ), "Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop"

#     def _train(self):
#         self.training_mode(False)

#         for epoch in gt.timed_for(
#             range(self._start_epoch, self.num_epochs), save_itrs=True
#         ):
#             print(f"in train, with eval to go: {self.num_eval_steps_per_epoch}")
#             for step in range(self.num_eval_steps_per_epoch):

#                 self.eval_data_collector.collect_one_step(
#                     step, self.num_eval_steps_per_epoch
#                 )
#             gt.stamp("evaluation sampling")
#             print("done with eval")

#             for _ in range(self.num_train_loops_per_epoch):
#                 # this if check could be moved inside the function
#                 if self.use_linear_lr_decay:
#                     # decrease learning rate linearly
#                     self.trainer.decay_lr(epoch, self.num_epochs)

#                 for step in range(self.num_expl_steps_per_train_loop):
#                     self.expl_data_collector.collect_one_step(
#                         step, self.num_expl_steps_per_train_loop
#                     )
#                     gt.stamp("data storing", unique=False)

#                 rollouts = self.expl_data_collector.get_rollouts()
#                 self.training_mode(True)
#                 self.trainer.train(rollouts)
#                 gt.stamp("training", unique=False)
#                 self.training_mode(False)

#             self._end_epoch(epoch)

#     def evaluate(self):
#         self._start_epoch = 0
#         self.training_mode(False)

#         for epoch in gt.timed_for(
#             range(self._start_epoch, self.num_epochs), save_itrs=True
#         ):

#             for step in range(self.num_eval_steps_per_epoch):
#                 self.eval_data_collector.collect_one_step(
#                     step, self.num_eval_steps_per_epoch
#                 )
#             gt.stamp("evaluation sampling")

#             self._end_epoch(epoch)
