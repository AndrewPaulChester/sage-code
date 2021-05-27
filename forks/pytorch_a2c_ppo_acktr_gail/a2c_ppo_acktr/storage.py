import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    """Houses experience buffer, general shape of (steps,processes,dimension). """

    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
    ):

        if isinstance(obs_shape[0], tuple):
            length = np.prod(obs_shape[0]) + obs_shape[1][0]
            self.obs = torch.zeros(num_steps + 1, num_processes, length)
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        # Why is this here? seems to cause problems with cross entropy
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.plan_length = torch.ones(num_steps, num_processes, 1)
        self.gamma = torch.ones(num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    @property
    def max_step(self):
        return self.step

    def to(self, device):
        """Sends all components of RolloutStorage to specified device"""
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.plan_length = self.plan_length.to(device)
        self.gamma = self.gamma.to(device)

    def insert(
        self,
        obs,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks,
        plan_length=None,
    ):
        """Inserts a new transition into the experience buffer"""
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if plan_length is not None:
            self.plan_length[self.step].copy_(plan_length)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """Resets experience buffer to start with current observation??"""
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True
    ):
        """ Calculates the discounted returns at each time step from rewards"""
        self.gamma.fill_(gamma)
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + torch.pow(self.gamma, self.plan_length[step])
                        * self.value_preds[step + 1]
                        * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = (
                        delta
                        + torch.pow(self.gamma, self.plan_length[step])
                        * gae_lambda
                        * self.masks[step + 1]
                        * gae
                    )
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        (
                            self.returns[step + 1]
                            * torch.pow(self.gamma, self.plan_length[step])
                            * self.masks[step + 1]
                            + self.rewards[step]
                        )
                        * self.bad_masks[step + 1]
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
                    )
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + torch.pow(self.gamma, self.plan_length[step])
                        * self.value_preds[step + 1]
                        * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = (
                        delta
                        + torch.pow(self.gamma, self.plan_length[step])
                        * gae_lambda
                        * self.masks[step + 1]
                        * gae
                    )
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1]
                        * torch.pow(self.gamma, self.plan_length[step])
                        * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None, valid_indices=None
    ):
        """only used for PPO to get random minibatches of current epoch"""
        num_steps, num_processes = self.rewards.size()[0:2]
        if valid_indices is None:
            valid_indices = range(num_processes * num_steps)
        batch_size = len(valid_indices)
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    num_processes, num_steps, num_processes * num_steps, num_mini_batch
                )
            )
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(valid_indices), mini_batch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        """only used for PPO to get random minibatches of current epoch"""
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class OptionRollouts(RolloutStorage):
    # TODO: finish this class - consider if it can be merged with standard rollout storage
    # not required if we use uvfa
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
    ):
        super().__init__(
            num_steps,
            num_processes,
            obs_shape,
            action_space,
            recurrent_hidden_state_size,
        )

        self.options = torch.zeros(num_steps, num_processes, 1)

    def to(self, device):
        """Sends all components of RolloutStorage to specified device"""
        super().to(device)
        self.options = self.options.to(device)

    def insert(
        self,
        obs,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks,
        option,
        plan_length=None,
    ):
        """Inserts a new transition into the experience buffer"""
        super().insert(
            obs,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks,
            plan_length,
        )
        self.options[self.step + 1].copy_(option)


class AsyncRollouts(RolloutStorage):
    # TODO: finish this class - consider if it can be merged with standard rollout storage
    # need to implement modified compute_returns and feed_forward generator, and to consider
    # if after_update needs to zero out the buffers (or at least the step pointers)
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        recurrent_hidden_state_size,
    ):
        super().__init__(
            num_steps,
            num_processes,
            obs_shape,
            action_space,
            recurrent_hidden_state_size,
        )
        self.step = np.zeros(num_processes, dtype=np.int)
        self.action_ready = np.ones(num_processes)

    @property
    def max_step(self):
        return max(self.step)

    def action_insert(
        self,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        step_masks,
    ):

        # print("action insert")
        # print(f"action_ready {self.action_ready}")
        # print(f"step_masks {step_masks}")
        if min(self.action_ready[step_masks]) == 0:
            raise ValueError("inserting actions when observations expected")

        self.recurrent_hidden_states[
            self.step[step_masks] + 1, step_masks
        ] = self.recurrent_hidden_states[self.step[step_masks] + 1, step_masks].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step[step_masks], step_masks] = self.actions[
            self.step[step_masks], step_masks
        ].copy_(actions[step_masks])
        self.action_log_probs[
            self.step[step_masks], step_masks
        ] = self.action_log_probs[self.step[step_masks], step_masks].copy_(
            action_log_probs[step_masks]
        )
        self.value_preds[self.step[step_masks], step_masks] = self.value_preds[
            self.step[step_masks], step_masks
        ].copy_(value_preds[step_masks])

        self.action_ready[step_masks] = 0

    def observation_insert(
        self, obs, rewards, masks, bad_masks, step_masks, plan_length=None
    ):
        # print("observation insert")
        # print(f"action_ready {self.action_ready}")
        # print(f"step_masks {step_masks}")
        if max(self.action_ready[step_masks]) == 1:
            raise ValueError("inserting observations when actions expected")
        # print(step_masks)
        # print(self.step)
        # print(self.obs.shape)
        # print(obs.shape)
        # print(self.step[step_masks] + 1)
        # print(self.obs[self.step[step_masks] + 1, step_masks].shape)
        # print(obs[step_masks].shape)

        self.obs[self.step[step_masks] + 1, step_masks] = self.obs[
            self.step[step_masks] + 1, step_masks
        ].copy_(obs[step_masks])
        self.rewards[self.step[step_masks], step_masks] = self.rewards[
            self.step[step_masks], step_masks
        ].copy_(rewards[step_masks])
        self.masks[self.step[step_masks] + 1, step_masks] = self.masks[
            self.step[step_masks] + 1, step_masks
        ].copy_(masks[step_masks])
        self.bad_masks[self.step[step_masks] + 1, step_masks] = self.bad_masks[
            self.step[step_masks] + 1, step_masks
        ].copy_(bad_masks[step_masks])
        if plan_length is not None:
            self.plan_length[self.step[step_masks], step_masks] = self.plan_length[
                self.step[step_masks], step_masks
            ].copy_(plan_length[step_masks])

        self.step[step_masks] = (self.step[step_masks] + 1) % self.num_steps
        self.action_ready[step_masks] = 1

    def insert(
        self,
        obs,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks,
        step_masks,
        plan_length=None,
    ):
        self.action_insert(
            recurrent_hidden_states, actions, action_log_probs, value_preds, step_masks
        )
        self.observation_insert(obs, rewards, masks, bad_masks, step_masks, plan_length)

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None, valid_indices=None
    ):
        valid_indices = []
        offset = 0
        for step in self.step:
            valid_indices.extend(range(offset, offset + step))
            offset += self.num_steps
        return super().feed_forward_generator(
            advantages, num_mini_batch, mini_batch_size, valid_indices=valid_indices
        )

    def after_update(self):
        """Zeros out masks to ensure that"""
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        masks = self.masks[-1]
        self.masks = torch.zeros_like(self.masks)
        self.masks[0].copy_(masks)
