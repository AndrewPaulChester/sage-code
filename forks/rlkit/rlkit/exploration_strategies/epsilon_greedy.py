import random

from forks.rlkit.rlkit.exploration_strategies.base import RawExplorationStrategy


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            # print("sampling random action")
            return self.action_space.sample(), True
        return action, False


class AnnealedEpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(
        self,
        action_space,
        prob_random_action=1,
        anneal_rate=0.99998,
        min_prob_random_action=0.01,
    ):
        self.prob_random_action = prob_random_action
        self.action_space = action_space
        self.min_prob_random_action = min_prob_random_action
        self.anneal_rate = anneal_rate
        print(self.anneal_rate)

    def get_action_from_raw_action(self, action, **kwargs):
        self.prob_random_action = max(
            self.min_prob_random_action, self.prob_random_action * self.anneal_rate
        )
        if random.random() <= self.prob_random_action:
            # print("sampling random action")
            return self.action_space.sample(), True
        return action, False

    def anneal_epsilon(self):
        pass


class LinearEpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(
        self,
        action_space,
        prob_random_action=1,
        anneal_schedule=200,
        min_prob_random_action=0.1,
    ):
        self.prob_random_action = prob_random_action
        self.action_space = action_space
        self.min_prob_random_action = min_prob_random_action
        self.anneal_schedule = anneal_schedule
        self.decay = (prob_random_action - min_prob_random_action) / anneal_schedule

    def get_action_from_raw_action(self, action, **kwargs):

        if random.random() <= self.prob_random_action:
            # print("sampling random action")
            return self.action_space.sample(), True
        return action, False

    def anneal_epsilon(self):
        print(f"annealing {self.prob_random_action}")
        self.prob_random_action = max(
            self.min_prob_random_action, self.prob_random_action - self.decay
        )
