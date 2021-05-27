from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLearner(metaclass=ABCMeta):
    """Base learner class that performs learning and acting.
    
    Designed to be extended with full functionality.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, ob, reward, done):
        pass


class QLearner(BaseLearner):
    """Base learner class that performs learning and acting.
    
    """

    def __init__(self, action_space, observation_space):
        # stores q values for each input, action pair.
        self.qtable = np.zeros((observation_space, action_space), dtype=np.float32)
        # self.qtable = joblib.load("tmp/archive/qtable_330000.joblib.pkl")
        self.prev_action = None
        self.prev_state = None
        self.learning_rate = 0.1
        self.discount = 0.99
        self.exploration_rate = 0.1

    def step(self, observation, reward, done):
        """Performs online qlearning over the provided observation
        
        :param observation: the current environment observation
        :param reward: the reward from the previous timestep

        :returns: The action to be taken by the agent
        """

        # argmax over the current observation
        action_code = np.argmax(self.qtable[observation])
        # Get value from current state
        nextq = self.qtable[observation][
            action_code
        ]  # take max value in qtable for the new state
        # Update qtable given the current results
        if self.prev_state is not None:
            index = (self.prev_state, self.prev_action)
            self.qtable[index] = (1 - self.learning_rate) * self.qtable[
                index
            ] + self.learning_rate * (reward + self.discount * nextq)

        action = _action_decode(action_code)
        # explore with probability epsilon
        if np.random.rand() < self.exploration_rate:
            empty, delivered, location = np.random.randint(0, 2, 3)
            action = {
                "empty": bool(empty),
                "delivered": bool(delivered),
                "location": None,
            }
            if location:
                x, y = np.random.randint(0, 5, 2)  # TODO: magic number
                action["location"] = (x, y)

        # store action and state
        self.prev_state = observation
        self.prev_action = _action_encode(action)
        # return action
        return action


def _action_encode(action):  # TODO: magic number
    empty = 2 if action["empty"] else 0
    delivered = 1 if action["delivered"] else 0
    if action["location"] is None:
        location = 25
    else:
        x, y = action["location"]
        location = 5 * y + x

    return (empty + delivered) * 26 + location


def _action_decode(action_code):  # TODO: magic number
    location = action_code % 26
    action_code = action_code // 26
    delivered = bool(action_code % 2)
    empty = bool(action_code // 2)

    action = {"empty": empty, "delivered": delivered, "location": None}
    if location != 25:
        x = location % 5
        y = location // 5
        action["location"] = (x, y)
    return action


# test action encodings.
# for i in range(104):
#   assert i == _action_encode(_action_decode(i)), f"conversion {i} failed"
