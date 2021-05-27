import numpy

import argparse
import sys

from sklearn.externals import joblib

import gym
import domains.gym_taxi
from gym import wrappers, logger

import matplotlib.pyplot as plt


class QAgent(object):
    """An agent specifically for solving the Discrete taxi world task."""

    def __init__(self, action_space, observation_space):
        self.action_space = action_space.n
        self.obs = observation_space.n
        # stores q values for each input, action pair.
        self.qtable = numpy.ones(
            (observation_space.n, action_space.n), dtype=numpy.float32
        )
        # self.qtable = joblib.load("tmp/archive/qtable_330000.joblib.pkl")
        self.prev_action = None
        self.prev_state = None
        self.learning_rate = 0.1
        self.discount = 0.99
        self.exploration_rate = 0.1
        self.steps = 0
        self.episodes = 0
        self.reward = 0
        self.rewards = []

    def qlearn_tabular(self, observation, reward):
        """Performs online qlearning over the provided observation
        
        :param observation: the current environment observation
        :param reward: the reward from the previous timestep

        :returns: The action to be taken by the agent
        """

        # argmax over the current observation
        action = numpy.argmax(self.qtable[observation])
        # Get value from current state
        nextq = self.qtable[observation][
            action
        ]  # take max value in qtable for the new state
        # Update qtable given the current results
        if self.prev_state is not None:
            index = (self.prev_state, self.prev_action)
            self.qtable[index] = (1 - self.learning_rate) * self.qtable[
                index
            ] + self.learning_rate * (reward + self.discount * nextq)

        # explore with probability epsilon
        if numpy.random.rand() < self.exploration_rate:
            action = numpy.random.randint(0, self.action_space)
        # store action and state
        self.prev_state = observation
        self.prev_action = action
        # return action
        return action

    def act(self, ob, reward, done):
        # every 100 episodes write q-table to file
        if self.steps == 0 and self.episodes % 100 == 0:
            filename = "data/taxi_qtable_{}.joblib.pkl".format(self.steps)
            joblib.dump(self.qtable, filename, compress=9)
            plt.plot(self.rewards)
            plt.show()

        # call parent class to update step numbers etc
        self.steps += 1

        self.reward += reward
        if done:
            print(
                f"episde {self.episodes} finished. Steps: {self.steps}, reward: {self.reward}"
            )
            self.rewards.append(self.reward)
            self.reward = 0
            self.episodes += 1
            self.steps = 0

        return self.qlearn_tabular(ob, reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "env_id",
        nargs="?",
        default="discrete-taxi-v0",
        help="Select the environment to run",
    )
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = "/tmp/Q-agent-results"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QAgent(env.action_space, env.observation_space)

    episode_count = 10000
    reward = 0
    done = False
    render = True
    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            if render:
                env.render()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
