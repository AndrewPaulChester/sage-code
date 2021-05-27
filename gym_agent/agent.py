import re
from domains.gym_taxi.utils.representations import (
    image_to_json,
    json_to_discrete,
    json_to_pddl,
    resize_image,
)
from domains.gym_taxi.utils.config import FIXED_GRID_SIZE, DISCRETE_ENVIRONMENT_STATES

from gym_agent.controller import TaxiController
from gym_agent.learner import QLearner
from gym_agent.planner import FDPlanner

SYMBOLIC_ACTION_COUNT = 4 * (FIXED_GRID_SIZE * FIXED_GRID_SIZE + 1)


class BaseAgent(object):
    """Base agent class that tracks environment stats.
    
    Designed to be extended with full functionality.
    """

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

        self.steps = 0
        self.episodes = 0
        self.reward = 0
        self.rewards = []

    def step(self, ob, reward, done):
        pass


class L2PAgent(BaseAgent):
    """Learn to plan agent coordinates learning and planning components.
    """

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)
        self.prev_action = None
        self.prev_state = None
        self.learning_rate = 0.1
        self.discount = 0.99
        self.exploration_rate = 0.1
        self.learner = QLearner(SYMBOLIC_ACTION_COUNT, DISCRETE_ENVIRONMENT_STATES)
        self.planner = FDPlanner()
        self.controller = TaxiController()
        self.actions = []
        self.r = 0

    def step(self, ob, reward, done):
        """Performs a single agent step which includes learning, planning and acting.
        
        :param ob: the current environment observation
        :param reward: the reward from the previous timestep
        :param done: indicates if the episode is over

        :returns: The action to be taken by the agent
        """
        self.r += reward
        # if no plan exists
        if len(self.actions) == 0:
            # preprocess observation
            js = image_to_json(resize_image(ob, (2 * FIXED_GRID_SIZE) - 1))
            discrete = json_to_discrete(js)
            # while no valid plan exists
            while len(self.actions) == 0:
                # get subgoal from learner
                subgoal = self.learner.step(discrete, self.r, done)

                # print(js)
                # print(subgoal)
                # produce planning problem given subgoal
                problem = generate_pddl(js, subgoal)
                # print(problem)
                # get plan from planner
                self.actions = self.planner.plan(problem)
                if len(self.actions) == 0:
                    self.r = -10
                    print("proposed invalid plan")

        self.r = 0
        # print(self.actions)
        # get atomic action from low-level controller for next action in sequence
        try:
            a = self.controller.step(self.actions.pop(0), ob)
        except IndexError:
            print("no plan found")
            # a = 0

        # perform learning

        # every 100 episodes write q-table to file
        if self.steps == 0 and self.episodes % 100 == 0:
            #     filename = "tmp/taxi_qtable_{}.joblib.pkl".format(self.steps)
            #     joblib.dump(self.planner.qtable, filename, compress=9)
            # plt.plot(self.rewards)
            # plt.show()
            pass

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

        # return action to environment
        return a, self.reward

