import numpy as np
import pandas as pd
import pickle
import sys
import argparse
import domains.gym_taxi.envs.taxi_env as taxi_env
import gym_agent.graph_planner as planner


class SDRLAgent(object):
    """An agent specifically for solving the Discrete taxi world task."""

    def __init__(
        self,
        action_space,
        observation_space,
        alpha=0.1,
        beta=0.01,
        alpha_final=0.01,
        anneal_episodes=200,
        run_id=0,
    ):
        self.run_id = run_id
        self.action_space = action_space.n
        self.obs = observation_space.n
        # stores q values for each input, action pair.
        self.rho_table = np.zeros(
            (observation_space.n, action_space.n), dtype=np.float32
        )
        self.unseen = np.ones((observation_space.n, action_space.n), dtype=np.uint8)
        self.r_table = np.zeros((observation_space.n, action_space.n), dtype=np.float32)
        self.plans = np.empty((observation_space.n), dtype=np.object)
        self.quality = np.zeros((observation_space.n), dtype=np.float32)
        self.prev_action = None
        self.prev_explored = False
        self.prev_state = None
        self.alpha = alpha
        self.alpha_final = alpha_final
        self.anneal_episodes = anneal_episodes
        self.increment = (alpha - alpha_final) / anneal_episodes
        self.beta = beta
        self.exploration_rate = 0.1
        self.steps = 0
        self.episodes = 0
        self.reward = 0
        self.passengers = 0
        self.rewards = []
        self.current_plan = []
        self.current_plan_step = 0

        with open("data/transitions.pickle", "rb") as handle:
            self.transitions = pickle.load(handle) #pre-calculated (s,a,s') transitions for taxi env

        self.planner = planner.Planner(self.transitions)

    def rlearn_tabular(self, observation, reward):
        """Performs online qlearning over the provided observation
        
        :param observation: the current environment observation
        :param reward: the reward from the previous timestep

        :returns: The action to be taken by the agent
        """

        # argmax over the current observation
        action = np.argmax(self.r_table[observation])
        # Get value from current state
        next_r = self.r_table[observation][action]
        next_r2 = np.max(self.r_table[observation])

        # take max value in r_table for the new state
        # Update qtable given the current results
        if self.prev_state is not None:
            index = (self.prev_state, self.prev_action)
            rho = self.rho_table[index]
            prev_max_r = np.max(self.r_table[observation])

            self.r_table[index] = (1 - self.alpha) * self.r_table[
                index
            ] + self.alpha * (reward - rho + next_r)

            prev_max_r = np.max(self.r_table[observation])

            if not self.prev_explored:
                self.rho_table[index] = (1 - self.beta) * self.rho_table[
                    index
                ] + self.beta * (reward - prev_max_r + next_r)
                self.unseen[index] = 0

        self.prev_explored = False
        self.prev_state = observation

    def act(self, ob, reward, done):
        # call parent class to update step numbers etc
        self.steps += 1
        self.reward += reward
        if reward > 0:
            self.passengers += 1

        # always update r-values
        self.rlearn_tabular(ob, reward)

        # check current plan to see if we are still going
        if self.current_plan_step < len(self.current_plan) - 1:
            self.current_plan_step += 1
            _, a = self.current_plan[self.current_plan_step]
            self.prev_action = a
            return a
        

        self.update_quality()
        # if not, create plan
        new_plan = self.planner.plan(ob, self.quality[ob], self.rho_table, self.unseen)

        if new_plan is None or new_plan == []:
            if self.plans[ob] is None:
                self.plans[ob] = []
            new_plan = self.plans[ob]
        else:
            self.plans[ob] = new_plan
        self.current_plan = new_plan
        self.current_plan_step = 0
        if self.current_plan:
            self.prev_action = self.current_plan[self.current_plan_step][1]
        else:
            self.prev_action = 7  # noop
        return self.prev_action

    def update_quality(self):
        if not self.current_plan:
            print("no plan to update")
            return
        s0 = self.current_plan[0][0]
        quality = 0
        for s, a in self.current_plan:
            quality += self.rho_table[s, a]
        self.quality[s0] = quality

    def episode_end(self):

        print(
            f"episde {self.episodes} finished. Steps: {self.steps}, reward: {self.reward}, passengers: {self.passengers}"
        )
        reward = self.reward
        episode = self.episodes
        steps = self.steps
        passengers = self.passengers
        self.rewards.append(self.reward)
        self.reward = 0
        self.episodes += 1
        self.steps = 0
        self.passengers = 0
        self.alpha = max(self.alpha - self.increment, self.alpha_final)

        return reward, episode, passengers


def run(
    run_id=1,
    alpha=0.1,
    alpha_final=0.1,
    anneal_schedule=200,
    beta=0.01,
    env_name="v1",
    seed=0,
):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})

    env = taxi_env.DiscretePredictableTaxiEnv(env_name)

    env.seed(seed)

    agent = SDRLAgent(
        env.action_space,
        env.observation_space,
        alpha,
        beta,
        alpha_final=alpha_final,
        anneal_episodes=anneal_schedule,
        run_id=run_id,
    )

    episode_count = 1000
    reward = 0
    done = False
    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                r, e, p = agent.episode_end()
                runlog_data = runlog_data.append(
                    {"metric": "passengers", "value": p, "step": e}, ignore_index=True
                )
                runlog_data = runlog_data.append(
                    {"metric": "reward", "value": r, "step": e}, ignore_index=True
                )
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        with open(f"data/{run_id}_data.csv", "w") as f:
            runlog_data.to_csv(f)
    # Close the env and write monitor result info to disk
    env.close()

    return runlog_data


def main(arglist):

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--run-id", type=int, help="ID for run")
    parser.add_argument("--env-name", help="Select the environment to run")
    parser.add_argument("--alpha", type=int, default=1, help="initial alpha value")
    parser.add_argument(
        "--anneal-schedule",
        type=float,
        default=200,
        help="number of episodes to linearly anneal over",
    )
    parser.add_argument("--beta", type=float, default=0.5, help="rho learning rate")
    parser.add_argument(
        "--alpha-final", type=float, default=0.01, help="final alpha value"
    )
    parser.add_argument("--seed", type=int, default=1, help="seed for run")

    args = parser.parse_args(arglist)

    run(
        args.run_id,
        args.alpha,
        args.alpha_final,
        args.anneal_schedule,
        args.beta,
        args.env_name,
        args.seed,
    )


if __name__ == "__main__":

    main(sys.argv[1:])
