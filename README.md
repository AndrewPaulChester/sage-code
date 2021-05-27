# Generating Symbolic Goals
This is the code for the submission of "SAGE: Generating Symbolic Goals for Myopic Models in Deep Reinforcement Learning".


## Installation
* Create a new conda virtual environment from sage.yml (conda env create -f sage.yml)
* For SAGE to operate in the Taxi and CraftWorld domains respectively, the [FastDownward](http://www.fast-downward.org/) and [ENHSP](https://gitlab.com/enricos83/ENHSP-Public) planners are required. Please create a planner folder and set an environment variable PLANNER_PATH pointing to it. Then follow their respective instructions to install them in subdirectories of that folder (see `gym_agent/planner.py` for details of how they are called)

## Credits
This code builds heavily on three key libraries:
* [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
* [rlkit](https://github.com/vitchyr/rlkit)
* [baselines](https://github.com/openai/baselines) - used for logging, not algorithm implementations.

Forked versions of these repositories used in this work are contained in the forks/ folder. Installation instructions for these libraries are left in their respective folders for completeness, but are not required for this project. Please simply follow the steps outlined in the installation section above.

## Training
The `train` script contains commands for reproducing the results of the paper. For commands with lists of taxi environments, make multiple seperate calls with one item per list.