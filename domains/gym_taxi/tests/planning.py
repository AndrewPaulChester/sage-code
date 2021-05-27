import domains.gym_taxi
from domains.gym_taxi.simulator.taxi_world import TaxiWorldSimulator
import domains.gym_taxi.utils.representations as convert
from domains.gym_taxi.utils.config import FUEL

world = TaxiWorldSimulator(**FUEL)

js = world._get_state_json()

pddl = convert.json_to_pddl(js)

problem_file = "fd-test.pddl"
with open(problem_file, "w") as pf:  # overwrite old plan file
    pf.write(pddl)
