import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

# Commands:
# python main.py --num-drones=2 --num-sensors=2 --run-name=centralized-critic --exp-name=yes --state_num_closest_sensors=2 --state-num-closest-drones=1 --min-sensor-priority=1 --centralized_critic --total-timesteps=10000000

# experiments = [
#     ["python", "main.py", "--num-drones=2", "--num-sensors=12", "--run-name=solution1", f"--exp-name=2-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=solution1", f"--exp-name=4-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=solution1", f"--exp-name=6-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
# ]

experiments = [
    # ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=scenario2", f"--exp-name=4-drone-punish-with-reward", "--state_num_closest_sensors=8", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000", "--punish-reward"],
    # ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=scenario2", f"--exp-name=6-drone", "--state_num_closest_sensors=8", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000"],
    ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=bench", f"--exp-name=6-drone", "--state_num_closest_sensors=8", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=50000", "--checkpoint-freq=100000"],
]

print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)