import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

# Commands:
# python main.py --num-drones=2 --num-sensors=2 --run-name=centralized-critic --exp-name=yes --state_num_closest_sensors=2 --state-num-closest-drones=1 --min-sensor-priority=1 --centralized_critic --total-timesteps=10000000

# Write the commands above in array form
# experiments = [
#     ["python", "main.py", "--num-drones=2", "--num-sensors=12", "--run-name=solution1", f"--exp-name=2-heuristics", "--state_num_closest_sensors=12", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics"],
#     ["python", "main.py", "--num-drones=2", "--num-sensors=12", "--run-name=solution1", f"--exp-name=2-learning", "--state_num_closest_sensors=12", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000", ],
#     ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=solution1", f"--exp-name=4-heuristics", "--state_num_closest_sensors=12", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics"],
#     ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=solution1", f"--exp-name=4-learning", "--state_num_closest_sensors=12", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000", ],
#     ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=solution1", f"--exp-name=6-heuristics", "--state_num_closest_sensors=12", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics"],
#     ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=solution1", f"--exp-name=6-learning", "--state_num_closest_sensors=10", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000", ],
# ]

experiments = [
    ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=optimized", f"--exp-name=6-learning", "--state_num_closest_sensors=8", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=50000000", "--checkpoint-freq=100000", "--punish-reward"],
]

print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)