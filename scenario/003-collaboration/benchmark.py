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
experiments = [
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=centralized", f"--exp-name=yes", f"--tau=0.005", f"--policy-frequency=3", f"--learning-rate=3e-5",   "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=centralized", f"--exp-name=no", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--no-centralized_critic", "--total-timesteps=10000000"]
]


print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)