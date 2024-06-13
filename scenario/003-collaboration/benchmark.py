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
]

excluded_combinations = []

for tau in [0.005, 0.0005]:
    excluded_combinations.append([tau, 2, 3e-4])

for policy_frequency in [2, 3]:
    excluded_combinations.append([0.005, policy_frequency, 3e-4])

for lr in [3e-4, 3e-5, 3e-6]:
    excluded_combinations.append([0.005, 2, lr])

for tau in [0.005, 0.0005]:
    for policy_frequency in [2, 3]:
        for lr in [3e-4, 3e-5, 3e-6]:
            if [tau, policy_frequency, lr] not in excluded_combinations:
                experiments.append(
                    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=parameter", f"--exp-name=tau={tau}--policy-frequency={policy_frequency}--lr={lr}", f"--tau={tau}", f"--policy-frequency={policy_frequency}", f"--learning-rate={lr}",   "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=1000000"]
                )

print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)