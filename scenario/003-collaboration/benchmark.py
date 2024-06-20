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
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=long", f"--exp-name=2-drones-2-sensors", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=4", "--run-name=long", f"--exp-name=2-drones-4-sensors", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=8", "--run-name=long", f"--exp-name=2-drones-8-sensors", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=4", "--num-sensors=8", "--run-name=more-agents", f"--exp-name=closest-2-sensors-1-drone", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=4", "--num-sensors=8", "--run-name=more-agents", f"--exp-name=closest-2-sensors-2-drone", "--state_num_closest_sensors=2", "--state-num-closest-drones=2", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=4", "--num-sensors=8", "--run-name=more-agents", f"--exp-name=closest-3-sensors-1-drone", "--state_num_closest_sensors=3", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=4", "--num-sensors=8", "--run-name=more-agents", f"--exp-name=closest-3-sensors-2-drone", "--state_num_closest_sensors=3", "--state-num-closest-drones=2", "--min-sensor-priority=1", "--total-timesteps=20000000"],
    ["python", "main.py", "--num-drones=4", "--num-sensors=8", "--run-name=more-agents", f"--exp-name=closest-3-sensors-3-drone", "--state_num_closest_sensors=3", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=20000000"],
]


print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)