import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

# Commands:
# python main.py --num-drones=1 --num-sensors=1 --state_num_closest_sensors=1 --state-num-closest-drones=0 --min-sensor-priority=1

# Write the commands above in array form
experiments = [
    ["python", "main.py", "--num-drones=1", "--num-sensors=2", "--state_num_closest_sensors=1", "--state-num-closest-drones=0", "--min-sensor-priority=1", "--exp-name", "1-drone-closest-1-sensor-0-drone"],
    ["python", "main.py", "--num-drones=1", "--num-sensors=2", "--state_num_closest_sensors=2", "--state-num-closest-drones=0", "--min-sensor-priority=1", "--exp-name", "1-drone-closest-2-sensor-0-drone"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--state_num_closest_sensors=1", "--state-num-closest-drones=0", "--min-sensor-priority=1", "--exp-name", "2-drone-closest-1-sensor-0-drone"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--state_num_closest_sensors=2", "--state-num-closest-drones=0", "--min-sensor-priority=1", "--exp-name", "2-drone-closest-2-sensor-0-drone"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--state_num_closest_sensors=1", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--exp-name", "2-drone-closest-1-sensor-1-drone"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--exp-name", "2-drone-closest-2-sensor-1-drone"],
]

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)