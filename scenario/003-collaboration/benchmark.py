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
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=tau", "--exp-name=0.005", "--tau=0.005",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=tau", "--exp-name=0.05", "--tau=0.05",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=tau", "--exp-name=0.5", "--tau=0.05",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=policy-frequency", "--exp-name=1", "--policy-frequency=1",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=policy-frequency", "--exp-name=2", "--policy-frequency=2",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=policy-frequency", "--exp-name=3", "--policy-frequency=3",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=lr", "--exp-name=3e-3", "--learning_rate=3e-3",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=lr", "--exp-name=3e-4", "--learning_rate=3e-4",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
    ["python", "main.py", "--num-drones=2", "--num-sensors=2", "--run-name=lr", "--exp-name=3e-5", "--learning_rate=3e-5",  "--state_num_closest_sensors=2", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--centralized_critic", "--total-timesteps=10000000"],
]



def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)