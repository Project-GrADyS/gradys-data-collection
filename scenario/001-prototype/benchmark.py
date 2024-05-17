import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

experiments = [
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=1", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=2", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=3", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=4", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=5", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=7", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=8", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=9", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=10", "--exp_name=3-sensor-hard", "--max_episode_length=200", "--no-checkpoint-visual-evaluation", "--run-name=SingleDrone", "--checkpoint-freq=1000000", "--total-timesteps=1000000"],
]

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)