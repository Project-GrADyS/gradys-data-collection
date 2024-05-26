import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

# Commands:
# python main.py --scenario_size=50 --num-drones=1 --num-sensors=1 --exp_name=1-sensor-1-drone-50-size --max_episode_length=60
# python main.py --scenario_size=50 --num-drones=1 --num-sensors=2 --exp_name=2-sensor-1-drone-50-size --max_episode_length=60
# python main.py --scenario_size=50 --num-drones=2 --num-sensors=1 --exp_name=1-sensor-2-drone-50-size --max_episode_length=60
# python main.py --scenario_size=50 --num-drones=2 --num-sensors=2 --exp_name=2-sensor-2-drone-50-size --max_episode_length=60
# python main.py --scenario_size=100 --num-drones=1 --num-sensors=1 --exp_name=1-sensor-1-drone-100-size --max_episode_length=60
# python main.py --scenario_size=100 --num-drones=1 --num-sensors=2 --exp_name=2-sensor-1-drone-100-size --max_episode_length=60
# python main.py --scenario_size=100 --num-drones=2 --num-sensors=1 --exp_name=1-sensor-2-drone-100-size --max_episode_length=60
# python main.py --scenario_size=100 --num-drones=2 --num-sensors=2 --exp_name=2-sensor-2-drone-100-size --max_episode_length=60

# Write the commands above in array form
experiments = [
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=4", "--exp_name=4-sensor", "--no-checkpoint-visual-evaluation", "--run-name=ExpNoise", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--exploration-noise=0.1"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=4", "--exp_name=4-sensor", "--no-checkpoint-visual-evaluation", "--run-name=ExpNoise", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--exploration-noise=0.01"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=4", "--exp_name=4-sensor", "--no-checkpoint-visual-evaluation", "--run-name=ExpNoise", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--exploration-noise=0.3"],
]

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)