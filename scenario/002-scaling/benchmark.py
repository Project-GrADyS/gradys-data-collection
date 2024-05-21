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
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=6-sensor-soft", "--no-checkpoint-visual-evaluation", "--run-name=StallFix", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--max_seconds_stalled=60"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=control", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=hard-reward", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--no-soft-reward"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=big-state", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--no-state_relative_positions"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=train_once", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--no-train_once_for_each_agent"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=punish-out-of-bounds", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--no-block_out_of_bounds"],
    ["python", "main.py", "--scenario_size=100", "--num-drones=1", "--num-sensors=6", "--exp_name=max-simulation_length", "--no-checkpoint-visual-evaluation", "--run-name=Ablation", "--checkpoint-freq=1000000", "--total-timesteps=10000000", "--max_episode_length=200", "--max_seconds_stalled=500"],

]

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)