import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3


experiments = []

for num_agents in [2, 4, 6, 8]:
    experiments.append([
        "python", "main.py",
        "--exp_name", "num_agents",
        "--run_name", f"{num_agents} agents",
        "--num_drones", "2",
        "--num_sensors", "12",
        "--state_num_closest_drones", "1",
        "--state_num_closest_sensors", "12",
        "--min_sensor_priority", "1",
        "--algorithm_iteration_interval", "0.5",
        "--actor_learning_rate", "0.00001",
        "--critic_learning_rate", "0.00001",
        "--end_when_all_collected", "False",
        "--scenario_size", "100",
        "--use_remote", "False",
        "--num_actors", f"{num_agents}",
        "--checkpoint_freq", "10000",
        "--total_learning_steps", "2500000"
    ])

print("Total experiments: ", len(experiments))


def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)

    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)
