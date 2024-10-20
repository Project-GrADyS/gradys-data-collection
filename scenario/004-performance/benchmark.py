import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3


experiments = []
#
# experiments.append([
#     "python", "main.py",
#     "--exp_name", "vary sensor",
#     "--run_name", "always 12",
#     "--num_drones", "2",
#     "--min_sensor_count", "12",
#     "--min_sensor_count", "12",
#     "--state_num_closest_drones", "1",
#     "--state_num_closest_sensors", "12",
#     "--min_sensor_priority", "1",
#     "--algorithm_iteration_interval", "0.5",
#     "--actor_learning_rate", "0.0001",
#     "--critic_learning_rate", "0.0001",
#     "--end_when_all_collected", "False",
#     "--scenario_size", "100",
#     "--num_actors", "5",
#     "--checkpoint_freq", "10000",
#     "--total_learning_steps", "500000"
# ])

experiments.append([
    "python", "main.py",
    "--exp_name", "vary agents",
    "--run_name", "between 1 and 2 active flag",
    "--min_drone_count", "1",
    "--max_drone_count", "2",
    "--min_sensor_count", "12",
    "--max_sensor_count", "12",
    "--state_num_closest_drones", "1",
    "--state_num_closest_sensors", "12",
    "--min_sensor_priority", "1",
    "--algorithm_iteration_interval", "0.5",
    "--actor_learning_rate", "0.0001",
    "--critic_learning_rate", "0.0001",
    "--end_when_all_collected", "False",
    "--scenario_size", "100",
    "--num_actors", "5",
    "--checkpoint_freq", "10000",
    "--total_learning_steps", "5000000"
])

print("Total experiments: ", len(experiments))


def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)

    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)
