import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3


experiments = []

experiments.append([
    "python", "main.py",
    "--exp_name", "learning rate",
    "--run_name", "big lr",
    "--num_drones", "2",
    "--num_sensors", "12",
    "--state_num_closest_drones", "1",
    "--state_num_closest_sensors", "12",
    "--min_sensor_priority", "1",
    "--algorithm_iteration_interval", "0.5",
    "--actor_learning_rate", "0.0001",
    "--critic_learning_rate", "0.0001",
    "--end_when_all_collected", "False",
    "--scenario_size", "100",
    "--use_remote", "False",
    "--num_actors", "5",
    "--checkpoint_freq", "10000",
    "--total_learning_steps", "5000000"
])

experiments.append([
    "python", "main.py",
    "--exp_name", "learning rate",
    "--run_name", "big lr with decay",
    "--num_drones", "2",
    "--num_sensors", "12",
    "--state_num_closest_drones", "1",
    "--state_num_closest_sensors", "12",
    "--min_sensor_priority", "1",
    "--algorithm_iteration_interval", "0.5",
    "--actor_learning_rate", "0.0001",
    "--critic_learning_rate", "0.0001",
    "--use-lr-decay", "True",
    "--end_when_all_collected", "False",
    "--scenario_size", "100",
    "--use_remote", "False",
    "--num_actors", "5",
    "--checkpoint_freq", "10000",
    "--total_learning_steps", "5000000"
])

experiments.append([
    "python", "main.py",
    "--exp_name", "learning rate",
    "--run_name", "normal lr",
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
    "--num_actors", "5",
    "--checkpoint_freq", "10000",
    "--total_learning_steps", "5000000"
])

experiments.append([
    "python", "main.py",
    "--exp_name", "learning rate",
    "--run_name", "even larger lr",
    "--num_drones", "2",
    "--num_sensors", "12",
    "--state_num_closest_drones", "1",
    "--state_num_closest_sensors", "12",
    "--min_sensor_priority", "1",
    "--algorithm_iteration_interval", "0.5",
    "--actor_learning_rate", "0.001",
    "--critic_learning_rate", "0.001",
    "--end_when_all_collected", "False",
    "--scenario_size", "100",
    "--use_remote", "False",
    "--num_actors", "5",
    "--checkpoint_freq", "10000",
    "--total_learning_steps", "5000000"
])

experiments.append([
    "python", "main.py",
    "--exp_name", "learning rate",
    "--run_name", "even larger lr with decay",
    "--num_drones", "2",
    "--num_sensors", "12",
    "--state_num_closest_drones", "1",
    "--state_num_closest_sensors", "12",
    "--min_sensor_priority", "1",
    "--algorithm_iteration_interval", "0.5",
    "--actor_learning_rate", "0.001",
    "--critic_learning_rate", "0.001",
    "--use-lr-decay", "True",
    "--end_when_all_collected", "False",
    "--scenario_size", "100",
    "--use_remote", "False",
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
