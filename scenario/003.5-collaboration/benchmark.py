import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3


# Commands:
# python main.py --num-drones=2 --num-sensors=2 --run-name=centralized-critic --exp-name=yes --state_num_closest_sensors=2 --state-num-closest-drones=1 --min-sensor-priority=1 --centralized_critic --total-timesteps=10000000

# experiments = [
#     ["python", "main.py", "--num-drones=2", "--num-sensors=12", "--run-name=solution1", f"--exp-name=2-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=solution1", f"--exp-name=4-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=solution1", f"--exp-name=6-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
# ]

experiments = [
]

for repeat in range(3):
    for alpha in [0, 0.5, 0.8]:
        for beta in [0, 0.6, 1.2]:
            experiments.append(
                ["python", "main.py", f"--num-drones=2", f"--num-sensors=12", f"--run-name=alpha_{alpha}-beta_{beta}-repeat_{repeat}",
                 f"--exp-name=priority_replay", "--min-sensor-priority=1", "--total-timesteps=5000000",
                 "--checkpoint-freq=100000", "--algorithm-iteration-interval=2",
                 "--actor-learning-rate=0.00001", "--critic-learning-rate=0.00001",
                 f"--state-num-closest-drones=1", f"--state-num-closest-sensors=12",
                 "--reward=punish", "--no-end-when-all-collected",
                 f"--priority-alpha={alpha}", f"--priority-beta={beta}", "--use-priority"])

for repeat in range(3):
    experiments.append(
        ["python", "main.py", f"--num-drones=2", f"--num-sensors=12", f"--run-name=no_priority-repeat{repeat}",
         f"--exp-name=priority_replay", "--min-sensor-priority=1", "--total-timesteps=5000000",
         "--checkpoint-freq=100000", "--algorithm-iteration-interval=2",
         "--actor-learning-rate=0.00001", "--critic-learning-rate=0.00001",
         f"--state-num-closest-drones=1", f"--state-num-closest-sensors=12",
         "--reward=punish", "--no-end-when-all-collected"])

print("Total experiments: ", len(experiments))


def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)

    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)