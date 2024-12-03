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

min_agents, max_agents = (4, 4)
min_sensors, max_sensors = (12, 36)
alpha, beta = (0.7, 1)
lr = 1e-6

for trajectory in [True]:
    experiments.append([
        "python", "main.py",
        f"--min-num-drones={min_agents}", f"--max-num-drones={max_agents}",
        f"--min-num-sensors={min_sensors}", f"--max-num-sensors={max_sensors}",
        "--run-name=trajectory",
        f"--exp-name=4 agents trajectory on actor" if trajectory else f"--exp-name=4 agents",
        f"--use-distributional-critic",
        f"--use-phantom-agents",
        f"--critic-use-active-agents",
        f"--use-priority",
        f"--trajectory-length=8" if trajectory else f"--trajectory-length=1",
        f"--batch-size=2048" if trajectory else f"--batch-size=256",
        f"--priority-alpha={alpha}", f"--priority-beta={beta}",
        "--min-sensor-priority=1", "--total-timesteps=50000",
        "--checkpoint-freq=100000", "--algorithm-iteration-interval=0.5",
        f"--actor-learning-rate={lr}", f"--critic-learning-rate={lr}",
        f"--state-num-closest-drones=7", f"--state-num-closest-sensors=12",
        "--reward=punish", "--no-end-when-all-collected"])

print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)

    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)