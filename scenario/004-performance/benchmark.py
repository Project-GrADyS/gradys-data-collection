import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

# Commands:
# python -m Pyro5.nameserver
# experiments = [
#     ["python", "main.py", "--num-drones=2", "--num-sensors=12", "--run-name=solution1", f"--exp-name=2-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=1", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=4", "--num-sensors=12", "--run-name=solution1", f"--exp-name=4-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=3", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
#     ["python", "main.py", "--num-drones=6", "--num-sensors=12", "--run-name=solution1", f"--exp-name=6-random", "--state_num_closest_sensors=12", "--state-num-closest-drones=5", "--min-sensor-priority=1", "--total-timesteps=500000", "--checkpoint-freq=100000", "--use-heuristics=random"],
# ]

experiments = [
["python","main.py","--run-name=speed-bench","--exp-name=pypy","--use-heuristics=random","--num-drones=20","--num-sensors=200","--scenario-size=1000","--use-pypy", "--use-remote", "--total-timesteps=150000"],
["python","main.py","--run-name=speed-bench","--exp-name=no-pypy","--use-heuristics=random","--num-drones=20","--num-sensors=200","--scenario-size=1000","--no-use-pypy", "--use-remote", "--total-timesteps=150000"],
["python","main.py","--run-name=speed-bench","--exp-name=no-remote","--use-heuristics=random","--num-drones=20","--num-sensors=200","--scenario-size=1000", "--total-timesteps=150000"],
]

# for agents in [8, 4, 2]:
#     for sensors in [12, 24, 36]:
#         experiments.append(["python", "main.py", f"--num-drones={agents}", f"--num-sensors={sensors}", "--run-name=results",
#             f"--exp-name=a_{agents}-s_{sensors}", "--min-sensor-priority=1", "--total-timesteps=50000000",
#             "--checkpoint-freq=100000", "--algorithm-iteration-interval=2",
#             "--actor-learning-rate=0.00001", "--critic-learning-rate=0.00001",
#             f"--state-num-closest-drones={agents-1}", f"--state-num-closest-sensors=12",
#             "--reward=punish", "--no-end-when-all-collected"])


print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)