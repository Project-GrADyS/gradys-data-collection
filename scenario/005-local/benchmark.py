import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

experiments = [
]

for agents in [4]:
    for sensors in [(12, 36)]:
        for local_obs in [True]:
            experiments.append(["python", "main.py", f"--num-drones={agents}", f"--min-num-sensors={sensors[0]}",
                "--exp-name=time_on_state", f"--max-num-sensors={sensors[1]}",
                f"--run-name={agents}a {sensors}s ", "--min-sensor-priority=1",
                "--total-timesteps=100000000",
                "--checkpoint-freq=1000000", "--algorithm-iteration-interval=2",
                "--actor-learning-rate=0.00001", "--critic-learning-rate=0.00001",
                f"--state-num-closest-drones={agents-1}", f"--state-num-closest-sensors=12",
                "--reward=punish", "--no-end-when-all-collected", "--no-speed-action",
                "--local-observation" if local_obs else "--no-local-observation",
                "--critic-global-state"
                ])


print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)