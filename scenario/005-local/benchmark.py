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
        for interval in [0.5]:
            experiments.append(["python3", "main.py", 
                f"--num-drones={agents}", 
                f"--min-num-sensors={sensors[0]}",
                f"--max-num-sensors={sensors[1]}",
                "--exp-name=local-benchmark", 
                f"--run-name={agents}a {sensors}s {interval} interval more memory", 
                "--buffer-size=10000000",
                "--min-sensor-priority=1",
                "--total-timesteps=100000000",
                "--checkpoint-freq=1000000",
                f"--algorithm-iteration-interval={interval}",
                "--actor-learning-rate=0.0001", 
                "--critic-learning-rate=0.0001",
                f"--state-num-closest-drones={agents-1}", 
                "--state-num-closest-sensors=12",
                "--reward=punish", 
                "--no-end-when-all-collected", 
                "--speed-action",
                "--local-observation",
                "--batch-size=512",
                "--critic-global-state",
                "--max-episode-length=800",
                "--max-seconds-stalled=80"
                ])


print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)