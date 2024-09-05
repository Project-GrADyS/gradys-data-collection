import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3

experiments = []
for num_actors in [1, 2, 4, 8]:
    for pypy in [True, False]:
        experiments.append(["python", "main.py", "--exp_name=bench", f"--run_name='num_actors={num_actors} pypy={pypy}'", f"--num_actors={num_actors}", 
         f"--use_remote={pypy}", f"--use_pypy={pypy}", f"--total_learning_steps=1000000"])

print("Total experiments: ", len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)