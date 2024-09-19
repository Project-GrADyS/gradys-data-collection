import multiprocessing
import subprocess
from dataclasses import dataclass

import tyro


@dataclass
class Args:
    concurrency: int = 3


experiments = [
["python","main.py","--exp_name","speed_bench","--run_name","pypy","--num_drones","20","--num_sensors","200","--scenario_size","1000","--use_pypy","True","--use_remote","True","--total_learning_steps","10000","--num_actors","7"],
]

print("Total experiments: ",  len(experiments))

def run_experiment(experiment):
    print("Running experiment: ", experiment)
    subprocess.run(experiment)


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    pool = multiprocessing.Pool(processes=args.concurrency)
    pool.map(run_experiment, experiments)
