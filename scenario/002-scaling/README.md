# Training
The `main.py` file contains all code used in the training process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

# Results
## Paths
The results of the training process are saved in the `runs` folder. 
The `runs/closest/closest-1-sensor-0-agent` training run was used to generate
a graph of UAV trajectories. The `plot_paths.py` script can be used to generate
the graph. The graph is saved in `exploratory-a4s8.png`.
