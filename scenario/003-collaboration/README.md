# Training
The `main.py` file contains all code used in the trainig process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

# Results
The results of the training process are saved in the `runs/results/` folder. 
All visualization and statistics below were generated from these results.

## Training Curves
The training data was processed to generate a graph with training curves in the
following manner:
1. The training data was loaded from the `runs/results/` folder.
2. Tensorboard was used to download the training data as csv files, which were
stored in the `statistics/curves/data/` folder.
3. The `statistics/curves/visualization.py` script was used to generate the graph, 
which is stored as `statistics/curves/rewards.png`
   
## Completion Times
Completion times were recorded using the `visualize_performance.py` script. 
This script runs several episodes of the environment and records the time 
taken to complete each episode. The results are stored in the 
`statistics/completion_times/` folder.

## Paths
The `plot_paths.py` script runs a single episode for each scenario variation, 
recording agent and sensor positions and saving them into plots stored in the
`statistics/paths/` folder.