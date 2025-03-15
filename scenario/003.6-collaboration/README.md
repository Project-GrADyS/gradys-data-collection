# Training
The `main.py` file contains all code used in the training process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

# Results

## Variable agent strategies

Several strategies for varying the number of agents were tested. The trained 
models and log files are stored in the `runs/agent_strats/`folder. 
Experiments testing each of them were compiled like such:

1. The average reward was downloaded using tensorboard and stored in the 
`visualization/variable agent strats/` folder
2. The `visualization/strats.py` script was used to generate the plot saved as
`visualization/variable agent strats/rewards.png`

## Variable agent results

Results from the best strat are saved in `runs/varying_agents/`, including the
saved models and log files. The results were analyzed following the steps:

1. The average reward and completion time data was downloaded using tensorboard
and stored in the `visualization/variable agent results/` folder
2. The `visualization/results.py` script was used to generate the
plots saved as `visualization/variable agent results/all_collected_rate.png` and
`visualization/variable agent results/completion_times.png`