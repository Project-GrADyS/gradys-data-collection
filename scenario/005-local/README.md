# Training
The `main.py` file contains all code used in the training process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

# Results

## Varying sensor results

The training curves and saved models from the varying sensor
results trials are available in the `runs/varying_sensor_results`
folder. The results were then processed by:

- The `compare_varying_sensor_results.py`
script, which compares the varying sensor solution to the fixed sensor solution.
It generates the `varying_sensor_completion_times.png`, 
`varying_sensor_completion_times_2_agents.png`, 
`varying_sensor_completion_times_4_agents.png` and, 
`varying_sensor_completion_times_8_agents.png` files on the 
`visualization/varying_sensor` folder.
- The `plot_varying_sensor_results.py` script, which plots the collection times
for sensor count between 12 and 36 and generates. It generates the 
`varying_sensor_completion_times_line.png` file on the 
`visualization/varying_sensor` folder.


## Mismatched agent numbers

An experiment evaluating the strategy of applying models
to scenarios where the number of agents differ from training.
Script `test_mismatch.py` was used to generate the results.
It generated a spreadsheet on `visualization/mismatched_completion_times.xlsx`,
which was modified using Excel to generate two pivot tables,
providing the data used on the paper.
