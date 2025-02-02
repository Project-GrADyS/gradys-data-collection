# Training
The `main.py` file contains all code used in the trainig process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

In the `runs` folder you can see the saved most relevant results obtained in this scenario. The code in `plot_paths.py` can be used to create plots containing the UAV's trajectories. Reward visualizations were generated with the script `runs/results/visualization/visualization.py`. Performance comparisons with the greedy heuristic were created with the `visualize_performance.py` script.

# Results

## Varying sensor results

The training curves and saved models from the varying sensor
results trials are available in the `runs/varying_sensor_results`
folder. The results were then processed by the `test_varying_sensor_results.py`
script, which generates the graphs in `visuzalization/varying_sensor`
graph.