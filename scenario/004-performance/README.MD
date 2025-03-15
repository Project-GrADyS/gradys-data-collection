# Training
The `main.py` file contains all code used in the training process for this scenario. The training process is configurable with several command line arguments. The `benchmark.py` file can be used to run several variations at the same time. The environment is defined inside `environment.py`, following the Pettingzoo specification.

# Results

## Unstable Asynchronous Training

The training curves and saved models that demonstrate the unstable training
results observed in the asynchronous training are available in the
`runs/unstable/` folder. The results were then processed by:

1. Using tensorboard to download the avg_reward curves saved as csv files
   on the `visualization/unstable_training/` folder.
2. Using the `visualization/unstable_training/unstable.py` script to generate the
   `unstable.png` file on the `visualization/unstable_training/` folder.