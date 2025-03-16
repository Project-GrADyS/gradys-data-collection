# GrADyS Data Collection

This repository contains the ongoing research on RL-powered UAV data collection from the [GrADyS](https://www.lac.inf.puc-rio.br/index.php/gradys/) research group. 

## Organization

The repository is organized in scenarios. Each scenario presents experiments
evaluating some hypothesis, each is generally an evolution of the previous. Not
all scenarios generated results of value, only the ones that did will be listed
in this documentation.

### 002-scaling
Explored the applicability of MADDPG to the UAV data collection problem with 
scale comparable to real world scenarios. Documentation for this scenario
can be found [here](scenario/002-scaling/README.md).

### 003-collaboration
Explored the ability of MADDPG to learn a collaborative strategy for data
collection. Documentation for this scenario can be found
[here](scenario/003-collaboration/README.md).

### 003.6-generalization
A continuation of the previous scenario, this one explores the effects of
varying the number of agents in the environment. Documentation for this scenario
can be found [here](scenario/003.6-generalization/README.md).

### 004-performance
Explored the performance of an asynchronous training framework (Ape-X) for the 
UAV data collection problem. Documentation for this scenario can be found
[here](scenario/004-performance/README.md).

### 005-local
Optimization and refactoring of 003-collaboration, also including experiments
with varying number of sensors and local communication. Documentation for this
scenario can be found [here](scenario/005-local/README.md).