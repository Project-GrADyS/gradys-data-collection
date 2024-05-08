# Participants
Multiple UAVs and Sensors (defined by variables `num_drones` and `num_sensors`) 

# Objectives
UAVs must collect data from all sensors at least once. UAVs are constantly attempting to collect 
information from the space around them, so data is automatically collected once they get close
enough to a sensor. When all sensors are visited an episode ends.

# Constraints
- Communication has limited range
- UAV speed is limited (and fixed at 10m/s)
- All participants must be within a square area of parametrized side.
- Episode length is limited

# Algorithm
The algorithm used to solve this problem was DDPG, adapted 
from https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy. A single RL agent is trained 
using information from all UAVs (Centralized training with parameter sharing).

The agents were integrated into GrADyS-SIM-Nextgen. Iterations happen every 0.5 seconds of simulation time.

## State
Agents (UAVs) know all agent's positions (including their own), they know their ID, they know the
positions of all sensors and they know if sensors were visited or not.

## Action
Agents chose a direction to travel

## Reward
- 1 to all agents if all sensor data has been collected
- -1 to a specific agent if it leaves the scenario area
- 0 otherwize

# Training
Episodes start with UAVs being placed randomly in a small area in the center of the scenario. Sensors are 
placed randomly outside of that small area. The simulation runs until the time limit. If an agent leaves
the scenario's area the simulation is terminated. If all sensors have been visited, the simulation is
terminated.

# Parameters
These are the parameters being studied:
- Number of drones [1, 2]
- Number of sensors [1, 2]
- Scenario size [50, 100] (size of the side of the scenario square)
- Training time [1_000_000, 10_000_000]
