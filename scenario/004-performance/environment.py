from typing import Optional, Literal

import gymnasium
from gymnasium.spaces import Box
import numpy as np
from pettingzoo import ParallelEnv
from remote_environment import GradysRemoteEnvironment, StateMode
import subprocess
import sys
import Pyro5.api

class GrADySEnvironment(ParallelEnv):
    remote_env: GradysRemoteEnvironment

    metadata = {"render_modes": ["visual", "console"], "name": "gradys-env"}

    def __init__(self,
                 render_mode: Optional[Literal["visual", "console"]] = None,
                 algorithm_iteration_interval: float = 0.5,
                 num_drones: int = 1,
                 num_sensors: int = 2,
                 scenario_size: float = 100,
                 max_episode_length: float = 500,
                 max_seconds_stalled: int = 30,
                 communication_range: float = 20,
                 state_num_closest_sensors: int = 2,
                 state_num_closest_drones: int = 2,
                 state_mode: StateMode = "relative",
                 id_on_state: bool = True,
                 min_sensor_priority: float = 0.1,
                 max_sensor_priority: float = 1,
                 full_random_drone_position: bool = False,
                 reward: Literal['punish', 'time-reward', 'reward'] = 'punish',
                 speed_action: bool = True,
                 end_when_all_collected: bool = True,
                 use_pypy: bool = False):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.render_mode = render_mode

        self.algorithm_iteration_interval = algorithm_iteration_interval

        self.num_sensors = num_sensors
        self.num_drones = num_drones
        self.possible_agents = [f"drone{i}" for i in range(num_drones)]
        self.max_episode_length = max_episode_length
        self.max_seconds_stalled = max_seconds_stalled
        self.scenario_size = scenario_size
        self.communication_range = communication_range
        self.state_num_closest_sensors = state_num_closest_sensors
        self.state_num_closest_drones = state_num_closest_drones
        self.state_mode = state_mode
        self.id_on_state = id_on_state
        self.min_sensor_priority = min_sensor_priority
        self.max_sensor_priority = max_sensor_priority
        self.full_random_drone_position = full_random_drone_position
        self.reward = reward
        self.speed_action = speed_action
        self.end_when_all_collected = end_when_all_collected
        self.agents = self.possible_agents.copy()

        import uuid
        object_name = f"environment{uuid.uuid4()}"

        command_args = [
            "pypy" if use_pypy else "python", "remote_environment.py",
            f"--object_name={object_name}",
            f"--render_mode={render_mode}" if render_mode else "",
            f"--algorithm_iteration_interval={algorithm_iteration_interval}",
            f"--num_drones={num_drones}",
            f"--num_sensors={num_sensors}",
            f"--scenario_size={scenario_size}",
            f"--max_episode_length={max_episode_length}",
            f"--max_seconds_stalled={max_seconds_stalled}",
            f"--communication_range={communication_range}",
            f"--state_num_closest_sensors={state_num_closest_sensors}",
            f"--state_num_closest_drones={state_num_closest_drones}",
            f"--state_mode={state_mode}",
            f"--id_on_state={id_on_state}",
            f"--min_sensor_priority={min_sensor_priority}",
            f"--max_sensor_priority={max_sensor_priority}",
            f"--full_random_drone_position={full_random_drone_position}",
            f"--reward={reward}",
            f"--speed_action={speed_action}",
            f"--end_when_all_collected={end_when_all_collected}"
        ]
        
        # Remove any empty strings
        command_args = [arg for arg in command_args if arg]

        print("Iniciando servidor com comando:")
        print(command_args)

        # Launch the remote environment process
        self.remote_env_process = subprocess.Popen(command_args, stderr=sys.stdout, stdout=sys.stdout)
        
        # Connect to it using Pyro5
        uri_string = f"PYRONAME:{object_name}"
        self.remote_env = Pyro5.api.Proxy(uri_string)

    def observation_space(self, agent):
        agent_id = 1 if self.id_on_state else 0
        if self.state_mode == "absolute":
            self_position = 2
            agent_positions = self.state_num_closest_drones * 2
            sensor_positions = self.state_num_closest_sensors * 2

            return Box(0, 1, shape=(self_position + agent_positions + sensor_positions + agent_id,))
        elif self.state_mode == "distance_angle" or self.state_mode == "relative":
            agent_positions = self.state_num_closest_drones * 2
            sensor_positions = self.state_num_closest_sensors * 2

            return Box(-1, 1, shape=(agent_positions + sensor_positions + agent_id,))
        elif self.state_mode == "angle":
            agent_positions = self.state_num_closest_drones * 1
            sensor_positions = self.state_num_closest_sensors * 1

            return Box(0, 1, shape=(agent_positions + sensor_positions + agent_id,))
        elif self.state_mode == "all_positions":
            # Observe locations of all agents
            agent_positions = self.num_drones * 2
            agent_index = 1
            sensor_positions = self.num_sensors * 2
            sensor_visited = self.num_sensors

            return Box(0, 1, shape=(agent_positions + agent_index + sensor_positions + sensor_visited + agent_id,))

    def action_space(self, agent):
        # Drone can move in any direction
        if self.speed_action:
            return Box(0, 1, shape=(2,))
        else:
            return Box(0, 1, shape=(1,))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # Visualization is handled by the simulator

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        # Closing the simulator
        self.remote_env.close()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """

        obs, info = self.remote_env.reset(seed, options)
        return {key: np.array(value) for key,value in obs.items()}, info

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        observations, rewards, terminations, truncations, infos = self.remote_env.step({
            key: value.tolist() for key, value in actions.items()
        })
        return {key: np.array(value) for key,value in observations.items()}, rewards, terminations, truncations, infos
