import math
import random
from time import sleep
from typing import List, Optional, Literal

import gymnasium
import numpy as np
from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv


class SensorProtocol(IProtocol):
    has_collected: bool

    def initialize(self) -> None:
        self.has_collected = False
        self.provider.tracked_variables["collected"] = self.has_collected

    def handle_packet(self, message: str) -> None:
        self.has_collected = True
        self.provider.tracked_variables["collected"] = self.has_collected

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass


class DroneProtocol(IProtocol):
    current_position: tuple[float, float, float]

    def act(self, action) -> None:
        direction: float = action[0] * 2 * np.pi

        unit_vector = [np.cos(direction), np.sin(direction)]

        # Waypoint really far into the direction of travel, same height
        destination = [
            float(self.current_position[0] + unit_vector[0] * 1e5),
            float(self.current_position[1] + unit_vector[1] * 1e5),
            float(self.current_position[2])
        ]

        # Start travelling in the direction of travel
        command = GotoCoordsMobilityCommand(*destination)
        self.provider.send_mobility_command(command)

    def initialize(self) -> None:
        self.current_position = (0, 0, 0)
        self._collect_packets()

    def handle_timer(self, timer: str) -> None:
        self._collect_packets()

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position

    def _collect_packets(self) -> None:
        command = BroadcastMessageCommand("")
        self.provider.send_communication_command(command)

        self.provider.schedule_timer("", self.provider.current_time() + 1)

    def finish(self) -> None:
        pass


class GrADySEnvironment(ParallelEnv):
    simulator: Simulator

    agent_node_ids: List[int]
    sensor_node_ids: List[int]

    # Indicates if an algorithm iteration has finished in the simulation
    algorithm_iteration_finished: bool

    reward_sum: float
    max_reward: float
    episode_duration: int

    sensors_collected: int

    metadata = {"render_modes": ["visual", "console"], "name": "gradys-env"}

    def __init__(self,
                 render_mode: Optional[Literal["visual", "console"]] = None,
                 algorithm_iteration_interval: float = 0.5,
                 num_drones: int = 1,
                 num_sensors: int = 2,
                 scenario_size: float = 100,
                 max_episode_length: float = 10_000,
                 randomize_sensor_positions: bool = False,
                 communication_range: float = 20,
                 soft_reward: bool = False,
                 state_num_closest_sensors: int = 2,
                 state_num_closest_drones: int = 2):
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
        self.scenario_size = scenario_size
        self.communication_range = communication_range
        self.randomize_sensor_positions = randomize_sensor_positions
        self.soft_reward = soft_reward
        self.state_num_closest_sensors = state_num_closest_sensors
        self.state_num_closest_drones = state_num_closest_drones

    def observation_space(self, agent):
        # Observe locations of all agents
        self_position = 2
        agent_positions = self.state_num_closest_drones * 2
        sensor_positions = self.state_num_closest_sensors * 2

        return Box(0, 1, shape=(self_position + agent_positions + sensor_positions,))

    def action_space(self, agent):
        # Drone can move in any direction
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
        self.simulator._finalize_simulation()

    def observe_simulation(self):
        unvisited_sensor_count = self.num_sensors - self.sensors_collected
        unvisited_sensor_locations = np.zeros((unvisited_sensor_count, 2))

        unvisited_index = 0
        for i in range(self.num_sensors):
            sensor_node = self.simulator.get_node(self.sensor_node_ids[i])
            if sensor_node.protocol_encapsulator.protocol.has_collected:
                continue

            unvisited_sensor_locations[unvisited_index] = np.array(sensor_node.position[:2]) / self.scenario_size
            unvisited_index += 1

        agent_positions = np.zeros((self.num_drones, 2))
        for i in range(self.num_drones):
            agent_node = self.simulator.get_node(self.agent_node_ids[i])
            agent_positions[i] = np.array(agent_node.position[:2]) / self.scenario_size

        state = {}
        for agent_index in range(self.num_drones):
            agent_distances = np.linalg.norm(
                agent_positions - agent_positions[agent_index], axis=1
            )
            closest_agent_indices = np.argsort(agent_distances)[1:self.state_num_closest_drones + 1]

            unvisited_sensor_distances = np.linalg.norm(
                unvisited_sensor_locations - agent_positions[agent_index], axis=1
            )
            closest_sensor_indices = np.argsort(unvisited_sensor_distances)[:self.state_num_closest_sensors]

            closest_agents = agent_positions[closest_agent_indices, :]
            padded_closes_agents = np.pad(closest_agents,
                                          ((0, self.state_num_closest_drones - len(closest_agents)),
                                           (0, 0)))
            closest_sensors = unvisited_sensor_locations[closest_sensor_indices, :]
            padded_closest_sensors = np.pad(closest_sensors,
                                            ((0, self.state_num_closest_sensors - len(closest_sensors)),
                                             (0, 0)))

            state[f"drone{agent_index}"] = np.concatenate([
                agent_positions[agent_index],
                padded_closes_agents.flatten(),
                padded_closest_sensors.flatten()
            ])
        return state

    def detect_out_of_bounds_agent(self, agent: Node) -> bool:
        return abs(agent.position[0]) > self.scenario_size or abs(agent.position[1]) > self.scenario_size


    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents.copy()

        builder = SimulationBuilder(SimulationConfiguration(
            duration=self.max_episode_length,
            debug=False,
            execution_logging=False
        ))
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=self.communication_range
        )))
        builder.add_handler(MobilityHandler(MobilityConfiguration(
            update_rate=self.algorithm_iteration_interval / 3
        )))
        builder.add_handler(TimerHandler())

        if self.render_mode == "visual":
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(
                open_browser=False,
                x_range=(-self.scenario_size, self.scenario_size),
                y_range=(-self.scenario_size, self.scenario_size),
                z_range=(0, self.scenario_size),
            )))

        class GrADySHandler(INodeHandler):
            event_loop: EventLoop

            @staticmethod
            def get_label() -> str:
                return "GrADySHandler"

            def inject(self, event_loop: EventLoop) -> None:
                self.event_loop = event_loop
                self.last_iteration = 0
                self.iterate_algorithm()

            def register_node(self, node: Node) -> None:
                pass

            def iterate_algorithm(handler_self):
                self.algorithm_iteration_finished = True

                handler_self.event_loop.schedule_event(
                    handler_self.event_loop.current_time + self.algorithm_iteration_interval,
                    handler_self.iterate_algorithm
                )

        builder.add_handler(GrADySHandler())

        self.sensor_node_ids = []

        fixed_positions = [(-self.scenario_size, 0), (self.scenario_size, 0), (0, -self.scenario_size),
                           (0, self.scenario_size), (self.scenario_size, self.scenario_size),
                           (-self.scenario_size, -self.scenario_size), (self.scenario_size, -self.scenario_size),
                           (-self.scenario_size, self.scenario_size)]
        for i in range(self.num_sensors):
            # Place sensors outside commuincation range but inside the scenario
            if self.randomize_sensor_positions:
                self.sensor_node_ids.append(builder.add_node(SensorProtocol, (
                    random.uniform(self.communication_range + 1, self.scenario_size) * (
                        1 if random.random() < 0.5 else -1),
                    random.uniform(self.communication_range + 1, self.scenario_size) * (
                        1 if random.random() < 0.5 else -1),
                    0
                )))
            else:
                self.sensor_node_ids.append(builder.add_node(SensorProtocol, (
                    fixed_positions[i][0],
                    fixed_positions[i][1],
                    0
                )))

        self.agent_node_ids = []
        for i in range(self.num_drones):
            self.agent_node_ids.append(builder.add_node(DroneProtocol, (
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                0
            )))

        self.simulator = builder.build()

        # Running a single simulation step to get the initial observations
        if not self.simulator.step_simulation():
            raise ValueError("Simulation failed to start")

        self.episode_duration = 0
        self.reward_sum = 0
        self.max_reward = -math.inf
        self.sensors_collected = 0

        return self.observe_simulation(), {}

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
        self.episode_duration += 1

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Acting
        for index, action in enumerate(actions.values()):
            agent_node = self.simulator.get_node(self.agent_node_ids[index])
            agent_node.protocol_encapsulator.protocol.act(action)

        # Simulating for a single iteration
        self.algorithm_iteration_finished = False
        simulation_ongoing = True
        while not self.algorithm_iteration_finished:
            if self.render_mode == "visual":
                current_time = self.simulator._current_timestamp
                next_time = self.simulator._event_loop.peek_event().timestamp
                sleep(max(0, next_time - current_time))

            simulation_ongoing = self.simulator.step_simulation()
            if not simulation_ongoing:
                break

        sensors_collected = sum(
            self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        )
        all_sensors_collected = sensors_collected == self.num_sensors

        if self.soft_reward:
            reward = sensors_collected / self.num_sensors if (all_sensors_collected or not simulation_ongoing) else 0
        else:
            reward = int(all_sensors_collected)
        rewards = {
            agent: reward for agent in self.agents
        }
        self.sensors_collected = sensors_collected

        for index in range(len(self.agents)):
            agent_node = self.simulator.get_node(self.agent_node_ids[index])
            if self.detect_out_of_bounds_agent(agent_node):
                simulation_ongoing = False
                rewards[self.agents[index]] = -1

        self.reward_sum += rewards[self.agents[0]]
        self.max_reward = max(self.max_reward, *rewards.values())

        terminations = {agent: all_sensors_collected or not simulation_ongoing for agent in self.agents}

        truncations = {agent: False for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = self.observe_simulation()

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {}

        if not simulation_ongoing or all_sensors_collected:
            infos = {
                agent: {
                    "avg_reward": self.reward_sum / self.episode_duration,
                    "max_reward": self.max_reward,
                    "episode_duration": self.episode_duration,
                    "success": all_sensors_collected,
                } for agent in self.agents
            }

        return observations, rewards, terminations, truncations, infos
