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
                 communication_range: float = 20):
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

    def observation_space(self, agent):
        # Observe locations of all agents
        agent_positions = self.num_drones * 2
        agent_index = 1
        sensor_positions = self.num_sensors * 2
        sensor_visited = self.num_sensors

        return Box(0, 1, shape=(agent_positions + agent_index + sensor_positions + sensor_visited,))

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
        sensor_locations = np.zeros(self.num_sensors * 2)
        sensor_visited = np.zeros(self.num_sensors)
        for i in range(self.num_sensors):
            sensor_node = self.simulator.get_node(self.sensor_node_ids[i])
            sensor_locations[i * 2] = sensor_node.position[0] / self.scenario_size
            sensor_locations[i * 2 + 1] = sensor_node.position[1] / self.scenario_size

            sensor_visited[i] = int(sensor_node.protocol_encapsulator.protocol.has_collected)

        agent_positions = np.zeros(self.num_drones * 2)
        for i in range(self.num_drones):
            agent_node = self.simulator.get_node(self.agent_node_ids[i])
            agent_positions[i * 2] = agent_node.position[0] / self.scenario_size
            agent_positions[i * 2 + 1] = agent_node.position[1] / self.scenario_size

        general_observations = np.concatenate([sensor_locations, sensor_visited, agent_positions])

        state = {}
        for agent_index in range(self.num_drones):
            # General observations and agent index
            state[f"drone{agent_index}"] = np.concatenate([general_observations, [agent_index / self.num_drones]])
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

        # reward = sensors_collected / self.num_sensors
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
                break

        self.reward_sum += rewards[self.agents[0]]

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
                    "episode_duration": self.episode_duration,
                    "success": all_sensors_collected,
                } for agent in self.agents
            }

        return observations, rewards, terminations, truncations, infos
