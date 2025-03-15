import math
import random
from time import sleep
from typing import List, Optional, Literal, TypedDict, cast

import gymnasium
import numpy as np
from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand, SendMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.position import squared_distance
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration, \
    VisualizationController
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from scipy.spatial import KDTree

StateMode = Literal["all_positions", "absolute", "relative", "distance_angle", "angle"]


class SensorBroadcast(TypedDict):
    id: int
    position: tuple[float, float, float]
    collected: bool
    type: Literal["S"]


class DroneBroadcast(TypedDict):
    id: int
    position: tuple[float, float, float]
    known_sensors: dict[int, tuple[float, float, float]]
    known_sensors_age: dict[int, float]
    known_drones: dict[int, tuple[float, float, float]]
    known_drones_age: dict[int, float]
    type: Literal["D"]

def create_sensor_protocol(min_priority: float, max_priority: float, drone_ids: List[int]):
    class SensorProtocol(IProtocol):
        has_collected: bool
        priority: float
        position: tuple[float, float, float] | None
        initialized: bool

        def __init__(self):
            super().__init__()
            
            self.min_priority: float = min_priority
            self.max_priority: float = max_priority
            self.drone_ids: List[int] = drone_ids

        def initialize(self) -> None:
            self.priority = random.uniform(self.min_priority, self.max_priority)
            self.provider.tracked_variables["priority"] = self.priority
            self.has_collected = False
            self.provider.tracked_variables["collected"] = self.has_collected
            self.position = None
            self.initialized = False

        def handle_packet(self, message: str) -> None:
            if message['type'] == 'D':
                self.has_collected = True
                self.provider.tracked_variables["collected"] = self.has_collected

        def send_heartbeat(self) -> None:
            message_content: SensorBroadcast = {
                "id": self.provider.get_id(),
                "position": cast(tuple[float, float, float], self.position),
                "collected": self.has_collected,
                "type": "S"
            }

            for drone in self.drone_ids:
                command = SendMessageCommand(message_content, drone)
                self.provider.send_communication_command(command)

            self.provider.schedule_timer("", self.provider.current_time() + 1)

        def handle_timer(self, timer: str) -> None:
            self.send_heartbeat()

        def handle_telemetry(self, telemetry: Telemetry) -> None:
            if self.position is None:
                self.position = telemetry.current_position
                self.send_heartbeat()
                self.initialized = True

        def finish(self) -> None:
            pass
    return SensorProtocol

def create_drone_protocol(
        speed_action: bool,
        algorithm_interval: float,
        num_closest_drones: int,
        num_closest_sensors: int,
        sensor_positions: dict[int, tuple[float, float, float]],
        scenario_size: float):
    class DroneProtocol(IProtocol):
        current_position: tuple[float, float, float] | None
        initialized: bool

        known_sensors: dict[int, tuple[float, float, float]]
        known_sensors_age: dict[int, float]
        known_drones: dict[int, tuple[float, float, float]]
        known_drones_age: dict[int, float]

        def __init__(self):
            super().__init__()
            self.speed_action = speed_action
            self.algorithm_interval = algorithm_interval
            self.num_closest_drones = num_closest_drones
            self.num_closest_sensors = num_closest_sensors
            self.sensor_positions = sensor_positions
            self.scenario_size = scenario_size

        def act(self, action: List[float], coordinate_limit: float) -> None:
            if isinstance(action, np.ndarray):
                action = action.tolist()
            self.provider.tracked_variables['current_action'] = action

            direction: float = action[0] * 2 * math.pi

            if self.speed_action:
                speed: float = action[1] * 15
                command = SetSpeedMobilityCommand(speed)
                self.provider.send_mobility_command(command)

            unit_vector = [math.cos(direction), math.sin(direction)]

            distance_to_x_edge = coordinate_limit - abs(self.current_position[0])
            distance_to_y_edge = coordinate_limit - abs(self.current_position[1])

            # Maintain direction but bound destination within scenario
            if distance_to_x_edge > 0 and distance_to_y_edge > 0:
                scale_x = distance_to_x_edge / (abs(unit_vector[0]) + 1e-10)
                scale_y = distance_to_y_edge / (abs(unit_vector[1]) + 1e-10)
                scale = min(scale_x, scale_y)

                destination = [
                    self.current_position[0] + unit_vector[0] * scale,
                    self.current_position[1] + unit_vector[1] * scale,
                    0
                ]

            # If the drone is at the edge of the scenario, prevent it from leaving
            else:
                destination = [
                    self.current_position[0] + unit_vector[0] * 1e5,
                    self.current_position[1] + unit_vector[1] * 1e5,
                    0
                ]
                # Bound destination within scenario
                destination[0] = max(-coordinate_limit, min(coordinate_limit, destination[0]))
                destination[1] = max(-coordinate_limit, min(coordinate_limit, destination[1]))

            # Start travelling in the direction of travel
            command = GotoCoordsMobilityCommand(*destination)
            self.provider.send_mobility_command(command)

        def observe(self):
            drone_keys = list(self.known_drones.keys())
            drone_positions = np.array([pos[:2] for pos in self.known_drones.values()]).reshape((-1, 2))

            sensor_keys = list(self.known_sensors.keys())
            sensor_positions = np.array([pos[:2] for pos in self.known_sensors.values()]).reshape((-1, 2))

            own_position = np.array(self.current_position[:2])

            drone_kdtree = KDTree(drone_positions)
            sensor_kdtree = KDTree(sensor_positions)

            drone_count = min(self.num_closest_drones, len(drone_positions))
            sensor_count = min(self.num_closest_sensors, len(sensor_positions))

            drone_observation = np.zeros((self.num_closest_drones, 2))
            drone_observation.fill(-1)
            drone_age_observation = np.zeros(self.num_closest_drones)
            drone_age_observation.fill(-1)
            sensor_observation = np.zeros((self.num_closest_sensors, 2))
            sensor_observation.fill(-1)
            sensor_age_observation = np.zeros(self.num_closest_sensors)

            if drone_count > 0:
                _, closest_drones = drone_kdtree.query(own_position, range(1, drone_count + 1))
                drone_observation[:drone_count] = (
                        own_position - drone_positions[closest_drones] + self.scenario_size) / (self.scenario_size * 2)

                # If k is one the kdtree does not return an array
                drone_keys = [drone_keys[closest_drone] for closest_drone in closest_drones]
                drone_age_observation[:drone_count] = [self.known_drones_age[drone_key] / 500 for drone_key in drone_keys]

            if sensor_count > 0:
                _, closest_sensors = sensor_kdtree.query(own_position, range(1, sensor_count + 1))
                sensor_observation[:sensor_count] = (
                        own_position - sensor_positions[closest_sensors] + self.scenario_size) / (self.scenario_size * 2)

                sensor_keys = [sensor_keys[closest_sensor] for closest_sensor in closest_sensors]
                sensor_age_observation[:sensor_count] = [self.known_sensors_age[sensor_key] / 500 for sensor_key in sensor_keys]

            return drone_observation.flatten(), sensor_observation.flatten(), drone_age_observation, sensor_age_observation

        def initialize(self) -> None:
            self.current_position = None

            self.known_sensors = {**self.sensor_positions}
            self.known_sensors_age = {
                sensor_id: self.provider.current_time()
                for sensor_id in self.sensor_positions.keys()
            }
            self.known_drones = {}
            self.known_drones_age = {}

            self.provider.tracked_variables['known_sensors'] = self.known_sensors
            self.provider.tracked_variables['known_drones'] = self.known_drones

            self.provider.schedule_timer("", self.provider.current_time())

            self.initialized = False

        def handle_timer(self, timer: str) -> None:
            self._collect_packets()

        def handle_packet(self, message: str) -> None:
            message_type = message['type']

            if message_type == 'S':
                sensor_message: SensorBroadcast = message
                self.known_sensors_age[sensor_message['id']] = self.provider.current_time()

                if sensor_message['collected'] and sensor_message['id'] in self.known_sensors:
                    del self.known_sensors[sensor_message['id']]
            else:
                drone_message: DroneBroadcast = message
                self.known_drones_age[drone_message['id']] = self.provider.current_time()
                self.known_drones[drone_message['id']] = drone_message['position']

                for other_drone in drone_message['known_drones_age'].keys():
                    if int(other_drone) == self.provider.get_id():
                        continue

                    if drone_message['known_drones_age'][other_drone] > self.known_drones_age.get(int(other_drone), -1):
                        self.known_drones_age[int(other_drone)] = drone_message['known_drones_age'][other_drone]
                        self.known_drones[int(other_drone)] = drone_message['known_drones'][other_drone]

                for other_sensor in drone_message['known_sensors_age'].keys():
                    if drone_message['known_sensors_age'][other_sensor] > self.known_sensors_age.get(int(other_sensor), -1):
                        self.known_sensors_age[int(other_sensor)] = drone_message['known_sensors_age'][other_sensor]
                        if other_sensor not in drone_message['known_sensors'] and other_sensor in self.known_sensors:
                            del self.known_sensors[int(other_sensor)]

        def handle_telemetry(self, telemetry: Telemetry) -> None:
            if self.current_position is None:
                self.current_position = telemetry.current_position
            else:
                self.current_position = telemetry.current_position
            self.initialized = True

        def _collect_packets(self) -> None:
            if self.current_position is not None:
                message_content: DroneBroadcast = {
                    "id": self.provider.get_id(),
                    "position": self.current_position,
                    "known_drones": self.known_drones,
                    "known_drones_age": self.known_drones_age,
                    "known_sensors": self.known_sensors,
                    "known_sensors_age": self.known_sensors_age,
                    "type": "D"
                }

                command = BroadcastMessageCommand(message_content)
                self.provider.send_communication_command(command)
            self.provider.schedule_timer("", self.provider.current_time() + 0.5)

        def finish(self) -> None:
            pass

    return DroneProtocol


class GrADySEnvironment(ParallelEnv):
    simulator: Simulator
    controller: VisualizationController

    agent_node_ids: List[int]
    sensor_node_ids: List[int]

    # Indicates if an algorithm iteration has finished in the simulation
    algorithm_iteration_finished: bool

    reward_sum: float
    max_reward: float
    episode_duration: int
    stall_duration: int

    sensors_collected: int
    collection_times: List[float]

    metadata = {"render_modes": ["visual", "console"], "name": "gradys-env"}

    def __init__(self,
                 render_mode: Optional[Literal["visual", "console"]] = None,
                 algorithm_iteration_interval: float = 0.5,
                 num_drones: int = 1,
                 num_sensors: int = 2,
                 max_sensor_count: int = 10,
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
                 local_observation: bool = False):
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
        self.max_sensor_count = max_sensor_count
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
        self.local_observation = local_observation

    def observation_space(self, agent):
        agent_id = 1 if self.id_on_state else 0

        agent_positions = self.state_num_closest_drones * 2
        agent_ages = self.state_num_closest_drones
        sensor_positions = self.state_num_closest_sensors * 2
        sensor_ages = self.state_num_closest_sensors

        return Box(-1, 1, shape=(agent_positions + sensor_positions + agent_ages + sensor_ages + agent_id,))


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
        self.simulator._finalize_simulation()

    def observe_global_state(self):
        """
        Global observation of the simulation state containing all unvisited sensor positions and all agent positions
        """
        sensor_nodes = np.array([self.simulator.get_node(sensor_id).position[:2] for sensor_id in self.sensor_node_ids])
        unvisited_sensor_mask = np.array(
            [not self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected for sensor_id in
             self.sensor_node_ids])
        unvisited_sensor_nodes = sensor_nodes[unvisited_sensor_mask]
        agent_nodes = np.array([self.simulator.get_node(agent_id).position[:2] for agent_id in self.agent_node_ids])

        # -1 padded position array
        all_sensor_positions = np.zeros((self.max_sensor_count, 2))
        all_sensor_positions.fill(-1)
        all_sensor_positions[:len(unvisited_sensor_nodes)] = unvisited_sensor_nodes

        # Agent array
        all_agent_positions = agent_nodes.copy()

        # Normalize the positions
        all_agent_positions = (all_agent_positions + self.scenario_size) / (self.scenario_size * 2)
        all_sensor_positions = (all_sensor_positions + self.scenario_size) / (self.scenario_size * 2)

        return np.concatenate([all_agent_positions.flatten(), all_sensor_positions.flatten()])



    def observe_simulation_relative_positions(self):
        sensor_nodes = np.array([self.simulator.get_node(sensor_id).position[:2] for sensor_id in self.sensor_node_ids])
        unvisited_sensor_mask = np.array(
            [not self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected for sensor_id in
             self.sensor_node_ids])
        unvisited_sensor_nodes = sensor_nodes[unvisited_sensor_mask]

        agent_nodes = np.array([self.simulator.get_node(agent_id).position[:2] for agent_id in self.agent_node_ids])

        if self.local_observation:
            node_observations = (
                self.simulator.get_node(node).protocol_encapsulator.protocol.observe() for node in self.agent_node_ids
            )
            state = {
                f"drone{index}":
                    np.concatenate([
                        drone_observation,
                        sensor_observation,
                        drone_age_observation,
                        sensor_age_observation,
                        [index / self.num_drones] if self.id_on_state else []
                    ])
                for index, (drone_observation, sensor_observation, drone_age_observation, sensor_age_observation)
                in enumerate(node_observations)
            }
        else:
            state = {}
            for agent_index in range(self.num_drones):
                agent_position = agent_nodes[agent_index]

                # Calculate distances to all unvisited sensors and sort them
                sensor_distances = np.linalg.norm(unvisited_sensor_nodes - agent_position, axis=1)
                sorted_sensor_indices = np.argsort(sensor_distances)

                # Select the closest sensors
                closest_unvisited_sensors = np.zeros((self.state_num_closest_sensors, 2))
                closest_unvisited_sensors.fill(-1)
                closest_unvisited_sensors[:len(sorted_sensor_indices)] = unvisited_sensor_nodes[
                    sorted_sensor_indices[:self.state_num_closest_sensors]]

                # Calculate distances to all other agents and sort them
                agent_distances = np.linalg.norm(agent_nodes - agent_position, axis=1)
                sorted_agent_indices = np.argsort(agent_distances)

                # Select the closest agents (excluding the agent itself)
                closest_agents = np.zeros((self.state_num_closest_drones, 2))
                closest_agents.fill(-1)
                closest_agents[:len(sorted_agent_indices) - 1] = agent_nodes[
                    sorted_agent_indices[1:self.state_num_closest_drones + 1]]

                # Normalize the positions
                closest_agents[:len(sorted_agent_indices) - 1] = (agent_position - closest_agents[:len(
                    sorted_agent_indices) - 1] + self.scenario_size) / (self.scenario_size * 2)
                closest_unvisited_sensors[:len(sorted_sensor_indices)] = \
                        (agent_position
                         - closest_unvisited_sensors[:len(sorted_sensor_indices)]
                         + self.scenario_size) / (self.scenario_size * 2)

                state[f"drone{agent_index}"] = np.concatenate([
                    closest_agents.flatten(),
                    closest_unvisited_sensors.flatten(),
                    np.array([agent_index / self.num_drones]) if self.id_on_state else []
                ])

        state['global'] = self.observe_global_state()
        return state

    def observe_simulation(self):
        if self.state_mode == "relative":
            return self.observe_simulation_relative_positions()
        else:
            raise NotImplementedError()

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
            debug=False,
            execution_logging=False,
            duration=self.max_episode_length,
        ))
        builder.add_handler(CommunicationHandler(CommunicationMedium(
            transmission_range=self.communication_range
        )))
        builder.add_handler(MobilityHandler(MobilityConfiguration(
            update_rate=self.algorithm_iteration_interval / 2
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
        self.agent_node_ids = []

        sensor_positions = {}
        for _ in range(self.num_sensors):
            x_pos = random.uniform(-self.scenario_size, self.scenario_size)
            y_pos = random.uniform(-self.scenario_size, self.scenario_size)
            node_id = builder.add_node(create_sensor_protocol(self.min_sensor_priority, self.max_sensor_priority, self.agent_node_ids), 
                                       (x_pos, y_pos, 0))
            self.sensor_node_ids.append(node_id)
            sensor_positions[node_id] = (x_pos, y_pos, 0)   
        
        for _ in range(self.num_drones):
            protocol = create_drone_protocol(self.speed_action, self.algorithm_iteration_interval, self.state_num_closest_drones, 
                                             self.state_num_closest_sensors, sensor_positions, self.scenario_size)
            if self.full_random_drone_position:
                self.agent_node_ids.append(builder.add_node(protocol, (
                    random.uniform(-self.scenario_size, self.scenario_size),
                    random.uniform(-self.scenario_size, self.scenario_size),
                    0
                )))
            else:
                self.agent_node_ids.append(builder.add_node(protocol, (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    0
                )))

        self.simulator = builder.build()
        if self.render_mode == "visual":
            self.controller = VisualizationController()

        # Running simulation until all positions are correctly set and initial messages have been exchanged
        while True:
            if not self.simulator.step_simulation():
                raise ValueError("Simulation failed to start")

            if all(self.simulator.get_node(sensor).protocol_encapsulator.protocol.initialized for sensor in
                   self.sensor_node_ids) \
                    and all(self.simulator.get_node(drone).protocol_encapsulator.protocol.initialized for drone in
                            self.agent_node_ids):
                break

        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0
        self.max_reward = -math.inf
        self.sensors_collected = 0
        self.collection_times = [self.max_episode_length for _ in range(self.num_sensors)]

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

        end_cause: str | None = None

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Acting
        for index, action in enumerate(actions.values()):
            agent_node = self.simulator.get_node(self.agent_node_ids[index])

            coordinate_limit = self.scenario_size
            agent_node.protocol_encapsulator.protocol.act(action, coordinate_limit)

        sensor_is_collected_before = [
            self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ]

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
                end_cause = "time_limit_exceeded"
                break

        sensor_is_collected = [
            self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.has_collected
            for sensor_id in self.sensor_node_ids
        ]

        if self.render_mode == "visual":
            collected_sensors = [sensor_id
                                 for index, sensor_id in enumerate(self.sensor_node_ids)
                                 if sensor_is_collected[index]]
            for sensor_id in collected_sensors:
                self.controller.paint_node(sensor_id, (0, 255, 0))

        if sum(sensor_is_collected) > sum(sensor_is_collected_before):
            self.stall_duration = 0
        else:
            self.stall_duration += self.algorithm_iteration_interval

        if self.stall_duration > self.max_seconds_stalled:
            simulation_ongoing = False
            end_cause = f"stalled ({sum(sensor_is_collected)}/{self.num_sensors})"

        all_sensors_collected = sum(sensor_is_collected) == self.num_sensors

        if all_sensors_collected:
            end_cause = "all_sensors_collected"

        # Calculating reward
        reward = 0
        if self.reward == 'punish':
            before = sum(sensor_is_collected_before)
            after = sum(sensor_is_collected)
            if after > before:
                reward = (after - before) * 10
            else:
                reward = -(self.num_sensors - sum(sensor_is_collected)) / self.num_sensors
        if self.reward == 'reward':
            reward = sum(sensor_is_collected) - sum(sensor_is_collected_before)

        current_timestamp = self.episode_duration * self.algorithm_iteration_interval
        for index, sensor_id in enumerate(self.sensor_node_ids):
            if sensor_is_collected[index] and self.collection_times[index] == self.max_episode_length:
                self.collection_times[index] = current_timestamp
                if self.reward == 'time-reward':
                    priority = self.simulator.get_node(sensor_id).protocol_encapsulator.protocol.priority
                    reward += priority * (1 - current_timestamp / self.max_episode_length)

        rewards = {
            agent: reward for agent in self.agents
        }
        self.sensors_collected = sum(sensor_is_collected)

        self.reward_sum += rewards[self.agents[0]]
        self.max_reward = max(self.max_reward, *rewards.values())

        simulation_ended = (all_sensors_collected and self.end_when_all_collected) or not simulation_ongoing
        terminations = {agent: simulation_ended for agent in self.agents}

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
                    "sum_reward": self.reward_sum,
                    "avg_collection_time": sum(self.collection_times) / self.num_sensors,
                    "episode_duration": self.episode_duration,
                    "completion_time": self.max_episode_length if not all_sensors_collected else self.simulator._current_timestamp,
                    "all_collected": all_sensors_collected,
                    "cause": end_cause,
                } for agent in self.agents
            }

        return observations, rewards, terminations, truncations, infos
