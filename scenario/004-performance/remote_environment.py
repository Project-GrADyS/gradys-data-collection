import math
import random
from time import sleep, time
from typing import List, Optional, Literal, Tuple

import Pyro5.server

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityHandler, MobilityConfiguration
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration, \
    VisualizationController
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import SimulationBuilder, Simulator, SimulationConfiguration
from gradysim.protocol.position import squared_distance

from arguments import StateMode, RemoteEnvironmentArgs, EnvironmentArgs

import numpy as np

from scipy.spatial import KDTree

class SensorProtocol(IProtocol):
    has_collected: bool

    min_priority: int = 0
    max_priority: int = 1

    def initialize(self) -> None:
        self.priority = random.uniform(self.min_priority, self.max_priority)
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
    current_position: Tuple[float, float, float]
    speed_action: bool = False
    algorithm_interval: float = 0.1

    def act(self, action: List[float], coordinate_limit: float) -> None:
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

        if self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + self.algorithm_interval - 0.1)

    def initialize(self) -> None:
        self.current_position = (0, 0, 0)
        self._collect_packets()
        if not self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + 0.1)

    def handle_timer(self, timer: str) -> None:
        self._collect_packets()
        if not self.speed_action:
            self.provider.schedule_timer("", self.provider.current_time() + 0.1)

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position

    def _collect_packets(self) -> None:
        command = BroadcastMessageCommand("")
        self.provider.send_communication_command(command)

    def finish(self) -> None:
        pass


class GrADySHandler(INodeHandler):
    event_loop: EventLoop

    def __init__(self, env):
        self.env = env

    @staticmethod
    def get_label() -> str:
        return "GrADySHandler"

    def inject(self, event_loop: EventLoop) -> None:
        self.event_loop = event_loop
        self.last_iteration = 0
        self.iterate_algorithm()

    def register_node(self, node: Node) -> None:
        pass

    def iterate_algorithm(self):
        self.env.algorithm_iteration_finished = True

        self.event_loop.schedule_event(
            self.event_loop.current_time + self.env.algorithm_iteration_interval,
            self.iterate_algorithm
        )


class GradysRemoteEnvironment:
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
                 end_when_all_collected: bool = True):
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

    @Pyro5.server.expose
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        # Closing the simulator
        self.simulator._finalize_simulation()

    def observe_simulation_angle_distance(self):
        sensor_nodes = [self.simulator.get_node(sensor_id) for sensor_id in self.sensor_node_ids]
        unvisited_sensor_nodes = [sensor_node for sensor_node in sensor_nodes
                                  if not sensor_node.protocol_encapsulator.protocol.has_collected]

        agent_nodes = [self.simulator.get_node(agent_id) for agent_id in self.agent_node_ids]

        state = {}
        for agent_index in range(self.num_drones):
            agent_position = agent_nodes[agent_index].position

            closest_unvisited_sensors = np.zeros((self.state_num_closest_sensors, 2))

            sorted_unvisited_sensors = sorted(unvisited_sensor_nodes,
                                              key=lambda sensor_node: squared_distance(sensor_node.position,
                                                                                       agent_position))

            for i, sensor_node in enumerate(sorted_unvisited_sensors[:self.state_num_closest_sensors]):
                angle = np.arctan2(sensor_node.position[1] - agent_position[1],
                                   sensor_node.position[0] - agent_position[0])
                distance = np.linalg.norm(np.array(sensor_node.position[:2]) - np.array(agent_position[:2]))
                closest_unvisited_sensors[i, 0] = angle / np.pi
                closest_unvisited_sensors[i, 1] = distance / self.scenario_size

            closest_agents = np.zeros((self.state_num_closest_drones, 2))

            sorted_agents = sorted(agent_nodes,
                                   key=lambda agent_node: squared_distance(agent_node.position, agent_position))

            for i, agent_node in enumerate(sorted_agents[1:self.state_num_closest_drones + 1]):
                angle = np.arctan2(agent_node.position[1] - agent_position[1],
                                   agent_node.position[0] - agent_position[0])
                distance = np.linalg.norm(np.array(agent_node.position[:2]) - np.array(agent_position[:2]))
                closest_agents[i, 0] = angle / np.pi
                closest_agents[i, 1] = distance / self.scenario_size

            state[f"drone{agent_index}"] = np.concatenate([
                closest_agents.flatten(),
                closest_unvisited_sensors.flatten(),
                np.array(agent_index).flatten() / self.num_drones if self.id_on_state else []
            ]).tolist()
        return state

    def observe_simulation_angle(self):
        sensor_nodes = [self.simulator.get_node(sensor_id) for sensor_id in self.sensor_node_ids]
        unvisited_sensor_nodes = [sensor_node for sensor_node in sensor_nodes
                                  if not sensor_node.protocol_encapsulator.protocol.has_collected]

        agent_nodes = [self.simulator.get_node(agent_id) for agent_id in self.agent_node_ids]

        state = {}
        for agent_index in range(self.num_drones):
            agent_position = agent_nodes[agent_index].position

            closest_unvisited_sensors = np.zeros((self.state_num_closest_sensors,))

            sorted_unvisited_sensors = sorted(unvisited_sensor_nodes,
                                              key=lambda sensor_node: squared_distance(sensor_node.position,
                                                                                       agent_position))

            for i, sensor_node in enumerate(sorted_unvisited_sensors[:self.state_num_closest_sensors]):
                angle = np.arctan2(sensor_node.position[1] - agent_position[1],
                                   sensor_node.position[0] - agent_position[0])
                closest_unvisited_sensors[i] = angle / np.pi

            closest_agents = np.zeros((self.state_num_closest_drones,))

            sorted_agents = sorted(agent_nodes,
                                   key=lambda agent_node: squared_distance(agent_node.position, agent_position))

            for i, agent_node in enumerate(sorted_agents[1:self.state_num_closest_drones + 1]):
                angle = np.arctan2(agent_node.position[1] - agent_position[1],
                                   agent_node.position[0] - agent_position[0])
                closest_agents[i] = angle / np.pi

            state[f"drone{agent_index}"] = np.concatenate([
                closest_agents,
                closest_unvisited_sensors,
                np.array(agent_index).flatten() / self.num_drones if self.id_on_state else []
            ]).tolist()
        return state

    def observe_simulation_relative_positions(self):
        unvisited_sensor_nodes = np.array([self.simulator.get_node(sensor_id).position[:2]
                                           for sensor_id in self.sensor_node_ids
                                           if not self.simulator.get_node(sensor_id) \
                                            .protocol_encapsulator.protocol.has_collected])
        sensor_kd_tree = KDTree(unvisited_sensor_nodes)


        agent_nodes = np.array([self.simulator.get_node(agent_id).position[:2] for agent_id in self.agent_node_ids])
        agent_kd_tree = KDTree(agent_nodes)

        closest_sensor_count = min(self.state_num_closest_sensors, len(unvisited_sensor_nodes))
        closest_agent_count = min(self.state_num_closest_drones, len(agent_nodes))

        agent_nodes_reshaped = agent_nodes.reshape((self.num_drones, 1, 2))

        # Calculating closest nodes to all agents
        _, all_closest_sensors = sensor_kd_tree.query(agent_nodes, closest_sensor_count)
        all_closest_sensors = all_closest_sensors.reshape((self.num_drones, closest_sensor_count))
        _, all_closest_agents = agent_kd_tree.query(agent_nodes, list(range(2, closest_agent_count+2)))
        all_closest_agents = all_closest_agents.reshape((self.num_drones, closest_agent_count))

        # Collecting the closest sensor coordinates
        closest_unvisited_sensors = unvisited_sensor_nodes[all_closest_sensors]
        # Normalizing sensor positions
        closest_unvisited_sensors *= -1
        closest_unvisited_sensors += agent_nodes_reshaped
        closest_unvisited_sensors += self.scenario_size
        closest_unvisited_sensors /= self.scenario_size * 2

        # Collecting the closest agent coordinates
        closest_agents = agent_nodes[all_closest_agents]
        # Normalizing agent positions
        closest_agents *= -1
        closest_agents += agent_nodes_reshaped
        closest_agents += self.scenario_size
        closest_agents /= self.scenario_size * 2

        closest_unvisited_sensor_list: list[list[float]] = closest_unvisited_sensors.reshape((self.num_drones, -1)).tolist()
        closest_agent_list: list[list[float]] = closest_agents.reshape((self.num_drones, -1)).tolist()

        state = {}
        for agent_index in range(self.num_drones):
            padded_closest_agents = [closest_agent_list[agent_index][i]
                                     if i < len(closest_agent_list[agent_index])
                                     else -1
                                     for i in range(self.state_num_closest_drones * 2)]
            padded_closest_sensors = [closest_unvisited_sensor_list[agent_index][i]
                                      if i < len(closest_unvisited_sensor_list[agent_index])
                                      else -1
                                      for i in range(self.state_num_closest_sensors * 2)]
            state[f"drone{agent_index}"] = \
                padded_closest_agents + \
                padded_closest_sensors + \
                ([agent_index / self.num_drones] if self.id_on_state else [])
        return state

    def observe_simulation_absolute_positions(self):
        sensor_nodes = [self.simulator.get_node(sensor_id) for sensor_id in self.sensor_node_ids]
        unvisited_sensor_nodes = [sensor_node for sensor_node in sensor_nodes
                                  if not sensor_node.protocol_encapsulator.protocol.has_collected]

        agent_nodes = [self.simulator.get_node(agent_id) for agent_id in self.agent_node_ids]

        state = {}
        for agent_index in range(self.num_drones):
            agent_position = agent_nodes[agent_index].position

            closest_unvisited_sensors = np.zeros((self.state_num_closest_sensors, 2))

            sorted_unvisited_sensors = sorted(unvisited_sensor_nodes,
                                              key=lambda sensor_node: squared_distance(sensor_node.position,
                                                                                       agent_position))

            for i, sensor_node in enumerate(sorted_unvisited_sensors[:self.state_num_closest_sensors]):
                closest_unvisited_sensors[i] = sensor_node.position[:2]

            closest_agents = np.zeros((self.state_num_closest_drones, 2))

            sorted_agents = sorted(agent_nodes,
                                   key=lambda agent_node: squared_distance(agent_node.position, agent_position))

            for i, agent_node in enumerate(sorted_agents[1:self.state_num_closest_drones + 1]):
                closest_agents[i] = agent_node.position[:2]

            state[f"drone{agent_index}"] = np.concatenate([
                np.array(agent_position[:2]) / self.scenario_size,
                closest_agents.flatten() / self.scenario_size,
                closest_unvisited_sensors.flatten() / self.scenario_size,
                np.array(agent_index).flatten() / self.num_drones if self.id_on_state else []
            ]).tolist()
        return state

    def observe_simulation_all_positions(self):
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
            state[f"drone{agent_index}"] = np.concatenate([
                general_observations,
                np.array(agent_index).flatten() / self.num_drones if self.id_on_state else []
            ]).tolist()
        return state

    def observe_simulation(self):
        if self.state_mode == "relative":
            return self.observe_simulation_relative_positions()
        elif self.state_mode == "distance_angle":
            return self.observe_simulation_angle_distance()
        elif self.state_mode == "angle":
            return self.observe_simulation_angle()
        elif self.state_mode == "absolute":
            return self.observe_simulation_absolute_positions()
        elif self.state_mode == "all_positions":
            return self.observe_simulation_all_positions()

    def detect_out_of_bounds_agent(self, agent: Node) -> bool:
        return abs(agent.position[0]) > self.scenario_size or abs(agent.position[1]) > self.scenario_size

    @Pyro5.server.expose
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
            execution_logging=False,
            duration=self.max_episode_length
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

        builder.add_handler(GrADySHandler(self))

        self.sensor_node_ids = []

        SensorProtocol.min_priority = self.min_sensor_priority
        SensorProtocol.max_priority = self.max_sensor_priority
        for i in range(self.num_sensors):

            self.sensor_node_ids.append(builder.add_node(SensorProtocol, (
                random.uniform(-self.scenario_size, self.scenario_size),
                random.uniform(-self.scenario_size, self.scenario_size),
                0
            )))

        self.agent_node_ids = []
        DroneProtocol.speed_action = self.speed_action
        DroneProtocol.algorithm_interval = self.algorithm_iteration_interval

        for i in range(self.num_drones):

            if self.full_random_drone_position:
                self.agent_node_ids.append(builder.add_node(DroneProtocol, (
                    random.uniform(-self.scenario_size, self.scenario_size),
                    random.uniform(-self.scenario_size, self.scenario_size),
                    0
                )))
            else:
                self.agent_node_ids.append(builder.add_node(DroneProtocol, (
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    0
                )))

        self.simulator = builder.build()
        if self.render_mode == "visual":
            self.controller = VisualizationController()

        # Running a single simulation step to get the initial observations
        if not self.simulator.step_simulation():
            raise ValueError("Simulation failed to start")

        self.episode_duration = 0
        self.stall_duration = 0
        self.reward_sum = 0
        self.max_reward = -math.inf
        self.sensors_collected = 0
        self.collection_times = [self.max_episode_length for _ in range(self.num_sensors)]

        return self.observe_simulation(), {}

    @Pyro5.server.expose
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
                    "all_collected": all_sensors_collected,
                    "cause": end_cause
                } for agent in self.agents
            }

        return observations, rewards, terminations, truncations, infos


if __name__ == "__main__":
    args = RemoteEnvironmentArgs().parse_args(known_only=True)
    env_args = EnvironmentArgs().parse_args(known_only=True)

    env = GradysRemoteEnvironment(env_args.render_mode, env_args.algorithm_iteration_interval, env_args.num_drones, env_args.num_sensors,
                                  env_args.scenario_size, env_args.max_episode_length, env_args.max_seconds_stalled, env_args.communication_range,
                                  env_args.state_num_closest_sensors, env_args.state_num_closest_drones, env_args.state_mode, env_args.id_on_state,
                                  env_args.min_sensor_priority, env_args.max_sensor_priority, env_args.full_random_drone_position, env_args.reward,
                                  env_args.speed_action, env_args.end_when_all_collected)

    Pyro5.server.serve(
        {
            env: args.object_name
        }, use_ns=True
    )

    # start = time()
    # index = 0
    #
    # obs, _ = env.reset()
    # while True:
    #     actions = {}
    #     for agent in obs:
    #         actions[agent] = np.random.uniform(0, 1, size=(2,))
    #     obs, rewards, dones, _, _ = env.step(actions)
    #     index += 1
    #     if dones["drone0"]:
    #         obs, _ = env.reset()
    #
    #     if index % 100 == 0:
    #         SPS = index / (time() - start)
    #         print(f"Index: {index}, SPS: {SPS}")

    # obs, _ = env.reset()
    # env.simulator.start_simulation()