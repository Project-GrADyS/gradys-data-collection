from typing import Dict

import numpy as np
from gradysim.protocol.position import squared_distance
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import Simulator, SimulationBuilder, SimulationConfiguration
from gymnasium.spaces import Box
from numpy._typing import ArrayLike

from base import GrADySEnvironment
from scenario.protocol import create_sensor_protocol, DroneProtocol


class CollectionEnvironment(GrADySEnvironment):
    slow_sensor: int
    fast_sensor: int
    drone: int

    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.possible_agents = ["drone"]
        self.agents = ["drone"]

    def observation_space(self, agent):
        return Box(0, 1, shape=(2,), dtype=np.float32)

    def action_space(self, agent):
        return Box(0, 1)

    def get_simulation_configuration(self) -> SimulationConfiguration:
        return SimulationConfiguration(
            real_time=True,
            duration=100
        )

    def create_simulation_scenario(self, builder: SimulationBuilder):
        self.slow_sensor = builder.add_node(create_sensor_protocol(
            buffer_size=100,
            min_packet_size=1,
            max_packet_size=2,
            min_packet_generation_interval=9.5,
            max_packet_generation_interval=10
        ), (100, 0, 0))

        self.fast_sensor = builder.add_node(create_sensor_protocol(
            buffer_size=100,
            min_packet_size=10,
            max_packet_size=12,
            min_packet_generation_interval=9.5,
            max_packet_generation_interval=10
        ), (-100, 0, 0))

        self.drone = builder.add_node(DroneProtocol, (0, 0, 0))

    def observe_simulation(self, simulator: Simulator) -> Dict[str, ArrayLike]:
        state = {}

        slow_sensor = simulator.get_node(self.slow_sensor)
        fast_sensor = simulator.get_node(self.fast_sensor)

        for agent in self.rl_agents:
            agent_node = simulator.get_node(self.drone)
            position = agent_node.position

            angle_to_first_sensor = (
                np.arctan2(slow_sensor.position[1] - position[1], slow_sensor.position[0] - position[0]))
            angle_to_second_sensor = (
                np.arctan2(fast_sensor.position[1] - position[1], fast_sensor.position[0] - position[0]))

            state[agent] = np.array([angle_to_first_sensor / (2 * np.pi), angle_to_second_sensor / (2 * np.pi)])

        return state

    def compute_agent_rewards(self, _previous, _next, actions) -> Dict[str, float]:
        # slow_sensor = self.simulator.get_node(self.slow_sensor)
        # fast_sensor = self.simulator.get_node(self.fast_sensor)
        #
        # lost_packets = slow_sensor.protocol_encapsulator.protocol.buffer.ovewritten_packets
        # lost_packets += fast_sensor.protocol_encapsulator.protocol.buffer.ovewritten_packets
        # lost_packets /= 2
        #
        # return {"drone": -lost_packets}
        return {"drone": -squared_distance(self.simulator.get_node(self.drone).position, self.simulator.get_node(self.fast_sensor).position) / 100*100}



