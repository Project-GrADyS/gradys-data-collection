from typing import List, Dict

import numpy as np
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.handler.assertion import assert_eventually_true_for_protocol, AssertionHandler
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import Simulator, SimulationBuilder, SimulationConfiguration
from gymnasium.spaces import Box, Discrete
from numpy.typing import ArrayLike
from pettingzoo.utils import wrappers, parallel_to_aec

from agent import RLAgentProtocol
from environment import GrADySEnvironment


class PingProtocol(RLAgentProtocol):
    received: List[str]
    sent: List[str]

    def act(self, action: ArrayLike) -> None:
        content = action
        command = BroadcastMessageCommand(str(content))
        self.provider.send_communication_command(command)
        self.sent.append(str(content))

    def initialize(self) -> None:
        self.received = []
        self.sent = []

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        self.received.append(message)

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass


@assert_eventually_true_for_protocol(PingProtocol, "received_sent_correct")
def assert_received_and_sent(node: Node[PingProtocol]):
    return len(node.protocol_encapsulator.protocol.sent) > 0 and len(node.protocol_encapsulator.protocol.received) > 0


class TestEnvironment(GrADySEnvironment):
    def __init__(self, render_mode=None):
        super().__init__(render_mode, algorithm_iteration_interval=1)

        self.possible_agents = [f"agent_{i}" for i in range(10)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = {
            f"agent_{i}": i for i in range(len(self.possible_agents))
        }

    def get_simulation_configuration(self):
        return SimulationConfiguration(
            duration=10
        )

    def create_simulation_scenario(self, builder: SimulationBuilder):
        builder.add_handler(AssertionHandler([assert_received_and_sent]))

        for i in range(10):
            builder.add_node(PingProtocol, (0, 0, 0))

    def observation_space(self, agent):
        return Box(0, 1)

    def action_space(self, agent):
        return Discrete(10, start=0)

    def observe_simulation(self, simulator: Simulator) -> Dict[str, ArrayLike]:
        return {
            agent: np.array([0]) for agent in self.agents
        }

    def compute_agent_rewards(self, previous: Simulator, next: Simulator, actions) -> Dict[str, float]:
        return {
            agent: 0 for agent in self.agents
        }


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    original_env = raw_env(render_mode=internal_render_mode)
    environment = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    # this wrapper helps error handling for discrete action spaces
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    environment = TestEnvironment(render_mode=render_mode)
    environment = parallel_to_aec(environment)
    return environment


def test_environment():
    env_instance = env()
    env_instance.reset()

    env_instance.reset(seed=42)

    for agent in env_instance.agent_iter():
        observation, reward, termination, truncation, info = env_instance.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env_instance.action_space(agent).sample()

        env_instance.step(action)

    raw_environment: GrADySEnvironment = env_instance.env.env.env.env
    for agent in raw_environment.simulator_agents:
        assert len(agent.protocol_encapsulator.protocol.received) == 90
        assert len(agent.protocol_encapsulator.protocol.sent) == 10
    env_instance.close()
