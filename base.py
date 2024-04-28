import copy
from abc import ABC, abstractmethod
from time import sleep
from typing import List, Optional, Literal, Dict, final

import gymnasium
from gradysim.protocol.interface import IProtocol
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler, VisualizationConfiguration
from gradysim.simulator.node import Node
from gradysim.simulator.simulation import Simulator, SimulationConfiguration, SimulationBuilder
from gymnasium.spaces import Discrete
from numpy._typing import ArrayLike
from pettingzoo import ParallelEnv


class RLAgentProtocol(IProtocol, ABC):
    @abstractmethod
    def act(self, action: ArrayLike) -> None:
        """
        Implement this method to handle the action dictated by the RL agent
        :param action: The action this agent should take
        """
        pass


class GrADySEnvironment(ParallelEnv, ABC):
    simulator: Simulator
    rl_agents: Dict[str, Node[RLAgentProtocol]]
    nodes: List[Node]

    episode_returns: Dict[str, float]
    episode_length: int

    # Indicates if an algorithm iteration has finished in the simulation
    algorithm_iteration_finished: bool


    metadata = {"render_modes": ["visual", "console"], "name": "gradys-env"}

    def __init__(self,
                 render_mode: Optional[Literal["visual", "console"]] =None,
                 algorithm_iteration_interval: float = 0.5):
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

    @abstractmethod
    def get_simulation_configuration(self) -> SimulationConfiguration:
        """
        Returns the simulation configuration to be used in simulations build with this environment
        """
        pass

    @abstractmethod
    def create_simulation_scenario(self, builder: SimulationBuilder):
        """
        Takes the SimulationBuilder object and creates the simulation scenario.
        """
        pass

    @abstractmethod
    def observe_simulation(self, simulator: Simulator) -> Dict[str, ArrayLike]:
        """
        Creates observations based on the current state of the simulation
        :return: A dictionary of observations for each agent in the format { "agent_id": observation }.
        """
        pass

    def collect_simulation_info(self, simulator: Simulator) -> Dict[str, dict]:
        """
        Collects additional information for each agent in the simulation.
        :return: A dictionary of dictionaries of information for each agent in the format { "agent_id": { "info_key": info } }
        """
        return {agent: {"teste": 1} for agent in self.agents}

    @abstractmethod
    def compute_agent_rewards(self, previous: Dict[str, ArrayLike], next: Dict[str, ArrayLike], actions) -> Dict[str, float]:
        """
        Computes the rewards for each agent based on the previous and next states of the simulation
        :param previous: The previous state of the simulation
        :param next: The next state of the simulation
        :param actions: The actions taken by each agent
        :return: A dictionary of rewards for each agent in the format { "agent_id": reward }
        """
        pass

    @abstractmethod
    def observation_space(self, agent):
        return Discrete(4)

    @abstractmethod
    def action_space(self, agent):
        return Discrete(3)

    @final
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

    @final
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    @final
    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.episode_returns = {agent: 0 for agent in self.possible_agents}
        self.episode_length = 0

        self.agents = self.possible_agents.copy()
        self.nodes = []

        builder = SimulationBuilder(self.get_simulation_configuration())
        builder.add_handler(CommunicationHandler())
        builder.add_handler(MobilityHandler())
        builder.add_handler(TimerHandler())

        if self.render_mode == "visual":
            builder.add_handler(VisualizationHandler(VisualizationConfiguration(open_browser=True)))

        agents = iter(self.agents)
        # Special handler to give us access to al RL agents. Also
        class GrADySHandler(INodeHandler):
            event_loop: EventLoop

            @staticmethod
            def get_label() -> str:
                return "GrADySHandler"

            def inject(self, event_loop: EventLoop) -> None:
                self.event_loop = event_loop
                self.last_iteration = 0
                self.iterate_algorithm()

            def register_node(handler_self, node: Node) -> None:
                if isinstance(node.protocol_encapsulator.protocol, RLAgentProtocol):
                    self.rl_agents[next(agents)] = node
                self.nodes.append(node)

            def iterate_algorithm(handler_self):
                self.algorithm_iteration_finished = True

                handler_self.event_loop.schedule_event(
                    handler_self.event_loop.current_time + self.algorithm_iteration_interval,
                    handler_self.iterate_algorithm
                )

        builder.add_handler(GrADySHandler())

        self.rl_agents = {}
        self.create_simulation_scenario(builder)

        self.simulator = builder.build()

        # Running a single simulation step to get the initial observations
        if not self.simulator.step_simulation():
            raise ValueError("Simulation failed to start")

        return self.observe_simulation(self.simulator), self.collect_simulation_info(self.simulator)

    @final
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
        self.episode_length += 1

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Saving current state
        state_before = self.observe_simulation(self.simulator)

        # Acting
        for agent, action in actions.items():
            agent_node = self.rl_agents[agent]
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

        # Collecting next state
        state_after = self.observe_simulation(self.simulator)

        rewards = self.compute_agent_rewards(state_before, state_after, actions)
        for agent, reward in rewards.items():
            self.episode_returns[agent] += reward

        terminations = {agent: False for agent in self.agents}

        truncations = {agent: not simulation_ongoing for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = self.observe_simulation(self.simulator)

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = self.collect_simulation_info(self.simulator)

        if not simulation_ongoing:
            self.agents = []

        return observations, rewards, terminations, truncations, infos
