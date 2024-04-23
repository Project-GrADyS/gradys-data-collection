from abc import ABC, abstractmethod

from gradysim.protocol.interface import IProtocol
from numpy._typing import ArrayLike


class RLAgentProtocol(IProtocol, ABC):
    @abstractmethod
    def act(self, action: ArrayLike) -> None:
        """
        Implement this method to handle the action dictated by the RL agent
        :param action: The action this agent should take
        """
        pass