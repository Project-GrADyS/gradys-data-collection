import random

import numpy as np
from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.protocol.messages.telemetry import Telemetry
from numpy._typing import ArrayLike

from base import RLAgentProtocol


class RollingPacketBuffer:
    def __init__(self, size: int):
        self.size = size

        self._buffer: np.array = np.zeros(size, dtype=np.uint16)
        self._buffer_cursor: int = 0

        self._packet_identifier: int = 1

        self._ovewritten_packets = 0

    def _nunique_in_range(self, start: int, end: int) -> int:
        """
        Returns the number of unique elements (excluding zero which marks empty spaces in the buffer) in the given range
        :param start: Start of the range
        :param end: End of the range (non-inclusive)
        :return: Number of unique elements
        """
        buffer_range = self._buffer[start:end]
        return len(np.unique(buffer_range[buffer_range != 0]))

    def add_packet(self, packet_size: int):
        if packet_size > self.size:
            raise ValueError("Packet size is larger than buffer size")

        if self._buffer_cursor + packet_size > self.size:
            self._buffer_cursor = 0

        self._ovewritten_packets += self._nunique_in_range(self._buffer_cursor, self._buffer_cursor + packet_size)

        self._buffer[self._buffer_cursor:self._buffer_cursor + packet_size] = self._packet_identifier
        self._packet_identifier += 1
        self._buffer_cursor += packet_size

    def clear_buffer(self) -> None:
        self._buffer[:] = 0
        self._buffer_cursor = 0
        self._packet_identifier = 1
        self._ovewritten_packets = 0

    @property
    def ovewritten_packets(self) -> int:
        return self._ovewritten_packets

    @property
    def packet_count(self) -> int:
        return self._nunique_in_range(0, self.size)

    @property
    def occupied_space(self) -> float:
        return np.sum(self._buffer != 0) / self.size

    @property
    def free_space(self) -> float:
        return 1 - self.occupied_space


def create_sensor_protocol(
        buffer_size: int,
        min_packet_size: int,
        max_packet_size: int,
        max_packet_generation_interval: float,
        min_packet_generation_interval: float
) -> type[IProtocol]:
    class SensorProtocol(IProtocol):
        buffer: RollingPacketBuffer

        def initialize(self) -> None:
            self.buffer = RollingPacketBuffer(buffer_size)
            self.generate_packet()

        def handle_timer(self, timer: str) -> None:
            if timer == "packet":
                self.generate_packet()

        def handle_packet(self, message: str) -> None:
            if message == "collect":
                self.buffer.clear_buffer()

        def handle_telemetry(self, telemetry: Telemetry) -> None:
            pass

        def finish(self) -> None:
            pass

        def generate_packet(self) -> None:
            packet_size = random.randint(min_packet_size, max_packet_size)
            self.buffer.add_packet(packet_size)

            interval = random.uniform(min_packet_generation_interval, max_packet_generation_interval)
            self.provider.schedule_timer("packet", self.provider.current_time() + interval)

    return SensorProtocol


class DroneProtocol(RLAgentProtocol):
    current_position: tuple[float, float, float]

    def act(self, action: ArrayLike) -> None:
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
        if timer == "collect":
            self._collect_packets()

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.current_position = telemetry.current_position

    def _collect_packets(self) -> None:
        command = BroadcastMessageCommand("collect")
        self.provider.send_communication_command(command)

        self.provider.schedule_timer("collect", self.provider.current_time() + 1)

    def finish(self) -> None:
        pass
