import numpy as np


def create_greedy_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(observation: np.ndarray) -> np.ndarray:
        closest_agent_position_array = observation[:num_closest_agents * 2]
        closest_sensor_position_array = observation[num_closest_agents * 2:num_closest_agents * 2 + num_closest_sensors * 2]
        agent_id = observation[-1]

        closest_agent_positions = closest_agent_position_array.reshape(-1, 2)
        closest_sensor_positions= closest_sensor_position_array.reshape(-1, 2)

        sensor_distances = np.linalg.norm(closest_sensor_positions, axis=1)

        picked_sensor = 0
        for agent in range(closest_agent_positions.shape[0]):
            agent_position = closest_agent_positions[agent]
            # Ignore empty positions
            if agent_position[0] == 0 and agent_position[1] == 0:
                continue

            sensor_positions_relative_to_agent = closest_sensor_positions - agent_position
            agent_sensor_distances = np.linalg.norm(sensor_positions_relative_to_agent, axis=1)

            if (sensor_distances[picked_sensor] > agent_sensor_distances[picked_sensor]
                    and picked_sensor + 1 < len(sensor_distances)):
                picked_sensor += 1

        picked_sensor_relative_position = closest_sensor_positions[picked_sensor]
        picked_sensor_angle = np.arctan2(picked_sensor_relative_position[1], picked_sensor_relative_position[0])
        return np.array((picked_sensor_angle + np.pi) / (2*np.pi)).reshape((1,))

    return heuristics