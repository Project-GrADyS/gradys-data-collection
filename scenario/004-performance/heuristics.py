import numpy as np


def create_greedy_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(observation: np.ndarray) -> np.ndarray:
        closest_sensor_position_array = observation[num_closest_agents * 2:num_closest_agents * 2 + num_closest_sensors * 2]

        closest_sensor_positions = closest_sensor_position_array.reshape(-1, 2)
        # Remove sensors at relative position (0, 0)
        closest_sensor_positions = closest_sensor_positions[~np.all(closest_sensor_positions == 0, axis=1)]

        # Find the closest sensor
        picked_sensor = np.argmin(np.linalg.norm(closest_sensor_positions, axis=1))

        # Find the relative position of the picked sensor
        picked_sensor_relative_position = closest_sensor_positions[picked_sensor]
        picked_sensor_angle = np.arctan2(picked_sensor_relative_position[1], picked_sensor_relative_position[0])
        return np.array((picked_sensor_angle + np.pi) / (2*np.pi)).reshape((1,))

    return heuristics

def create_random_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(_observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, 1, size=(2,))
    return heuristics