import numpy as np
import math

def create_greedy_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(observation: np.ndarray) -> np.ndarray:
        closest_sensor_position_array = observation[num_closest_agents * 2:num_closest_agents * 2 + num_closest_sensors * 2] * 2 -1

        closest_sensor_positions = closest_sensor_position_array.reshape(-1, 2)
        # Remove sensors at relative position (0, 0)
        closest_sensor_positions = closest_sensor_positions[~np.all(closest_sensor_positions == 0, axis=1)]

        # Find the closest sensor
        picked_sensor = np.argmin(np.linalg.norm(closest_sensor_positions, axis=1))

        # Find the relative position of the picked sensor
        picked_sensor_relative_position = closest_sensor_positions[picked_sensor]
        picked_sensor_angle = np.arctan2(picked_sensor_relative_position[1], picked_sensor_relative_position[0])
        return np.array([(picked_sensor_angle + np.pi) / (2*np.pi), 1]).reshape((2,))

    return heuristics

def create_smart_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(observation: np.ndarray) -> np.ndarray:
        closest_sensor_position_array = observation[num_closest_agents * 2:num_closest_agents * 2 + num_closest_sensors * 2]
        closest_sensor_positions = closest_sensor_position_array.reshape(-1, 2).tolist()
        closest_sensor_positions = [pos for pos in closest_sensor_positions if pos[0] != -1 or pos[1] != -1]

        closest_agent_position_array = observation[:num_closest_agents * 2]
        closest_agent_positions = closest_agent_position_array.reshape(-1, 2).tolist()
        closest_agent_positions = [pos for pos in closest_agent_positions if pos[0] != -1 or pos[1] != -1]
        closest_agent_positions = [[0.5, 0.5]] + closest_agent_positions

        distances = []
        for a in closest_agent_positions:
            a_list = []
            for s in closest_sensor_positions:
                a_list.append(math.sqrt((a[0] - s[0]) ** 2 + (a[1] - s[1]) ** 2))
            distances.append(a_list)

        sensor_assigments: list[None | int] = [
            None for _ in range(len(closest_agent_positions))
        ]

        max_steps = 100
        unmodified_count = 0
        for i in range(max_steps):
            previous_assignments = sensor_assigments.copy()
            agent = i % len(closest_agent_positions)
            free_sensors = [s
                            for s in range(len(closest_sensor_positions))
                            if s not in sensor_assigments
                            or distances[agent][s] < distances[sensor_assigments.index(s)][s]]

            # Nothing to consider
            if len(free_sensors) == 0:
                continue

            # New sensor is better than last one
            chosen_sensor = min(free_sensors, key=lambda s: distances[agent][s])
            if sensor_assigments[agent] is None or distances[agent][sensor_assigments[agent]] > distances[agent][chosen_sensor]:
                if chosen_sensor in sensor_assigments:
                    old_index = sensor_assigments.index(chosen_sensor)
                    sensor_assigments[old_index] = None

                sensor_assigments[agent] = chosen_sensor

            # Guy who took my sensor is better than me


            if previous_assignments == sensor_assigments:
                unmodified_count += 1
                if unmodified_count >= len(closest_agent_positions):
                    break
            else:
                unmodified_count = 0

        # No more alternatives
        if sensor_assigments[0] is None:
            sensor_assigments[0] = int(np.argmin(distances[0]))

        my_choice = closest_sensor_positions[sensor_assigments[0]]
        picked_sensor_angle = np.arctan2(my_choice[1] * 2 - 1, my_choice[0] * 2 - 1)
        return np.array([(picked_sensor_angle + np.pi) / (2*np.pi), 1]).reshape((2,))
    return heuristics

def create_random_heuristics(num_closest_agents: int, num_closest_sensors: int):
    def heuristics(_observation: np.ndarray) -> np.ndarray:
        return np.random.uniform(0, 1, size=(2,))
    return heuristics