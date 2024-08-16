from main import Actor
from environment import GrADySEnvironment
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the model
model_path = f"runs/long-run__optimized-punish-no-end__1__1722784275/optimized-punish-no-end-checkpoint1800.cleanrl_model"

print(f"Loading model from {model_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_model = torch.load(model_path, map_location=device)[0]

def get_agent_positions(environment: GrADySEnvironment):
    agent_nodes = [environment.simulator.get_node(agent_id) for agent_id in environment.agent_node_ids]
    return {f"Agent {agent_id}": agent_node.position for agent_id, agent_node in enumerate(agent_nodes)}

def get_sensor_positions(environment: GrADySEnvironment):
    sensor_nodes = [environment.simulator.get_node(sensor_id) for sensor_id in environment.sensor_node_ids]
    return {f"Sensor {sensor_id}": sensor_node.position for sensor_id, sensor_node in enumerate(sensor_nodes)}

if __name__ == "__main__":
    env = GrADySEnvironment(
        algorithm_iteration_interval=0.5,
        num_drones=4,
        num_sensors=12,
        max_seconds_stalled=30,
        scenario_size=100,
        render_mode="visual",
        state_num_closest_sensors=12,
        state_num_closest_drones=3,
        min_sensor_priority=1,
        full_random_drone_position=False,
        speed_action=True
    )
    actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
    actor.load_state_dict(actor_model)

    success_count = 0
    collection_times_sum = 0

    # Running the model
    obs, _ = env.reset()

    position_list = []

    for agent, position in get_agent_positions(env).items():
        position_list.append({
            "agent": agent,
            "x": position[0],
            "y": position[1],
            "timestamp": env.simulator._current_timestamp
        })

    sensor_positions = []
    for sensor, position in get_sensor_positions(env).items():
        sensor_positions.append({
            "sensor": sensor,
            "x": position[0],
            "y": position[1]
        })

    while True:
        with torch.no_grad():
            actions = {}
            for agent in env.agents:
                actions[agent] = actor(torch.Tensor(obs[agent]).to(device)).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent, position in get_agent_positions(env).items():
            position_list.append({
                "agent": agent,
                "x": position[0],
                "y": position[1],
                "timestamp": env.simulator._current_timestamp
            })

        obs = next_obs
        if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
            break
    env.close()


    position_df = pd.DataFrame.from_records(position_list)
    position_df = position_df.set_index("timestamp")

    sensor_df = pd.DataFrame.from_records(sensor_positions)

    # Plot agents paths over time
    # Overlay sensor positions
    sns.set_theme()
    sns.set_context("talk")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.scatterplot(data=sensor_df, x="x", y="y", ax=ax, marker='x', color='black',
                    label='Sensors Positions', s=100, linewidth=2)

    grouped = position_df.groupby("agent")
    # Plot a line for each agent
    for name, group in grouped:
        plt.plot(group['x'], group['y'], marker='o', linestyle='-', ms=5, label=name)

    plt.legend()

    plt.savefig("path.png")
