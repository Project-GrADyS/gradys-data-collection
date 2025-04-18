from model import Actor
from environment import GrADySEnvironment, observation_space_from_args, action_space_from_args
from heuristics import *
from arguments import *
import torch

# Loading the model
models = [
    (2, 12, "runs/better results/12 to 36 sensors-1728613610.0721927/12 to 36 sensors-checkpoint11000000.cleanrl_model"),
    (2, 24, "runs/better results/12 to 36 sensors-1728613610.0721927/12 to 36 sensors-checkpoint11000000.cleanrl_model"),
    (2, 36, "runs/better results/12 to 36 sensors-1728613610.0721927/12 to 36 sensors-checkpoint11000000.cleanrl_model"),
    #(2, 12, "runs/vary sensor/between 2 and 12-1728347973.973188/between 2 and 12-checkpoint500000.cleanrl_model"),
]

completion_times = []

for num_agents, num_sensors, model_path in models:
    print(f"Running with {num_agents} agents and {num_sensors} sensors")
    print(f"Loading model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_model = torch.load(model_path, map_location=device, weights_only=True)[0]

    # Running model
    env = GrADySEnvironment(
        algorithm_iteration_interval=0.5,
        num_drones=num_agents,
        min_sensor_count=num_sensors,
        max_sensor_count=num_sensors,
        max_seconds_stalled=30,
        scenario_size=100,
        render_mode=None,
        state_num_closest_sensors=12,
        state_num_closest_drones=num_agents - 1,
        min_sensor_priority=1,
        full_random_drone_position=False,
        speed_action=True
    )

    env_args = EnvironmentArgs()
    env_args.state_mode = 'relative'
    env_args.id_on_state = True
    env_args.state_num_closest_sensors = 12
    env_args.state_num_closest_drones = num_agents - 1

    model_args = ModelArgs()

    actor = Actor(action_space_from_args(env_args).shape[0],
                  observation_space_from_args(env_args).shape[0],
                  model_args).to(device)
    actor.load_state_dict(actor_model)

    completion_time_sum = 0

    total_runs = 500
    for i in range(total_runs):
        if i % 100 == 0:
            print(f"Running maddpg: repetition {i}/{total_runs}")

        # Running the model
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                actions = {}
                for agent in env.agents:
                    actions[agent] = actor(torch.Tensor(obs[agent]).to(device)).cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            obs = next_obs
            if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                completion_time_sum += infos[env.agents[0]]['completion_time']

                break
        env.close()

    print(f"Running maddpg: repetition {total_runs}/{total_runs}")

    completion_times.append({
        "Agent Count": num_agents,
        "Sensor Count": num_sensors,
        "Method": "MADDPG",
        "Completion Time (s)": completion_time_sum / total_runs
    })

    # Running heuristic
    heuristics = create_greedy_heuristics(num_agents - 1, 12)

    env = GrADySEnvironment(
        algorithm_iteration_interval=0.5,
        num_drones=num_agents,
        min_sensor_count=num_sensors,
        max_sensor_count=num_sensors,
        max_seconds_stalled=30,
        scenario_size=100,
        render_mode=None,
        state_num_closest_sensors=12,
        state_num_closest_drones=num_agents - 1,
        min_sensor_priority=1,
        full_random_drone_position=False,
        speed_action=True
    )

    completion_time_sum = 0
    for i in range(total_runs):
        if i % 100 == 0:
            print(f"Running heuristic: repetition {i}/{total_runs}")

        # Running the model
        obs, _ = env.reset()
        while True:
            actions = {}
            for agent in env.agents:
                actions[agent] = heuristics(obs[agent])

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            obs = next_obs
            if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                completion_time_sum += infos[env.agents[0]]['completion_time']
                break
        env.close()

    print(f"Running heuristic: repetition {total_runs}/{total_runs}")

    completion_times.append({
        "Agent Count": num_agents,
        "Sensor Count": num_sensors,
        "Method": "Heuristic",
        "Completion Time (s)": completion_time_sum / total_runs
    })

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
ct_df = pd.DataFrame.from_records(completion_times)
ct_df.to_csv("completion_times.csv")
plt.set_loglevel('WARNING')

max_completion_time = ct_df['Completion Time (s)'].max()
for num_agents in [2, 4, 8]:
    ax = sns.barplot(ct_df[ct_df['Agent Count'] == num_agents], x='Sensor Count', y='Completion Time (s)', hue='Method', palette='tab10')
    ax.set_ylim(0, max_completion_time)
    plt.savefig(f"completion_times-{num_agents}a-greedy.png")
    plt.close()

for num_sensors in [12, 24, 36]:
    ax = sns.barplot(ct_df[ct_df['Sensor Count'] == num_sensors], x='Agent Count', y='Completion Time (s)', hue='Method', palette='tab10')
    ax.set_ylim(0, max_completion_time)
    plt.savefig(f"completion_times-{num_sensors}s-greedy.png")
    plt.close()

# Average completion time over method
ax = sns.barplot(ct_df, x='Method', y='Completion Time (s)')
ax.set_ylim(0, max_completion_time)
plt.savefig("completion_times-average-greedy.png")