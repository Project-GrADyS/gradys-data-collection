from collections import defaultdict

from main import Actor
from environment import GrADySEnvironment
import torch

# Loading the model
models = [
    (2, 12, "runs/varying_sensor_results/a_2-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (2, 24, "runs/varying_sensor_results/a_2-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (2, 36, "runs/varying_sensor_results/a_2-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (4, 12, "runs/varying_sensor_results/a_4-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (4, 24, "runs/varying_sensor_results/a_4-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (4, 36, "runs/varying_sensor_results/a_4-s_(12, 36)/checkpoint4999.cleanrl_model"),
    (8, 12, "runs/varying_sensor_results/a_8-s_(12,36)/checkpoint10000.cleanrl_model"),
    (8, 24, "runs/varying_sensor_results/a_8-s_(12,36)/checkpoint10000.cleanrl_model"),
    (8, 36, "runs/varying_sensor_results/a_8-s_(12,36)/checkpoint10000.cleanrl_model"),
]

completion_times = []

for num_agents, num_sensors, model_path in models:
    print(f"Running with {num_agents} agents and {num_sensors} sensors")
    print(f"Loading model from {model_path}")
    device = torch.device("cpu")
    actor_model = torch.load(model_path, map_location=device, weights_only=True)[0]

    if __name__ == "__main__":
        failure_causes = defaultdict(int)

        env = GrADySEnvironment(
            algorithm_iteration_interval=0.5,
            num_drones=num_agents,
            num_sensors=num_sensors,
            max_seconds_stalled=30,
            scenario_size=100,
            render_mode=None,
            state_num_closest_sensors=12,
            state_num_closest_drones=num_agents - 1,
            min_sensor_priority=1,
            full_random_drone_position=False,
            speed_action=True,
            max_sensor_count=num_sensors
        )
        actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
        actor.load_state_dict(actor_model)

        success_count = 0
        collection_times_sum = 0
        completion_time_sum = 0

        total_runs = 100
        for i in range(total_runs):
            if i % 10 == 0:
                print(f"Running experiment: repetition {i}/{total_runs}")

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
                    cause = infos[env.agents[0]]["cause"]
                    failure_causes[cause] += 1

                    success_count += infos[env.agents[0]]['all_collected']
                    collection_times_sum += infos[env.agents[0]]['avg_collection_time']
                    completion_time_sum += infos[env.agents[0]]['completion_time']

                    break
            env.close()

        print(f"Running experiment: repetition {total_runs}/{total_runs}")

        print("-" * 80)
        print("Episode ending causes:")
        for cause, count in failure_causes.items():
            print(f"{cause}: {count/total_runs:.2%}")

        print("-" * 80)
        print(f"Average collection time: {collection_times_sum / total_runs:.2f}s", )
        print(f"Success rate: {success_count / total_runs:.2%}", )
        print(f"Average completion time: {completion_time_sum / total_runs: .2f}s")

        completion_times.append({
            "Agent Count": num_agents,
            "Sensor Count": num_sensors,
            "Completion Time (s)": completion_time_sum / total_runs
        })
        print("\n\n")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
ct_df = pd.DataFrame.from_records(completion_times)
plt.set_loglevel('WARNING')
sns.barplot(ct_df, x='Sensor Count', y='Completion Time (s)', hue='Agent Count', palette='tab10')
plt.savefig("visualization/varying_sensor_completion_times.png")