from collections import defaultdict

from environment import GrADySEnvironment
from heuristics import *

models = [
    (2, 12),
    (2, 24),
    (2, 36),
    (4, 12),
    (4, 24),
    (4, 36),
    (8, 12),
    (8, 24),
    (8, 36),
]


completion_times = []

for num_agents, num_sensors in models:
    print(f"Running with {num_agents} agents and {num_sensors} sensors")
    # Loading the model
    heuristics = create_smart_heuristics(num_agents - 1, 12)

    if __name__ == "__main__":
        failure_causes = defaultdict(int)

        env = GrADySEnvironment(
            algorithm_iteration_interval=0.5,
            num_drones=num_agents,
            num_sensors=num_sensors,
            max_seconds_stalled=30,
            scenario_size=100,
            render_mode="visual",
            state_num_closest_sensors=12,
            state_num_closest_drones=num_agents - 1,
            min_sensor_priority=1,
            full_random_drone_position=False,
            speed_action=True
        )

        success_count = 0
        collection_times_sum = 0
        completion_time_sum = 0

        total_runs = 200
        for i in range(total_runs):
            if i % 100 == 0:
                print(f"Running experiment: repetition {i}/{total_runs}")

            # Running the model
            obs, _ = env.reset()
            while True:
                actions = {}
                for agent in env.agents:
                    actions[agent] = heuristics(obs[agent])

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
plt.savefig("completion_times_heuristics.png")