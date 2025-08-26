from main import Actor
from environment import GrADySEnvironment
from heuristics import *
import torch

# Loading the model
models = [
    (2, 12, "runs/results/results__a_2-s_12__1__1723034080/a_2-s_12-checkpoint4999.cleanrl_model"),
    (2, 24, "runs/results/results__a_2-s_24__1__1723034081/a_2-s_24-checkpoint4999.cleanrl_model"),
    (2, 36, "runs/results/results__a_2-s_36__1__1723034081/a_2-s_36-checkpoint4999.cleanrl_model"),
    (4, 12, "runs/results/results__a_4-s_12__1__1723034081/a_4-s_12-checkpoint4999.cleanrl_model"),
    (4, 24, "runs/results/results__a_4-s_24__1__1723034081/a_4-s_24-checkpoint4999.cleanrl_model"),
    (4, 36, "runs/results/results__a_4-s_36__1__1723034081/a_4-s_36-checkpoint4999.cleanrl_model"),
    (8, 12, "runs/results/results__a_8-s_12__1__1726583488/a_8-s_12-checkpoint4999.cleanrl_model"),
    (8, 24, "runs/results/results__a_8-s_24__1__1723034081/a_8-s_24-checkpoint4999.cleanrl_model"),
    (8, 36, "runs/results/results__a_8-s_36__1__1723034081/a_8-s_36-checkpoint4999.cleanrl_model"),
]

completion_time_sum = 0
successful_runs = 0
total_runs = 0

for num_agents, num_sensors, model_path in models:
    print(f"Running with {num_agents} agents and {num_sensors} sensors")
    print(f"Loading model from {model_path}")
    device = torch.device("cpu")
    actor_model = torch.load(model_path, map_location=device, weights_only=True)[0]

    # Running model
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
        speed_action=True
    )
    actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
    actor.load_state_dict(actor_model)

    run_count = 500
    for i in range(run_count):
        if i % 100 == 0:
            print(f"Running maddpg: repetition {i}/{run_count}")

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
                total_runs += 1
                if infos[env.agents[0]]['all_collected']:
                    completion_time_sum += infos[env.agents[0]]['completion_time']
                    successful_runs += 1

                break
        env.close()

    print(f"Running maddpg: repetition {run_count}/{run_count}")

print("------------------------------")
print(f"Total successful runs: {successful_runs}")
print(f"Total runs: {total_runs}")
print(f"Success rate: {successful_runs / total_runs * 100:.2f}%")
print(f"Average completion time: {completion_time_sum/successful_runs:.2f}s")