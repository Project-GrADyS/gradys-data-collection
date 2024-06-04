from collections import defaultdict

from main import Actor
from environment import GrADySEnvironment
import torch

# Loading the model
model_path = f"runs/GrADyS__2-drone-closest-2-sensor-0-drone__1__1717379166/2-drone-closest-2-sensor-0-drone-checkpoint99.cleanrl_model"
print(f"Loading model from {model_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_model = torch.load(model_path, map_location=device)[0]

if __name__ == "__main__":
    failure_causes = defaultdict(int)

    env = GrADySEnvironment(
        algorithm_iteration_interval=0.5,
        num_drones=2,
        num_sensors=2,
        max_seconds_stalled=30,
        scenario_size=100,
        render_mode="visual",
        randomize_sensor_positions=True,
        state_num_closest_sensors=2,
        state_num_closest_drones=0,
        min_sensor_priority=1
    )
    actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
    actor.load_state_dict(actor_model)

    total_runs = 1_000
    for i in range(total_runs):
        if i % 100 == 0:
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
                break
        env.close()

    print("-" * 80)
    print("Episode ending causes:")
    for cause, count in failure_causes.items():
        print(f"{cause}: {count/total_runs:.2%}")