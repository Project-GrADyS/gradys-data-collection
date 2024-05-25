from main import Actor
from environment import GrADySEnvironment
import torch

# Loading the model
model_path = f"runs/sensors/3-sensor-soft/3-sensor-soft-checkpoint223.cleanrl_model"
print(f"Loading model from {model_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_model = torch.load(model_path, map_location=device)[0]

if __name__ == "__main__":
    for i in range(10):
        print(f"Running experiment: repetition {i}")
        # Creating environment
        env = GrADySEnvironment(
            algorithm_iteration_interval=0.5,
            render_mode="visual",
            num_drones=1,
            num_sensors=3,
            max_seconds_stalled=30,
            scenario_size=100,
            randomize_sensor_positions=True,
            soft_reward=True,
            state_num_closest_sensors=2,
            state_num_closest_drones=2,
        )

        actor = Actor(env.action_space(0), env.observation_space(0)).to(device)
        actor.load_state_dict(actor_model)

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
                print(f"Experiment finished. Success: {infos[env.agents[0]]['success']}")
                break
        env.close()