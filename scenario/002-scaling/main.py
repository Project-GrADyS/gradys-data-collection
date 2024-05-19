# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from environment import GrADySEnvironment


@dataclass
class Args:
    run_name: str = "GrADyS"
    """the name of the run"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoints: bool = True
    """whether to save model checkpoints"""
    checkpoint_freq: int = 1_000_000
    """the frequency of checkpoints"""
    checkpoint_visual_evaluation: bool = False
    """whether to visually evaluate the model at each checkpoint"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    early_stopping: bool = True
    """if toggled, early stopping will be enabled based on the success rate"""
    early_stopping_beginning: int = 1_000
    """the beginning of early stopping"""
    early_stopping_patience: int = 10_000
    """the patience of early stopping"""
    early_stopping_minimum: float = 0.7
    """the minimum success rate to consider early stopping"""
    early_stopping_tolerance: float = 0.02
    """the tolerance of early stopping"""

    soft_reward: bool = True
    """if toggled, soft reward will be used"""
    state_num_closest_sensors: int = 2
    """the number of closest sensors to consider in the state"""
    state_num_closest_drones: int = 2
    """the number of closest drones to consider in the state"""

    algorithm_iteration_interval: float = 0.5

    max_seconds_stalled: int = 30
    num_drones: int = 1
    num_sensors: int = 2
    scenario_size: float = 100
    randomize_sensor_positions: bool = True


def make_env(render_mode=None):
    return GrADySEnvironment(
        algorithm_iteration_interval=args.algorithm_iteration_interval,
        render_mode=render_mode,
        num_drones=args.num_drones,
        num_sensors=args.num_sensors,
        max_seconds_stalled=args.max_seconds_stalled,
        scenario_size=args.scenario_size,
        randomize_sensor_positions=args.randomize_sensor_positions,
        soft_reward=args.soft_reward,
        state_num_closest_sensors=args.state_num_closest_sensors,
        state_num_closest_drones=args.state_num_closest_drones,
    )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


args = tyro.cli(Args)


class EarlyStopping:
    def __init__(self):
        self.counter = 0
        self.best_score = None
        self.blocked = True

    def __call__(self, score, step):
        if step % 100 == 0:
            writer.add_scalar("charts/early_stopping_counter", self.counter, step)
            writer.add_scalar("charts/early_stopping_best_score", self.best_score or 0, step)

        if score >= args.early_stopping_minimum and step >= args.early_stopping_beginning:
            self.blocked = False

        if self.blocked:
            return False

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score * (1 + args.early_stopping_tolerance):
            self.counter += 1
            if self.counter >= args.early_stopping_patience:
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False

run_name = f"{args.run_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")

def main():
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Statistics
    episode_count = 0
    number_of_successes = 0

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env()

    observation_space = env.observation_space(0)
    action_space = env.action_space(0)

    assert isinstance(action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(action_space, observation_space).to(device)
    qf1 = QNetwork(action_space, observation_space).to(device)
    qf1_target = QNetwork(action_space, observation_space).to(device)
    target_actor = Actor(action_space, observation_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    early_stopping = EarlyStopping()

    def save_checkpoint():
        print("Reached checkpoint at step", global_step)
        model_path = f"runs/{run_name}/{args.exp_name}-checkpoint{global_step // 10_000}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

        if args.checkpoint_visual_evaluation:
            print("Visually evaluating the model")
            temp_env = make_env("visual")
            obs, _ = temp_env.reset(seed=args.seed)
            for _ in range(100):
                with torch.no_grad():
                    actions = {}
                    for agent in temp_env.agents:
                        actions[agent] = actor(torch.Tensor(obs[agent]).to(device))
                        actions[agent] += torch.normal(0, actor.action_scale * args.exploration_noise)
                        actions[agent] = actions[agent].cpu().numpy().clip(action_space.low, action_space.high)

                next_obs, rewards, terminations, truncations, infos = temp_env.step(actions)
                obs = next_obs
                if len(infos) > 0 and "avg_reward" in infos[temp_env.agents[0]]:
                    break
            temp_env.close()
        print("Checkpoint evaluation done")


    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    terminated = False
    for global_step in range(args.total_timesteps):
        step_start = time.time()

        if terminated:
            obs, _ = env.reset(seed=args.seed)
            terminated = False

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = {
                agent: action_space.sample() for agent in env.agents
            }
        else:
            with torch.no_grad():
                actions = {}
                all_obs = torch.tensor(np.array([obs[agent] for agent in env.agents]),
                                       device=device,
                                       dtype=torch.float32)
                all_actions: torch.Tensor = actor(all_obs)
                all_actions.add_(torch.normal(torch.zeros_like(all_actions, device=device),
                                              actor.action_scale * args.exploration_noise))
                all_actions.clip_(torch.tensor(action_space.low, device=device),
                                  torch.tensor(action_space.high, device=device))
                for index, agent in enumerate(env.agents):
                    actions[agent] = all_actions[index].cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
            episode_count += 1
            terminated = True

            info = infos[env.agents[0]]

            avg_reward = sum([info["avg_reward"] for info in infos.values()]) / len(infos)
            max_reward = max([info["max_reward"] for info in infos.values()])

            writer.add_scalar(
                "charts/avg_reward",
                avg_reward,
                global_step,
            )
            writer.add_scalar(
                "charts/max_reward",
                max_reward,
                global_step,
            )
            writer.add_scalar(
                "charts/episode_duration",
                info["episode_duration"],
                global_step,
            )
            number_of_successes += info["success"]
            writer.add_scalar(
                "charts/success_rate",
                number_of_successes / episode_count,
                global_step,
            )

            if args.early_stopping:
                if early_stopping(number_of_successes / episode_count, global_step):
                    print("Early stopping after", episode_count, "episodes")
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        for agent in env.agents:
            rb.add(obs[agent],
                   real_next_obs[agent],
                   actions[agent],
                   np.array([rewards[agent]]),
                   np.array([terminations[agent]]),
                   [infos.get(agent, {})])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for _ in env.agents:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        qf1_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.lerp_(param.data, args.tau)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.lerp_(param.data, args.tau)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/step_duration", time.time() - step_start, global_step)
                print(f"{args.exp_name} - SPS:", int(global_step / (time.time() - start_time)))

        if args.checkpoints and global_step % args.checkpoint_freq == 0 and global_step > 0:
            save_checkpoint()

    save_checkpoint()
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
