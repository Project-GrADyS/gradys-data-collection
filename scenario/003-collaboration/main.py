# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from environment import GrADySEnvironment, StateMode
from heuristics import create_greedy_heuristics


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
    learning_rate: float = 3e-6
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
    policy_frequency: int = 3
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Legacy options
    train_once_for_each_agent: bool = True
    """if toggled, a training iteration will be done for each agent at each timestep"""
    max_episode_length: float = 500
    """the maximum length of the episode"""

    state_mode: StateMode = "relative"
    """chooses the state mode to use"""
    id_on_state: bool = True
    """if toggled, the state will be modified to include the agent's ID"""
    state_num_closest_sensors: int = 2
    """the number of closest sensors to consider in the state"""
    state_num_closest_drones: int = 2
    """the number of closest drones to consider in the state"""

    centralized_critic: bool = True

    algorithm_iteration_interval: float = 0.5
    max_seconds_stalled: int = 30
    num_drones: int = 2
    num_sensors: int = 5
    scenario_size: float = 100
    min_sensor_priority: float = 0.1
    max_sensor_priority: float = 1.0
    full_random_drone_position: bool = True

    critic_model_size: int = 512
    actor_model_size: int = 256

    use_heuristics: bool = False

args = tyro.cli(Args)


def make_env(render_mode=None):
    return GrADySEnvironment(
        algorithm_iteration_interval=args.algorithm_iteration_interval,
        render_mode=render_mode,
        num_drones=args.num_drones,
        num_sensors=args.num_sensors,
        max_episode_length=args.max_episode_length,
        max_seconds_stalled=args.max_seconds_stalled,
        scenario_size=args.scenario_size,
        state_num_closest_sensors=args.state_num_closest_sensors,
        state_num_closest_drones=args.state_num_closest_drones,
        state_mode=args.state_mode,
        id_on_state=args.id_on_state,
        min_sensor_priority=args.min_sensor_priority,
        max_sensor_priority=args.max_sensor_priority,
        full_random_drone_position=args.full_random_drone_position
    )


# ALGO LOGIC: initialize agent here:
if args.centralized_critic:
    class Critic(nn.Module):
        def __init__(self, action_space, observation_space):
            super().__init__()
            self.fc1 = nn.Linear(
                np.array(observation_space.shape).prod() * args.num_drones + np.prod(action_space.shape) * args.num_drones, args.critic_model_size)
            self.fc2 = nn.Linear(args.critic_model_size, args.critic_model_size)
            self.fc3 = nn.Linear(args.critic_model_size, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
else:
    class Critic(nn.Module):
        def __init__(self, action_space, observation_space):
            super().__init__()
            self.fc1 = nn.Linear(
                np.array(observation_space.shape).prod() + np.prod(action_space.shape), args.critic_model_size)
            self.fc2 = nn.Linear(args.critic_model_size, args.critic_model_size)
            self.fc3 = nn.Linear(args.critic_model_size, 1)

        def forward(self, x, a):
            x = torch.cat([x, a], 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

class Actor(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), args.actor_model_size)
        self.fc2 = nn.Linear(args.actor_model_size, args.actor_model_size)
        self.fc_mu = nn.Linear(args.actor_model_size, np.prod(action_space.shape))
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


heuristics = create_greedy_heuristics(args.state_num_closest_drones, args.state_num_closest_sensors)

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
    all_collected_count = 0
    all_avg_collection_times = 0

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
    qf1 = Critic(action_space, observation_space).to(device)
    qf1_target = Critic(action_space, observation_space).to(device)
    target_actor = Actor(action_space, observation_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    if args.centralized_critic:
        extended_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            dtype=np.float32,
            shape=(args.num_drones, *observation_space.shape),
        )
        extended_action_space = gym.spaces.Box(
            low=action_space.low[0],
            high=action_space.high[0],
            dtype=np.float32,
            shape=(args.num_drones, *action_space.shape),
        )
        rb = ReplayBuffer(
            args.buffer_size,
            extended_observation_space,
            extended_action_space,
            device,
            handle_timeout_termination=False,
        )
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            observation_space,
            action_space,
            device,
            handle_timeout_termination=False,
        )

    start_time = time.time()

    def evaluate_checkpoint():
        print("Reached checkpoint at step", global_step)
        model_path = f"runs/{run_name}/{args.exp_name}-checkpoint{global_step // 10_000}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

        temp_env = make_env("visual" if args.checkpoint_visual_evaluation else None)

        actor.eval()
        target_actor.eval()
        qf1.eval()
        qf1_target.eval()

        sum_avg_reward = 0
        sum_max_reward = 0
        sum_sum_reward = 0
        sum_episode_duration = 0
        sum_avg_collection_time = 0
        sum_all_collected = 0

        evaluation_runs = 500
        for i in range(evaluation_runs):
            if i % 100 == 0:
                print(f"Evaluating model ({i+1}/{evaluation_runs})")
            obs, _ = temp_env.reset(seed=args.seed)
            while True:
                actions = {}
                with torch.no_grad():
                    for agent in temp_env.agents:
                        if args.use_heuristics:
                            actions[agent] = heuristics(obs[agent])
                        else:
                            actions[agent] = actor(torch.Tensor(obs[agent]).to(device))
                            actions[agent] += torch.normal(0, actor.action_scale * args.exploration_noise)
                            actions[agent] = actions[agent].cpu().numpy().clip(action_space.low, action_space.high)

                next_obs, _, _, _, infos = temp_env.step(actions)
                obs = next_obs

                if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                    info = infos[env.agents[0]]

                    sum_avg_reward += info["avg_reward"]
                    sum_max_reward += info["max_reward"]
                    sum_sum_reward += info["sum_reward"]
                    sum_episode_duration += info["episode_duration"]
                    sum_avg_collection_time += info["avg_collection_time"]
                    sum_all_collected += info["all_collected"]
                    break
            temp_env.close()
        
        writer.add_scalar(
            "eval/avg_reward",
            sum_avg_reward / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/max_reward",
            sum_max_reward / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/sum_reward",
            sum_sum_reward / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/episode_duration",
            sum_episode_duration / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/avg_collection_time",
            sum_avg_collection_time / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/all_collected_rate",
            sum_all_collected / evaluation_runs,
            global_step,
        )
        temp_env.close()

        actor.train()
        target_actor.train()
        qf1.train()
        qf1_target.train()

        print("Checkpoint evaluation done")

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    terminated = False
    for global_step in range(args.total_timesteps):
        step_start = time.time()

        if terminated:
            obs, _ = env.reset(seed=args.seed)
            terminated = False

        if args.use_heuristics:
            actions = {
                agent: heuristics(obs[agent]) for agent in env.agents
            }
        else:
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

            avg_reward = info["avg_reward"]
            max_reward = info["max_reward"]
            sum_reward = info["sum_reward"]

            avg_collection_time = info["avg_collection_time"]

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
                "charts/sum_reward",
                sum_reward,
                global_step,
            )
            writer.add_scalar(
                "charts/episode_duration",
                info["episode_duration"],
                global_step,
            )
            all_avg_collection_times += avg_collection_time
            writer.add_scalar(
                "charts/avg_collection_time",
                all_avg_collection_times / episode_count,
                global_step,
            )
            all_collected_count += info["all_collected"]
            writer.add_scalar(
                "charts/all_collected_rate",
                all_collected_count / episode_count,
                global_step,
            )

        if args.use_heuristics:
            obs = next_obs
        elif args.centralized_critic:
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            all_agent_next_obs = np.stack([next_obs[agent] for agent in env.agents])
            all_agent_obs = np.stack([obs[agent] for agent in env.agents])
            all_agent_actions = np.stack([actions[agent] for agent in env.agents])
            rb.add(
                all_agent_obs,
                all_agent_next_obs,
                all_agent_actions,
                np.array(rewards[env.agents[0]]),
                np.array(terminations[env.agents[0]]),
                [infos.get(agent, {}) for agent in env.agents]
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step <= args.learning_starts:
                continue
            
            training_step_count = args.num_drones if args.train_once_for_each_agent else 1
            data = rb.sample(args.batch_size * training_step_count)

            with torch.no_grad():
                next_actions = target_actor(data.next_observations).view(data.next_observations.shape[0], -1)
                next_state = data.next_observations.reshape(data.next_observations.shape[0], -1)
                qf1_next_target = qf1_target(next_state, next_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

            current_state = data.observations.reshape(data.observations.shape[0], -1)
            all_actions = data.actions.reshape(data.actions.shape[0], -1)
            qf1_a_values = qf1(current_state, all_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_actions = actor(data.observations).view(data.observations.shape[0], -1)

                actor_loss = -qf1(current_state, actor_actions).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.lerp_(param.data, args.tau)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.lerp_(param.data, args.tau)
        else:
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
            if global_step <= args.learning_starts:
                continue

            training_step_count = args.num_drones if args.train_once_for_each_agent else 1
            for _ in range(training_step_count):
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
            if not args.use_heuristics:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/step_duration", time.time() - step_start, global_step)
            print(f"{args.exp_name} - SPS:", int(global_step / (time.time() - start_time)))

        if args.checkpoints and global_step % args.checkpoint_freq == 0 and global_step > 0:
            evaluate_checkpoint()

    evaluate_checkpoint()
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
