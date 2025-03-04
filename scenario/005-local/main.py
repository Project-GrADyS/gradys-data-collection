# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from environment import GrADySEnvironment, StateMode
from heuristics import create_greedy_heuristics, create_random_heuristics
from fast_replay import FastReplayBuffer

torch.set_float32_matmul_precision('high')

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
    checkpoints: bool = True
    """whether to save model checkpoints"""
    checkpoint_freq: int = 1_000_000
    """the frequency of checkpoints"""
    checkpoint_visual_evaluation: bool = False
    """whether to visually evaluate the model at each checkpoint"""
    statistics_frequency: float = 10_000
    """statistics will be saved to the tensorboard logs with this frequency"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 3e-6
    """the learning rate of the actor optimizer"""
    critic_learning_rate: float = 3e-6
    """the learning rate of the critic optimizer"""
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
    local_observation: bool = False
    critic_global_state: bool = False

    algorithm_iteration_interval: float = 0.5
    max_seconds_stalled: int = 30
    end_when_all_collected: bool = False
    num_drones: int = 2
    min_num_sensors: int = 5
    max_num_sensors: int = 12
    scenario_size: float = 100
    min_sensor_priority: float = 0.1
    max_sensor_priority: float = 1.0
    full_random_drone_position: bool = False

    reward: Literal['punish', 'time-reward', 'reward'] = 'punish'

    speed_action: bool = True

    critic_model_size: int = 512
    actor_model_size: int = 256

    use_heuristics: None | Literal['greedy', 'random'] = None


args = tyro.cli(Args)


def make_env(evaluation=False):
    num_sensors = random.randint(args.min_num_sensors, args.max_num_sensors)
    return GrADySEnvironment(
        algorithm_iteration_interval=args.algorithm_iteration_interval,
        render_mode="visual" if evaluation and args.checkpoint_visual_evaluation else None,
        num_drones=args.num_drones,
        num_sensors=num_sensors,
        max_sensor_count=args.max_num_sensors,
        max_episode_length=args.max_episode_length,
        max_seconds_stalled=args.max_seconds_stalled,
        scenario_size=args.scenario_size,
        state_num_closest_sensors=args.state_num_closest_sensors,
        state_num_closest_drones=args.state_num_closest_drones,
        state_mode=args.state_mode,
        id_on_state=args.id_on_state,
        min_sensor_priority=args.min_sensor_priority,
        max_sensor_priority=args.max_sensor_priority,
        full_random_drone_position=False if evaluation else args.full_random_drone_position,
        reward=args.reward,
        speed_action=args.speed_action,
        end_when_all_collected=args.end_when_all_collected,
        local_observation=args.local_observation
    )


# ALGO LOGIC: initialize agent here:
if args.critic_global_state:
    class Critic(nn.Module):
        def __init__(self, action_space, observation_space):
            super().__init__()
            self.fc1 = nn.Linear(
                2 * args.num_drones + 2 * args.max_num_sensors + np.prod(action_space.shape) * args.num_drones,
                args.critic_model_size)
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
                np.array(observation_space.shape).prod() * args.num_drones + np.prod(
                    action_space.shape) * args.num_drones,
                args.critic_model_size)
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


if args.use_heuristics == 'greedy':
    heuristics = create_greedy_heuristics(args.state_num_closest_drones, args.state_num_closest_sensors)
if args.use_heuristics == 'random':
    heuristics = create_random_heuristics(args.state_num_closest_drones, args.state_num_closest_sensors)

run_name = f"{args.run_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{args.exp_name}/{run_name}")

def main():
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Statistics
    episode_count = 0
    all_collected_count = 0
    all_avg_collection_times = 0
    all_avg_reward = 0
    all_max_reward = 0
    all_sum_reward = 0
    all_episode_duration = 0
    all_completion_time = 0
    sensor_completion_time = defaultdict(int)
    sensor_episode_counts = defaultdict(int)

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

    # Creating tensor for optimization reasons, they are used to clip
    # clip and normalize actions
    zeros_like_all_actions = torch.zeros((args.num_drones, *action_space.shape), device=device)
    noise_buffer = torch.zeros_like(zeros_like_all_actions)
    low_action_space_tensor = torch.tensor(action_space.low, device=device)
    high_action_space_tensor = torch.tensor(action_space.high, device=device)
    gamma_tensor = torch.tensor(args.gamma, device=device)

    assert isinstance(action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(action_space, observation_space).to(device)
    qf1 = Critic(action_space, observation_space).to(device)
    qf1_target = Critic(action_space, observation_space).to(device)
    target_actor = Actor(action_space, observation_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.AdamW(list(qf1.parameters()), lr=args.critic_learning_rate, fused=True)
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.actor_learning_rate, fused=True)

    replay_buffer = FastReplayBuffer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        num_agents=args.num_drones,
        max_num_sensors=args.max_num_sensors,
        obs_shape=observation_space.shape,
        action_shape=action_space.shape,
        device=device
    )

    start_time = time.time()

    def evaluate_checkpoint():
        print("Reached checkpoint at step", global_step)
        model_path = f"runs/{args.exp_name}/{run_name}/checkpoint{global_step // 10_000}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

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
        sum_all_completion_time = 0
        eval_sensor_completion_time = defaultdict(int)
        eval_sensor_episode_counts = defaultdict(int)

        evaluation_runs = 200
        for i in range(evaluation_runs):
            if i % 100 == 0:
                print(f"Evaluating model ({i + 1}/{evaluation_runs})")
            temp_env = make_env(True)
            temp_obs, _ = temp_env.reset(seed=args.seed)
            while True:
                actions = {}
                with torch.no_grad():
                    for agent in temp_env.agents:
                        if args.use_heuristics:
                            actions[agent] = heuristics(temp_obs[agent])
                        else:
                            actions[agent] = actor(torch.tensor(temp_obs[agent], device=device, dtype=torch.float32))
                            actions[agent] += torch.normal(0, actor.action_scale * args.exploration_noise)
                            actions[agent] = actions[agent].cpu().numpy().clip(action_space.low, action_space.high)

                next_obs, _, _, _, infos = temp_env.step(actions)
                temp_obs = next_obs

                if len(infos) > 0 and "avg_reward" in infos[temp_env.agents[0]]:
                    info = infos[temp_env.agents[0]]

                    sum_avg_reward += info["avg_reward"]
                    sum_max_reward += info["max_reward"]
                    sum_sum_reward += info["sum_reward"]
                    sum_episode_duration += info["episode_duration"]
                    sum_avg_collection_time += info["avg_collection_time"]
                    sum_all_collected += info["all_collected"]
                    sum_all_completion_time += info["completion_time"]
                    eval_sensor_episode_counts[temp_env.num_sensors] += 1
                    eval_sensor_completion_time[temp_env.num_sensors] += info["completion_time"]
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
        writer.add_scalar(
            "eval/all_completion_time",
            sum_all_completion_time / evaluation_runs,
            global_step,
        )

        for num_sensors, count in sensor_episode_counts.items():
            writer.add_scalar(
                f"eval/sensor_completion_time/{num_sensors}",
                sensor_completion_time[num_sensors] / count,
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
    all_agent_obs = (torch.from_numpy(np.stack([obs[agent] for agent in env.agents]))
                     .to(device, dtype=torch.float32, non_blocking=True))
    terminated = False

    # Forward declaration for statistics
    qf1_a_values =  None
    qf1_loss = None
    actor_loss = None

    def train():
        nonlocal qf1_a_values, qf1_loss, actor_loss

        data = replay_buffer.sample()

        with torch.no_grad():
            next_actions = target_actor(data["next_observations"]).view(data["next_observations"].shape[0], -1)
            if args.critic_global_state:
                next_state = data["next_global_state"].view(data["next_global_state"].shape[0], -1)
            else:
                next_state = data["next_observations"].view(data["next_observations"].shape[0], -1)
            qf1_next_target = qf1_target(next_state, next_actions)
            next_q_value = data["reward"].view((data["reward"].shape[0],)) \
                           + (1 - data["done"].view((data["done"].shape[0],))) \
                           * gamma_tensor \
                           * qf1_next_target.view(-1)

        if args.critic_global_state:
            current_state = data["global_state"].view(data["global_state"].shape[0], -1)
        else:
            current_state = data["observations"].view(data["observations"].shape[0], -1)
        all_actions = data["actions"].view(data["actions"].shape[0], -1)
        qf1_a_values = qf1(current_state, all_actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

        # optimize the model
        q_optimizer.zero_grad()
        qf1_loss.backward()
        q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_actions = actor(data["observations"]).view(data["observations"].shape[0], -1)

            actor_loss = -qf1(current_state, actor_actions).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            with torch.no_grad():
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.lerp_(param.data, args.tau)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.lerp_(param.data, args.tau)

    train = torch.compile(train)

    for global_step in range(args.total_timesteps):
        step_start = time.time()

        if terminated:
            env = make_env()
            obs, _ = env.reset(seed=args.seed)
            terminated = False

        if args.use_heuristics:
            actions_dict = {
                agent: heuristics(obs[agent]) for agent in env.agents
            }
            all_agent_numpy_actions = np.stack([actions_dict[agent] for agent in env.agents], dtype=np.float32)
            all_agent_torch_actions = torch.from_numpy(all_agent_numpy_actions).to(device, dtype=torch.float32, non_blocking=True)
        else:
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions_dict = {
                    agent: action_space.sample() for agent in env.agents
                }
                all_agent_numpy_actions = np.stack([actions_dict[agent] for agent in env.agents], dtype=np.float32)
                all_agent_torch_actions = torch.from_numpy(all_agent_numpy_actions).to(device, dtype=torch.float32, non_blocking=True)
            else:
                with torch.no_grad():
                    actions_dict = {}
                    all_actions: torch.Tensor = actor(all_agent_obs)
                    torch.normal(zeros_like_all_actions, actor.action_scale * args.exploration_noise, out=noise_buffer)
                    all_actions.add_(noise_buffer)
                    all_actions.clip_(low_action_space_tensor, high_action_space_tensor)
                    all_agent_numpy_actions = all_actions.cpu().numpy()
                    all_agent_torch_actions = all_actions
                    for index, agent in enumerate(env.agents):
                        actions_dict[agent] = all_agent_numpy_actions[index]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions_dict)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
            episode_count += 1
            terminated = True

            info = infos[env.agents[0]]

            all_avg_reward += info["avg_reward"]
            all_max_reward += info["max_reward"]
            all_sum_reward += info["sum_reward"]
            all_episode_duration += info["episode_duration"]
            all_avg_collection_times += info["avg_collection_time"]
            all_collected_count += info["all_collected"]
            all_completion_time += info["completion_time"]

            sensor_episode_counts[env.num_sensors] += 1
            sensor_completion_time[env.num_sensors] += info["completion_time"]

        if args.use_heuristics:
            obs = next_obs
            all_agent_obs = torch.from_numpy(np.stack([obs[agent] for agent in env.agents])) \
                .to(device, dtype=np.float32, non_blocking=True)
        else:
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            all_agent_next_obs = torch.from_numpy(np.stack([next_obs[agent] for agent in env.agents])) \
                .to(device, dtype=torch.float32, non_blocking=True)

            reward_tensor = torch.tensor([rewards[env.agents[0]]], dtype=torch.float32)
            done_tensor = torch.tensor([int(terminations[env.agents[0]])], dtype=torch.float32)
            global_state_tensor = torch.tensor(obs['global'], dtype=torch.float32)
            next_global_state_tensor = torch.tensor(next_obs['global'], dtype=torch.float32)

            replay_buffer.add(
                obs=all_agent_obs,
                next_obs=all_agent_next_obs,
                action=all_agent_torch_actions,
                reward=reward_tensor.to(device, non_blocking=True),
                done=done_tensor.to(device, non_blocking=True),
                global_states=global_state_tensor.to(device, non_blocking=True),
                next_global_states=next_global_state_tensor.to(device, non_blocking=True)
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            all_agent_obs = all_agent_next_obs

            # ALGO LOGIC: training.
            if global_step <= args.learning_starts:
                continue

            train()

        if global_step > 0 and global_step % args.statistics_frequency == 0:
            if not args.use_heuristics:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

            sps = args.statistics_frequency / (time.time() - start_time)
            start_time = time.time()

            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/step_duration", time.time() - step_start, global_step)

            writer.add_scalar(
                "charts/avg_reward",
                all_avg_reward / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/max_reward",
                all_max_reward / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/sum_reward",
                all_sum_reward / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/episode_duration",
                all_episode_duration / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/avg_collection_time",
                all_avg_collection_times / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/all_collected_rate",
                all_collected_count / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/completion_time",
                all_completion_time / episode_count,
                global_step,
            )

            for num_sensors, count in sensor_episode_counts.items():
                writer.add_scalar(
                    f"charts/sensor_completion_time/{num_sensors}",
                    sensor_completion_time[num_sensors] / count,
                    global_step,
                )
            
            episode_count = 0
            all_collected_count = 0
            all_avg_collection_times = 0
            all_avg_reward = 0
            all_max_reward = 0
            all_sum_reward = 0
            all_episode_duration = 0
            all_completion_time = 0
            sensor_completion_time = defaultdict(int)
            sensor_episode_counts = defaultdict(int)

            print(f"{args.exp_name}|{args.run_name} - SPS:", sps)

        if args.checkpoints and global_step % args.checkpoint_freq == 0 and global_step > 0:
            evaluate_checkpoint()

    evaluate_checkpoint()
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
