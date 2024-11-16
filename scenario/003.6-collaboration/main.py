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
from stable_baselines3.common.buffers import ReplayBuffer
from tensordict import TensorDict
from torch.nn import ZeroPad1d
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, PrioritizedSampler

from environment import GrADySEnvironment, StateMode
from heuristics import create_greedy_heuristics, create_random_heuristics


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
    checkpoint_model_freq: int = 10
    """every checkpoint_model_freq checkpoints the model will be saved"""
    checkpoint_visual_evaluation: bool = False
    """whether to visually evaluate the model at each checkpoint"""
    statistics_frequency: float = 10_000
    """statistics will be saved to the tensorboard logs with this frequency"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 3e-6
    """the learning rate of the actor optimizer"""
    critic_learning_rate: float = 3e-6
    """the learning rate of the critic optimizer"""
    critic_use_active_agents: bool = False
    """if toggled, the critic will use the number of active agents as an input"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    use_priority: bool = False
    """if toggled, the replay buffer will use priority sampling"""
    priority_alpha: float = 0.7
    priority_beta: float = 1
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

    algorithm_iteration_interval: float = 0.5
    max_seconds_stalled: int = 30
    end_when_all_collected: bool = False
    min_num_drones: int = 2
    max_num_drones: int = 4
    use_phantom_agents: bool = True
    """if toggled, phantom agents will be used when the number of agents is less than the max"""
    min_num_sensors: int = 5
    max_num_sensors: int = 5
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


def make_env(evaluation=False, max_possible=False):
    num_sensors = random.randint(args.min_num_sensors, args.max_num_sensors) if not max_possible else args.max_num_sensors
    num_drones = random.randint(args.min_num_drones, args.max_num_drones) if not max_possible else args.max_num_drones

    return GrADySEnvironment(
        algorithm_iteration_interval=args.algorithm_iteration_interval,
        render_mode="visual" if evaluation and args.checkpoint_visual_evaluation else None,
        num_drones=num_drones,
        num_sensors=num_sensors,
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
        end_when_all_collected=args.end_when_all_collected
    )


# ALGO LOGIC: initialize agent here:
class Critic(nn.Module):
    def __init__(self, action_space, observation_space):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_space.shape).prod() * args.max_num_drones + np.prod(action_space.shape) * args.max_num_drones + args.critic_use_active_agents,
            args.critic_model_size)
        self.fc2 = nn.Linear(args.critic_model_size, args.critic_model_size)
        self.fc3 = nn.Linear(args.critic_model_size, 1)

    def forward(self, x, a, active_agents):
        x = torch.cat([x, a, active_agents] if args.critic_use_active_agents else [x, a], 1)
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

run_name = f"{args.run_name}/{args.exp_name}__{int(time.time())}"
writer = SummaryWriter(f"runs/{run_name}")

def main():
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Statistics
    episode_count = 0
    all_collected_count = defaultdict(lambda: 0)
    all_avg_collection_times = defaultdict(lambda: 0)
    all_avg_reward = defaultdict(lambda: 0)
    all_max_reward = defaultdict(lambda: 0)
    all_sum_reward = defaultdict(lambda: 0)
    all_episode_duration = defaultdict(lambda: 0)
    all_completion_times = defaultdict(lambda: 0)

    runs_per_configuration = defaultdict(lambda: 0)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    max_env = make_env(max_possible=True)
    observation_space = max_env.observation_space(0)
    action_space = max_env.action_space(0)

    assert isinstance(action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(action_space, observation_space).to(device)
    qf1 = Critic(action_space, observation_space).to(device)
    qf1_target = Critic(action_space, observation_space).to(device)
    target_actor = Actor(action_space, observation_space).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.critic_learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.actor_learning_rate)

    if args.use_priority:
        replay_buffer = TensorDictReplayBuffer(batch_size=args.batch_size,
                                               storage=LazyTensorStorage(args.buffer_size, device=device),
                                               sampler=PrioritizedSampler(args.buffer_size,
                                                                          alpha=args.priority_alpha,
                                                                          beta=args.priority_beta),
                                               prefetch=10,
                                               priority_key="priority")
    else:
        replay_buffer = TensorDictReplayBuffer(batch_size=args.batch_size,
                                               storage=LazyTensorStorage(args.buffer_size, device=device))

    start_time = time.time()

    def evaluate_checkpoint(save_movel: bool):
        print("Reached checkpoint at step", global_step)
        model_path = f"runs/{run_name}/{args.exp_name}-checkpoint{global_step // 10_000}.cleanrl_model"
        if save_movel:
            torch.save((actor.state_dict(), qf1.state_dict()), model_path)
            print(f"model saved to {model_path}")

        actor.eval()
        target_actor.eval()
        qf1.eval()
        qf1_target.eval()

        sum_avg_reward = defaultdict(lambda: 0)
        sum_max_reward = defaultdict(lambda: 0)
        sum_sum_reward = defaultdict(lambda: 0)
        sum_episode_duration = defaultdict(lambda: 0)
        sum_avg_collection_time = defaultdict(lambda: 0)
        sum_all_collected = defaultdict(lambda: 0)
        sum_completion_time = defaultdict(lambda: 0)

        eval_runs_per_config = defaultdict(lambda: 0)

        evaluation_runs = 50
        for i in range(evaluation_runs):
            if i % 10 == 0:
                print(f"Evaluating model ({i}/{evaluation_runs})")
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

                if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                    info = infos[env.agents[0]]

                    n_agents = temp_env.num_drones

                    sum_avg_reward[n_agents] += info["avg_reward"]
                    sum_max_reward[n_agents] += info["max_reward"]
                    sum_sum_reward[n_agents] += info["sum_reward"]
                    sum_episode_duration[n_agents] += info["episode_duration"]
                    sum_avg_collection_time[n_agents] += info["avg_collection_time"]
                    sum_all_collected[n_agents] += info["all_collected"]
                    sum_completion_time[n_agents] += info["completion_time"]
                    eval_runs_per_config[n_agents] += 1
                    break
            temp_env.close()

        print(f"Evaluating model ({evaluation_runs}/{evaluation_runs})")
        
        writer.add_scalar(
            "eval/avg_reward",
            sum(sum_avg_reward.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/max_reward",
            sum(sum_max_reward.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/sum_reward",
            sum(sum_sum_reward.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/episode_duration",
            sum(sum_episode_duration.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/avg_collection_time",
            sum(sum_avg_collection_time.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/all_collected_rate",
            sum(sum_all_collected.values()) / evaluation_runs,
            global_step,
        )
        writer.add_scalar(
            "eval/completion_time",
            sum(sum_completion_time.values()) / evaluation_runs,
            global_step,
        )

        for n_a in sum_avg_reward.keys():
            writer.add_scalar(
                f"eval/{n_a} agents/avg_reward_agents",
                sum_avg_reward[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/max_reward_agents",
                sum_max_reward[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/sum_reward_agents",
                sum_sum_reward[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/episode_duration_agents",
                sum_episode_duration[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/avg_collection_time_agents",
                sum_avg_collection_time[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/all_collected_rate_agents",
                sum_all_collected[n_a] / eval_runs_per_config[n_a],
                global_step,
            )
            writer.add_scalar(
                f"eval/{n_a} agents/completion_time_agents",
                sum_completion_time[n_a] / eval_runs_per_config[n_a],
                global_step,
            )

        temp_env.close()

        actor.train()
        target_actor.train()
        qf1.train()
        qf1_target.train()

        print("Checkpoint evaluation done")

    def stack_all_agent_information(information: dict[str, np.ndarray]) -> np.ndarray:
        if args.use_phantom_agents:
            all_possible_agents = [f"drone{i}" for i in range(args.max_num_drones)]
            available_agents = list(information.keys())
            all_agent_information = np.stack([information.get(agent, information[random.choice(available_agents)]) for agent in all_possible_agents])
        else:
            all_agent_information = np.stack([information[agent] for agent in env.agents])
            all_agent_information = np.pad(all_agent_information, ((0, args.max_num_drones - len(env.agents)), (0, 0)),
                                           mode='constant', constant_values=0)
        return all_agent_information

    # TRY NOT TO MODIFY: start the game
    env = make_env()
    terminated = True
    for global_step in range(args.total_timesteps):
        step_start = time.time()

        if terminated:
            env = make_env()
            obs, _ = env.reset(seed=args.seed)
            all_agent_obs = stack_all_agent_information(obs)
            terminated = False

        if args.use_heuristics:
            actions = {
                agent: heuristics(obs[agent]) for agent in env.agents
            }
            all_agent_actions = stack_all_agent_information(actions)
        else:
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = {
                    agent: action_space.sample() for agent in env.agents
                }
                all_agent_actions = stack_all_agent_information(actions)
            else:
                with torch.no_grad():
                    actions = {}
                    all_obs = torch.tensor(all_agent_obs,
                                           device=device,
                                           dtype=torch.float32)
                    all_actions: torch.Tensor = actor(all_obs)
                    all_actions.add_(torch.normal(torch.zeros_like(all_actions, device=device),
                                                  actor.action_scale * args.exploration_noise))
                    all_actions.clip_(torch.tensor(action_space.low, device=device),
                                      torch.tensor(action_space.high, device=device))
                    for index, agent in enumerate(env.agents):
                        actions[agent] = all_actions[index].cpu().numpy()
                    all_agent_actions = stack_all_agent_information(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
            episode_count += 1
            terminated = True

            info = infos[env.agents[0]]

            num_agents = env.num_drones

            all_avg_reward[num_agents] += info["avg_reward"]
            all_max_reward[num_agents] += info["max_reward"]
            all_sum_reward[num_agents] += info["sum_reward"]
            all_episode_duration[num_agents] += info["episode_duration"]
            all_avg_collection_times[num_agents] += info["avg_collection_time"]
            all_collected_count[num_agents] += info["all_collected"]
            all_completion_times[num_agents] += info["completion_time"]

            runs_per_configuration[num_agents] += 1

        if args.use_heuristics:
            obs = next_obs
            all_agent_obs = stack_all_agent_information(obs)
        else:
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            all_agent_next_obs = stack_all_agent_information(next_obs)

            experience = TensorDict({
                "state": all_agent_obs,
                "actions": all_agent_actions,
                "reward": rewards[env.agents[0]],
                "next_state": all_agent_next_obs,
                "done": int(terminations[env.agents[0]]),
                "active_agents": [len(env.agents) / args.max_num_drones]
            }, device=device).to(torch.float32)
            replay_buffer.add(experience)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            all_agent_obs = all_agent_next_obs

            # ALGO LOGIC: training.
            if global_step <= args.learning_starts:
                continue
            
            training_step_count = args.max_num_drones if args.train_once_for_each_agent else 1
            samples = replay_buffer.sample()



            with torch.no_grad():
                next_actions = target_actor(samples["next_state"]).view(samples["next_state"].shape[0], -1)
                next_state = samples["next_state"].reshape(samples["next_state"].shape[0], -1)
                qf1_next_target = qf1_target(next_state, next_actions, samples["active_agents"])
                next_q_value = samples["reward"].flatten() + (1 - samples["done"].flatten()) * args.gamma * (
                    qf1_next_target).view(-1)

            current_state = samples["state"].reshape(samples["state"].shape[0], -1)
            all_actions = samples["actions"].reshape(samples["actions"].shape[0], -1)
            qf1_a_values = qf1(current_state, all_actions, samples["active_agents"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if args.use_priority:
                samples['priority'] = qf1_loss.expand(args.batch_size)
                replay_buffer.update_tensordict_priority(samples)

            if global_step % args.policy_frequency == 0:
                actor_actions = actor(samples["state"]).view(samples["state"].shape[0], -1)

                actor_loss = -qf1(current_state, actor_actions, samples["active_agents"]).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.lerp_(param.data, args.tau)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.lerp_(param.data, args.tau)

        if global_step > 0 and global_step % args.statistics_frequency == 0:
            if not args.use_heuristics:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)
            writer.add_scalar("charts/step_duration", time.time() - step_start, global_step)

            writer.add_scalar(
                "charts/avg_reward",
                sum(all_avg_reward.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/max_reward",
                sum(all_max_reward.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/sum_reward",
                sum(all_sum_reward.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/episode_duration",
                sum(all_episode_duration.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/avg_collection_time",
                sum(all_avg_collection_times.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/all_collected_rate",
                sum(all_collected_count.values()) / episode_count,
                global_step,
            )
            writer.add_scalar(
                "charts/completion_time",
                sum(all_completion_times.values()) / episode_count,
                global_step,
            )

            for n_a in all_avg_reward.keys():
                writer.add_scalar(
                    f"charts/{n_a} agents/avg_reward_agents",
                    all_avg_reward[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/max_reward_agents",
                    all_max_reward[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/sum_reward_agents",
                    all_sum_reward[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/episode_duration_agents",
                    all_episode_duration[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/avg_collection_time_agents",
                    all_avg_collection_times[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/all_collected_rate_agents",
                    all_collected_count[n_a] / runs_per_configuration[n_a],
                    global_step,
                )
                writer.add_scalar(
                    f"charts/{n_a} agents/completion_time_agents",
                    all_completion_times[n_a] / runs_per_configuration[n_a],
                    global_step,
                )

            print(f"{args.exp_name} - SPS:", global_step / (time.time() - start_time))

        if args.checkpoints and global_step % args.checkpoint_freq == 0 and global_step > 0:
            evaluate_checkpoint((global_step / args.checkpoint_freq) % args.checkpoint_model_freq == 0)

    evaluate_checkpoint(True)
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
