import os
from time import time
import math

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from arguments import ActorArgs, EnvironmentArgs, LoggingArgs, ModelArgs, CoordinationArgs, LearnerArgs
from environment import action_space_from_args, observation_space_from_args, make_env
from heuristics import create_greedy_heuristics, create_random_heuristics
from model import Actor, Critic

# print = lambda *args: args

def clone_state_dict(state_dict: dict):
    cloned_dict = {}
    while state_dict:
        key, value = state_dict.popitem()
        cloned_dict[key] = value.clone()

    return cloned_dict


@torch.no_grad()
def execute_actor(current_step: torch.multiprocessing.Value,
                  current_sps: torch.multiprocessing.Value,
                  actor_id: int,
                  model_args: ModelArgs,
                  actor_args: ActorArgs,
                  learner_args: LearnerArgs,
                  logging_args: LoggingArgs,
                  environment_args: EnvironmentArgs,
                  coordination_args: CoordinationArgs,
                  experience_queue: torch.multiprocessing.JoinableQueue,
                  model_queue: torch.multiprocessing.Queue,
                  start_lock: torch.multiprocessing.Barrier):
    total_process_count = os.cpu_count()
    torch.set_num_threads(math.floor(total_process_count / coordination_args.num_actors))

    writer = SummaryWriter(logging_args.get_path())

    print(f"ACTOR {actor_id} - " f"Actor {actor_id} started, awaiting lock release...")
    start_lock.wait()

    observation_space = observation_space_from_args(environment_args)
    action_space = action_space_from_args(environment_args)

    device = torch.device("cuda" if actor_args.actor_use_cuda else "cpu")
    print(f"ACTOR {actor_id} - " f"Using device {device}")

    actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    target_actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)
    target_critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)

    def receive_models():
        critic_state_dict, target_critic_state_dict, actor_state_dict, target_actor_state_dict = model_queue.get()
        critic_model.load_state_dict(clone_state_dict(critic_state_dict))
        target_critic_model.load_state_dict(clone_state_dict(target_critic_state_dict))
        actor_model.load_state_dict(clone_state_dict(actor_state_dict))
        target_actor_model.load_state_dict(clone_state_dict(target_actor_state_dict))
        del critic_state_dict
        del target_critic_state_dict
        del actor_state_dict
        del target_actor_state_dict

    print(f"ACTOR {actor_id} - " "Received first model")
    receive_models()

    sample = TensorDict({
        "state": np.stack([observation_space.sample() for _ in range(environment_args.num_drones)]),
        "actions": np.stack([action_space.sample() for _ in range(environment_args.num_drones)]),
        "reward": torch.rand(1),
        "next_state": np.stack([observation_space.sample() for _ in range(environment_args.num_drones)]),
        "done": torch.rand(1),
        "priority": torch.rand(1),
    })
    buffer = sample.expand(actor_args.experience_buffer_size)
    buffer = buffer.clone()
    buffer = buffer.zero_()
    buffer = buffer.share_memory_()

    env = make_env(environment_args)

    try:
        heuristics = None
        if actor_args.use_heuristics == 'greedy':
            heuristics = create_greedy_heuristics(environment_args.state_num_closest_drones,
                                                  environment_args.state_num_closest_sensors)
        if actor_args.use_heuristics == 'random':
            heuristics = create_random_heuristics(environment_args.state_num_closest_drones,
                                                  environment_args.state_num_closest_sensors)

        # Variables used for statistics
        episode_count = 1
        all_collected_count = 0
        all_avg_collection_times = 0
        all_avg_reward = 0
        all_max_reward = 0
        all_sum_reward = 0
        all_episode_duration = 0
        sps_start_time = time()

        obs, _ = env.reset()
        all_agent_obs = np.stack([obs[agent] for agent in env.agents])
        terminated = False
        cursor = 0
        action_step = 0
        while True:
            action_step += 1

            # Update model if available
            if not model_queue.empty():
                receive_models()

            if terminated:
                obs, _ = env.reset()
                terminated = False

            if actor_args.use_heuristics:
                actions = {
                    agent: heuristics(obs[agent]) for agent in env.agents
                }
                all_agent_actions = np.stack([actions[agent] for agent in env.agents])
            else:
                # ALGO LOGIC: put action logic here
                with torch.no_grad():
                    actions = {}
                    all_obs = torch.tensor(all_agent_obs,
                                           device=device,
                                           dtype=torch.float32)
                    all_actions: torch.Tensor = actor_model(all_obs)
                    all_actions.add_(torch.normal(torch.zeros_like(all_actions, device=device),
                                                  actor_model.action_scale * actor_args.exploration_noise))
                    all_actions.clip_(torch.tensor(action_space.low, device=device),
                                      torch.tensor(action_space.high, device=device))

                    all_agent_actions = all_actions.cpu().numpy()

                    for index, agent in enumerate(env.agents):
                        actions[agent] = all_actions[index].cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
                terminated = True
                episode_count += 1

                info = infos[env.agents[0]]

                all_avg_reward += info["avg_reward"]
                all_max_reward += info["max_reward"]
                all_sum_reward += info["sum_reward"]
                all_episode_duration += info["episode_duration"]
                all_avg_collection_times += info["avg_collection_time"]
                all_collected_count += info["all_collected"]

            all_agent_next_obs = np.stack([next_obs[agent] for agent in env.agents])
            reward = torch.tensor(rewards[env.agents[0]]).to(device)
            done = torch.tensor(int(terminations[env.agents[0]])).to(device)

            # Estimating TD error for prioritized experience replay
            all_next_obs = torch.tensor(all_agent_next_obs, dtype=torch.float32).to(device)
            next_actions = target_actor_model(all_next_obs).view(1, -1)
            next_state = torch.tensor(all_agent_next_obs.reshape(1, -1), dtype=torch.float32).to(device)
            qf1_next_target = target_critic_model(next_state, next_actions)
            next_q_value = reward.flatten() + (1 - done.flatten()) * learner_args.gamma * (
                qf1_next_target).view(1, -1)
            current_state = torch.tensor(all_agent_obs.reshape(1, -1), dtype=torch.float32).to(device)
            all_current_actions = torch.tensor(all_agent_actions.reshape(1, -1), dtype=torch.float32).to(device)
            qf1_a_values = critic_model(current_state, all_current_actions).view(1, -1)
            qf1_loss = mse_loss(qf1_a_values, next_q_value)

            buffer[cursor] = TensorDict({
                "state": all_agent_obs,
                "actions": all_agent_actions,
                "reward": rewards[env.agents[0]],
                "next_state": all_agent_next_obs,
                "done": int(terminations[env.agents[0]]),
                "priority": qf1_loss.item(),
            })
            cursor += 1

            obs = next_obs
            all_agent_obs = all_agent_next_obs

            if cursor >= actor_args.experience_buffer_size:
                print(f"ACTOR {actor_id} - " f"Sending buffer at step {action_step}")
                experience_queue.put(buffer)
                experience_queue.join()
                buffer.zero_()
                cursor = 0

            if action_step % actor_args.actor_statistics_frequency == 0:
                writer.add_scalar(
                    f"actor{actor_id}/avg_reward",
                    all_avg_reward / episode_count,
                    action_step,
                )
                writer.add_scalar(
                    f"actor{actor_id}/max_reward",
                    all_max_reward / episode_count,
                    action_step,
                )
                writer.add_scalar(
                    f"actor{actor_id}/sum_reward",
                    all_sum_reward / episode_count,
                    action_step,
                )
                writer.add_scalar(
                    f"actor{actor_id}/episode_duration",
                    all_episode_duration / episode_count,
                    action_step,
                )
                writer.add_scalar(
                    f"actor{actor_id}/avg_collection_time",
                    all_avg_collection_times / episode_count,
                    action_step,
                )
                writer.add_scalar(
                    f"actor{actor_id}/all_collected_rate",
                    all_collected_count / episode_count,
                    action_step,
                )

                sps = actor_args.actor_statistics_frequency / (time() - sps_start_time)
                sps_start_time = time()
                writer.add_scalar(f"actor{actor_id}/sps", sps, action_step)
                print(f"ACTOR {actor_id} " f"SPS: {sps}; STEP: {action_step}")

                # Updating shared values
                current_step.value = action_step
                current_sps.value = sps

        print(f"ACTOR {actor_id} - " f"Finished training at step {action_step}")
    except:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        env.kill()
        writer.close()