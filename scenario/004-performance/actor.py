import os
from time import time
import math

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from arguments import ActorArgs, EnvironmentArgs, LoggingArgs, ModelArgs, CoordinationArgs
from environment import action_space_from_args, observation_space_from_args, make_env
from heuristics import create_greedy_heuristics, create_random_heuristics
from model import Actor

# print = lambda *args: args

@torch.no_grad()
def execute_actor(current_step: torch.multiprocessing.Value,
                  current_sps: torch.multiprocessing.Value,
                  actor_id: int,
                  model_args: ModelArgs,
                  actor_args: ActorArgs,
                  logging_args: LoggingArgs,
                  environment_args: EnvironmentArgs,
                  coordination_args: CoordinationArgs,
                  experience_queue: torch.multiprocessing.Queue,
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

    state_dict = model_queue.get()
    print(f"ACTOR {actor_id} - " "Received first model")
    actor_model.load_state_dict(state_dict)
    del state_dict

    sample = TensorDict({
        "state": np.stack([observation_space.sample() for _ in range(environment_args.num_drones)]),
        "actions": np.stack([action_space.sample() for _ in range(environment_args.num_drones)]),
        "reward": torch.rand(1),
        "next_state": np.stack([observation_space.sample() for _ in range(environment_args.num_drones)]),
        "done": torch.rand(1),
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
                received_state_dict = model_queue.get()
                actor_model.load_state_dict(received_state_dict)
                # print(f"ACTOR {actor_id} - " f"Updating model at step {action_step}")
                del received_state_dict
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

            buffer[cursor] = TensorDict({
                "state": all_agent_obs,
                "actions": all_agent_actions,
                "reward": rewards[env.agents[0]],
                "next_state": all_agent_next_obs,
                "done": int(terminations[env.agents[0]]),
            })
            cursor += 1

            obs = next_obs
            all_agent_obs = all_agent_next_obs

            if cursor >= actor_args.experience_buffer_size:
                print(f"ACTOR {actor_id} - " f"Sending buffer at step {action_step}")
                experience_queue.put(buffer)
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