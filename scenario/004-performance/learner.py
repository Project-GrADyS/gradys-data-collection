import multiprocessing
import os
import time
from copy import deepcopy

import math
import torch
from tensordict import TensorDictBase
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyTensorStorage, PrioritizedSampler, TensorDictReplayBuffer

from arguments import LearnerArgs, ExperienceArgs, ActorArgs, LoggingArgs, EnvironmentArgs, ModelArgs, CoordinationArgs
from environment import make_env, observation_space_from_args, action_space_from_args
from heuristics import create_greedy_heuristics, create_random_heuristics
from model import Actor, Critic

# print = lambda *args: args


class EnvScaler:
    def __init__(self, env_args: EnvironmentArgs):
        self.env_args = env_args

        self._scaling_steps = []
        for a in range(self.env_args.min_drone_count, self.env_args.max_drone_count + 1):
            for s in range(self.env_args.min_sensor_count, self.env_args.max_sensor_count + 1):
                self._scaling_steps.append((a, s))

        self._current_scaling_step = 0 if self.env_args.progressive_scaling else len(self._scaling_steps) - 1
        self._current_confidence = 0

    def scale(self, all_collected_rate: float):
        if not self.env_args.progressive_scaling:
            return

        if all_collected_rate >= self.env_args.progressive_scaling_cutoff:
            self._current_confidence += 1
        else:
            self._current_confidence = 0

        if (self._current_confidence >= self.env_args.progressive_scaling_confidence
                and self._current_scaling_step < len(self._scaling_steps) - 1):
            self._current_scaling_step += 1
            self._current_confidence = 0

    @property
    def max_drone_count(self):
        return self._scaling_steps[self._current_scaling_step][0]

    @property
    def max_sensor_count(self):
        return self._scaling_steps[self._current_scaling_step][1]

    @property
    def current_scaling_step(self):
        return self._current_scaling_step

    @property
    def current_confidence(self):
        return self._current_confidence



def evaluate_checkpoint(learner_step: int,
                        writer: SummaryWriter,
                        device: torch.device,
                        logging_args: LoggingArgs,
                        env_args: EnvironmentArgs,
                        max_drone_count: int,
                        max_sensor_count: int,
                        actor_args: ActorArgs,
                        actor_model_dict: dict,
                        critic_model_dict: dict,
                        save_model: bool = False):
    print("LEARNER - " "Reached checkpoint at step", learner_step)
    if save_model:
        model_path = f"{logging_args.get_path()}/{logging_args.run_name}-checkpoint{learner_step}.cleanrl_model"
        torch.save((actor_model_dict, critic_model_dict), model_path)

    temp_env = make_env(env_args, max_drone_count, max_sensor_count)

    action_space = action_space_from_args(env_args)
    observation_space = observation_space_from_args(env_args)

    actor_model = Actor(action_space.shape[0], observation_space.shape[0], ModelArgs()).to(device)
    critic_model = Critic(action_space.shape[0], observation_space.shape[0], env_args, ModelArgs()).to(device)

    actor_model.load_state_dict(actor_model_dict)
    critic_model.load_state_dict(critic_model_dict)

    actor_model.eval()
    critic_model.eval()

    sum_avg_reward = 0
    sum_max_reward = 0
    sum_sum_reward = 0
    sum_episode_duration = 0
    sum_avg_collection_time = 0
    sum_all_collected = 0
    sum_completion_time = 0

    heuristics = None
    if actor_args.use_heuristics == 'greedy':
        heuristics = create_greedy_heuristics(env_args.state_num_closest_drones,
                                              env_args.state_num_closest_sensors)
    if actor_args.use_heuristics == 'random':
        heuristics = create_random_heuristics(env_args.state_num_closest_drones,
                                              env_args.state_num_closest_sensors)

    evaluation_runs = 50
    for i in range(evaluation_runs):
        if i % 10 == 0:
            print("LEARNER - "f"Evaluating model ({i}/{evaluation_runs})")
        temp_obs, _ = temp_env.reset()
        while True:
            actions = {}
            with torch.no_grad():
                for agent in temp_env.agents:
                    if actor_args.use_heuristics:
                        actions[agent] = heuristics(temp_obs[agent])
                    else:
                        actions[agent] = actor_model(torch.tensor(temp_obs[agent], device=device, dtype=torch.float32))

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
                sum_completion_time += info["completion_time"]
                break
        temp_env.close()

    writer.add_scalar(
        "eval/avg_reward",
        sum_avg_reward / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/max_reward",
        sum_max_reward / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/sum_reward",
        sum_sum_reward / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/episode_duration",
        sum_episode_duration / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/avg_collection_time",
        sum_avg_collection_time / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/all_collected_rate",
        sum_all_collected / evaluation_runs,
        learner_step,
    )
    writer.add_scalar(
        "eval/completion_time",
        sum_completion_time / evaluation_runs,
        learner_step,
    )
    temp_env.close()

    actor_model.train()
    critic_model.train()

    print("Checkpoint evaluation done")
    return {
        "avg_reward": sum_avg_reward / evaluation_runs,
        "max_reward": sum_max_reward / evaluation_runs,
        "sum_reward": sum_sum_reward / evaluation_runs,
        "episode_duration": sum_episode_duration / evaluation_runs,
        "avg_collection_time": sum_avg_collection_time / evaluation_runs,
        "all_collected_rate": sum_all_collected / evaluation_runs,
    }

def state_dict_to_cpu(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}

def execute_learner(current_step: torch.multiprocessing.Value,
                    current_sps: torch.multiprocessing.Value,
                    learner_args: LearnerArgs,
                    actor_args: ActorArgs,
                    experience_args: ExperienceArgs,
                    logging_args: LoggingArgs,
                    environment_args: EnvironmentArgs,
                    max_scaling_drone_count: torch.multiprocessing.Value,
                    max_scaling_sensor_count: torch.multiprocessing.Value,
                    model_args: ModelArgs,
                    coordination_args: CoordinationArgs,
                    experience_queue: torch.multiprocessing.JoinableQueue,
                    actor_model_queues: list[torch.multiprocessing.Queue],
                    start_lock: torch.multiprocessing.Barrier):
    total_process_count = os.cpu_count()
    total_threads = coordination_args.num_actors + 1
    # Learner gets the same amount of threads as the actor but also gets the remainer
    torch.set_num_threads(
        math.floor(total_process_count / total_threads) + total_process_count % total_threads
    )
    print("LEARNER - " f"Using {torch.get_num_threads()} threads")


    print("LEARNER " "Learner ready, awaiting lock release...")
    start_lock.wait()

    writer = SummaryWriter(logging_args.get_path())

    received_experiences = 0

    observation_space = observation_space_from_args(environment_args)
    action_space = action_space_from_args(environment_args)

    device = torch.device("cuda" if learner_args.learner_cuda else "cpu")
    print("LEARNER - " f"Using device {device}")

    actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)
    target_actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    target_critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)

    actor_model.compile(backend='cudagraphs')
    critic_model.compile(backend='cudagraphs')
    target_actor_model.compile(backend='cudagraphs')
    target_critic_model.compile(backend='cudagraphs')

    target_actor_model.load_state_dict(actor_model.state_dict())
    target_critic_model.load_state_dict(critic_model.state_dict())
    critic_optimizer = optim.AdamW(list(critic_model.parameters()), lr=learner_args.critic_learning_rate, fused=True)
    actor_optimizer = optim.AdamW(list(actor_model.parameters()), lr=learner_args.actor_learning_rate, fused=True)

    critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(critic_optimizer, mode='max', factor=learner_args.decay_factor, patience=learner_args.decay_patience, min_lr=learner_args.min_lr_decay)
    actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, mode='max', factor=learner_args.decay_factor, patience=learner_args.decay_patience, min_lr=learner_args.min_lr_decay)

    scaler = EnvScaler(environment_args)

    def upload_models():
        critic_state_dict = state_dict_to_cpu(critic_model.state_dict())
        target_critic_state_dict = state_dict_to_cpu(target_critic_model.state_dict())
        actor_state_dict = state_dict_to_cpu(actor_model.state_dict())
        target_actor_state_dict = state_dict_to_cpu(target_actor_model.state_dict())

        for queue in actor_model_queues:
            queue.put((critic_state_dict, target_critic_state_dict, actor_state_dict, target_actor_state_dict))


    print("LEARNER - " "Sending first models to actors")
    upload_models()

    replay_buffer = TensorDictReplayBuffer(batch_size=experience_args.batch_size,
                                           storage=LazyTensorStorage(experience_args.buffer_size, device=device),
                                           sampler=PrioritizedSampler(experience_args.buffer_size, alpha=experience_args.priority_alpha, beta=experience_args.priority_beta),
                                           prefetch=10,
                                           priority_key="priority")


    print("LEARNER - " f"Waiting for {learner_args.learning_starts} experiences before starting learning loop")
    while received_experiences < learner_args.learning_starts:
        experience = experience_queue.get()
        experience_clone = experience.to(device)
        del experience
        replay_buffer.extend(experience_clone)
        experience_queue.task_done()
        received_experiences += len(experience_clone)
        print("LEARNER - " f"Waiting for experiences before learning loop starts ({received_experiences}/{learner_args.learning_starts})")

    print("LEARNER - " f"Learning loop starting")
    sps_start_time = time.time()
    for learning_step in range(1, learner_args.total_learning_steps + 1):
        received = False
        while not experience_queue.empty():
            received = True
            experience = experience_queue.get()
            experience_clone = experience.clone().to(device)
            del experience

            replay_buffer.extend(experience_clone.to(device))

            experience_queue.task_done()
            received_experiences += len(experience_clone)
        if received:
            print("LEARNER - " f"Receiving experiences at step {learning_step} - new total {received_experiences}")

        if received_experiences < learner_args.learning_starts or actor_args.use_heuristics:
            continue

        data: TensorDictBase = replay_buffer.sample().to(device)

        with torch.no_grad():
            next_actions = target_actor_model(data["next_state"]).view(experience_args.batch_size, -1)
            next_observations = data["next_state"].view(experience_args.batch_size, -1)
            next_state = torch.cat([next_observations, data["active_agents"]], dim=1)
            qf1_next_target = target_critic_model(next_state, next_actions)
            next_q_value = data["reward"].view(-1) + (1 - data["done"].view(-1)) * learner_args.gamma * qf1_next_target.view(-1)


        current_observations = data["state"].view(experience_args.batch_size, -1)
        current_state = torch.cat([current_observations, data["active_agents"]], dim=1)
        all_actions = data["actions"].view(experience_args.batch_size, -1)
        qf1_a_values = critic_model(current_state, all_actions).view(-1)
        qf1_loss = mse_loss(qf1_a_values, next_q_value)

        data["priority"] = qf1_loss.expand(experience_args.batch_size)
        replay_buffer.update_tensordict_priority(data)

        # optimize the model
        critic_optimizer.zero_grad(set_to_none=True)
        qf1_loss.backward()
        critic_optimizer.step()

        if learning_step % learner_args.policy_frequency == 0:
            actor_actions = actor_model(data["state"]).view(data["state"].shape[0], -1)
            actor_loss = -critic_model(current_state, actor_actions).mean()

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()

            # update the target network
            for param, target_param in zip(actor_model.parameters(), target_actor_model.parameters()):
                target_param.data.lerp_(param.data, learner_args.tau)
            for param, target_param in zip(critic_model.parameters(), target_critic_model.parameters()):
                target_param.data.lerp_(param.data, learner_args.tau)

        if learning_step % learner_args.actor_model_upload_frequency == 0:
            print("LEARNER - " f"Uploading actor model at step {learning_step}")
            upload_models()

        if learning_step % learner_args.learner_statistics_frequency == 0:
            if not actor_args.use_heuristics:
                writer.add_scalar("learner/critic_values", qf1_a_values.mean().item(), learning_step)
                writer.add_scalar("learner/critic_loss", qf1_loss.item(), learning_step)
                writer.add_scalar("learner/actor_loss", actor_loss.item(), learning_step)
                writer.add_scalar("learner/critic_lr", critic_scheduler.get_last_lr()[0], learning_step)
                writer.add_scalar("learner/actor_lr", actor_scheduler.get_last_lr()[0], learning_step)

            sps = learner_args.learner_statistics_frequency / (time.time() - sps_start_time)
            sps_start_time = time.time()
            writer.add_scalar("learner/sps", sps, learning_step)
            print("LEARNER - " f"SPS: {sps}; STEP: {learning_step}")

            # Updating shared values
            current_step.value = learning_step
            current_sps.value = sps

        if learner_args.checkpoints and learning_step % learner_args.checkpoint_freq == 0:
            eval_results = evaluate_checkpoint(
                learning_step,
                writer,
                device,
                logging_args,
                environment_args,
                max_scaling_drone_count.value,
                max_scaling_sensor_count.value,
                actor_args,
                actor_model.state_dict(),
                critic_model.state_dict(),
                (learning_step / learner_args.checkpoint_freq) % learner_args.model_save_freq == 0
            )
            if learner_args.use_lr_decay:
                critic_scheduler.step(eval_results["avg_reward"])
                actor_scheduler.step(eval_results["avg_reward"])

            scaler.scale(eval_results["all_collected_rate"])
            max_scaling_drone_count.value = scaler.max_drone_count
            max_scaling_sensor_count.value = scaler.max_sensor_count

            writer.add_scalar("learner/max_drone_count", max_scaling_drone_count.value, learning_step)
            writer.add_scalar("learner/max_sensor_count", max_scaling_sensor_count.value, learning_step)
            writer.add_scalar("learner/scaler_confidence", scaler._current_confidence, learning_step)

    print("LEARNER - " f"Finished learning at step {learning_step}")
    evaluate_checkpoint(
        learning_step,
        writer,
        device,
        logging_args,
        environment_args,
        max_scaling_drone_count.value,
        max_scaling_sensor_count.value,
        actor_args,
        actor_model.state_dict(),
        critic_model.state_dict(),
        True
    )