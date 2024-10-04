
import time
from copy import deepcopy
import torch
from line_profiler import line_profiler
from tensordict import TensorDict
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import ReplayBuffer, LazyTensorStorage, PrioritizedSampler, TensorDictReplayBuffer

from arguments import LearnerArgs, ExperienceArgs, ActorArgs, LoggingArgs, EnvironmentArgs, ModelArgs
from environment import make_env, observation_space_from_args, action_space_from_args
from heuristics import create_greedy_heuristics, create_random_heuristics
from model import Actor, Critic

# print = lambda *args: args

def evaluate_checkpoint(learner_step: int,
                        writer: SummaryWriter,
                        device: torch.device,
                        logging_args: LoggingArgs,
                        env_args: EnvironmentArgs,
                        actor_args: ActorArgs,
                        actor_model_dict: dict,
                        critic_model_dict: dict):
    print("LEARNER - " "Reached checkpoint at step", learner_step)
    model_path = f"{logging_args.get_path()}/{logging_args.run_name}-checkpoint{learner_step}.cleanrl_model"
    torch.save((actor_model_dict, critic_model_dict), model_path)

    env_args = deepcopy(env_args)
    env_args.use_remote = False
    temp_env = make_env(env_args, True)

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
    temp_env.close()

    actor_model.train()
    critic_model.train()

    print("Checkpoint evaluation done")

def state_dict_to_cpu(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}

def execute_learner(current_step: torch.multiprocessing.Value,
                    current_sps: torch.multiprocessing.Value,
                    learner_args: LearnerArgs,
                    actor_args: ActorArgs,
                    experience_args: ExperienceArgs,
                    logging_args: LoggingArgs,
                    environment_args: EnvironmentArgs,
                    model_args: ModelArgs,
                    experience_queue: torch.multiprocessing.JoinableQueue,
                    actor_model_queues: list[torch.multiprocessing.Queue],
                    start_lock: torch.multiprocessing.Barrier):
    print("LEARNER " "Learner ready, awaiting lock release...")
    start_lock.wait()

    writer = SummaryWriter(logging_args.get_path())

    received_experiences = 0

    observation_space = observation_space_from_args(environment_args)
    action_space = action_space_from_args(environment_args)

    device = torch.device("cuda" if learner_args.learner_cuda else "cpu")
    print("LEARNER - " f"Using device {device}")

    actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    actor_model.share_memory()
    critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)
    critic_model.share_memory()
    target_actor_model = Actor(action_space.shape[0], observation_space.shape[0], model_args).to(device)
    target_actor_model.share_memory()
    target_critic_model = Critic(action_space.shape[0], observation_space.shape[0], environment_args, model_args).to(device)
    target_critic_model.share_memory()
    target_actor_model.load_state_dict(actor_model.state_dict())
    target_critic_model.load_state_dict(critic_model.state_dict())
    critic_optimizer = optim.Adam(list(critic_model.parameters()), lr=learner_args.critic_learning_rate, fused=True)
    actor_optimizer = optim.Adam(list(actor_model.parameters()), lr=learner_args.actor_learning_rate, fused=True)

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
                                           sampler=PrioritizedSampler(experience_args.buffer_size, alpha=0.8, beta=1.1),
                                           priority_key="priority")

    learning_step = 0

    print("LEARNER - " f"Waiting for {learner_args.learning_starts} experiences before starting learning loop")
    while received_experiences < learner_args.learning_starts:
        experience = experience_queue.get()
        experience_clone = experience.clone().to(device)
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

            replay_buffer.extend(experience_clone.to(device, non_blocking=True))

            experience_queue.task_done()
            received_experiences += len(experience_clone)
        if received:
            print("LEARNER - " f"Receiving experiences at step {learning_step} - new total {received_experiences}")

        if received_experiences < learner_args.learning_starts or actor_args.use_heuristics:
            continue

        data: TensorDict = replay_buffer.sample()

        with torch.no_grad():
            next_actions = target_actor_model(data["next_state"]).view(data["next_state"].shape[0], -1)
            next_state = data["next_state"].reshape(data["next_state"].shape[0], -1)
            qf1_next_target = target_critic_model(next_state, next_actions)
            next_q_value = data["reward"].flatten() + (1 - data["done"].flatten()) * learner_args.gamma * (
                qf1_next_target).view(-1)

        current_state = data["state"].reshape(data["state"].shape[0], -1)
        all_actions = data["actions"].reshape(data["actions"].shape[0], -1)
        qf1_a_values = critic_model(current_state, all_actions).view(-1)
        qf1_loss = mse_loss(qf1_a_values, next_q_value)

        data["priority"] = qf1_loss.expand(experience_args.batch_size)

        # optimize the model
        critic_optimizer.zero_grad()
        qf1_loss.backward()
        critic_optimizer.step()

        actor_actions = actor_model(data["state"]).view(data["state"].shape[0], -1)
        actor_loss = -critic_model(current_state, actor_actions).mean()
        if learning_step % learner_args.policy_frequency == 0:
            actor_optimizer.zero_grad()
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

            sps = learner_args.learner_statistics_frequency / (time.time() - sps_start_time)
            sps_start_time = time.time()
            writer.add_scalar("learner/sps", sps, learning_step)
            print("LEARNER - " f"SPS: {sps}; STEP: {learning_step}")

            # Updating shared values
            current_step.value = learning_step
            current_sps.value = sps

        if learner_args.checkpoints and learning_step % learner_args.checkpoint_freq == 0:
            evaluate_checkpoint(
                learning_step,
                writer,
                device,
                logging_args,
                environment_args,
                actor_args,
                actor_model.state_dict(),
                critic_model.state_dict()
            )

    evaluate_checkpoint(
        learning_step,
        writer,
        device,
        logging_args,
        environment_args,
        actor_args,
        actor_model.state_dict(),
        critic_model.state_dict()
    )
    print("LEARNER - " f"Finished learning at step {learning_step}")
    writer.close()