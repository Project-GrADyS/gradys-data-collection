import logging
import multiprocessing
import subprocess
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from actor import execute_actor
from arguments import (ActorArgs, EnvironmentArgs, CoordinationArgs,
                       LoggingArgs, LearnerArgs, ExperienceArgs, ModelArgs, AllArgs)
from learner import execute_learner

# print = lambda *args: args

def start_namespace_server() -> subprocess.Popen:
    executable = sys.executable
    return subprocess.Popen([f"{executable}", "-m", "Pyro5.nameserver"], stdout=sys.stdout, stderr=sys.stderr)


def main():
    AllArgs().parse_args()
    actor_args = ActorArgs().parse_args(known_only=True)
    environment_args = EnvironmentArgs().parse_args(known_only=True)
    coordination_args = CoordinationArgs().parse_args(known_only=True)
    logging_args = LoggingArgs().parse_args(known_only=True)
    learner_args = LearnerArgs().parse_args(known_only=True)
    experience_args = ExperienceArgs().parse_args(known_only=True)
    model_args = ModelArgs().parse_args(known_only=True)

    print("MAIN - " f"Running experiment {logging_args.get_path()}")
    writer = SummaryWriter(logging_args.get_path())

    all_arg_properties = (vars(model_args) | vars(actor_args) | vars(environment_args) | vars(coordination_args) |
                          vars(logging_args) | vars(learner_args) | vars(experience_args))

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in all_arg_properties.items()])),
    )

    # Beginning coordination process
    print("MAIN - " "Starting nameserver")
    ns_process = start_namespace_server()
    print("MAIN - " "Sleeping until nameserver starts...")
    time.sleep(1)

    ctx = torch.multiprocessing.get_context('spawn')

    print("MAIN - " "Creating synchronization structures")
    synchronization_lock = ctx.Barrier(2 + coordination_args.num_actors)

    actor_model_queues = [ctx.Queue() for _ in range(coordination_args.num_actors)]
    experience_queue = ctx.Queue()

    learner_step = torch.multiprocessing.Value('i', 0)
    learner_sps = torch.multiprocessing.Value('d', 0.0)
    actor_steps = [torch.multiprocessing.Value('i', 0) for _ in range(coordination_args.num_actors)]
    actor_sps = [torch.multiprocessing.Value('d', 0.0) for _ in range(coordination_args.num_actors)]

    print("MAIN - " "Starting learner process")
    learner_process: multiprocessing.Process = \
        ctx.Process(target=execute_learner,
                    args=(learner_step, learner_sps, learner_args, actor_args, experience_args,
                          logging_args, environment_args, model_args, experience_queue, actor_model_queues,
                          synchronization_lock))
    learner_process.start()

    actor_processes = []
    for i in range(coordination_args.num_actors):
        print("MAIN - " f"Starting actor process {i}")

        actor_process = \
            ctx.Process(target=execute_actor,
                        args=(actor_steps[i], actor_sps[i], i, model_args, actor_args, logging_args, environment_args,
                              experience_queue, actor_model_queues[i], synchronization_lock))
        actor_processes.append(actor_process)
        actor_process.start()

    def terminate_all():
        print("MAIN - " "Terminating all processes")
        ns_process.terminate()
        learner_process.terminate()
        for actor in actor_processes:
            actor.terminate()
    try:
        print("MAIN - " f"Waiting until processes are ready...")
        synchronization_lock.wait()
        print("MAIN - " "Processes ready")

        time_start = time.time()
        while True:
            live_processes = []
            dead_processes = []

            learner_label = f"L(s={learner_step.value};sps={learner_sps.value:.2f})"
            if learner_process.is_alive():
                live_processes.append(learner_label)
            else:
                dead_processes.append(learner_label)

            for index, actor in enumerate(actor_processes):
                actor_label = f"A{index}(s={actor_steps[index].value};sps={actor_sps[index].value:.2f})"
                if actor.is_alive():
                    live_processes.append(actor_label)
                else:
                    dead_processes.append(actor_label)

            experience_queue_size = experience_queue.qsize()
            actor_model_queues_sizes = [q.qsize() for q in actor_model_queues]

            time_elapsed = time.time() - time_start
            formatted_time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

            main_status_string = f"""
--------- ELAPSED: {formatted_time_elapsed} ---------
MAIN -  Live processes: {live_processes}
MAIN -  Dead processes: {dead_processes}
MAIN -  Experience queue size: {experience_queue_size}
MAIN -  Actor model queue sizes: {actor_model_queues_sizes}
            """
            print(main_status_string)

            if not learner_process.is_alive():
                break

            time.sleep(1)
    finally:
        terminate_all()

if __name__ == "__main__":
    main()