# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy
import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from environment import GrADySEnvironment


@dataclass
class Args:
    exp_name: str = "main_jax"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "GrADyS"
    """the environment id of the Atari game"""
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

    algorithm_iteration_interval: float = 0.5
    max_episode_length: float = 30
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
        max_episode_length=args.max_episode_length,
        scenario_size=args.scenario_size,
        randomize_sensor_positions=args.randomize_sensor_positions
    )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
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
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    # env setup
    env = make_env()

    observation_space = env.observation_space(0)
    action_space = env.action_space(0)

    assert isinstance(action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = action_space.high[0]
    observation_space.dtype = np.float32

    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(action_space.shape),
        action_scale=jnp.array((action_space.high - action_space.low) / 2.0),
        action_bias=jnp.array((action_space.high + action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs[env.agents[0]]),
        target_params=actor.init(actor_key, obs[env.agents[0]]),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs[env.agents[0]], action_space.sample()),
        target_params=qf.init(qf1_key, obs[env.agents[0]], action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        next_state_actions = (actor.apply(actor_state.target_params, next_observations)).clip(0, 1)  # TODO: proper clip
        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
        next_q_value = (rewards + (1 - terminations) * args.gamma * (qf1_next_target)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
        )
        return actor_state, qf1_state, actor_loss_value

    start_time = time.time()
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
            actions = {}
            for index, agent in enumerate(env.agents):
                actions[agent] = actor.apply(actor_state.params, obs[agent])
                actions[agent] = np.array(
                    [
                        (jax.device_get(actions[agent])[0] + np.random.normal(0, actor.action_scale * args.exploration_noise)[0]).clip(
                            action_space.low, action_space.high
                        )
                    ]
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(infos) > 0 and "avg_reward" in infos[env.agents[0]]:
            episode_count += 1
            terminated = True

            info = infos[env.agents[0]]

            avg_reward = sum([info["avg_reward"] for info in infos.values()]) / len(infos)

            writer.add_scalar(
                f"charts/avg_reward",
                avg_reward,
                global_step,
            )
            writer.add_scalar(
                f"charts/episode_duration",
                info["episode_duration"],
                global_step,
            )
            number_of_successes += info["success"]
            writer.add_scalar(
                f"charts/success_rate",
                number_of_successes / episode_count,
                global_step,
            )

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        for agent in env.agents:
            rb.add(obs[agent],
                   next_obs[agent],
                   actions[agent],
                   np.array([rewards[agent]]),
                   np.array([terminations[agent]]),
                   [infos.get(agent, {})])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
            )
            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/step_duration", time.time() - step_start, global_step)
                print(f"{args.exp_name} - SPS:", int(global_step / (time.time() - start_time)))

    # if args.checkpoints and global_step % args.checkpoint_freq == 0 and global_step > 0:
    #     print("Reached checkpoint at step", global_step)
    #
    #     print(f"model saved to {model_path}")
    #
    #     if args.checkpoint_visual_evaluation:
    #         print("Visually evaluating the model")
    #         temp_env = make_env("visual")
    #         obs, _ = temp_env.reset(seed=args.seed)
    #         for _ in range(100):
    #             with torch.no_grad():
    #                 actions = {}
    #                 for agent in temp_env.agents:
    #                     actions[agent] = actor(torch.Tensor(obs[agent]).to(device))
    #                     actions[agent] += torch.normal(0, actor.action_scale * args.exploration_noise)
    #                     actions[agent] = actions[agent].cpu().numpy().clip(action_space.low, action_space.high)
    #
    #             next_obs, rewards, terminations, truncations, infos = temp_env.step(actions)
    #             obs = next_obs
    #             if len(infos) > 0 and "avg_reward" in infos[temp_env.agents[0]]:
    #                 break
    #         temp_env.close()
    #     print("Checkpoint evaluation done")

    env.close()
    writer.close()