from time import time
from typing import Literal, Optional

StateMode = Literal["all_positions", "absolute", "relative", "distance_angle", "angle"]

from tap import Tap

class EnvironmentArgs(Tap):
    def __init__(self):
        Tap.__init__(self, explicit_bool=True)

    max_episode_length: float = 500
    """the maximum length of the episode"""

    state_mode: StateMode = "relative"
    """chooses the state mode to use"""
    id_on_state: bool = True
    """if toggled, the state will be modified to include the agent's ID"""
    state_num_closest_sensors: int = 2
    """the number of closest sensors to consider in the state"""
    state_num_closest_drones: int = 1
    """the number of closest drones to consider in the state"""

    algorithm_iteration_interval: float = 2
    max_seconds_stalled: int = 30
    end_when_all_collected: bool = False
    num_drones: int = 2
    min_sensor_count: int = 12
    max_sensor_count: int = 12
    scenario_size: float = 100
    min_sensor_priority: float = 0.1
    max_sensor_priority: float = 1.0
    full_random_drone_position: bool = False
    communication_range: float = 20
    render_mode: Optional[str] = None

    reward: Literal['punish', 'time-reward', 'reward'] = 'punish'
    speed_action: bool = True
    use_pypy: bool = False

class ExperienceArgs(Tap):
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""

class RemoteEnvironmentArgs(Tap):
    object_name: str = "environment"

class ActorArgs(Tap):
    def __init__(self):
        Tap.__init__(self, explicit_bool=True)

    experience_buffer_size: int = 10_000
    """the actor will accumulate this many experiences before sending them to the central experience replay"""
    use_heuristics: Literal['greedy', 'random', ''] = ''
    actor_use_cuda: bool = False
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    actor_statistics_frequency: int = 10_000

class CoordinationArgs(Tap):
    num_actors: int = 8
    """number of actors collecting experiences"""

class LearnerArgs(Tap):
    def __init__(self):
        Tap.__init__(self, explicit_bool=True)

    learner_cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 3
    """the frequency of training policy (delayed)"""
    actor_model_upload_frequency: int = 1000
    """the frequency of uploading the actor model to the actors"""
    learner_statistics_frequency: int = 10_000
    checkpoints: bool = True
    """whether to save model checkpoints"""
    checkpoint_freq: int = 100_000
    """the frequency of checkpoints"""
    model_save_freq: int = 100
    """will save model every model_save_freq checkpoints"""
    checkpoint_visual_evaluation: bool = False
    """whether to visually evaluate the model at each checkpoint"""
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3

    use_lr_decay: bool = False
    decay_patience: int = 20
    decay_factor: float = 0.9
    min_lr_decay: float = 1e-4

    total_learning_steps: int = 1_000_000
    """the total number of learning steps"""


class LoggingArgs(Tap):
    exp_name: str = "no_name"
    run_name: str = "no_name"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = time()

    def get_path(self):
        return f"runs/{self.exp_name}/{self.run_name}-{self.time}"


class ModelArgs(Tap):
    actor_model_size: int = 256
    critic_model_size: int = 512


class AllArgs(EnvironmentArgs, ExperienceArgs, ActorArgs, CoordinationArgs, LearnerArgs, LoggingArgs, ModelArgs):
    pass