from pettingzoo.utils import wrappers, parallel_to_aec

from base import GrADySEnvironment
from scenario.environment import CollectionEnvironment


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    original_env = raw_env(render_mode=internal_render_mode)
    environment = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    # this wrapper helps error handling for discrete action spaces
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    environment = CollectionEnvironment(render_mode=render_mode)
    environment = parallel_to_aec(environment)
    return environment


def main():
    env_instance = env("visual")
    env_instance.reset()

    for agent in env_instance.agent_iter():
        observation, reward, termination, truncation, info = env_instance.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env_instance.action_space(agent).sample()

        env_instance.step(action)

    env_instance.close()


if __name__ == "__main__":
    main()