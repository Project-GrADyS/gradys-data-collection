import torch


class FastReplayBuffer:
    def __init__(self, buffer_size, batch_size, num_agents,
                 max_num_sensors, obs_shape, action_shape,
                 device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        # Store everything directly on GPU
        self.observations = torch.zeros((buffer_size, num_agents, *obs_shape), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, num_agents, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_agents, *action_shape), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.global_states = torch.zeros((buffer_size, 2 * num_agents + 2 * max_num_sensors), dtype=torch.float32, device=device)
        self.next_global_states = torch.zeros((buffer_size, 2 * num_agents + 2 * max_num_sensors), dtype=torch.float32, device=device)

    def add(self, obs, next_obs, action, reward, done, global_states, next_global_states):
        # Assuming inputs are already torch tensors on GPU
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.global_states[self.ptr] = global_states
        self.next_global_states[self.ptr] = next_global_states

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        # Generate indices on GPU
        indices = torch.randint(0, self.size, (self.batch_size,), device=self.observations.device)
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "reward": self.rewards[indices],
            "done": self.dones[indices],
            "global_state": self.global_states[indices],
            "next_global_state": self.next_global_states[indices],
        }