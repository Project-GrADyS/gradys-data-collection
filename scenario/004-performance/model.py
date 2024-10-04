import torch
from torch import nn
import torch.nn.functional as F

from arguments import EnvironmentArgs, ModelArgs

torch.backends.cuda.matmul.allow_tf32 = True

class Critic(nn.Module):
    def __init__(self, action_space_size, observation_space_size, env_args: EnvironmentArgs, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(
            observation_space_size * env_args.num_drones + action_space_size * env_args.num_drones, args.critic_model_size)
        self.fc2 = nn.Linear(args.critic_model_size, args.critic_model_size)
        self.fc3 = nn.Linear(args.critic_model_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, action_space_size, observation_space_size, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(observation_space_size, args.actor_model_size)
        self.fc2 = nn.Linear(args.actor_model_size, args.actor_model_size)
        self.fc_mu = nn.Linear(args.actor_model_size, action_space_size)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((1 - 0) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((1 + 0) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias