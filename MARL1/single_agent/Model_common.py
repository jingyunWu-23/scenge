"""Network definitions required by the bundled CARLA Evolution MAPPO agent."""

import torch as th
from torch import nn


class ActorNetwork(nn.Module):
    """Actor MLP used by the original MARL1 MAPPO checkpoints."""

    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.output_act = output_act

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        return self.output_act(self.fc3(out), dim=1)


class CriticNetwork(nn.Module):
    """Critic MLP kept for compatibility with older MARL1 imports."""

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        return self.fc3(out)


class ActorCriticNetwork(nn.Module):
    """Shared-body actor critic kept for compatibility with MARL1 imports."""

    def __init__(self, state_dim, action_dim, hidden_size, actor_output_act, critic_output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def forward(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val
