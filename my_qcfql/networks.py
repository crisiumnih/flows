import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, chunk_size=5, hidden_dim=512):
        super(Critic, self).__init__()
        input_dim = obs_dim + action_dim * chunk_size

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        return self.q1(x), self.q2(x)

    def Q1(self, obs, action):
        x = torch.cat([obs, action], 1)
        return self.q1(x)

class FlowVectorField(nn.Module):
    def __init__(self, obs_dim, action_dim, chunk_size=5, hidden_dim=512):
        super(FlowVectorField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim * chunk_size + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
    def forward(self, obs, x_t, t):
        x = torch.cat([obs, x_t, t], dim=-1)
        return self.net(x)
        
        
        
class OneStepFlow(nn.Module):
    def __init__(self, obs_dim, action_dim, chunk_size=5, hidden_dim=512):
        super(OneStepFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim * chunk_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * chunk_size)
        )
    def forward(self, obs, noise):
        x = torch.cat([obs, noise], dim=-1)
        return self.net(x)