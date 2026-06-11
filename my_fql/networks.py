import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        q1 = F.gelu(self.l1(x))
        q1 = F.gelu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.gelu(self.l4(x))
        q2 = F.gelu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    
    def Q1(self, obs, action):
        x = torch.cat([obs, action], 1)
        q1 = F.gelu(self.l1(x))
        q1 = F.gelu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class FlowVectorField(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(FlowVectorField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, obs, x_t, t):
        x = torch.cat([obs, x_t, t], dim=-1)
        return self.net(x)
        
        
        
class OneStepFlow(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(OneStepFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, obs, noise):
        x = torch.cat([obs, noise], dim=-1)
        return self.net(x)