import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks import Critic, FlowVectorField, OneStepFlow

class FQL():
    def __init__(self, obs_dim, action_dim, max_action, chunk_size, discount=0.99,
                tau=0.005, alpha=10.0, flow_steps=10):
                
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.flow = FlowVectorField(obs_dim, action_dim, chunk_size).to(self.device)
        self.critic = Critic(obs_dim, action_dim, chunk_size).to(self.device)
        self.one_step_flow = OneStepFlow(obs_dim, action_dim, chunk_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_params = list(self.flow.parameters()) + list(self.one_step_flow.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.action_dim = action_dim
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.flow_steps = flow_steps
        self.chunk_size = chunk_size
        
        self.total_it = 0
        
    def compute_flow_actions(self, obs, noise):
        action = noise
        for i in range(self.flow_steps):
            t = torch.full((obs.shape[0], 1), i / self.flow_steps).to(self.device)
            vel = self.flow(obs, action, t)
            action = action + vel/self.flow_steps
        action = torch.clamp(action, -1, 1)
        return action
    
    def select_action(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        noise = torch.randn(1, self.action_dim * self.chunk_size).to(self.device)
        action = self.one_step_flow(obs, noise)
        action = torch.clamp(action, -1, 1)
        r_action = action.cpu().data.numpy().reshape(self.chunk_size, self.action_dim)
        return r_action
    
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, mask = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = torch.randn(batch_size, self.action_dim * self.chunk_size).to(self.device)
            next_action = torch.clamp(self.one_step_flow(next_state, noise), -1, 1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + mask * (self.discount**self.chunk_size) * target_Q
            target_Q = torch.clamp(target_Q, -500, 0)
            
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # BC flow loss
        x_0 = torch.randn_like(action)
        x_1 = action
        t = torch.rand(batch_size, 1).to(self.device)
        x_t = (1 - t) * x_0 + t * x_1
        velocity = x_1 - x_0
        
        pred_vel = self.flow(state, x_t, t)
        bc_flow_loss = F.mse_loss(pred_vel, velocity)
        
        # Distillation loss
        
        noise = torch.randn(batch_size, self.action_dim * self.chunk_size).to(self.device)
        with torch.no_grad():
            target_actions = self.compute_flow_actions(state, noise)
        pred_actions = self.one_step_flow(state, noise)
        distill_loss = F.mse_loss(pred_actions, target_actions)
        
        # Q loss (with normalization to prevent Q-value explosion)
        pred_actions_clipped = torch.clamp(pred_actions, -1, 1)
        q1, q2 = self.critic(state, pred_actions_clipped)
        q = (q1 + q2) / 2
        lam = (1 / q.abs().mean()).detach()
        q_loss = -lam * q.mean()
        
        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, 1.0)
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 5000 == 0:
            print(f"  [losses] critic={critic_loss.item():.4f} bc_flow={bc_flow_loss.item():.4f} "
                  f"distill={distill_loss.item():.4f} q={q_loss.item():.4f} "
                  f"reward_mean={reward.mean().item():.4f} Q_mean={((q1+q2)/2).mean().item():.4f}")

    def save(self, path):
        torch.save({
            'flow': self.flow.state_dict(),
            'one_step_flow': self.one_step_flow.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['flow'])
        self.one_step_flow.load_state_dict(checkpoint['one_step_flow'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']
            