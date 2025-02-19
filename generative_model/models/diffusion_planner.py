import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiffusionTrajectoryPlanner(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=256, num_steps=1000, beta_schedule='linear'):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Define noise schedule
        if beta_schedule == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, num_steps)
        else:
            self.beta = torch.exp(torch.linspace(np.log(1e-4), np.log(0.02), num_steps))
            
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Denoising network
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, x, t):
        """
        Args:
            x: Trajectory tensor [batch_size, sequence_length, state_dim]
            t: Timestep tensor [batch_size]
        """
        t_emb = t.float() / self.num_steps
        t_emb = t_emb.view(-1, 1).expand(-1, x.shape[1]).unsqueeze(-1)
        
        # Concatenate time embedding
        x_t = torch.cat([x, t_emb], dim=-1)
        
        # Predict noise
        noise_pred = self.net(x_t)
        return noise_pred
    
    def sample(self, batch_size, sequence_length, context=None):
        """Generate trajectories using reverse diffusion process"""
        device = next(self.parameters()).device
        
        # Start from random noise
        x = torch.randn(batch_size, sequence_length, self.state_dim).to(device)
        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict and denoise
            noise_pred = self(x, t_batch)
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            # Update sample
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + \
                torch.sqrt(beta_t) * noise
                
        return x
    
    def compute_loss(self, x, noise=None):
        """Compute diffusion loss for training"""
        batch_size = x.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=x.device)
        
        # Add noise
        if noise is None:
            noise = torch.randn_like(x)
            
        # Noisy samples
        x_noisy = torch.sqrt(self.alpha_bar[t].view(-1, 1, 1)) * x + \
                  torch.sqrt(1 - self.alpha_bar[t].view(-1, 1, 1)) * noise
                  
        # Predict noise
        noise_pred = self(x_noisy, t)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss 