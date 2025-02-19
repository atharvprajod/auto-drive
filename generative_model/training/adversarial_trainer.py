import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, trajectories):
        """
        Args:
            trajectories: [batch_size, sequence_length, state_dim]
        Returns:
            Scalar discriminator predictions
        """
        batch_size = trajectories.shape[0]
        flattened = trajectories.view(batch_size, -1)
        return self.net(flattened)

class AdversarialTrainer:
    def __init__(self, 
                 generator: nn.Module,
                 discriminator: TrajectoryDiscriminator,
                 gen_optimizer: optim.Optimizer,
                 disc_optimizer: optim.Optimizer,
                 device: torch.device):
        """
        Initialize adversarial trainer for trajectory generation
        
        Args:
            generator: Trajectory generator model
            discriminator: Trajectory discriminator model
            gen_optimizer: Generator optimizer
            disc_optimizer: Discriminator optimizer
            device: torch device
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.device = device
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_step(self, real_trajectories: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step
        
        Args:
            real_trajectories: Ground truth trajectories [batch_size, sequence_length, state_dim]
            conditions: Dictionary of conditioning information
            
        Returns:
            Dictionary of loss values
        """
        batch_size = real_trajectories.shape[0]
        
        # Train discriminator
        self.disc_optimizer.zero_grad()
        
        # Real samples
        real_pred = self.discriminator(real_trajectories)
        real_target = torch.ones(batch_size, 1).to(self.device)
        d_loss_real = self.criterion(real_pred, real_target)
        
        # Fake samples
        fake_trajectories = self.generator(**conditions)
        fake_pred = self.discriminator(fake_trajectories.detach())
        fake_target = torch.zeros(batch_size, 1).to(self.device)
        d_loss_fake = self.criterion(fake_pred, fake_target)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.disc_optimizer.step()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        
        # Generate fake trajectories
        fake_trajectories = self.generator(**conditions)
        fake_pred = self.discriminator(fake_trajectories)
        
        # Generator wants discriminator to predict real
        g_loss = self.criterion(fake_pred, real_target)
        g_loss.backward()
        self.gen_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item()
        }
        
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate models on validation dataset
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for real_trajectories, conditions in val_loader:
                batch_size = real_trajectories.shape[0]
                
                # Generate fake trajectories
                fake_trajectories = self.generator(**conditions)
                
                # Discriminator predictions
                real_pred = self.discriminator(real_trajectories)
                fake_pred = self.discriminator(fake_trajectories)
                
                # Compute losses
                real_target = torch.ones(batch_size, 1).to(self.device)
                fake_target = torch.zeros(batch_size, 1).to(self.device)
                
                d_loss = self.criterion(real_pred, real_target) + \
                        self.criterion(fake_pred, fake_target)
                g_loss = self.criterion(fake_pred, real_target)
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                num_batches += 1
                
        self.generator.train()
        self.discriminator.train()
        
        return {
            'val_g_loss': total_g_loss / num_batches,
            'val_d_loss': total_d_loss / num_batches
        } 