import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

class ComfortCost(nn.Module):
    def __init__(self,
                 lateral_jerk_weight: float = 1.0,
                 longitudinal_jerk_weight: float = 1.0,
                 lateral_accel_weight: float = 0.5,
                 longitudinal_accel_weight: float = 0.5,
                 device: Optional[torch.device] = None):
        """
        Initialize comfort cost function for trajectory optimization
        
        Args:
            lateral_jerk_weight: Weight for lateral jerk term
            longitudinal_jerk_weight: Weight for longitudinal jerk term
            lateral_accel_weight: Weight for lateral acceleration term
            longitudinal_accel_weight: Weight for longitudinal acceleration term
            device: Torch device
        """
        super().__init__()
        self.lateral_jerk_weight = lateral_jerk_weight
        self.longitudinal_jerk_weight = longitudinal_jerk_weight
        self.lateral_accel_weight = lateral_accel_weight
        self.longitudinal_accel_weight = longitudinal_accel_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_accelerations(self, trajectories: torch.Tensor, dt: float) -> Dict[str, torch.Tensor]:
        """
        Compute lateral and longitudinal accelerations from trajectories
        
        Args:
            trajectories: Position trajectories [batch_size, num_timesteps, 2]
            dt: Time step
            
        Returns:
            Dictionary of acceleration components
        """
        # Compute velocities
        velocities = (trajectories[:, 1:] - trajectories[:, :-1]) / dt
        
        # Compute accelerations
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
        
        # Decompose into lateral and longitudinal components
        heading = torch.atan2(velocities[:, :, 1], velocities[:, :, 0])
        
        # Rotation matrices for each timestep
        cos_theta = torch.cos(heading)
        sin_theta = torch.sin(heading)
        
        # Compute longitudinal (tangential) and lateral (normal) components
        longitudinal_accel = (accelerations[:, :, 0] * cos_theta[:, 1:] + 
                            accelerations[:, :, 1] * sin_theta[:, 1:])
        
        lateral_accel = (-accelerations[:, :, 0] * sin_theta[:, 1:] + 
                        accelerations[:, :, 1] * cos_theta[:, 1:])
        
        return {
            'longitudinal': longitudinal_accel,
            'lateral': lateral_accel,
            'total': accelerations
        }
        
    def compute_jerks(self, accelerations: Dict[str, torch.Tensor], dt: float) -> Dict[str, torch.Tensor]:
        """
        Compute jerk components from accelerations
        
        Args:
            accelerations: Dictionary of acceleration components
            dt: Time step
            
        Returns:
            Dictionary of jerk components
        """
        # Compute jerk as derivative of acceleration
        longitudinal_jerk = (accelerations['longitudinal'][:, 1:] - 
                           accelerations['longitudinal'][:, :-1]) / dt
        
        lateral_jerk = (accelerations['lateral'][:, 1:] - 
                       accelerations['lateral'][:, :-1]) / dt
        
        total_jerk = (accelerations['total'][:, 1:] - 
                     accelerations['total'][:, :-1]) / dt
        
        return {
            'longitudinal': longitudinal_jerk,
            'lateral': lateral_jerk,
            'total': total_jerk
        }
        
    def forward(self, trajectories: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Compute comfort cost for trajectories
        
        Args:
            trajectories: Position trajectories [batch_size, num_timesteps, 2]
            dt: Time step
            
        Returns:
            Comfort costs [batch_size]
        """
        # Compute accelerations and jerks
        accelerations = self.compute_accelerations(trajectories, dt)
        jerks = self.compute_jerks(accelerations, dt)
        
        # Compute weighted cost components
        lateral_jerk_cost = self.lateral_jerk_weight * torch.sum(jerks['lateral']**2, dim=(1, 2))
        longitudinal_jerk_cost = self.longitudinal_jerk_weight * torch.sum(jerks['longitudinal']**2, dim=(1, 2))
        
        lateral_accel_cost = self.lateral_accel_weight * torch.sum(accelerations['lateral']**2, dim=(1, 2))
        longitudinal_accel_cost = self.longitudinal_accel_weight * torch.sum(accelerations['longitudinal']**2, dim=(1, 2))
        
        # Total cost
        total_cost = (lateral_jerk_cost + longitudinal_jerk_cost + 
                     lateral_accel_cost + longitudinal_accel_cost)
        
        return total_cost
    
    def get_metrics(self, trajectories: torch.Tensor, dt: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute detailed comfort metrics for trajectories
        
        Args:
            trajectories: Position trajectories [batch_size, num_timesteps, 2]
            dt: Time step
            
        Returns:
            Dictionary of comfort metrics
        """
        accelerations = self.compute_accelerations(trajectories, dt)
        jerks = self.compute_jerks(accelerations, dt)
        
        metrics = {
            'max_lateral_accel': torch.max(torch.abs(accelerations['lateral']), dim=1)[0],
            'max_longitudinal_accel': torch.max(torch.abs(accelerations['longitudinal']), dim=1)[0],
            'max_lateral_jerk': torch.max(torch.abs(jerks['lateral']), dim=1)[0],
            'max_longitudinal_jerk': torch.max(torch.abs(jerks['longitudinal']), dim=1)[0],
            'rms_lateral_accel': torch.sqrt(torch.mean(accelerations['lateral']**2, dim=1)),
            'rms_longitudinal_accel': torch.sqrt(torch.mean(accelerations['longitudinal']**2, dim=1)),
            'rms_lateral_jerk': torch.sqrt(torch.mean(jerks['lateral']**2, dim=1)),
            'rms_longitudinal_jerk': torch.sqrt(torch.mean(jerks['longitudinal']**2, dim=1))
        }
        
        return metrics 