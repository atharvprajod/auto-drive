import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class SignedDistanceField:
    def __init__(self, resolution: float = 0.1, device: Optional[torch.device] = None):
        """
        Initialize signed distance field for collision checking
        
        Args:
            resolution: Grid resolution in meters
            device: Torch device
        """
        self.resolution = resolution
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize empty SDF grid
        self.grid_size = [100, 100]  # 10m x 10m area
        self.grid = torch.zeros(self.grid_size, device=self.device)
        self.origin = torch.tensor([-5.0, -5.0], device=self.device)
        
    def update_obstacles(self, obstacles: List[Dict[str, torch.Tensor]]):
        """
        Update SDF grid with new obstacle positions
        
        Args:
            obstacles: List of obstacle dictionaries with position and size
        """
        # Reset grid
        self.grid.fill_(float('inf'))
        
        # Update for each obstacle
        for obstacle in obstacles:
            pos = obstacle['position']  # [x, y]
            size = obstacle['size']     # [width, height]
            
            # Convert to grid coordinates
            grid_pos = ((pos - self.origin) / self.resolution).long()
            grid_size = (size / self.resolution).long()
            
            # Create obstacle mask
            x = torch.arange(self.grid_size[0], device=self.device)
            y = torch.arange(self.grid_size[1], device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Compute distances to obstacle boundaries
            dx = torch.clamp(torch.abs(X - grid_pos[0]) - grid_size[0]/2, min=0)
            dy = torch.clamp(torch.abs(Y - grid_pos[1]) - grid_size[1]/2, min=0)
            dist = torch.sqrt(dx**2 + dy**2) * self.resolution
            
            # Update grid with minimum distance
            self.grid = torch.minimum(self.grid, dist)
            
        # Compute sign (negative inside obstacles)
        inside_mask = self.grid < self.resolution
        self.grid[inside_mask] *= -1
        
    def get_distance(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get signed distances at query positions
        
        Args:
            positions: Query positions [batch_size, num_points, 2]
            
        Returns:
            Signed distances [batch_size, num_points]
        """
        batch_size, num_points, _ = positions.shape
        
        # Convert to grid coordinates
        grid_positions = (positions - self.origin) / self.resolution
        
        # Interpolate distances
        grid_positions = grid_positions.view(-1, 1, 1, 2)  # [batch*points, 1, 1, 2]
        grid_positions = 2.0 * grid_positions / (torch.tensor(self.grid_size, device=self.device) - 1) - 1.0
        
        distances = F.grid_sample(
            self.grid.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            grid_positions,
            mode='bilinear',
            align_corners=True
        )
        
        return distances.view(batch_size, num_points)

class CollisionCost(nn.Module):
    def __init__(self, 
                 sdf: SignedDistanceField,
                 collision_threshold: float = 0.5,
                 collision_weight: float = 1.0):
        """
        Initialize collision cost function
        
        Args:
            sdf: Signed distance field for collision checking
            collision_threshold: Minimum safe distance to obstacles
            collision_weight: Weight of collision cost term
        """
        super().__init__()
        self.sdf = sdf
        self.collision_threshold = collision_threshold
        self.collision_weight = collision_weight
        
    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute collision costs for trajectories
        
        Args:
            trajectories: Batch of trajectories [batch_size, num_timesteps, 2]
            
        Returns:
            Collision costs [batch_size]
        """
        # Get signed distances along trajectories
        distances = self.sdf.get_distance(trajectories)
        
        # Compute collision cost (higher cost for closer distances)
        costs = torch.relu(self.collision_threshold - distances)
        
        # Sum over timesteps
        total_costs = self.collision_weight * torch.sum(costs, dim=1)
        
        return total_costs
    
    def gradient(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of collision cost with respect to trajectories
        
        Args:
            trajectories: Batch of trajectories [batch_size, num_timesteps, 2]
            
        Returns:
            Cost gradients [batch_size, num_timesteps, 2]
        """
        trajectories.requires_grad_(True)
        
        # Forward pass
        costs = self.forward(trajectories)
        
        # Compute gradients
        grads = torch.autograd.grad(costs.sum(), trajectories)[0]
        
        return grads 