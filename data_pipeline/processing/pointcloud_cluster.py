import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ClusterConfig:
    """Configuration for point cloud clustering"""
    min_points: int = 10
    max_points: int = 10000
    distance_threshold: float = 0.5
    cluster_radius: float = 1.0
    min_cluster_size: int = 50
    max_cluster_size: int = 5000

class GPUVoxelGrid:
    def __init__(self,
                 voxel_size: float = 0.1,
                 max_points_per_voxel: int = 32,
                 device: Optional[torch.device] = None):
        """
        Initialize GPU-accelerated voxel grid
        
        Args:
            voxel_size: Size of voxel cells
            max_points_per_voxel: Maximum number of points per voxel
            device: Torch device
        """
        self.voxel_size = voxel_size
        self.max_points_per_voxel = max_points_per_voxel
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert point cloud to voxel grid
        
        Args:
            points: [N, D] point cloud
            
        Returns:
            voxels: [M, max_points, D] voxelized points
            coordinates: [M, 3] voxel coordinates
            num_points: [M] number of points in each voxel
        """
        # Compute voxel coordinates for each point
        coords = torch.floor(points[:, :3] / self.voxel_size).int()
        
        # Get unique voxels
        unq_coords, inverse_indices, voxel_counts = torch.unique(
            coords, dim=0, return_inverse=True, return_counts=True)
            
        # Initialize voxel grid
        num_voxels = len(unq_coords)
        voxels = torch.zeros(num_voxels, self.max_points_per_voxel, points.shape[1],
                           device=self.device)
        num_points = torch.zeros(num_voxels, device=self.device)
        
        # Fill voxels
        for i, point_idx in enumerate(inverse_indices):
            if num_points[point_idx] < self.max_points_per_voxel:
                voxels[point_idx, int(num_points[point_idx])] = points[i]
                num_points[point_idx] += 1
                
        return voxels, unq_coords, num_points

class CUDAClusteringModule(nn.Module):
    def __init__(self, config: ClusterConfig, device: Optional[torch.device] = None):
        """
        Initialize CUDA-accelerated clustering module
        
        Args:
            config: Clustering configuration
            device: Torch device
        """
        super().__init__()
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize voxel grid
        self.voxel_grid = GPUVoxelGrid(
            voxel_size=config.distance_threshold,
            device=self.device
        )
        
    def compute_adjacency(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute adjacency matrix for voxel coordinates
        
        Args:
            coords: [N, 3] voxel coordinates
            
        Returns:
            [N, N] adjacency matrix
        """
        # Compute pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.sum(diff**2, dim=-1)
        
        # Create adjacency matrix
        radius = int(self.config.cluster_radius / self.voxel_grid.voxel_size)
        adj_matrix = (dist <= radius**2).float()
        
        return adj_matrix
        
    def find_clusters(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Find connected components using GPU-accelerated BFS
        
        Args:
            adj_matrix: [N, N] adjacency matrix
            
        Returns:
            [N] cluster labels
        """
        N = adj_matrix.shape[0]
        labels = -torch.ones(N, device=self.device, dtype=torch.long)
        current_label = 0
        
        for i in range(N):
            if labels[i] >= 0:
                continue
                
            # Initialize queue for BFS
            queue = [i]
            labels[i] = current_label
            
            while queue:
                node = queue.pop(0)
                neighbors = torch.where(adj_matrix[node] > 0)[0]
                
                for neighbor in neighbors:
                    if labels[neighbor] < 0:
                        labels[neighbor] = current_label
                        queue.append(neighbor.item())
                        
            current_label += 1
            
        return labels
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform point cloud clustering
        
        Args:
            points: [N, D] point cloud
            
        Returns:
            Dictionary containing cluster information
        """
        # Voxelize point cloud
        voxels, coords, num_points = self.voxel_grid.voxelize(points)
        
        # Filter voxels by point count
        valid_mask = (num_points >= self.config.min_points)
        valid_voxels = voxels[valid_mask]
        valid_coords = coords[valid_mask]
        
        # Compute adjacency matrix
        adj_matrix = self.compute_adjacency(valid_coords)
        
        # Find clusters
        cluster_labels = self.find_clusters(adj_matrix)
        
        # Compute cluster statistics
        unique_labels = torch.unique(cluster_labels)
        num_clusters = len(unique_labels)
        
        cluster_sizes = torch.zeros(num_clusters, device=self.device)
        cluster_centers = torch.zeros(num_clusters, 3, device=self.device)
        
        for i, label in enumerate(unique_labels):
            mask = (cluster_labels == label)
            cluster_points = valid_voxels[mask].reshape(-1, points.shape[1])
            
            cluster_sizes[i] = len(cluster_points)
            cluster_centers[i] = torch.mean(cluster_points[:, :3], dim=0)
            
        # Filter clusters by size
        valid_clusters = (cluster_sizes >= self.config.min_cluster_size) & \
                        (cluster_sizes <= self.config.max_cluster_size)
                        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers[valid_clusters],
            'cluster_sizes': cluster_sizes[valid_clusters],
            'valid_clusters': valid_clusters
        }
        
    @torch.no_grad()
    def visualize_clusters(self, points: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
        """
        Generate colored point cloud for visualization
        
        Args:
            points: [N, D] point cloud
            cluster_labels: [N] cluster labels
            
        Returns:
            [N, 6] colored point cloud (xyz + rgb)
        """
        # Generate random colors for clusters
        num_clusters = torch.max(cluster_labels) + 1
        colors = torch.rand(num_clusters, 3, device=self.device)
        
        # Assign colors to points
        point_colors = colors[cluster_labels]
        
        # Combine points and colors
        colored_points = torch.cat([points[:, :3], point_colors], dim=1)
        
        return colored_points 