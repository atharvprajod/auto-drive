import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BCConfig:
    """Configuration for behavior cloning"""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

class BehaviorCloningPolicy(nn.Module):
    def __init__(self, config: BCConfig):
        """
        Initialize behavior cloning policy
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # Build policy network
        layers = []
        input_dim = config.state_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = config.hidden_dim
            
        layers.append(nn.Linear(config.hidden_dim, config.action_dim))
        
        # Add final activation if action bounds are specified
        if config.action_bounds is not None:
            layers.append(nn.Tanh())
            
        self.policy = nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute actions from states
        
        Args:
            states: [batch_size, state_dim] state tensor
            
        Returns:
            [batch_size, action_dim] action tensor
        """
        actions = self.policy(states)
        
        # Scale actions to bounds if specified
        if self.config.action_bounds is not None:
            action_low, action_high = self.config.action_bounds
            actions = 0.5 * (action_high + action_low) + \
                     0.5 * (action_high - action_low) * actions
                     
        return actions
        
    def compute_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute behavior cloning loss
        
        Args:
            states: [batch_size, state_dim] state tensor
            actions: [batch_size, action_dim] action tensor
            
        Returns:
            Dictionary of loss terms
        """
        # Predict actions
        pred_actions = self(states)
        
        # Compute MSE loss
        mse_loss = F.mse_loss(pred_actions, actions)
        
        # Add L2 regularization
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param)
            
        total_loss = mse_loss + self.config.weight_decay * l2_loss
        
        return {
            'mse_loss': mse_loss,
            'l2_loss': l2_loss,
            'total_loss': total_loss
        }

class MultiModalBCPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 image_encoder: nn.Module,
                 point_cloud_encoder: nn.Module,
                 fusion_dim: int = 512,
                 config: BCConfig = None):
        """
        Initialize multi-modal behavior cloning policy
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            image_encoder: CNN for processing images
            point_cloud_encoder: PointNet for processing point clouds
            fusion_dim: Dimension of fused features
            config: Policy configuration
        """
        super().__init__()
        self.config = config or BCConfig(state_dim, action_dim)
        
        # Sensor encoders
        self.image_encoder = image_encoder
        self.point_cloud_encoder = point_cloud_encoder
        
        # Feature fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(state_dim + image_encoder.output_dim + 
                     point_cloud_encoder.output_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, action_dim)
        )
        
        if config.action_bounds is not None:
            self.policy_head.append(nn.Tanh())
            
    def forward(self,
                states: torch.Tensor,
                images: torch.Tensor,
                point_clouds: torch.Tensor) -> torch.Tensor:
        """
        Compute actions from multi-modal inputs
        
        Args:
            states: [batch_size, state_dim] state tensor
            images: [batch_size, C, H, W] image tensor
            point_clouds: [batch_size, N, 3] point cloud tensor
            
        Returns:
            [batch_size, action_dim] action tensor
        """
        # Encode sensor inputs
        image_features = self.image_encoder(images)
        point_cloud_features = self.point_cloud_encoder(point_clouds)
        
        # Concatenate features
        combined_features = torch.cat([
            states,
            image_features,
            point_cloud_features
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion_net(combined_features)
        
        # Predict actions
        actions = self.policy_head(fused_features)
        
        # Scale actions to bounds if specified
        if self.config.action_bounds is not None:
            action_low, action_high = self.config.action_bounds
            actions = 0.5 * (action_high + action_low) + \
                     0.5 * (action_high - action_low) * actions
                     
        return actions
        
    def compute_loss(self,
                    states: torch.Tensor,
                    images: torch.Tensor,
                    point_clouds: torch.Tensor,
                    actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-modal behavior cloning loss
        
        Args:
            states: [batch_size, state_dim] state tensor
            images: [batch_size, C, H, W] image tensor
            point_clouds: [batch_size, N, 3] point cloud tensor
            actions: [batch_size, action_dim] action tensor
            
        Returns:
            Dictionary of loss terms
        """
        # Predict actions
        pred_actions = self(states, images, point_clouds)
        
        # Compute MSE loss
        mse_loss = F.mse_loss(pred_actions, actions)
        
        # Add L2 regularization
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param)
            
        total_loss = mse_loss + self.config.weight_decay * l2_loss
        
        return {
            'mse_loss': mse_loss,
            'l2_loss': l2_loss,
            'total_loss': total_loss
        } 