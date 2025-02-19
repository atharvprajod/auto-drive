import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class StructuralPruneConfig:
    """Configuration for structural pruning"""
    target_sparsity: float = 0.5  # Target channel sparsity
    pruning_steps: int = 10       # Number of pruning steps
    min_channels: int = 4         # Minimum channels per layer
    importance_criterion: str = 'l1'  # ['l1', 'l2', 'random']
    retrain_steps: int = 1000     # Steps to retrain after pruning

class ChannelPruner:
    def __init__(self, 
                 model: nn.Module,
                 config: StructuralPruneConfig):
        """
        Initialize channel pruner
        
        Args:
            model: Neural network model
            config: Pruning configuration
        """
        self.model = model
        self.config = config
        
        # Get prunable layers (Conv2d and Linear)
        self.prunable_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.prunable_layers.append((name, module))
                
    def compute_importance(self, layer: nn.Module) -> torch.Tensor:
        """
        Compute channel importance scores
        
        Args:
            layer: Neural network layer
            
        Returns:
            Channel importance scores
        """
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data
            if self.config.importance_criterion == 'l1':
                importance = torch.norm(weights, p=1, dim=(1, 2, 3))
            elif self.config.importance_criterion == 'l2':
                importance = torch.norm(weights, p=2, dim=(1, 2, 3))
            else:  # random
                importance = torch.rand(weights.size(0))
        else:  # Linear layer
            weights = layer.weight.data
            if self.config.importance_criterion == 'l1':
                importance = torch.norm(weights, p=1, dim=1)
            elif self.config.importance_criterion == 'l2':
                importance = torch.norm(weights, p=2, dim=1)
            else:  # random
                importance = torch.rand(weights.size(0))
                
        return importance
        
    def prune_layer(self, 
                    layer: nn.Module,
                    keep_mask: torch.Tensor) -> nn.Module:
        """
        Prune channels from layer
        
        Args:
            layer: Neural network layer
            keep_mask: Boolean mask for channels to keep
            
        Returns:
            Pruned layer
        """
        device = next(layer.parameters()).device
        
        if isinstance(layer, nn.Conv2d):
            # Create new layer with fewer channels
            new_out_channels = keep_mask.sum().item()
            new_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=new_out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None
            ).to(device)
            
            # Copy remaining weights and bias
            new_layer.weight.data = layer.weight.data[keep_mask]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_mask]
                
        else:  # Linear layer
            # Create new layer with fewer features
            new_out_features = keep_mask.sum().item()
            new_layer = nn.Linear(
                in_features=layer.in_features,
                out_features=new_out_features,
                bias=layer.bias is not None
            ).to(device)
            
            # Copy remaining weights and bias
            new_layer.weight.data = layer.weight.data[keep_mask]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[keep_mask]
                
        return new_layer
        
    def prune_model(self) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Prune channels from model
        
        Returns:
            Pruned model and channel masks
        """
        channel_masks = {}
        
        # Compute importance scores for each layer
        for name, layer in self.prunable_layers:
            importance = self.compute_importance(layer)
            
            # Determine number of channels to keep
            num_channels = importance.size(0)
            keep_channels = max(
                int(num_channels * (1 - self.config.target_sparsity)),
                self.config.min_channels
            )
            
            # Get top channels
            _, indices = torch.sort(importance, descending=True)
            keep_mask = torch.zeros_like(importance, dtype=torch.bool)
            keep_mask[indices[:keep_channels]] = True
            
            # Store mask
            channel_masks[name] = keep_mask
            
            # Prune layer
            pruned_layer = self.prune_layer(layer, keep_mask)
            
            # Replace layer in model
            parent_name = '.'.join(name.split('.')[:-1])
            layer_name = name.split('.')[-1]
            if parent_name:
                parent = self.model.get_submodule(parent_name)
                setattr(parent, layer_name, pruned_layer)
            else:
                setattr(self.model, layer_name, pruned_layer)
                
        return self.model, channel_masks
        
    def get_pruning_stats(self) -> Dict[str, float]:
        """Get pruning statistics"""
        total_params = 0
        remaining_params = 0
        
        for name, layer in self.prunable_layers:
            if isinstance(layer, nn.Conv2d):
                params = np.prod(layer.weight.shape)
                if layer.bias is not None:
                    params += layer.weight.size(0)
            else:  # Linear
                params = np.prod(layer.weight.shape)
                if layer.bias is not None:
                    params += layer.weight.size(0)
                    
            total_params += params
            
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                params = np.prod(layer.weight.shape)
                if layer.bias is not None:
                    params += layer.weight.size(0)
                remaining_params += params
                
        return {
            'total_params': total_params,
            'remaining_params': remaining_params,
            'compression_ratio': total_params / remaining_params,
            'sparsity': 1 - (remaining_params / total_params)
        } 