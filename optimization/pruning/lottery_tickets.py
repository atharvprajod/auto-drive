import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy
import numpy as np

@dataclass
class LotteryConfig:
    """Configuration for lottery ticket hypothesis"""
    pruning_rounds: int = 10
    prune_rate: float = 0.2      # Fraction of weights to prune per round
    rewind_to_step: int = 1000   # Training step to rewind to
    retrain_steps: int = 5000    # Steps to retrain after pruning
    save_tickets: bool = True    # Whether to save winning tickets

class LotteryTicketPruner:
    def __init__(self,
                 model: nn.Module,
                 config: LotteryConfig):
        """
        Initialize lottery ticket pruner
        
        Args:
            model: Neural network model
            config: Lottery ticket configuration
        """
        self.model = model
        self.config = config
        
        # Initialize weight masks
        self.masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.masks[name] = torch.ones_like(param)
                
        # Save initial weights
        self.initial_state = copy.deepcopy(model.state_dict())
        
    def compute_mask(self,
                    weight: torch.Tensor,
                    current_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pruning mask based on weight magnitudes
        
        Args:
            weight: Weight tensor
            current_mask: Current pruning mask
            
        Returns:
            Updated pruning mask
        """
        # Get current active weights
        active_weights = weight[current_mask.bool()]
        
        # Compute number of weights to prune
        n_prune = int(len(active_weights) * self.config.prune_rate)
        
        if n_prune == 0:
            return current_mask
            
        # Get magnitude threshold
        threshold = torch.sort(torch.abs(active_weights))[0][n_prune]
        
        # Update mask
        new_mask = current_mask * (torch.abs(weight) > threshold)
        return new_mask
        
    def apply_masks(self):
        """Apply masks to model weights"""
        for name, mask in self.masks.items():
            param = self.model.get_parameter(name)
            param.data *= mask
            
    def run_lottery(self,
                    train_fn: callable,
                    rewind_state: Optional[Dict] = None) -> List[Dict]:
        """
        Run lottery ticket pruning process
        
        Args:
            train_fn: Function to train model for specified steps
            rewind_state: Optional state to rewind to (if None, use initial state)
            
        Returns:
            List of pruning statistics for each round
        """
        stats = []
        
        # Use provided rewind state or initial state
        rewind_state = rewind_state or self.initial_state
        
        for round in range(self.config.pruning_rounds):
            # Train model
            train_fn(self.model, self.config.retrain_steps)
            
            # Compute new masks
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    self.masks[name] = self.compute_mask(
                        param.data,
                        self.masks[name]
                    )
                    
            # Get statistics
            round_stats = self.get_pruning_stats()
            stats.append(round_stats)
            
            # Save winning ticket if requested
            if self.config.save_tickets:
                self.save_winning_ticket(f'ticket_round_{round}.pt', round_stats)
                
            # Rewind weights and apply masks
            self.model.load_state_dict(rewind_state)
            self.apply_masks()
            
        return stats
        
    def get_pruning_stats(self) -> Dict[str, float]:
        """Get pruning statistics"""
        total_weights = 0
        remaining_weights = 0
        
        for name, mask in self.masks.items():
            total_weights += mask.numel()
            remaining_weights += mask.sum().item()
            
        return {
            'total_weights': total_weights,
            'remaining_weights': remaining_weights,
            'sparsity': 1 - (remaining_weights / total_weights),
            'compression_ratio': total_weights / remaining_weights
        }
        
    def save_winning_ticket(self, filename: str, stats: Dict[str, float]):
        """Save winning ticket (pruned model and masks)"""
        torch.save({
            'model_state': self.model.state_dict(),
            'masks': self.masks,
            'stats': stats,
            'config': self.config
        }, filename)
        
    @staticmethod
    def load_winning_ticket(filename: str) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Load winning ticket"""
        checkpoint = torch.load(filename)
        return checkpoint['model_state'], checkpoint['masks']
        
class IterativeMagnitudePruning:
    def __init__(self,
                 model: nn.Module,
                 config: LotteryConfig,
                 optimizer: torch.optim.Optimizer):
        """
        Initialize iterative magnitude pruning
        
        Args:
            model: Neural network model
            config: Lottery ticket configuration
            optimizer: Model optimizer
        """
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.pruner = LotteryTicketPruner(model, config)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step
        
        Args:
            batch: Dictionary of input tensors
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Apply masks
        self.pruner.apply_masks()
        
        return {'loss': loss.item()}
        
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              steps: int) -> Dict[str, float]:
        """
        Train model for specified number of steps
        
        Args:
            train_loader: Training data loader
            steps: Number of training steps
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        stats = []
        
        for step in range(steps):
            # Get batch
            try:
                batch = next(train_iter)
            except (StopIteration, NameError):
                train_iter = iter(train_loader)
                batch = next(train_iter)
                
            # Training step
            step_stats = self.train_step(batch)
            stats.append(step_stats)
            
        # Aggregate statistics
        mean_stats = {
            k: np.mean([s[k] for s in stats])
            for k in stats[0].keys()
        }
        
        return mean_stats 