import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

class TrajectoryEnsemble:
    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        """
        Initialize ensemble of trajectory prediction models
        
        Args:
            models: List of trajectory prediction models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0/len(models)] * len(models)
        assert len(self.weights) == len(self.models), "Number of weights must match number of models"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate ensemble prediction by weighted averaging of individual model predictions
        
        Args:
            inputs: Dictionary of input tensors required by models
            
        Returns:
            Weighted average trajectory prediction
        """
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(**inputs)
                predictions.append(pred)
                
        # Stack predictions [num_models, batch_size, sequence_length, state_dim]
        stacked_preds = torch.stack(predictions)
        
        # Weighted average [batch_size, sequence_length, state_dim]
        weights = torch.tensor(self.weights, device=stacked_preds.device)
        weighted_pred = torch.sum(weights.view(-1, 1, 1, 1) * stacked_preds, dim=0)
        
        return weighted_pred
    
    def update_weights(self, validation_errors: List[float]):
        """
        Update ensemble weights based on validation performance
        
        Args:
            validation_errors: List of validation errors for each model
        """
        # Convert errors to weights (lower error -> higher weight)
        inv_errors = 1.0 / (np.array(validation_errors) + 1e-6)
        self.weights = (inv_errors / inv_errors.sum()).tolist()
        
class AdaptiveEnsemble(TrajectoryEnsemble):
    def __init__(self, models: List[nn.Module], context_net: nn.Module):
        """
        Initialize adaptive ensemble with context-dependent weights
        
        Args:
            models: List of trajectory prediction models
            context_net: Neural network that predicts weights given context
        """
        super().__init__(models)
        self.context_net = context_net
        
    def predict(self, inputs: Dict[str, torch.Tensor], context: torch.Tensor) -> torch.Tensor:
        """
        Generate context-dependent ensemble prediction
        
        Args:
            inputs: Dictionary of input tensors required by models
            context: Context tensor for weight prediction
            
        Returns:
            Context-weighted trajectory prediction
        """
        # Predict weights from context
        weights = self.context_net(context)  # [batch_size, num_models]
        weights = torch.softmax(weights, dim=-1)
        
        # Get individual predictions
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(**inputs)
                predictions.append(pred)
                
        # Stack predictions [num_models, batch_size, sequence_length, state_dim]
        stacked_preds = torch.stack(predictions)
        
        # Batch-wise weighted average
        weighted_pred = torch.sum(weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1) * stacked_preds, dim=0)
        
        return weighted_pred 