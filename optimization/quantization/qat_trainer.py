import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

@dataclass
class QATConfig:
    """Configuration for quantization-aware training"""
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    dtype: torch.dtype = torch.qint8
    qscheme: torch.qscheme = torch.per_tensor_affine
    freeze_bn_steps: int = 1000
    eval_freq: int = 100

class QuantizationAwareTrainer:
    def __init__(self,
                 model: nn.Module,
                 config: QATConfig):
        """
        Initialize quantization-aware trainer
        
        Args:
            model: Neural network model
            config: QAT configuration
        """
        self.model = model
        self.config = config
        
        # Prepare model for QAT
        self.prepare_qat_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.qat_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def prepare_qat_model(self):
        """Prepare model for quantization-aware training"""
        # Specify quantization configuration
        self.qconfig = torch.quantization.get_default_qat_qconfig(
            'fbgemm' if self.config.dtype == torch.qint8 else 'qnnpack'
        )
        
        # Prepare model for QAT
        self.qat_model = torch.quantization.prepare_qat(
            self.model,
            self.qconfig
        )
        
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
        outputs = self.qat_model(**batch)
        loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
        
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training statistics
        """
        self.qat_model.train()
        stats = []
        
        for i, batch in enumerate(train_loader):
            # Freeze batch norm statistics after certain steps
            if epoch * len(train_loader) + i > self.config.freeze_bn_steps:
                self.qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                
            # Training step
            step_stats = self.train_step(batch)
            stats.append(step_stats)
            
        # Aggregate statistics
        mean_stats = {
            k: sum(s[k] for s in stats) / len(stats)
            for k in stats[0].keys()
        }
        
        return mean_stats
        
    def validate(self,
                val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation statistics
        """
        self.qat_model.eval()
        stats = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                outputs = self.qat_model(**batch)
                loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
                
                stats.append({'loss': loss.item()})
                
        # Aggregate statistics
        mean_stats = {
            k: sum(s[k] for s in stats) / len(stats)
            for k in stats[0].keys()
        }
        
        return mean_stats
        
    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, List[float]]:
        """
        Train model with quantization awareness
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [] if val_loader else None
        }
        
        best_val_loss = float('inf')
        best_model = None
        
        for epoch in range(self.config.num_epochs):
            # Training epoch
            train_stats = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_stats['loss'])
            
            # Validation
            if val_loader and epoch % self.config.eval_freq == 0:
                val_stats = self.validate(val_loader)
                history['val_loss'].append(val_stats['loss'])
                
                # Save best model
                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    best_model = copy.deepcopy(self.qat_model.state_dict())
                    
        # Restore best model if available
        if best_model is not None:
            self.qat_model.load_state_dict(best_model)
            
        return history
        
    def convert_model(self) -> nn.Module:
        """
        Convert QAT model to quantized model
        
        Returns:
            Quantized model
        """
        self.qat_model.eval()
        return torch.quantization.convert(self.qat_model)
        
    def get_quantization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get quantization statistics for each layer"""
        stats = {}
        
        for name, module in self.qat_model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                obs = module.weight_fake_quant
                stats[name] = {
                    'scale': obs.scale.item(),
                    'zero_point': obs.zero_point.item(),
                    'dtype': str(obs.dtype),
                    'qscheme': str(obs.qscheme)
                }
                
        return stats
        
class QATCallback:
    def __init__(self,
                 eval_freq: int = 100,
                 save_best: bool = True):
        """
        Callback for quantization-aware training
        
        Args:
            eval_freq: Evaluation frequency
            save_best: Whether to save best model
        """
        self.eval_freq = eval_freq
        self.save_best = save_best
        self.best_loss = float('inf')
        self.best_model = None
        
    def on_epoch_end(self,
                    trainer: QuantizationAwareTrainer,
                    epoch: int,
                    logs: Dict[str, float]):
        """Called at the end of each epoch"""
        # Save best model
        if self.save_best and logs.get('val_loss', float('inf')) < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_model = copy.deepcopy(trainer.qat_model.state_dict())
            
    def on_train_end(self, trainer: QuantizationAwareTrainer):
        """Called at the end of training"""
        if self.best_model is not None:
            trainer.qat_model.load_state_dict(self.best_model) 