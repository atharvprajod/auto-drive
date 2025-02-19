import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..policies.behavior_cloning import BehaviorCloningPolicy, MultiModalBCPolicy
from ..policies.dagger import DAgger
from ..dataset.driving_dataset import DrivingDataset, DataTransform
from .batch_processor import BatchProcessorConfig, CUDABatchProcessor, AsyncBatchLoader, CUDAMemoryManager
from .optimizers import OptimizerConfig, create_optimizer, create_scheduler, apply_gradient_updates

@dataclass
class TrainingConfig:
    """Configuration for imitation learning training"""
    # General training settings
    method: str = 'behavior_cloning'  # ['behavior_cloning', 'dagger']
    num_epochs: int = 100
    batch_size: int = 32
    
    # Model architecture
    state_dim: int = 6
    action_dim: int = 2
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Multi-modal settings
    use_images: bool = True
    use_lidar: bool = True
    image_size: Tuple[int, int] = (224, 224)
    num_points: int = 2048
    
    # Optimization settings
    optimizer: str = 'lamb'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Learning rate scheduling
    lr_schedule: str = 'cosine'
    warmup_steps: int = 1000
    decay_steps: int = 10000
    min_lr_ratio: float = 0.1
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None
    
    # CUDA optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    cuda_cache_clear_freq: int = 100
    prefetch_factor: int = 2
    num_workers: int = 4
    
    # DAgger specific
    dagger_beta_schedule: str = 'linear'
    dagger_beta_start: float = 1.0
    dagger_beta_end: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 10
    use_wandb: bool = True
    experiment_name: str = 'imitation_learning'

class ImitationLearningTrainer:
    def __init__(self,
                 config: TrainingConfig,
                 data_dir: str,
                 checkpoint_dir: str):
        """
        Initialize imitation learning trainer
        
        Args:
            config: Training configuration
            data_dir: Path to dataset
            checkpoint_dir: Path to save checkpoints
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize policy
        self.policy = self._create_policy()
        self.policy.to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer_config = OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            lr_schedule=config.lr_schedule,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            min_lr_ratio=config.min_lr_ratio,
            clip_grad_norm=config.clip_grad_norm,
            clip_grad_value=config.clip_grad_value
        )
        
        self.optimizer = create_optimizer(self.policy.parameters(), optimizer_config)
        self.scheduler = create_scheduler(self.optimizer, optimizer_config)
        
        # Initialize CUDA optimizations
        batch_processor_config = BatchProcessorConfig(
            use_mixed_precision=config.use_mixed_precision,
            gradient_clip_val=config.clip_grad_norm,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            cuda_cache_clear_freq=config.cuda_cache_clear_freq
        )
        
        self.batch_processor = CUDABatchProcessor(batch_processor_config)
        self.memory_manager = CUDAMemoryManager()
        
        # Initialize data loaders with async loading
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.train_loader = AsyncBatchLoader(
            self.train_loader,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor
        )
        
        # Initialize DAgger if needed
        if config.method == 'dagger':
            self.dagger = DAgger(
                policy=self.policy,
                config=self._create_dagger_config()
            )
            
        # Initialize logging
        if config.use_wandb:
            wandb.init(
                project=config.experiment_name,
                config=vars(config)
            )
            
        # Apply CUDA memory optimizations to model
        self.policy = self.batch_processor.optimize_memory(self.policy)
        
    def _create_policy(self) -> nn.Module:
        """Create policy network based on configuration"""
        if self.config.use_images or self.config.use_lidar:
            # Create encoders
            image_encoder = self._create_image_encoder() if self.config.use_images else None
            point_cloud_encoder = self._create_point_cloud_encoder() if self.config.use_lidar else None
            
            return MultiModalBCPolicy(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                image_encoder=image_encoder,
                point_cloud_encoder=point_cloud_encoder,
                fusion_dim=self.config.hidden_dim * 2,
                config=self._create_bc_config()
            )
        else:
            return BehaviorCloningPolicy(self._create_bc_config())
            
    def _create_bc_config(self):
        """Create behavior cloning configuration"""
        from ..policies.behavior_cloning import BCConfig
        return BCConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
    def _create_dagger_config(self):
        """Create DAgger configuration"""
        from ..policies.dagger import DAggerConfig
        return DAggerConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            beta_schedule=self.config.dagger_beta_schedule,
            beta_start=self.config.dagger_beta_start,
            beta_end=self.config.dagger_beta_end
        )
        
    def _create_image_encoder(self) -> nn.Module:
        """Create CNN encoder for images"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.config.hidden_dim)
        )
        
    def _create_point_cloud_encoder(self) -> nn.Module:
        """Create PointNet encoder for point clouds"""
        return nn.Sequential(
            # MLP for point features
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            
            # Max pooling to get global features
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, self.config.hidden_dim)
        )
        
    def _create_data_loaders(self):
        """Create data loaders for training and validation"""
        # Create data transform
        transform = DataTransform(
            image_size=self.config.image_size,
            num_points=self.config.num_points,
            normalize=True
        )
        
        # Create datasets
        train_dataset = DrivingDataset(
            data_path=self.data_dir,
            split='train',
            transform=transform
        )
        
        val_dataset = DrivingDataset(
            data_path=self.data_dir,
            split='val',
            transform=transform
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()
        epoch_stats = []
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # Process batch with CUDA optimizations
            outputs, loss = self.batch_processor.process_batch(
                model=self.policy,
                batch=batch,
                optimizer=self.optimizer,
                training=True
            )
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Check and optimize memory usage
            self.memory_manager.check_memory()
            
            # Record statistics
            stats = {k: v.item() if torch.is_tensor(v) else v 
                    for k, v in outputs.items()}
            stats['loss'] = loss.item()
            epoch_stats.append(stats)
            
        # Aggregate statistics
        mean_stats = {
            k: np.mean([s[k] for s in epoch_stats])
            for k in epoch_stats[0].keys()
        }
        
        # Add memory stats
        mean_stats.update(self.memory_manager.get_memory_stats())
        
        return mean_stats
        
    def validate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.policy.eval()
        val_stats = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Process batch with CUDA optimizations
                outputs, loss = self.batch_processor.process_batch(
                    model=self.policy,
                    batch=batch,
                    optimizer=self.optimizer,
                    training=False
                )
                
                # Record statistics
                stats = {k: v.item() if torch.is_tensor(v) else v 
                        for k, v in outputs.items()}
                stats['loss'] = loss.item()
                val_stats.append(stats)
                
        # Aggregate statistics
        mean_stats = {
            k: np.mean([s[k] for s in val_stats])
            for k in val_stats[0].keys()
        }
        
        return mean_stats
        
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_stats = self.train_epoch()
            
            # Validation
            val_stats = self.validate()
            
            # Logging
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_stats.items()},
                    **{f'val/{k}': v for k, v in val_stats.items()}
                })
                
            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_{epoch}.pt')
                
            # Save best model
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                self.save_checkpoint('best_model.pt')
                
            # Print progress
            print(f'Epoch {epoch}:')
            print(f'Train loss: {train_stats["loss"]:.4f}')
            print(f'Val loss: {val_stats["loss"]:.4f}')
            print(f'GPU Memory: {train_stats["allocated"]:.1f}MB allocated, '
                  f'{train_stats["cached"]:.1f}MB cached')
            
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] 