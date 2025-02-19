import torch
import torch.nn as nn
import torch.cuda.amp as amp
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BatchProcessorConfig:
    """Configuration for batch processing"""
    use_mixed_precision: bool = True
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    cuda_cache_clear_freq: int = 100  # Clear CUDA cache every N batches

class CUDABatchProcessor:
    def __init__(self, config: BatchProcessorConfig):
        """
        Initialize CUDA-optimized batch processor
        
        Args:
            config: Batch processor configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize mixed precision training
        self.scaler = amp.GradScaler(enabled=config.use_mixed_precision)
        self.batch_count = 0
        
    @torch.cuda.amp.autocast()
    def process_batch(self,
                     model: nn.Module,
                     batch: Dict[str, torch.Tensor],
                     optimizer: torch.optim.Optimizer,
                     training: bool = True) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Process a single batch with CUDA optimizations
        
        Args:
            model: Neural network model
            batch: Dictionary of input tensors
            optimizer: Model optimizer
            training: Whether in training mode
            
        Returns:
            Dictionary of model outputs and loss value
        """
        # Move batch to device and convert to float16 for mixed precision
        batch = {k: v.to(self.device, non_blocking=True) 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # Compute model outputs and loss
            outputs = model(**batch)
            loss = outputs['total_loss'] if isinstance(outputs, dict) else outputs
            
            # Scale loss for gradient accumulation
            if self.config.accumulate_grad_batches > 1:
                loss = loss / self.config.accumulate_grad_batches
        
        # Backward pass during training
        if training:
            # Scale gradients for mixed precision
            self.scaler.scale(loss).backward()
            
            # Accumulate gradients
            if (self.batch_count + 1) % self.config.accumulate_grad_batches == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(optimizer)
                
                # Clip gradients
                if self.config.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Update weights
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            self.batch_count += 1
            
            # Clear CUDA cache periodically
            if self.batch_count % self.config.cuda_cache_clear_freq == 0:
                torch.cuda.empty_cache()
        
        return outputs, loss
    
    def optimize_memory(self, model: nn.Module):
        """Apply memory optimizations to model"""
        # Enable CUDA memory optimizations
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        # Use channels last memory format for CNN layers
        model = model.to(memory_format=torch.channels_last)
        
        return model

class AsyncBatchLoader:
    def __init__(self,
                 dataloader: torch.utils.data.DataLoader,
                 num_workers: int = 4,
                 prefetch_factor: int = 2):
        """
        Initialize asynchronous batch loader
        
        Args:
            dataloader: PyTorch dataloader
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch per worker
        """
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        
        # Configure dataloader for asynchronous loading
        self.dataloader.num_workers = num_workers
        self.dataloader.prefetch_factor = prefetch_factor
        self.dataloader.pin_memory = True
        
    def preload_next(self):
        """Preload next batch asynchronously"""
        try:
            self.next_batch = next(self.dataloader_iter)
        except (StopIteration, AttributeError):
            self.dataloader_iter = iter(self.dataloader)
            self.next_batch = next(self.dataloader_iter)
            
        # Preload to GPU asynchronously
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.cuda(non_blocking=True)
                for k, v in self.next_batch.items()
            }
            
    def get_next_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch, ensuring it's ready on GPU"""
        # Wait for current stream to finish
        torch.cuda.current_stream().wait_stream(self.stream)
        
        # Get current batch
        batch = self.next_batch
        
        # Start loading next batch
        self.preload_next()
        
        return batch

class CUDAMemoryManager:
    def __init__(self, 
                 target_memory_usage: float = 0.8,
                 check_interval: int = 100):
        """
        Initialize CUDA memory manager
        
        Args:
            target_memory_usage: Target GPU memory usage (0-1)
            check_interval: Check memory every N iterations
        """
        self.target_memory_usage = target_memory_usage
        self.check_interval = check_interval
        self.iteration = 0
        
    def check_memory(self):
        """Check and optimize CUDA memory usage"""
        self.iteration += 1
        
        if self.iteration % self.check_interval == 0:
            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            # Clear cache if usage is too high
            if current_memory > self.target_memory_usage:
                torch.cuda.empty_cache()
                
    def optimize_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Optimize tensor memory layout"""
        optimized = {}
        
        for key, tensor in tensors.items():
            if tensor.dim() == 4:  # For 4D tensors (images)
                optimized[key] = tensor.contiguous(memory_format=torch.channels_last)
            else:
                optimized[key] = tensor.contiguous()
                
        return optimized
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current CUDA memory statistics"""
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
        } 