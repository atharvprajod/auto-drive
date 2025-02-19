import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import psutil
import gc

@dataclass
class MemoryConfig:
    """Configuration for memory analysis"""
    track_per_layer: bool = True
    track_activation_memory: bool = True
    track_peak_memory: bool = True
    track_cuda_cache: bool = True
    clear_cache_freq: int = 100

class MemoryTracker:
    def __init__(self):
        """Initialize memory tracker"""
        self.reset()
        
    def reset(self):
        """Reset memory statistics"""
        self.curr_memory = 0
        self.peak_memory = 0
        self.memory_timeline = []
        
    def update(self, memory_allocated: int):
        """Update memory statistics"""
        self.curr_memory = memory_allocated
        self.peak_memory = max(self.peak_memory, memory_allocated)
        self.memory_timeline.append(memory_allocated)
        
class MemoryAnalyzer:
    def __init__(self,
                 model: nn.Module,
                 config: MemoryConfig):
        """
        Initialize memory analyzer
        
        Args:
            model: Neural network model
            config: Memory analysis configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize trackers
        self.layer_memory = defaultdict(MemoryTracker)
        self.activation_memory = MemoryTracker()
        self.total_memory = MemoryTracker()
        
        if config.track_per_layer:
            self._register_hooks()
            
    def _register_hooks(self):
        """Register hooks for memory tracking"""
        def pre_hook(name):
            def hook(module, input):
                if torch.cuda.is_available():
                    memory = torch.cuda.memory_allocated()
                    self.layer_memory[name].update(memory)
            return hook
            
        def post_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    # Track activation memory
                    if self.config.track_activation_memory:
                        if isinstance(output, torch.Tensor):
                            activation_size = output.nelement() * output.element_size()
                            self.activation_memory.update(activation_size)
                        elif isinstance(output, (tuple, list)):
                            activation_size = sum(
                                t.nelement() * t.element_size()
                                for t in output if isinstance(t, torch.Tensor)
                            )
                            self.activation_memory.update(activation_size)
                            
                    # Track total memory
                    memory = torch.cuda.memory_allocated()
                    self.total_memory.update(memory)
                    
            return hook
            
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.RNN, nn.LSTM, nn.GRU)):
                module.register_forward_pre_hook(pre_hook(name))
                module.register_forward_hook(post_hook(name))
                
    def analyze_memory(self,
                      input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Analyze memory usage during forward pass
        
        Args:
            input_tensors: Dictionary of input tensors
            
        Returns:
            Dictionary of memory statistics
        """
        # Clear memory before analysis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Reset trackers
        self.activation_memory.reset()
        self.total_memory.reset()
        for tracker in self.layer_memory.values():
            tracker.reset()
            
        # Forward pass
        with torch.no_grad():
            _ = self.model(**input_tensors)
            
        # Collect statistics
        stats = {
            'total': {
                'current_mb': self.total_memory.curr_memory / 1024**2,
                'peak_mb': self.total_memory.peak_memory / 1024**2
            },
            'activation': {
                'current_mb': self.activation_memory.curr_memory / 1024**2,
                'peak_mb': self.activation_memory.peak_memory / 1024**2
            }
        }
        
        # Layer-wise statistics
        if self.config.track_per_layer:
            stats['layers'] = {
                name: {
                    'current_mb': tracker.curr_memory / 1024**2,
                    'peak_mb': tracker.peak_memory / 1024**2
                }
                for name, tracker in self.layer_memory.items()
            }
            
        # CUDA cache statistics
        if self.config.track_cuda_cache and torch.cuda.is_available():
            stats['cuda_cache'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
            }
            
        return stats
        
    def profile_memory_growth(self,
                            input_tensors: Dict[str, torch.Tensor],
                            num_iterations: int = 100) -> Dict[str, List[float]]:
        """
        Profile memory growth over multiple iterations
        
        Args:
            input_tensors: Dictionary of input tensors
            num_iterations: Number of iterations
            
        Returns:
            Dictionary of memory growth statistics
        """
        memory_growth = {
            'allocated_mb': [],
            'cached_mb': []
        }
        
        for i in range(num_iterations):
            # Forward pass
            with torch.no_grad():
                _ = self.model(**input_tensors)
                
            # Record memory
            if torch.cuda.is_available():
                memory_growth['allocated_mb'].append(
                    torch.cuda.memory_allocated() / 1024**2
                )
                memory_growth['cached_mb'].append(
                    torch.cuda.memory_reserved() / 1024**2
                )
                
            # Clear cache periodically
            if (i + 1) % self.config.clear_cache_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        return memory_growth
        
    def suggest_optimizations(self,
                            stats: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Suggest memory optimizations based on analysis
        
        Args:
            stats: Memory statistics
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check total memory usage
        total_memory = stats['total']['peak_mb']
        available_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**2
            if torch.cuda.is_available() else
            psutil.virtual_memory().total / 1024**2
        )
        
        if total_memory > 0.8 * available_memory:
            suggestions.append(
                "High memory usage detected. Consider reducing batch size or "
                "using gradient checkpointing."
            )
            
        # Check activation memory
        if stats['activation']['peak_mb'] > 0.5 * total_memory:
            suggestions.append(
                "Large activation memory detected. Consider using activation "
                "checkpointing or reducing intermediate feature sizes."
            )
            
        # Check layer memory
        if 'layers' in stats:
            max_layer_memory = max(
                layer['peak_mb'] for layer in stats['layers'].values()
            )
            if max_layer_memory > 0.3 * total_memory:
                suggestions.append(
                    "Some layers use excessive memory. Consider reducing layer "
                    "sizes or using more efficient architectures."
                )
                
        # Check CUDA cache
        if 'cuda_cache' in stats:
            if stats['cuda_cache']['cached_mb'] > 2 * stats['cuda_cache']['allocated_mb']:
                suggestions.append(
                    "Large CUDA cache detected. Consider using empty_cache() "
                    "more frequently or limiting cache size."
                )
                
        return suggestions
        
class MemoryOptimizer:
    def __init__(self,
                 model: nn.Module,
                 target_memory_mb: float = None):
        """
        Initialize memory optimizer
        
        Args:
            model: Neural network model
            target_memory_mb: Target memory usage in MB
        """
        self.model = model
        self.target_memory_mb = target_memory_mb
        
    def optimize_batch_size(self,
                          sample_input: Dict[str, torch.Tensor],
                          min_batch_size: int = 1,
                          max_batch_size: int = 128) -> Tuple[int, float]:
        """
        Find optimal batch size for memory constraints
        
        Args:
            sample_input: Sample input tensors
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            
        Returns:
            Optimal batch size and memory usage
        """
        analyzer = MemoryAnalyzer(self.model, MemoryConfig())
        
        left = min_batch_size
        right = max_batch_size
        best_batch_size = left
        best_memory = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test batch size
            batch_input = {
                k: torch.cat([v] * (mid // v.shape[0] + 1), dim=0)[:mid]
                for k, v in sample_input.items()
            }
            
            # Analyze memory
            stats = analyzer.analyze_memory(batch_input)
            memory_usage = stats['total']['peak_mb']
            
            # Update search
            if self.target_memory_mb is None or memory_usage <= self.target_memory_mb:
                if mid > best_batch_size:
                    best_batch_size = mid
                    best_memory = memory_usage
                left = mid + 1
            else:
                right = mid - 1
                
        return best_batch_size, best_memory 