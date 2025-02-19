import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np
from contextlib import contextmanager
import torch.cuda.nvtx as nvtx

@dataclass
class ProfilingConfig:
    """Configuration for latency profiling"""
    num_warmup: int = 10
    num_iterations: int = 100
    use_cuda_events: bool = True
    profile_layers: bool = True
    trace_cuda: bool = False

@contextmanager
def cuda_timer():
    """Context manager for CUDA timing"""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        start = time.perf_counter()
        yield
        elapsed_time = (time.perf_counter() - start) * 1000  # Convert to ms
        
    return elapsed_time

class LatencyProfiler:
    def __init__(self,
                 model: nn.Module,
                 config: ProfilingConfig):
        """
        Initialize latency profiler
        
        Args:
            model: Neural network model
            config: Profiling configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize layer timings
        self.layer_times = {}
        if config.profile_layers:
            self._register_hooks()
            
    def _register_hooks(self):
        """Register forward hooks for layer timing"""
        def hook_fn(name):
            def hook(module, input, output):
                if name not in self.layer_times:
                    self.layer_times[name] = []
                    
                with cuda_timer() as elapsed:
                    pass  # Time is measured in context manager
                    
                self.layer_times[name].append(elapsed)
                
            return hook
            
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.RNN, nn.LSTM, nn.GRU)):
                module.register_forward_hook(hook_fn(name))
                
    def profile_forward(self,
                       input_tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Profile forward pass latency
        
        Args:
            input_tensors: Dictionary of input tensors
            
        Returns:
            Dictionary of timing statistics
        """
        latencies = []
        
        # Warmup runs
        for _ in range(self.config.num_warmup):
            with torch.no_grad():
                _ = self.model(**input_tensors)
                
        # Profile runs
        for _ in range(self.config.num_iterations):
            with torch.no_grad(), cuda_timer() as elapsed:
                _ = self.model(**input_tensors)
                
            latencies.append(elapsed)
            
        # Compute statistics
        latencies = np.array(latencies)
        stats = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p90_ms': np.percentile(latencies, 90),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99)
        }
        
        return stats
        
    def profile_layers(self) -> Dict[str, Dict[str, float]]:
        """Profile latency of individual layers"""
        if not self.config.profile_layers:
            return {}
            
        layer_stats = {}
        for name, times in self.layer_times.items():
            times = np.array(times)
            layer_stats[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'percentage': np.mean(times) / self.total_time * 100
            }
            
        return layer_stats
        
    def trace_cuda_events(self,
                         input_tensors: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """
        Trace CUDA events for detailed profiling
        
        Args:
            input_tensors: Dictionary of input tensors
            
        Returns:
            List of CUDA event timings
        """
        if not self.config.trace_cuda or not torch.cuda.is_available():
            return []
            
        events = []
        
        # Enable CUDA profiling
        torch.cuda.synchronize()
        
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            with torch.no_grad():
                _ = self.model(**input_tensors)
                
        # Extract event timings
        for evt in prof.key_averages():
            events.append({
                'name': evt.key,
                'cuda_time_ms': evt.cuda_time_total / 1000,
                'cpu_time_ms': evt.cpu_time_total / 1000,
                'count': evt.count
            })
            
        return events
        
    def benchmark_throughput(self,
                           input_tensors: Dict[str, torch.Tensor],
                           batch_sizes: List[int]) -> Dict[int, float]:
        """
        Benchmark throughput for different batch sizes
        
        Args:
            input_tensors: Dictionary of input tensors
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary of throughput (samples/sec) for each batch size
        """
        throughput = {}
        
        for batch_size in batch_sizes:
            # Resize input tensors
            batch_inputs = {
                k: torch.cat([v] * (batch_size // v.shape[0] + 1), dim=0)[:batch_size]
                for k, v in input_tensors.items()
            }
            
            # Profile forward pass
            stats = self.profile_forward(batch_inputs)
            
            # Compute throughput
            throughput[batch_size] = batch_size / (stats['mean_ms'] / 1000)  # samples/sec
            
        return throughput
        
    def export_chrome_trace(self, filename: str):
        """Export Chrome trace file for visualization"""
        if not self.config.trace_cuda or not torch.cuda.is_available():
            return
            
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            prof.export_chrome_trace(filename)
            
class BatchLatencyOptimizer:
    def __init__(self,
                 model: nn.Module,
                 min_batch_size: int = 1,
                 max_batch_size: int = 128,
                 target_latency: float = None):
        """
        Initialize batch size optimizer
        
        Args:
            model: Neural network model
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_latency: Target latency in milliseconds
        """
        self.model = model
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        
    def find_optimal_batch_size(self,
                              sample_input: Dict[str, torch.Tensor]) -> Tuple[int, float]:
        """
        Find optimal batch size using binary search
        
        Args:
            sample_input: Sample input tensor
            
        Returns:
            Optimal batch size and corresponding latency
        """
        left = self.min_batch_size
        right = self.max_batch_size
        best_batch_size = left
        best_latency = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test batch size
            batch_input = {
                k: torch.cat([v] * (mid // v.shape[0] + 1), dim=0)[:mid]
                for k, v in sample_input.items()
            }
            
            # Profile latency
            profiler = LatencyProfiler(self.model, ProfilingConfig())
            stats = profiler.profile_forward(batch_input)
            latency = stats['mean_ms']
            
            # Update best result
            if self.target_latency is None or latency <= self.target_latency:
                if mid > best_batch_size:
                    best_batch_size = mid
                    best_latency = latency
                left = mid + 1
            else:
                right = mid - 1
                
        return best_batch_size, best_latency 