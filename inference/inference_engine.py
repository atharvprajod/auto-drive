import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    """Base configuration for inference engines"""
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "fp32"  # ["fp32", "fp16", "int8"]
    enable_profiling: bool = False
    warmup_iterations: int = 5
    timeout_ms: Optional[int] = None
    max_workspace_size: Optional[int] = None  # In MB

class InferenceEngine(ABC):
    """Base class for all inference engines"""
    
    def __init__(self, model: nn.Module, config: InferenceConfig):
        """
        Initialize inference engine
        
        Args:
            model: PyTorch model
            config: Inference configuration
        """
        self.model = model
        self.config = config
        self.is_optimized = False
        
    @abstractmethod
    def optimize(self) -> None:
        """Optimize model for inference"""
        pass
        
    @abstractmethod
    def infer(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Any:
        """
        Run inference on inputs
        
        Args:
            inputs: Model inputs
            
        Returns:
            Model outputs
        """
        pass
        
    def warmup(self, sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
        """
        Warm up the inference engine
        
        Args:
            sample_input: Sample input for warmup
        """
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.infer(sample_input)
                
    def benchmark(self, 
                 sample_input: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                 num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance
        
        Args:
            sample_input: Sample input for benchmarking
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        if not self.is_optimized:
            raise RuntimeError("Model must be optimized before benchmarking")
            
        # Warmup
        self.warmup(sample_input)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_event.record()
                _ = self.infer(sample_input)
                end_event.record()
                
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
                
        import numpy as np
        latencies = np.array(latencies)
        
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "throughput_ips": float(1000 / np.mean(latencies) * self.config.batch_size)
        } 