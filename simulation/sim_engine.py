import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import os
import sys
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import time

@dataclass
class SimulationConfig:
    """Base configuration for simulation engines"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    physics_timestep: float = 0.01  # Physics simulation timestep in seconds
    render_fps: int = 60  # Target rendering framerate
    enable_gpu_physics: bool = True
    enable_gpu_rendering: bool = True
    max_substeps: int = 10  # Maximum physics substeps per frame
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    enable_profiling: bool = False
    profile_physics: bool = True
    profile_rendering: bool = True
    cache_dir: str = "sim_cache"
    
class SimulationEngine(ABC):
    """Base class for all simulation engines"""
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation engine
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.is_initialized = False
        self.physics_engine = None
        self.render_engine = None
        self.profiling_stats = {}
        self.time_stats = {
            'physics': [],
            'rendering': [],
            'total': []
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize simulation engines and resources"""
        pass
        
    @abstractmethod
    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Step the simulation forward
        
        Args:
            action: Control action to apply
            
        Returns:
            Dictionary containing simulation state and observations
        """
        pass
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset simulation to initial state
        
        Returns:
            Dictionary containing initial simulation state
        """
        pass
        
    @abstractmethod
    def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, None]:
        """
        Render the current simulation state
        
        Args:
            mode: Rendering mode ('rgb_array', 'human', etc.)
            
        Returns:
            Rendered frame if mode is 'rgb_array', None otherwise
        """
        pass
        
    def close(self) -> None:
        """Clean up simulation resources"""
        if self.physics_engine is not None:
            self.physics_engine.close()
        if self.render_engine is not None:
            self.render_engine.close()
            
    def get_state(self) -> Dict[str, Any]:
        """
        Get current simulation state
        
        Returns:
            Dictionary containing simulation state
        """
        raise NotImplementedError
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set simulation state
        
        Args:
            state: Dictionary containing simulation state
        """
        raise NotImplementedError
        
    def start_profiling(self) -> None:
        """Start collecting profiling statistics"""
        self.config.enable_profiling = True
        self.profiling_stats.clear()
        self.time_stats = {
            'physics': [],
            'rendering': [],
            'total': []
        }
        
    def stop_profiling(self) -> Dict[str, Any]:
        """
        Stop profiling and return statistics
        
        Returns:
            Dictionary containing profiling statistics
        """
        self.config.enable_profiling = False
        
        # Process timing statistics
        stats = {}
        for key, times in self.time_stats.items():
            if times:
                stats[f'{key}_mean_ms'] = float(np.mean(times))
                stats[f'{key}_std_ms'] = float(np.std(times))
                stats[f'{key}_min_ms'] = float(np.min(times))
                stats[f'{key}_max_ms'] = float(np.max(times))
                stats[f'{key}_p95_ms'] = float(np.percentile(times, 95))
                
        return {**stats, **self.profiling_stats}
        
    def _profile_section(self, section: str) -> float:
        """
        Profile a section of code
        
        Args:
            section: Name of the section being profiled
            
        Returns:
            Elapsed time in milliseconds
        """
        if not self.config.enable_profiling:
            return 0.0
            
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            yield
            
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            start_time = time.perf_counter()
            yield
            elapsed_time = (time.perf_counter() - start_time) * 1000
            
        self.time_stats[section].append(elapsed_time)
        return elapsed_time 