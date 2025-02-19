import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class CurriculumStage:
    """Defines a stage in the curriculum learning process"""
    name: str
    difficulty: float  # 0.0 to 1.0
    min_epochs: int
    performance_threshold: float
    data_sampler: Callable
    augmentation_params: Dict[str, Any]

class CurriculumScheduler:
    def __init__(self, 
                 stages: List[CurriculumStage],
                 evaluation_metric: Callable,
                 smoothing_factor: float = 0.95):
        """
        Initialize curriculum learning scheduler
        
        Args:
            stages: List of curriculum stages in order of increasing difficulty
            evaluation_metric: Function that computes performance metric
            smoothing_factor: EMA smoothing factor for performance tracking
        """
        self.stages = stages
        self.evaluation_metric = evaluation_metric
        self.smoothing_factor = smoothing_factor
        
        self.current_stage_idx = 0
        self.epochs_in_stage = 0
        self.smoothed_performance = 0.0
        
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get current training parameters based on curriculum stage"""
        stage = self.current_stage
        return {
            'difficulty': stage.difficulty,
            'data_sampler': stage.data_sampler,
            **stage.augmentation_params
        }
    
    def step(self, performance: float) -> bool:
        """
        Update curriculum state based on current performance
        
        Args:
            performance: Current performance metric
            
        Returns:
            Boolean indicating whether curriculum stage has changed
        """
        # Update smoothed performance
        if self.epochs_in_stage == 0:
            self.smoothed_performance = performance
        else:
            self.smoothed_performance = (self.smoothing_factor * self.smoothed_performance + 
                                       (1 - self.smoothing_factor) * performance)
        
        self.epochs_in_stage += 1
        stage = self.current_stage
        
        # Check if ready to advance to next stage
        if (self.epochs_in_stage >= stage.min_epochs and 
            self.smoothed_performance >= stage.performance_threshold):
            
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.epochs_in_stage = 0
                self.smoothed_performance = 0.0
                return True
                
        return False
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current curriculum progress statistics"""
        return {
            'current_stage': self.current_stage.name,
            'stage_progress': self.epochs_in_stage,
            'total_stages': len(self.stages),
            'current_stage_idx': self.current_stage_idx,
            'smoothed_performance': self.smoothed_performance
        }

class ScenarioDifficultyScheduler:
    """Scheduler for progressively increasing scenario difficulty"""
    
    def __init__(self,
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 growth_rate: float = 0.1):
        """
        Args:
            initial_difficulty: Starting difficulty level (0-1)
            max_difficulty: Maximum difficulty level
            growth_rate: Rate of difficulty increase
        """
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.growth_rate = growth_rate
        
    def get_scenario_params(self) -> Dict[str, float]:
        """Get current scenario generation parameters"""
        return {
            'num_obstacles': int(10 * self.current_difficulty),
            'min_obstacle_distance': 5.0 * (1 - self.current_difficulty),
            'road_curvature': 0.5 * self.current_difficulty,
            'weather_intensity': self.current_difficulty
        }
        
    def step(self, success_rate: float):
        """
        Update difficulty based on agent performance
        
        Args:
            success_rate: Rate of successful scenario completions (0-1)
        """
        target_difficulty = self.current_difficulty
        
        if success_rate > 0.8:  # Too easy
            target_difficulty += self.growth_rate
        elif success_rate < 0.2:  # Too hard
            target_difficulty -= self.growth_rate
            
        self.current_difficulty = np.clip(target_difficulty, 0.0, self.max_difficulty) 