import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

@dataclass
class SamplingConfig:
    """Configuration for scenario sampling"""
    num_bins: int = 50
    min_samples_per_bin: int = 10
    difficulty_weight: float = 2.0
    novelty_weight: float = 1.0
    kde_bandwidth: float = 0.1

class ScenarioSampler:
    def __init__(self, config: SamplingConfig):
        """
        Initialize scenario sampler
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        
        # Initialize density estimators
        self.state_kde = None
        self.action_kde = None
        
        # Initialize statistics
        self.difficulty_scores = []
        self.novelty_scores = []
        self.sampling_weights = None
        
    def update_density_estimation(self,
                                states: torch.Tensor,
                                actions: torch.Tensor):
        """
        Update density estimators with new data
        
        Args:
            states: [N, state_dim] state vectors
            actions: [N, action_dim] action vectors
        """
        # Fit KDE to states
        self.state_kde = KernelDensity(
            bandwidth=self.config.kde_bandwidth,
            kernel='gaussian'
        ).fit(states.numpy())
        
        # Fit KDE to actions
        self.action_kde = KernelDensity(
            bandwidth=self.config.kde_bandwidth,
            kernel='gaussian'
        ).fit(actions.numpy())
        
    def compute_scenario_scores(self,
                              states: torch.Tensor,
                              actions: torch.Tensor,
                              metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute difficulty and novelty scores for scenarios
        
        Args:
            states: [N, state_dim] state vectors
            actions: [N, action_dim] action vectors
            metrics: Dictionary of scenario metrics
            
        Returns:
            Dictionary of scenario scores
        """
        # Compute state-action density scores
        state_log_density = self.state_kde.score_samples(states.numpy())
        action_log_density = self.action_kde.score_samples(actions.numpy())
        
        # Compute novelty scores (negative log density)
        novelty_scores = -(state_log_density + action_log_density)
        
        # Compute difficulty scores
        difficulty_scores = self._compute_difficulty_scores(metrics)
        
        # Combine scores
        total_scores = (self.config.difficulty_weight * difficulty_scores +
                       self.config.novelty_weight * novelty_scores)
        
        return {
            'novelty_scores': torch.from_numpy(novelty_scores),
            'difficulty_scores': difficulty_scores,
            'total_scores': torch.from_numpy(total_scores)
        }
        
    def _compute_difficulty_scores(self, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute scenario difficulty scores from metrics"""
        # Example difficulty metrics:
        # - Minimum distance to obstacles
        # - Maximum required deceleration
        # - Path curvature
        # - Number of dynamic objects
        
        difficulty = torch.zeros(len(next(iter(metrics.values()))))
        
        if 'min_distance' in metrics:
            difficulty += torch.exp(-metrics['min_distance'])
            
        if 'max_decel' in metrics:
            difficulty += torch.abs(metrics['max_decel'])
            
        if 'curvature' in metrics:
            difficulty += torch.abs(metrics['curvature'])
            
        if 'num_objects' in metrics:
            difficulty += metrics['num_objects'] / 10.0
            
        return difficulty
        
    def update_sampling_weights(self, scores: Dict[str, torch.Tensor]):
        """Update scenario sampling weights based on scores"""
        total_scores = scores['total_scores']
        
        # Compute sampling weights using softmax
        self.sampling_weights = torch.softmax(total_scores, dim=0)
        
    def sample_scenarios(self,
                        scenarios: List[Dict],
                        num_samples: int) -> List[Dict]:
        """
        Sample scenarios based on weights
        
        Args:
            scenarios: List of scenario dictionaries
            num_samples: Number of scenarios to sample
            
        Returns:
            Sampled scenarios
        """
        if self.sampling_weights is None:
            return np.random.choice(scenarios, num_samples, replace=True)
            
        indices = torch.multinomial(
            self.sampling_weights,
            num_samples,
            replacement=True
        )
        
        return [scenarios[i] for i in indices]

class AdaptiveScenarioSampler(ScenarioSampler):
    def __init__(self, config: SamplingConfig):
        """
        Initialize adaptive scenario sampler
        
        Args:
            config: Sampling configuration
        """
        super().__init__(config)
        
        # Initialize difficulty bins
        self.difficulty_bins = np.linspace(0, 1, config.num_bins + 1)
        self.bin_counts = np.zeros(config.num_bins)
        self.bin_weights = np.ones(config.num_bins)
        
    def update_bin_weights(self, difficulty_scores: torch.Tensor):
        """Update sampling weights for difficulty bins"""
        # Compute bin assignments
        bin_indices = np.digitize(difficulty_scores, self.difficulty_bins) - 1
        
        # Update bin counts
        for bin_idx in range(self.config.num_bins):
            self.bin_counts[bin_idx] = np.sum(bin_indices == bin_idx)
            
        # Compute target distribution (uniform)
        target_count = max(
            self.config.min_samples_per_bin,
            np.mean(self.bin_counts)
        )
        
        # Update bin weights
        self.bin_weights = np.clip(
            target_count / (self.bin_counts + 1e-6),
            0.1,
            10.0
        )
        
    def sample_scenarios(self,
                        scenarios: List[Dict],
                        difficulty_scores: torch.Tensor,
                        num_samples: int) -> List[Dict]:
        """
        Sample scenarios using difficulty-based adaptive sampling
        
        Args:
            scenarios: List of scenario dictionaries
            difficulty_scores: Difficulty scores for scenarios
            num_samples: Number of scenarios to sample
            
        Returns:
            Sampled scenarios
        """
        # Update bin weights
        self.update_bin_weights(difficulty_scores)
        
        # Compute bin assignments
        bin_indices = np.digitize(difficulty_scores, self.difficulty_bins) - 1
        
        # Compute scenario weights
        scenario_weights = torch.from_numpy(self.bin_weights[bin_indices])
        
        if self.sampling_weights is not None:
            scenario_weights *= self.sampling_weights
            
        # Normalize weights
        scenario_weights = scenario_weights / torch.sum(scenario_weights)
        
        # Sample scenarios
        indices = torch.multinomial(
            scenario_weights,
            num_samples,
            replacement=True
        )
        
        return [scenarios[i] for i in indices] 