import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MPCConfig:
    """Configuration for MPC controller"""
    horizon: int = 20
    dt: float = 0.1
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    
    # State constraints
    max_velocity: float = 20.0  # m/s
    max_acceleration: float = 3.0  # m/s^2
    max_steering_angle: float = 0.5  # rad
    max_steering_rate: float = 0.3  # rad/s
    
    # Cost weights
    tracking_weight: float = 1.0
    control_weight: float = 0.1
    smoothness_weight: float = 0.01

class MPCController:
    def __init__(self, 
                 config: MPCConfig,
                 dynamics_model: torch.nn.Module,
                 device: Optional[torch.device] = None):
        """
        Initialize MPC controller
        
        Args:
            config: MPC configuration
            dynamics_model: Differentiable dynamics model
            device: Torch device (GPU/CPU)
        """
        self.config = config
        self.dynamics_model = dynamics_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dynamics_model.to(self.device)
        
    def optimize_trajectory(self,
                          initial_state: torch.Tensor,
                          reference_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimize control sequence using MPC
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            reference_trajectory: Reference trajectory [batch_size, horizon, state_dim]
            
        Returns:
            Dictionary containing optimal controls and predicted states
        """
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]
        control_dim = self.dynamics_model.control_dim
        
        # Initialize control sequence
        controls = torch.zeros(batch_size, self.config.horizon, control_dim,
                             device=self.device, requires_grad=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([controls], lr=0.01)
        
        best_cost = float('inf')
        best_controls = None
        best_states = None
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Forward simulate trajectory
            states = self.rollout_trajectory(initial_state, controls)
            
            # Compute costs
            tracking_cost = torch.mean((states - reference_trajectory) ** 2)
            control_cost = torch.mean(controls ** 2)
            smoothness_cost = torch.mean((controls[:, 1:] - controls[:, :-1]) ** 2)
            
            total_cost = (self.config.tracking_weight * tracking_cost +
                         self.config.control_weight * control_cost +
                         self.config.smoothness_weight * smoothness_cost)
            
            # Backward pass
            total_cost.backward()
            
            # Update controls
            optimizer.step()
            
            # Project controls to feasible set
            with torch.no_grad():
                controls.data = self.project_controls(controls)
            
            # Check convergence
            if total_cost.item() < best_cost:
                best_cost = total_cost.item()
                best_controls = controls.detach().clone()
                best_states = states.detach().clone()
                
            if iteration > 0 and abs(total_cost.item() - best_cost) < self.config.convergence_threshold:
                break
                
        return {
            'optimal_controls': best_controls,
            'predicted_states': best_states,
            'cost': best_cost
        }
        
    def rollout_trajectory(self,
                          initial_state: torch.Tensor,
                          controls: torch.Tensor) -> torch.Tensor:
        """
        Rollout trajectory using dynamics model
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            controls: Control sequence [batch_size, horizon, control_dim]
            
        Returns:
            Predicted states [batch_size, horizon + 1, state_dim]
        """
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[1]
        
        states = torch.zeros(batch_size, self.config.horizon + 1, state_dim,
                           device=self.device)
        states[:, 0] = initial_state
        
        for t in range(self.config.horizon):
            states[:, t + 1] = self.dynamics_model(states[:, t], controls[:, t])
            
        return states
    
    def project_controls(self, controls: torch.Tensor) -> torch.Tensor:
        """Project controls to satisfy constraints"""
        # Clip control magnitudes
        controls = torch.clamp(controls, -self.config.max_steering_angle, self.config.max_steering_angle)
        
        # Clip control rates
        control_rates = (controls[:, 1:] - controls[:, :-1]) / self.config.dt
        max_rates = torch.tensor([self.config.max_steering_rate, self.config.max_acceleration],
                               device=self.device)
        
        control_rates = torch.clamp(control_rates, -max_rates, max_rates)
        
        # Reconstruct controls from rates
        controls_new = torch.zeros_like(controls)
        controls_new[:, 0] = controls[:, 0]
        for t in range(1, self.config.horizon):
            controls_new[:, t] = controls_new[:, t-1] + control_rates[:, t-1] * self.config.dt
            
        return controls_new
    
    def get_control(self, 
                   current_state: torch.Tensor,
                   reference_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Get optimal control input for current state
        
        Args:
            current_state: Current state [batch_size, state_dim]
            reference_trajectory: Reference trajectory [batch_size, horizon, state_dim]
            
        Returns:
            Optimal control input [batch_size, control_dim]
        """
        # Optimize trajectory
        result = self.optimize_trajectory(current_state, reference_trajectory)
        
        # Return first control input
        return result['optimal_controls'][:, 0] 