import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class VehicleDynamics(nn.Module):
    def __init__(self, config: Dict = None):
        """
        Initialize vehicle dynamics model
        
        Args:
            config: Dictionary of vehicle parameters
        """
        super().__init__()
        self.config = config or {}
        
        # Vehicle parameters
        self.mass = self.config.get('mass', 1500.0)  # kg
        self.wheelbase = self.config.get('wheelbase', 2.7)  # meters
        self.max_steering = self.config.get('max_steering', 0.5)  # radians
        self.max_acceleration = self.config.get('max_acceleration', 3.0)  # m/s^2
        
        # State dimensions
        self.state_dim = 6  # [x, y, heading, velocity, steering_angle, acceleration]
        self.control_dim = 2  # [steering_rate, acceleration_rate]
        
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Forward dynamics model
        
        Args:
            state: Current state [batch_size, state_dim]
            control: Control input [batch_size, control_dim]
            
        Returns:
            Next state [batch_size, state_dim]
        """
        # Extract states
        x = state[:, 0]
        y = state[:, 1]
        theta = state[:, 2]
        v = state[:, 3]
        delta = state[:, 4]
        a = state[:, 5]
        
        # Extract controls
        delta_dot = control[:, 0]  # steering rate
        jerk = control[:, 1]  # acceleration rate
        
        # Compute state derivatives
        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = v * torch.tan(delta) / self.wheelbase
        v_dot = a
        delta_dot = torch.clamp(delta_dot, -self.max_steering, self.max_steering)
        a_dot = torch.clamp(jerk, -self.max_acceleration, self.max_acceleration)
        
        # Integrate using Euler method
        dt = 0.1  # time step
        x_next = x + x_dot * dt
        y_next = y + y_dot * dt
        theta_next = theta + theta_dot * dt
        v_next = v + v_dot * dt
        delta_next = delta + delta_dot * dt
        a_next = a + a_dot * dt
        
        # Stack next state
        next_state = torch.stack([x_next, y_next, theta_next, v_next, delta_next, a_next], dim=1)
        
        return next_state
    
    def get_jacobians(self, state: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Jacobian matrices of dynamics
        
        Args:
            state: Current state [batch_size, state_dim]
            control: Control input [batch_size, control_dim]
            
        Returns:
            A: State Jacobian [batch_size, state_dim, state_dim]
            B: Control Jacobian [batch_size, state_dim, control_dim]
        """
        batch_size = state.shape[0]
        
        # Create Jacobian matrices
        A = torch.zeros(batch_size, self.state_dim, self.state_dim, device=state.device)
        B = torch.zeros(batch_size, self.state_dim, self.control_dim, device=state.device)
        
        # Extract states
        theta = state[:, 2]
        v = state[:, 3]
        delta = state[:, 4]
        
        # Fill state Jacobian A
        A[:, 0, 2] = -v * torch.sin(theta)  # dx/dtheta
        A[:, 0, 3] = torch.cos(theta)       # dx/dv
        A[:, 1, 2] = v * torch.cos(theta)   # dy/dtheta
        A[:, 1, 3] = torch.sin(theta)       # dy/dv
        A[:, 2, 3] = torch.tan(delta) / self.wheelbase  # dtheta/dv
        A[:, 2, 4] = v / (self.wheelbase * torch.cos(delta)**2)  # dtheta/ddelta
        A[:, 3, 5] = 1.0  # dv/da
        
        # Fill control Jacobian B
        B[:, 4, 0] = 1.0  # ddelta/ddelta_dot
        B[:, 5, 1] = 1.0  # da/djerk
        
        return A, B
    
class BicycleModel(VehicleDynamics):
    """Kinematic bicycle model for vehicle dynamics"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        # Additional bicycle model parameters
        self.cf = self.config.get('cornering_stiffness_front', 80000.0)  # N/rad
        self.cr = self.config.get('cornering_stiffness_rear', 80000.0)   # N/rad
        self.lf = self.config.get('distance_cog_front', 1.3)  # meters
        self.lr = self.config.get('distance_cog_rear', 1.4)   # meters
        self.iz = self.config.get('yaw_inertia', 2500.0)  # kg*m^2
        
    def forward(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """
        Forward dynamics using bicycle model
        
        Args:
            state: Current state [batch_size, state_dim]
            control: Control input [batch_size, control_dim]
            
        Returns:
            Next state [batch_size, state_dim]
        """
        # Extract states
        x, y, theta, v, delta, a = torch.split(state, 1, dim=1)
        delta_dot, jerk = torch.split(control, 1, dim=1)
        
        # Slip angles
        alpha_f = delta - torch.atan2(self.lf * theta, v)
        alpha_r = -torch.atan2(self.lr * theta, v)
        
        # Lateral forces
        Fyf = self.cf * alpha_f
        Fyr = self.cr * alpha_r
        
        # State derivatives
        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = v * (Fyf * torch.cos(delta) + Fyr) / (self.mass * v**2)
        v_dot = a
        delta_dot = torch.clamp(delta_dot, -self.max_steering, self.max_steering)
        a_dot = torch.clamp(jerk, -self.max_acceleration, self.max_acceleration)
        
        # Integrate
        dt = 0.1
        next_state = state + dt * torch.cat([x_dot, y_dot, theta_dot, v_dot, delta_dot, a_dot], dim=1)
        
        return next_state 