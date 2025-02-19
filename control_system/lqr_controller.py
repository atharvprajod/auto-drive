import torch
import numpy as np
from typing import Tuple, Dict

class LQRController:
    def __init__(self, 
                 state_dim: int,
                 control_dim: int,
                 horizon: int,
                 device: torch.device = None):
        """
        Initialize LQR controller with GPU acceleration
        
        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            horizon: Time horizon for optimization
            device: Torch device (GPU/CPU)
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize cost matrices
        self.Q = torch.eye(state_dim, device=self.device)  # State cost
        self.R = torch.eye(control_dim, device=self.device)  # Control cost
        self.Qf = torch.eye(state_dim, device=self.device)  # Terminal state cost
        
    def set_cost_matrices(self, Q: torch.Tensor, R: torch.Tensor, Qf: torch.Tensor = None):
        """Set cost matrices for LQR"""
        self.Q = Q.to(self.device)
        self.R = R.to(self.device)
        self.Qf = Qf.to(self.device) if Qf is not None else Q.to(self.device)
        
    def linearize_dynamics(self, 
                          state: torch.Tensor, 
                          control: torch.Tensor,
                          dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linearize system dynamics around operating point
        
        Args:
            state: Current state [batch_size, state_dim]
            control: Current control [batch_size, control_dim]
            dt: Time step
            
        Returns:
            A: State transition matrix [batch_size, state_dim, state_dim]
            B: Control matrix [batch_size, state_dim, control_dim]
        """
        batch_size = state.shape[0]
        
        # Example linear bicycle model
        # State: [x, y, heading, velocity]
        # Control: [steering_angle, acceleration]
        
        v = state[:, 3:4]  # Velocity
        theta = state[:, 2:3]  # Heading
        
        # State transition matrix
        A = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        A[:, 0, 0] = 1.0  # x
        A[:, 1, 1] = 1.0  # y
        A[:, 2, 2] = 1.0  # heading
        A[:, 3, 3] = 1.0  # velocity
        A[:, 0, 2] = -v * torch.sin(theta) * dt
        A[:, 0, 3] = torch.cos(theta) * dt
        A[:, 1, 2] = v * torch.cos(theta) * dt
        A[:, 1, 3] = torch.sin(theta) * dt
        
        # Control matrix
        B = torch.zeros(batch_size, self.state_dim, self.control_dim, device=self.device)
        B[:, 2, 0] = v * dt  # steering affects heading
        B[:, 3, 1] = dt      # acceleration affects velocity
        
        return A, B
        
    def solve(self, 
             initial_state: torch.Tensor,
             target_state: torch.Tensor,
             dt: float) -> Dict[str, torch.Tensor]:
        """
        Solve LQR problem for optimal control sequence
        
        Args:
            initial_state: Initial state [batch_size, state_dim]
            target_state: Target state [batch_size, state_dim]
            dt: Time step
            
        Returns:
            Dictionary containing optimal control sequence and predicted trajectory
        """
        batch_size = initial_state.shape[0]
        
        # Initialize trajectory
        states = torch.zeros(batch_size, self.horizon + 1, self.state_dim, device=self.device)
        controls = torch.zeros(batch_size, self.horizon, self.control_dim, device=self.device)
        states[:, 0] = initial_state
        
        # Initialize cost-to-go matrices
        P = [torch.zeros_like(self.Q) for _ in range(self.horizon + 1)]
        P[-1] = self.Qf
        
        K = torch.zeros(self.horizon, batch_size, self.control_dim, self.state_dim, device=self.device)
        k = torch.zeros(self.horizon, batch_size, self.control_dim, device=self.device)
        
        # Backward pass
        for t in reversed(range(self.horizon)):
            state = states[:, t]
            control = controls[:, t]
            
            # Linearize dynamics
            A, B = self.linearize_dynamics(state, control, dt)
            
            # Compute optimal feedback gain
            Qxx = self.Q
            Quu = self.R
            Qux = torch.zeros(batch_size, self.control_dim, self.state_dim, device=self.device)
            
            # Ricatti equation
            P[t] = Qxx + A.transpose(1, 2) @ P[t + 1] @ A - \
                   (Qux + B.transpose(1, 2) @ P[t + 1] @ A).transpose(1, 2) @ \
                   torch.inverse(Quu + B.transpose(1, 2) @ P[t + 1] @ B) @ \
                   (Qux + B.transpose(1, 2) @ P[t + 1] @ A)
            
            # Compute feedback gains
            K[t] = -torch.inverse(Quu + B.transpose(1, 2) @ P[t + 1] @ B) @ \
                   (Qux + B.transpose(1, 2) @ P[t + 1] @ A)
            k[t] = -torch.inverse(Quu + B.transpose(1, 2) @ P[t + 1] @ B) @ \
                   (B.transpose(1, 2) @ P[t + 1] @ (target_state - state))
        
        # Forward pass
        for t in range(self.horizon):
            state = states[:, t]
            
            # Compute optimal control
            controls[:, t] = (K[t] @ (state - target_state).unsqueeze(-1)).squeeze(-1) + k[t]
            
            # Simulate dynamics
            A, B = self.linearize_dynamics(state, controls[:, t], dt)
            states[:, t + 1] = (A @ state.unsqueeze(-1)).squeeze(-1) + \
                              (B @ controls[:, t].unsqueeze(-1)).squeeze(-1)
        
        return {
            'optimal_controls': controls,
            'predicted_states': states,
            'feedback_gains': K
        } 