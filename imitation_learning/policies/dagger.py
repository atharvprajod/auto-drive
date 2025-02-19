import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class DAggerConfig:
    """Configuration for DAgger algorithm"""
    state_dim: int
    action_dim: int
    buffer_size: int = 100000
    batch_size: int = 256
    num_epochs: int = 100
    beta_schedule: str = 'linear'  # ['linear', 'exponential']
    beta_start: float = 1.0
    beta_end: float = 0.0
    expert_policy: Optional[Callable] = None

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int):
        """
        Initialize replay buffer for DAgger
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim))
        self.expert_actions = np.zeros((buffer_size, action_dim))
        self.observations = {}  # Additional observation modalities
        
    def add(self,
            state: np.ndarray,
            expert_action: np.ndarray,
            **observations):
        """Add transition to buffer"""
        self.states[self.ptr] = state
        self.expert_actions[self.ptr] = expert_action
        
        # Store additional observations
        for key, value in observations.items():
            if key not in self.observations:
                self.observations[key] = np.zeros((self.buffer_size,) + value.shape)
            self.observations[key][self.ptr] = value
            
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Create batch
        observations = {
            'states': torch.FloatTensor(self.states[indices])
        }
        
        # Add additional observation modalities
        for key, value in self.observations.items():
            observations[key] = torch.FloatTensor(value[indices])
            
        expert_actions = torch.FloatTensor(self.expert_actions[indices])
        
        return observations, expert_actions

class DAgger:
    def __init__(self,
                 policy: nn.Module,
                 config: DAggerConfig):
        """
        Initialize DAgger algorithm
        
        Args:
            policy: Policy network
            config: DAgger configuration
        """
        self.policy = policy
        self.config = config
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            config.state_dim,
            config.action_dim,
            config.buffer_size
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(policy.parameters())
        
        # Beta schedule for mixing expert/policy
        if config.beta_schedule == 'linear':
            self.betas = np.linspace(
                config.beta_start,
                config.beta_end,
                config.num_epochs
            )
        else:  # exponential
            self.betas = np.exp(
                np.linspace(
                    np.log(config.beta_start),
                    np.log(config.beta_end),
                    config.num_epochs
                )
            )
            
    def collect_data(self,
                    env,
                    num_episodes: int,
                    epoch: int) -> List[Dict[str, float]]:
        """
        Collect data using mixed expert/policy actions
        
        Args:
            env: Environment
            num_episodes: Number of episodes to collect
            epoch: Current epoch (for beta scheduling)
            
        Returns:
            List of episode statistics
        """
        episode_stats = []
        beta = self.betas[epoch]
        
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                # Get policy action
                with torch.no_grad():
                    policy_action = self.policy(
                        torch.FloatTensor(obs['states']).unsqueeze(0)
                    ).squeeze(0).numpy()
                    
                # Get expert action
                expert_action = self.config.expert_policy(obs)
                
                # Mix expert and policy actions
                if np.random.random() < beta:
                    action = expert_action
                else:
                    action = policy_action
                    
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                # Store transition with expert action
                self.replay_buffer.add(
                    obs['states'],
                    expert_action,
                    **{k: v for k, v in obs.items() if k != 'states'}
                )
                
                obs = next_obs
                episode_return += reward
                
            episode_stats.append({
                'return': episode_return,
                'beta': beta
            })
            
        return episode_stats
        
    def update_policy(self, num_updates: int) -> Dict[str, float]:
        """
        Update policy using collected data
        
        Args:
            num_updates: Number of gradient updates
            
        Returns:
            Dictionary of training statistics
        """
        total_loss = 0
        
        for _ in range(num_updates):
            # Sample batch
            observations, expert_actions = self.replay_buffer.sample(
                self.config.batch_size
            )
            
            # Compute loss
            pred_actions = self.policy(**observations)
            loss = torch.mean((pred_actions - expert_actions)**2)
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {
            'loss': total_loss / num_updates
        }
        
    def train(self,
              env,
              num_episodes_per_epoch: int,
              num_updates_per_epoch: int) -> List[Dict[str, float]]:
        """
        Train policy using DAgger
        
        Args:
            env: Environment
            num_episodes_per_epoch: Number of episodes to collect per epoch
            num_updates_per_epoch: Number of gradient updates per epoch
            
        Returns:
            List of training statistics
        """
        training_stats = []
        
        for epoch in range(self.config.num_epochs):
            # Collect data
            episode_stats = self.collect_data(
                env,
                num_episodes_per_epoch,
                epoch
            )
            
            # Update policy
            update_stats = self.update_policy(num_updates_per_epoch)
            
            # Aggregate statistics
            stats = {
                'epoch': epoch,
                'mean_return': np.mean([s['return'] for s in episode_stats]),
                'beta': self.betas[epoch],
                **update_stats
            }
            
            training_stats.append(stats)
            
        return training_stats 