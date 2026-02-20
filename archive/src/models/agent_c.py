"""
Agent C: The autonomous agent that learns to act based on value function.

Agent C maintains internal state and selects actions to maximize
expected future intrinsic rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AgentC(nn.Module):
    """
    Agent C: Recurrent State-Space Model (RSSM) based agent.
    
    Maintains internal state (h, z) and outputs actions/filters.
    """
    
    def __init__(
        self,
        observation_dim: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        action_dim: int = 8
    ):
        """
        Args:
            observation_dim: Dimension of observations (from F)
            hidden_dim: Hidden state dimension
            latent_dim: Latent state dimension
            action_dim: Action dimension
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Recurrent state update (h)
        self.gru = nn.GRUCell(observation_dim + latent_dim, hidden_dim)
        
        # Latent state encoder (z | h, observation)
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim + observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # mean and log_std
        )
        
        # Policy network (action | h, z)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize agent state.
        
        Args:
            batch_size: Batch size
            device: Device
        
        Returns:
            state: Dictionary containing h and z
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return {'h': h, 'z': z}
    
    def forward(
        self,
        observation: torch.Tensor,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            observation: Current observation (B, observation_dim)
            state: Current agent state (h, z)
        
        Returns:
            action: Selected action (B, action_dim)
            next_state: Updated agent state
        """
        h = state['h']
        z = state['z']
        
        # Update recurrent state
        h_next = self.gru(torch.cat([observation, z], dim=-1), h)
        
        # Encode latent state
        latent_params = self.latent_encoder(torch.cat([h_next, observation], dim=-1))
        z_mean, z_log_std = torch.chunk(latent_params, 2, dim=-1)
        z_std = torch.exp(z_log_std)
        
        # Sample latent state (reparameterization trick)
        if self.training:
            eps = torch.randn_like(z_mean)
            z_next = z_mean + z_std * eps
        else:
            z_next = z_mean
        
        # Generate action
        action = self.policy(torch.cat([h_next, z_next], dim=-1))
        
        # Update state
        next_state = {'h': h_next, 'z': z_next}
        
        return action, next_state
