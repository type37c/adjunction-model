"""
Value Function

Estimates the expected cumulative intrinsic reward from a given state.
"""

import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    """
    Value Function: V(s) → Expected future intrinsic reward
    
    A simple MLP that maps agent state to a scalar value.
    """
    
    def __init__(
        self,
        state_dim: int = 96,  # hidden_dim + latent_dim from Agent C
        hidden_dim: int = 64
    ):
        """
        Args:
            state_dim: Dimension of agent state (h + z concatenated)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Agent state (B, state_dim)
        
        Returns:
            value: Estimated value (B, 1)
        """
        return self.network(state)
