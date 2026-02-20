"""
Adjunction Model: F ⊣ G with Agent C

This module integrates Functor F, Functor G, and Agent C into a unified model.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG
from src.models.agent_c import AgentC


class AdjunctionModel(nn.Module):
    """
    Adjunction Model: Integrates F, G, and Agent C.
    
    The model computes:
    1. F: Shape → Affordance
    2. G: Affordance → Shape (reconstruction)
    3. Coherence Signal (η): ||Shape - G(F(Shape))||²
    4. Agent C: Maintains state and selects actions
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        affordance_dim: int = 16,
        agent_hidden_dim: int = 64,
        agent_latent_dim: int = 32,
        agent_action_dim: int = 8
    ):
        """
        Args:
            input_dim: Input dimension (3 for xyz)
            hidden_dim: Hidden dimension for F/G
            affordance_dim: Affordance dimension
            agent_hidden_dim: Agent C hidden dimension
            agent_latent_dim: Agent C latent dimension
            agent_action_dim: Agent C action dimension
        """
        super().__init__()
        
        # Functor F: Shape → Affordance
        self.F = FunctorF(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            affordance_dim=affordance_dim
        )
        
        # Functor G: Affordance → Shape
        self.G = FunctorG(
            affordance_dim=affordance_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim
        )
        
        # Agent C
        self.agent_c = AgentC(
            observation_dim=affordance_dim,
            hidden_dim=agent_hidden_dim,
            latent_dim=agent_latent_dim,
            action_dim=agent_action_dim
        )
    
    def compute_coherence_signal(
        self,
        pos: torch.Tensor,
        reconstructed_pos: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence signal (η) per batch.
        
        η = ||Shape - G(F(Shape))||² (reconstruction error)
        
        Args:
            pos: Original positions (N, 3)
            reconstructed_pos: Reconstructed positions (N, 3)
            batch: Batch indices (N,)
        
        Returns:
            coherence_signal: η per batch (B, 1)
        """
        # Compute per-point reconstruction error
        point_errors = torch.sum((pos - reconstructed_pos) ** 2, dim=-1)  # (N,)
        
        # Average per batch
        batch_size = batch.max().item() + 1
        coherence_signal = torch.zeros(batch_size, 1, device=pos.device)
        
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() > 0:
                coherence_signal[b, 0] = point_errors[mask].mean()
        
        return coherence_signal
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        agent_state: Dict[str, torch.Tensor],
        coherence_signal_prev: torch.Tensor,
        coherence_spatial_prev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the adjunction model.
        
        Args:
            pos: Point positions (N, 3)
            batch: Batch indices (N,)
            agent_state: Agent C state
            coherence_signal_prev: Previous coherence signal (B, 1)
            coherence_spatial_prev: Previous spatial coherence (N,)
        
        Returns:
            results: Dictionary containing:
                - affordances: Predicted affordances (N, affordance_dim)
                - reconstructed_pos: Reconstructed positions (N, 3)
                - coherence_signal: Current coherence signal (B, 1)
                - coherence_spatial: Spatial coherence (N,)
                - agent_action: Agent action (B, action_dim)
                - agent_state: Updated agent state
        """
        # 1. F: Shape → Affordance
        affordances = self.F(pos)  # (N, affordance_dim)
        
        # 2. G: Affordance → Shape (reconstruction)
        reconstructed_pos = self.G(affordances)  # (N, 3)
        
        # 3. Compute coherence signal (η)
        coherence_signal = self.compute_coherence_signal(pos, reconstructed_pos, batch)  # (B, 1)
        
        # 4. Compute spatial coherence (per-point reconstruction error)
        coherence_spatial = torch.sum((pos - reconstructed_pos) ** 2, dim=-1)  # (N,)
        
        # 5. Agent C: Aggregate affordances per batch for observation
        batch_size = batch.max().item() + 1
        batch_affordances = torch.zeros(batch_size, affordances.size(-1), device=pos.device)
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() > 0:
                batch_affordances[b] = affordances[mask].mean(dim=0)
        
        # 6. Agent C forward pass
        agent_action, agent_state_next = self.agent_c(batch_affordances, agent_state)
        
        return {
            'affordances': affordances,
            'reconstructed_pos': reconstructed_pos,
            'coherence_signal': coherence_signal,
            'coherence_spatial': coherence_spatial,
            'agent_action': agent_action,
            'agent_state': agent_state_next
        }
