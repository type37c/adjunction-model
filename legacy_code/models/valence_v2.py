"""
Valence Memory V2 - Intrinsic Reward-based Valence Update

This module implements the valence update mechanism using intrinsic rewards.
Instead of directly mapping coherence changes to valence, it uses three
types of intrinsic motivation to drive valence updates.

Key change from V1:
- V1: valence = f(delta_coherence, attention_weight)
- V2: valence = f(R_intrinsic) where R_intrinsic = alpha*R_curiosity + beta*R_competence + gamma*R_novelty

V2 also introduces confidence as a meta-indicator of P:
- Confidence = 1 - normalized_entropy(priority_distribution)
- Curiosity = confidence(t) - confidence(t-1)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .intrinsic_reward import IntrinsicRewardComputation


class ValenceMemoryV2(nn.Module):
    """
    Valence Memory with intrinsic reward-based updates.
    
    Uses IntrinsicRewardComputation to compute rewards, then projects
    the total intrinsic reward into a valence update.
    """
    
    def __init__(
        self,
        valence_dim: int = 32,
        decay_rate: float = 0.1,
        alpha_curiosity: float = 0.3,
        beta_competence: float = 0.5,
        gamma_novelty: float = 0.2,
        novelty_scale: float = 0.1,
        competence_scale: float = 100.0,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.valence_dim = valence_dim
        self.decay_rate = decay_rate
        
        # Intrinsic reward computation
        self.intrinsic_reward = IntrinsicRewardComputation(
            alpha_curiosity=alpha_curiosity,
            beta_competence=beta_competence,
            gamma_novelty=gamma_novelty,
            novelty_scale=novelty_scale,
            competence_scale=competence_scale
        )
        
        # Project intrinsic reward to valence space
        self.reward_to_valence = nn.Sequential(
            nn.Linear(1, valence_dim),
            nn.Tanh()
        ).to(device)
        
        # Initialize with small weights for stability
        for m in self.reward_to_valence.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
        # Store initial valence template
        self.register_buffer(
            '_initial_valence_template',
            torch.zeros(1, valence_dim, dtype=torch.float32)
        )
    
    def get_initial_valence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get initial valence vector for a batch."""
        return torch.zeros(
            batch_size, self.valence_dim,
            device=device,
            dtype=torch.float32
        )
    
    def forward(
        self,
        valence_prev: torch.Tensor,
        uncertainty_prev: torch.Tensor,
        uncertainty_curr: torch.Tensor,
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor,
        attention_weight: torch.Tensor,
        kl_divergence: torch.Tensor,
        confidence_prev: Optional[torch.Tensor] = None,
        confidence_curr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update valence based on intrinsic rewards.
        
        Args:
            valence_prev: Previous valence vector (B, valence_dim)
            uncertainty_prev: Previous uncertainty (B,)
            uncertainty_curr: Current uncertainty (B,)
            coherence_prev: Previous coherence (B, 1)
            coherence_curr: Current coherence (B, 1)
            attention_weight: Attention weight (B, 1)
            kl_divergence: KL(posterior || prior) (B,)
            confidence_prev: Previous confidence (B,) - from prev step's priority distribution
            confidence_curr: Current confidence (B,) - from current step's priority distribution
        
        Returns:
            results: Dictionary with updated valence and reward components
        """
        # Compute intrinsic rewards
        rewards = self.intrinsic_reward(
            uncertainty_prev,
            uncertainty_curr,
            coherence_prev,
            coherence_curr,
            attention_weight,
            kl_divergence,
            confidence_prev,
            confidence_curr
        )
        
        R_intrinsic = rewards['R_intrinsic']  # (B,)
        
        # Project intrinsic reward to valence update
        valence_update = self.reward_to_valence(R_intrinsic.unsqueeze(-1))  # (B, valence_dim)
        
        # Update valence with exponential decay
        valence_new = (1 - self.decay_rate) * valence_prev + self.decay_rate * valence_update
        
        return {
            'valence': valence_new,
            'R_intrinsic': R_intrinsic,
            'R_curiosity': rewards['R_curiosity'],
            'R_competence': rewards['R_competence'],
            'R_novelty': rewards['R_novelty'],
            'confidence': rewards['confidence_curr']
        }
