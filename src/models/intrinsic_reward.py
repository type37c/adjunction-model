"""
Intrinsic Reward Module

Computes intrinsic rewards based on competence and novelty.
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import defaultdict


class IntrinsicRewardModule:
    """
    Computes intrinsic rewards for the agent.
    
    R_intrinsic = R_competence + R_novelty
    
    - R_competence: Reward for improving coherence (reducing η)
    - R_novelty: Reward for exploring new states
    """
    
    def __init__(
        self,
        competence_weight: float = 1.0,
        novelty_weight: float = 0.1,
        novelty_decay: float = 0.99
    ):
        """
        Args:
            competence_weight: Weight for competence reward
            novelty_weight: Weight for novelty reward
            novelty_decay: Decay factor for visitation counts
        """
        self.competence_weight = competence_weight
        self.novelty_weight = novelty_weight
        self.novelty_decay = novelty_decay
        
        # State visitation counts (for novelty)
        self.visitation_counts = defaultdict(int)
        self.total_steps = 0
    
    def compute_competence_reward(
        self,
        eta_prev: torch.Tensor,
        eta_curr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute competence reward.
        
        R_competence = η(t-1) - η(t)
        
        Positive when η decreases (coherence improves).
        
        Args:
            eta_prev: Previous coherence signal (B, 1)
            eta_curr: Current coherence signal (B, 1)
        
        Returns:
            reward: Competence reward (B, 1)
        """
        return eta_prev - eta_curr
    
    def compute_novelty_reward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute novelty reward.
        
        R_novelty = -log(visitation_count + 1)
        
        Higher reward for less visited states.
        
        Args:
            state: Agent state (B, state_dim)
        
        Returns:
            reward: Novelty reward (B, 1)
        """
        batch_size = state.size(0)
        rewards = torch.zeros(batch_size, 1, device=state.device)
        
        for i in range(batch_size):
            # Discretize state for counting (simple hash)
            state_hash = hash(tuple(state[i].detach().cpu().numpy().round(2)))
            
            # Get visitation count
            count = self.visitation_counts[state_hash]
            
            # Compute novelty reward
            rewards[i, 0] = -torch.log(torch.tensor(count + 1.0))
            
            # Update count
            self.visitation_counts[state_hash] += 1
        
        # Decay all counts periodically
        self.total_steps += 1
        if self.total_steps % 100 == 0:
            for key in self.visitation_counts:
                self.visitation_counts[key] = int(self.visitation_counts[key] * self.novelty_decay)
        
        return rewards
    
    def compute_intrinsic_reward(
        self,
        eta_prev: torch.Tensor,
        eta_curr: torch.Tensor,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total intrinsic reward.
        
        Args:
            eta_prev: Previous coherence signal (B, 1)
            eta_curr: Current coherence signal (B, 1)
            state: Agent state (B, state_dim)
        
        Returns:
            rewards: Dictionary containing:
                - total: Total intrinsic reward
                - competence: Competence component
                - novelty: Novelty component
        """
        # Compute components
        r_competence = self.compute_competence_reward(eta_prev, eta_curr)
        r_novelty = self.compute_novelty_reward(state)
        
        # Weighted sum
        r_total = (
            self.competence_weight * r_competence +
            self.novelty_weight * r_novelty
        )
        
        return {
            'total': r_total,
            'competence': r_competence,
            'novelty': r_novelty
        }
    
    def reset(self):
        """Reset visitation counts."""
        self.visitation_counts.clear()
        self.total_steps = 0
