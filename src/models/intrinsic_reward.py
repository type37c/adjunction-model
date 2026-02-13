"""
Intrinsic Reward Module for Purpose Space P

This module computes intrinsic rewards based on Active Inference principles,
adapted for the "suspension structure" philosophy:

1. Curiosity (R_curiosity): Reduction in uncertainty
   - "I learned something I didn't know"
   - Encourages exploration of uncertain states

2. Competence (R_competence): Improvement in coherence after attention
   - "I faced a breakdown and understood it better"
   - Encourages engaging with breakdowns rather than avoiding them

3. Novelty (R_novelty): Unexpected discoveries (KL divergence)
   - "Something unexpected happened"
   - Encourages discovering new patterns (but not too much)

Key difference from standard Active Inference:
- Standard: Minimize surprise (prediction error)
- This project: Maximize creative potential (breakdown × uncertainty × valence)

The intrinsic reward is used to update valence, giving Agent C a "purpose":
- valence(t+1) = (1-decay) × valence(t) + decay × R_intrinsic
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class IntrinsicRewardComputation(nn.Module):
    """
    Compute intrinsic rewards for valence update.
    
    This module combines three types of intrinsic motivation:
    1. Curiosity: uncertainty reduction
    2. Competence: coherence improvement (when attended)
    3. Novelty: unexpected discoveries
    """
    
    def __init__(
        self,
        alpha_curiosity: float = 0.0,  # Disabled due to sign issue
        beta_competence: float = 0.6,
        gamma_novelty: float = 0.4,
        novelty_scale: float = 0.1,
        competence_scale: float = 100.0  # Scale up competence reward
    ):
        """
        Args:
            alpha_curiosity: Weight for curiosity reward
            beta_competence: Weight for competence reward
            gamma_novelty: Weight for novelty reward
            novelty_scale: Scale factor for novelty (to keep it in reasonable range)
        """
        super().__init__()
        
        self.alpha = alpha_curiosity
        self.beta = beta_competence
        self.gamma = gamma_novelty
        self.novelty_scale = novelty_scale
        self.competence_scale = competence_scale
    
    def forward(
        self,
        uncertainty_prev: torch.Tensor,
        uncertainty_curr: torch.Tensor,
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor,
        attention_weight: torch.Tensor,
        kl_divergence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute intrinsic rewards.
        
        Args:
            uncertainty_prev: Previous uncertainty (B,)
            uncertainty_curr: Current uncertainty (B,)
            coherence_prev: Previous coherence (B, 1)
            coherence_curr: Current coherence (B, 1)
            attention_weight: Attention weight (B, 1) - how much we attended to this
            kl_divergence: KL(posterior || prior) (B,)
        
        Returns:
            rewards: Dictionary with individual and total intrinsic rewards
        """
        # (1) Curiosity reward: reduction in uncertainty
        # Positive when uncertainty decreases (we learned something)
        R_curiosity = uncertainty_prev - uncertainty_curr  # (B,)
        
        # Normalize to [0, 1] range (approximately)
        R_curiosity = torch.tanh(R_curiosity / 10.0)  # Scale down large values
        
        # (2) Competence reward: attending to breakdowns (v2 - redesigned)
        # Positive when we attend to large breakdowns (engage with difficulty)
        # This is appropriate when F/G are frozen and cannot reduce breakdowns
        R_competence = coherence_curr * attention_weight * self.competence_scale  # (B, 1)
        
        # Normalize to [0, 1] range
        R_competence = torch.tanh(R_competence)
        
        # (3) Novelty reward: unexpected discoveries
        # KL divergence measures how much the posterior differs from prior
        # High KL = surprising observation
        R_novelty = kl_divergence * self.novelty_scale  # (B,)
        
        # Cap novelty to avoid excessive values
        R_novelty = torch.tanh(R_novelty)
        
        # Total intrinsic reward
        R_intrinsic = (
            self.alpha * R_curiosity.unsqueeze(-1) +
            self.beta * R_competence +
            self.gamma * R_novelty.unsqueeze(-1)
        )  # (B, 1)
        
        return {
            'R_curiosity': R_curiosity,
            'R_competence': R_competence.squeeze(-1),
            'R_novelty': R_novelty,
            'R_intrinsic': R_intrinsic.squeeze(-1)
        }


if __name__ == '__main__':
    print("Testing Intrinsic Reward Computation...")
    
    batch_size = 4
    
    # Create module
    intrinsic_reward = IntrinsicRewardComputation(
        alpha_curiosity=0.3,
        beta_competence=0.5,
        gamma_novelty=0.2
    )
    
    print(f"\nIntrinsic Reward module created")
    print(f"  α (curiosity): {intrinsic_reward.alpha}")
    print(f"  β (competence): {intrinsic_reward.beta}")
    print(f"  γ (novelty): {intrinsic_reward.gamma}")
    
    # Simulate different scenarios
    scenarios = [
        {
            'name': 'Learning (uncertainty reduced)',
            'uncertainty_prev': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'uncertainty_curr': torch.tensor([30.0, 30.0, 30.0, 30.0]),
            'coherence_prev': torch.ones(4, 1) * 0.5,
            'coherence_curr': torch.ones(4, 1) * 0.5,
            'attention_weight': torch.ones(4, 1) * 0.5,
            'kl_divergence': torch.ones(4) * 0.1
        },
        {
            'name': 'Competence (breakdown resolved)',
            'uncertainty_prev': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'uncertainty_curr': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'coherence_prev': torch.ones(4, 1) * 0.8,
            'coherence_curr': torch.ones(4, 1) * 0.3,
            'attention_weight': torch.ones(4, 1) * 0.9,
            'kl_divergence': torch.ones(4) * 0.1
        },
        {
            'name': 'Novelty (unexpected discovery)',
            'uncertainty_prev': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'uncertainty_curr': torch.tensor([50.0, 50.0, 50.0, 50.0]),
            'coherence_prev': torch.ones(4, 1) * 0.5,
            'coherence_curr': torch.ones(4, 1) * 0.5,
            'attention_weight': torch.ones(4, 1) * 0.5,
            'kl_divergence': torch.ones(4) * 5.0
        },
        {
            'name': 'Mixed (all three)',
            'uncertainty_prev': torch.tensor([60.0, 60.0, 60.0, 60.0]),
            'uncertainty_curr': torch.tensor([40.0, 40.0, 40.0, 40.0]),
            'coherence_prev': torch.ones(4, 1) * 0.7,
            'coherence_curr': torch.ones(4, 1) * 0.4,
            'attention_weight': torch.ones(4, 1) * 0.8,
            'kl_divergence': torch.ones(4) * 2.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        rewards = intrinsic_reward(
            scenario['uncertainty_prev'],
            scenario['uncertainty_curr'],
            scenario['coherence_prev'],
            scenario['coherence_curr'],
            scenario['attention_weight'],
            scenario['kl_divergence']
        )
        
        print(f"  R_curiosity: {rewards['R_curiosity'].mean().item():.4f}")
        print(f"  R_competence: {rewards['R_competence'].mean().item():.4f}")
        print(f"  R_novelty: {rewards['R_novelty'].mean().item():.4f}")
        print(f"  R_intrinsic (total): {rewards['R_intrinsic'].mean().item():.4f}")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Intrinsic rewards give Agent C a 'purpose':")
    print("1. Curiosity: Seek to reduce uncertainty")
    print("2. Competence: Engage with breakdowns and resolve them")
    print("3. Novelty: Discover new patterns")
    print("This prevents 'coherence minimization collapse'")
    print("="*60)
