"""
Intrinsic Reward Module for Purpose Space P

This module computes intrinsic rewards based on Active Inference principles,
adapted for the "suspension structure" philosophy:

1. Curiosity (R_curiosity): Confidence change (narrowing down choices)
   - "Choices are narrowing — I am becoming more certain"
   - Confidence = 1 - normalized_entropy(priority_distribution)
   - Curiosity = confidence(t) - confidence(t-1)
   - This is a META-INDICATOR of P, not an axis of P

2. Competence (R_competence): Attending to breakdowns
   - "I face difficulty and engage with it"
   - Encourages engaging with breakdowns rather than avoiding them

3. Novelty (R_novelty): Unexpected discoveries (KL divergence)
   - "Something unexpected happened"
   - Encourages discovering new patterns (but not too much)

Key insights from development:
- Confidence = "other choices decrease" (entropy reduction of priority distribution)
- Confidence is an OUTPUT indicator of P, not an INPUT axis
- P has 3 axes: Coherence, Uncertainty, Valence (inputs to Priority)
- Confidence emerges FROM Priority distribution (output)
- Curiosity = change in confidence (delta of meta-indicator)
- Competence = "attending to breakdowns" (not reducing them, when F/G frozen)

The intrinsic reward is used to update valence, giving Agent C a "purpose":
- valence(t+1) = (1-decay) * valence(t) + decay * R_intrinsic
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class IntrinsicRewardComputation(nn.Module):
    """
    Compute intrinsic rewards for valence update.
    
    This module combines three types of intrinsic motivation:
    1. Curiosity: confidence change (entropy reduction of priority distribution)
    2. Competence: attending to breakdowns (engaging with difficulty)
    3. Novelty: unexpected discoveries (KL divergence)
    
    Confidence is computed as:
        confidence = 1 - entropy(priority_normalized) / log(num_points)
    
    Curiosity reward is:
        R_curiosity = confidence(t) - confidence(t-1)
        (positive when choices narrow down)
    """
    
    def __init__(
        self,
        alpha_curiosity: float = 0.3,
        beta_competence: float = 0.5,
        gamma_novelty: float = 0.2,
        novelty_scale: float = 0.1,
        competence_scale: float = 100.0
    ):
        """
        Args:
            alpha_curiosity: Weight for curiosity reward
            beta_competence: Weight for competence reward
            gamma_novelty: Weight for novelty reward
            novelty_scale: Scale factor for novelty (to keep it in reasonable range)
            competence_scale: Scale factor for competence (to amplify small coherence values)
        """
        super().__init__()
        
        self.alpha = alpha_curiosity
        self.beta = beta_competence
        self.gamma = gamma_novelty
        self.novelty_scale = novelty_scale
        self.competence_scale = competence_scale
    
    @staticmethod
    def compute_confidence(priority_normalized: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence as 1 - normalized entropy of priority distribution.
        
        Confidence = 1 means all attention on one point (maximum certainty).
        Confidence = 0 means uniform attention (maximum uncertainty).
        
        Args:
            priority_normalized: Normalized priority scores (N,), sums to 1 per batch
            batch: Batch assignment (N,)
        
        Returns:
            confidence: Per-batch confidence (B,)
        """
        batch_size = int(batch.max().item()) + 1
        confidence = torch.zeros(batch_size, device=batch.device)
        
        for b in range(batch_size):
            mask = (batch == b)
            n_points = mask.sum()
            if n_points > 1:
                p = priority_normalized[mask]
                # Entropy of the distribution
                entropy = -(p * torch.log(p + 1e-8)).sum()
                # Maximum entropy for uniform distribution
                max_entropy = torch.log(torch.tensor(float(n_points), device=batch.device))
                # Confidence = 1 - normalized entropy
                confidence[b] = 1.0 - (entropy / (max_entropy + 1e-8))
            else:
                # Single point: maximum confidence
                confidence[b] = 1.0
        
        return confidence
    
    def forward(
        self,
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
        Compute intrinsic rewards.
        
        Args:
            uncertainty_prev: Previous uncertainty (B,)
            uncertainty_curr: Current uncertainty (B,)
            coherence_prev: Previous coherence (B, 1)
            coherence_curr: Current coherence (B, 1)
            attention_weight: Attention weight (B, 1)
            kl_divergence: KL(posterior || prior) (B,)
            confidence_prev: Previous confidence (B,) - from prev step's priority distribution
            confidence_curr: Current confidence (B,) - from current step's priority distribution
        
        Returns:
            rewards: Dictionary with individual and total intrinsic rewards
        """
        batch_size = uncertainty_curr.size(0)
        device = uncertainty_curr.device
        
        # (1) Curiosity reward: confidence change (narrowing down choices)
        # Confidence = 1 - normalized_entropy(priority_distribution)
        # Curiosity = confidence(t) - confidence(t-1)
        # Positive when choices narrow down (becoming more certain)
        
        if confidence_prev is not None and confidence_curr is not None:
            # Curiosity = increase in confidence
            R_curiosity = torch.clamp(confidence_curr - confidence_prev, min=0.0)
        else:
            # Fallback: no confidence data available yet (first step)
            R_curiosity = torch.zeros(batch_size, device=device)
        
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
            'R_intrinsic': R_intrinsic.squeeze(-1),
            'confidence_curr': confidence_curr if confidence_curr is not None else torch.zeros(batch_size, device=device)
        }


if __name__ == '__main__':
    print("Testing Intrinsic Reward Computation (v5: Confidence-based Curiosity)...")
    
    batch_size = 2
    num_points = 10
    
    # Create module
    intrinsic_reward = IntrinsicRewardComputation(
        alpha_curiosity=0.3,
        beta_competence=0.5,
        gamma_novelty=0.2
    )
    
    print(f"\nIntrinsic Reward module created")
    print(f"  alpha (curiosity): {intrinsic_reward.alpha}")
    print(f"  beta (competence): {intrinsic_reward.beta}")
    print(f"  gamma (novelty): {intrinsic_reward.gamma}")
    
    # Test confidence computation
    print(f"\n{'='*60}")
    print("Testing confidence computation")
    print(f"{'='*60}")
    
    batch = torch.tensor([0]*num_points + [1]*num_points)
    
    # Uniform distribution (low confidence)
    p_uniform = torch.ones(num_points * 2) / num_points
    conf_uniform = IntrinsicRewardComputation.compute_confidence(p_uniform, batch)
    print(f"  Uniform distribution: confidence = {conf_uniform}")
    
    # Concentrated distribution (high confidence)
    p_concentrated = torch.zeros(num_points * 2)
    p_concentrated[0] = 0.9
    p_concentrated[1:num_points] = 0.1 / (num_points - 1)
    p_concentrated[num_points] = 0.9
    p_concentrated[num_points+1:] = 0.1 / (num_points - 1)
    conf_concentrated = IntrinsicRewardComputation.compute_confidence(p_concentrated, batch)
    print(f"  Concentrated distribution: confidence = {conf_concentrated}")
    
    # Delta-like distribution (maximum confidence)
    p_delta = torch.zeros(num_points * 2)
    p_delta[0] = 1.0
    p_delta[num_points] = 1.0
    conf_delta = IntrinsicRewardComputation.compute_confidence(p_delta, batch)
    print(f"  Delta distribution: confidence = {conf_delta}")
    
    # Test curiosity reward (confidence change)
    print(f"\n{'='*60}")
    print("Testing curiosity reward (confidence change)")
    print(f"{'='*60}")
    
    rewards = intrinsic_reward(
        uncertainty_prev=torch.tensor([50.0, 50.0]),
        uncertainty_curr=torch.tensor([50.0, 50.0]),
        coherence_prev=torch.ones(2, 1) * 0.5,
        coherence_curr=torch.ones(2, 1) * 0.5,
        attention_weight=torch.ones(2, 1) * 0.5,
        kl_divergence=torch.ones(2) * 0.1,
        confidence_prev=conf_uniform,
        confidence_curr=conf_concentrated
    )
    
    print(f"  Confidence: {conf_uniform[0]:.4f} -> {conf_concentrated[0]:.4f}")
    print(f"  R_curiosity: {rewards['R_curiosity'].mean().item():.4f}")
    print(f"  R_competence: {rewards['R_competence'].mean().item():.4f}")
    print(f"  R_novelty: {rewards['R_novelty'].mean().item():.4f}")
    print(f"  R_intrinsic: {rewards['R_intrinsic'].mean().item():.4f}")
    
    # Test no confidence change
    print(f"\n{'='*60}")
    print("Testing no confidence change")
    print(f"{'='*60}")
    
    rewards2 = intrinsic_reward(
        uncertainty_prev=torch.tensor([50.0, 50.0]),
        uncertainty_curr=torch.tensor([50.0, 50.0]),
        coherence_prev=torch.ones(2, 1) * 0.5,
        coherence_curr=torch.ones(2, 1) * 0.5,
        attention_weight=torch.ones(2, 1) * 0.5,
        kl_divergence=torch.ones(2) * 0.1,
        confidence_prev=conf_concentrated,
        confidence_curr=conf_concentrated
    )
    
    print(f"  Confidence: {conf_concentrated[0]:.4f} -> {conf_concentrated[0]:.4f}")
    print(f"  R_curiosity: {rewards2['R_curiosity'].mean().item():.4f}")
    print(f"  R_intrinsic: {rewards2['R_intrinsic'].mean().item():.4f}")
    
    print(f"\n{'='*60}")
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Confidence = 1 - normalized_entropy(priority_distribution)")
    print("  High confidence = choices narrowed down")
    print("  Low confidence = choices spread out")
    print("Curiosity = confidence(t) - confidence(t-1)")
    print("  Positive when becoming more certain")
    print("  Zero when confidence unchanged or decreasing")
    print("Confidence is an OUTPUT indicator of P, not an INPUT axis")
    print("="*60)
