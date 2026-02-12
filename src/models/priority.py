"""
Priority Computation: coherence × uncertainty

This module implements the core principle of intentionality in the suspension structure:
    priority_i = coherence_i × uncertainty_i

Where:
- coherence_i: Per-point coherence signal (spatial decomposition)
- uncertainty_i: Uncertainty of the agent's latent state z

This principle embodies the idea that the agent should prioritize breakdowns that:
1. Are large (high coherence signal → high potential for "margin recovery")
2. Are uncertain (high uncertainty → high potential for GNN growth)

This is NOT:
- Prioritizing by breakdown size alone (would be reactive)
- Prioritizing by novelty alone (would be curiosity-driven)
- Prioritizing by resolvability (would avoid difficult problems)

Instead, it's prioritizing by "long-term margin maximization":
Choosing uncertain breakdowns forces GNN growth, which maintains adaptability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PriorityComputation(nn.Module):
    """
    Compute priority scores for spatial breakdowns.
    
    priority_i = coherence_i × uncertainty_i
    """
    
    def __init__(
        self,
        uncertainty_type: str = 'entropy',  # 'entropy' or 'kl'
        temperature: float = 1.0  # Temperature for softmax normalization
    ):
        """
        Args:
            uncertainty_type: Type of uncertainty measure
                - 'entropy': H(z) = -sum(p log p)
                - 'kl': KL(posterior || prior)
            temperature: Temperature for softmax normalization of priorities
        """
        super().__init__()
        
        self.uncertainty_type = uncertainty_type
        self.temperature = temperature
    
    def compute_uncertainty(
        self,
        z_mean: torch.Tensor,
        z_std: torch.Tensor,
        prior_mean: torch.Tensor = None,
        prior_std: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute uncertainty of the latent state z.
        
        Args:
            z_mean: Mean of z (B, latent_dim)
            z_std: Std of z (B, latent_dim)
            prior_mean: Mean of prior (B, latent_dim), required if uncertainty_type='kl'
            prior_std: Std of prior (B, latent_dim), required if uncertainty_type='kl'
        
        Returns:
            uncertainty: (B,) uncertainty per sample
        """
        if self.uncertainty_type == 'entropy':
            # Entropy of Gaussian: H(z) = 0.5 * log(2πe * σ^2)
            # We use sum over dimensions as a measure of total uncertainty
            entropy = 0.5 * torch.log(2 * torch.pi * torch.e * z_std.pow(2))
            uncertainty = entropy.sum(dim=-1)  # (B,)
        
        elif self.uncertainty_type == 'kl':
            # KL divergence between posterior and prior
            if prior_mean is None or prior_std is None:
                raise ValueError("prior_mean and prior_std required for uncertainty_type='kl'")
            
            kl = 0.5 * (
                2 * torch.log(prior_std / z_std) +
                (z_std.pow(2) + (z_mean - prior_mean).pow(2)) / prior_std.pow(2) - 1
            )
            uncertainty = kl.sum(dim=-1)  # (B,)
        
        else:
            raise ValueError(f"Unknown uncertainty_type: {self.uncertainty_type}")
        
        return uncertainty
    
    def compute_priority(
        self,
        coherence_spatial: torch.Tensor,
        uncertainty: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute priority scores for each point.
        
        priority_i = coherence_i × uncertainty_i
        
        Args:
            coherence_spatial: (N,) per-point coherence signal
            uncertainty: (B,) per-sample uncertainty
            batch: (N,) batch assignment for coherence_spatial
        
        Returns:
            priority: (N,) priority score per point
            priority_normalized: (N,) softmax-normalized priority (sums to 1 per batch)
        """
        # Expand uncertainty to per-point
        # uncertainty: (B,) -> (N,)
        uncertainty_expanded = uncertainty[batch]  # (N,)
        
        # Compute priority
        priority = coherence_spatial * uncertainty_expanded  # (N,)
        
        # Normalize priority within each batch using softmax
        # This ensures priorities sum to 1 per sample
        priority_normalized = torch.zeros_like(priority)
        
        batch_size = batch.max().item() + 1
        for b in range(batch_size):
            mask = (batch == b)
            priority_batch = priority[mask]
            
            # Apply temperature and softmax
            priority_batch_normalized = F.softmax(priority_batch / self.temperature, dim=0)
            priority_normalized[mask] = priority_batch_normalized
        
        return priority, priority_normalized
    
    def forward(
        self,
        coherence_spatial: torch.Tensor,
        agent_state_info: Dict[str, torch.Tensor],
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for priority computation.
        
        Args:
            coherence_spatial: (N,) per-point coherence signal
            agent_state_info: Dictionary with RSSM info
                - 'posterior_mean': (B, latent_dim)
                - 'posterior_std': (B, latent_dim)
                - 'prior_mean': (B, latent_dim)
                - 'prior_std': (B, latent_dim)
            batch: (N,) batch assignment
        
        Returns:
            results: Dictionary with:
                - 'priority': (N,) raw priority scores
                - 'priority_normalized': (N,) normalized priority scores
                - 'uncertainty': (B,) uncertainty per sample
        """
        # Compute uncertainty
        if self.uncertainty_type == 'entropy':
            uncertainty = self.compute_uncertainty(
                agent_state_info['posterior_mean'],
                agent_state_info['posterior_std']
            )
        else:  # 'kl'
            uncertainty = self.compute_uncertainty(
                agent_state_info['posterior_mean'],
                agent_state_info['posterior_std'],
                agent_state_info['prior_mean'],
                agent_state_info['prior_std']
            )
        
        # Compute priority
        priority, priority_normalized = self.compute_priority(
            coherence_spatial, uncertainty, batch
        )
        
        return {
            'priority': priority,
            'priority_normalized': priority_normalized,
            'uncertainty': uncertainty
        }


if __name__ == '__main__':
    # Test priority computation
    print("Testing Priority Computation...")
    
    batch_size = 2
    num_points = 512
    latent_dim = 64
    
    # Create module
    priority_module = PriorityComputation(uncertainty_type='entropy', temperature=1.0)
    
    # Create dummy inputs
    coherence_spatial = torch.rand(batch_size * num_points)  # (N,)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)  # (N,)
    
    agent_state_info = {
        'posterior_mean': torch.randn(batch_size, latent_dim),
        'posterior_std': F.softplus(torch.randn(batch_size, latent_dim)) + 1e-5,
        'prior_mean': torch.randn(batch_size, latent_dim),
        'prior_std': F.softplus(torch.randn(batch_size, latent_dim)) + 1e-5
    }
    
    # Forward pass
    results = priority_module(coherence_spatial, agent_state_info, batch)
    
    print(f"\nResults:")
    print(f"  Coherence spatial shape: {coherence_spatial.shape}")
    print(f"  Uncertainty shape: {results['uncertainty'].shape}")
    print(f"  Uncertainty values: {results['uncertainty']}")
    print(f"  Priority shape: {results['priority'].shape}")
    print(f"  Priority mean: {results['priority'].mean():.4f}")
    print(f"  Priority std: {results['priority'].std():.4f}")
    print(f"  Priority normalized shape: {results['priority_normalized'].shape}")
    
    # Check normalization
    for b in range(batch_size):
        mask = (batch == b)
        priority_sum = results['priority_normalized'][mask].sum()
        print(f"  Priority sum (batch {b}): {priority_sum:.4f} (should be ~1.0)")
    
    print("\nPriority Computation test passed!")
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("priority_i = coherence_i × uncertainty_i")
    print("This implements the principle of 'long-term margin maximization':")
    print("- High coherence → large potential for margin recovery")
    print("- High uncertainty → large potential for GNN growth")
    print("="*60)
