"""
Valence Memory V3 - Action-Slack Memory for Emergent Attention

This module implements a simplified valence mechanism that records the association
between actions and Slack changes (Δη).

Key differences from V2:
- V2: valence = f(R_intrinsic) where R_intrinsic combines curiosity, competence, novelty
- V3: valence = memory of (action, Δη) pairs

The valence vector encodes "what happened when I did what", allowing the agent
to learn which actions lead to stable or unstable Slack dynamics.

This is designed for Condition 2 (Emergent Valence), where the agent learns
how to use valence on its own, rather than having the usage prescribed by
a Priority = coherence × uncertainty × valence formula.
"""

import torch
import torch.nn as nn
from typing import Dict


class ValenceMemoryV3(nn.Module):
    """
    Valence Memory that records associations between actions and Slack changes.
    
    The valence vector is updated based on:
    - The action taken (or internal state change)
    - The resulting change in Slack (Δη)
    
    This creates a memory of "action-consequence" pairs that the agent can use
    to guide future behavior.
    """
    
    def __init__(
        self,
        valence_dim: int = 32,
        decay_rate: float = 0.1,
        learning_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Args:
            valence_dim: Dimension of valence vector
            decay_rate: Exponential decay rate for old memories (β)
            learning_rate: Learning rate for valence updates (α)
            device: Device to place the module on
        """
        super().__init__()
        
        self.valence_dim = valence_dim
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        
        # Project Δη (scalar) to valence space
        # This learns a representation of "what Δη means"
        self.delta_eta_to_valence = nn.Sequential(
            nn.Linear(1, valence_dim),
            nn.Tanh()
        ).to(device)
        
        # Initialize with small weights for stability
        for m in self.delta_eta_to_valence.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
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
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update valence based on Slack change.
        
        Args:
            valence_prev: Previous valence vector (B, valence_dim)
            coherence_prev: Previous coherence (η at t-1) (B, 1)
            coherence_curr: Current coherence (η at t) (B, 1)
        
        Returns:
            results: Dictionary with updated valence and Δη
        """
        # Compute Δη (change in Slack)
        # Positive Δη means Slack increased (moved away from coherent state)
        # Negative Δη means Slack decreased (moved toward coherent state)
        delta_eta = coherence_curr - coherence_prev  # (B, 1)
        
        # Project Δη to valence space
        # This learns what different Δη values "mean" in terms of valence
        valence_update = self.delta_eta_to_valence(delta_eta)  # (B, valence_dim)
        
        # Update valence with exponential decay and learning rate
        # valence(t) = (1 - β) * valence(t-1) + α * Δvalence
        valence_new = (
            (1 - self.decay_rate) * valence_prev + 
            self.learning_rate * valence_update
        )
        
        return {
            'valence': valence_new,
            'delta_eta': delta_eta.squeeze(-1),  # (B,) for logging
            'valence_update_norm': valence_update.norm(dim=-1)  # (B,) for logging
        }


if __name__ == '__main__':
    # Test ValenceMemoryV3
    print("Testing ValenceMemoryV3...")
    
    batch_size = 4
    valence_dim = 32
    
    # Create module
    valence_memory = ValenceMemoryV3(
        valence_dim=valence_dim,
        decay_rate=0.1,
        learning_rate=0.1
    )
    
    # Initialize valence
    device = torch.device('cpu')
    valence_prev = valence_memory.get_initial_valence(batch_size, device)
    
    print(f"\nInitial valence shape: {valence_prev.shape}")
    print(f"Initial valence norm: {valence_prev.norm(dim=-1)}")
    
    # Simulate a sequence of Slack changes
    coherence_prev = torch.tensor([[0.5], [0.6], [0.4], [0.7]], device=device)
    coherence_curr = torch.tensor([[0.4], [0.5], [0.5], [0.6]], device=device)
    
    # Update valence
    results = valence_memory(valence_prev, coherence_prev, coherence_curr)
    
    print(f"\nΔη: {results['delta_eta']}")
    print(f"Valence update norm: {results['valence_update_norm']}")
    print(f"New valence shape: {results['valence'].shape}")
    print(f"New valence norm: {results['valence'].norm(dim=-1)}")
    
    # Test multiple updates
    valence = valence_prev
    print(f"\nSimulating 10 steps...")
    for step in range(10):
        coherence_prev = torch.rand(batch_size, 1, device=device) * 0.5 + 0.3
        coherence_curr = torch.rand(batch_size, 1, device=device) * 0.5 + 0.3
        
        results = valence_memory(valence, coherence_prev, coherence_curr)
        valence = results['valence']
        
        if step % 3 == 0:
            print(f"  Step {step}: Δη mean={results['delta_eta'].mean():.4f}, "
                  f"valence norm mean={valence.norm(dim=-1).mean():.4f}")
    
    print("\nValenceMemoryV3 test passed!")
