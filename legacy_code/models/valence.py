"""
Valence Memory: Experiential Value Accumulation

This module implements the third axis of the Purpose Space P: Valence.

Valence represents the agent's learned value judgment based on past experiences:
"When I attended to this kind of breakdown, did coherence improve or worsen?"

Key design principles:
1. Valence is a DESIGNED FRAMEWORK (the axis itself)
2. The CONTENT of valence (specific values) EMERGES from experience
3. Valence is accumulated through attention-weighted coherence changes
4. Old experiences decay exponentially to allow adaptation

Update rule:
    valence(t+1) = (1-β) × valence(t) + β × Δcoherence × attention_weight(t)

Where:
- Δcoherence = coherence(t) - coherence(t+1): positive if coherence improved
- attention_weight(t): how much attention was paid to that point
- β: learning rate (decay factor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ValenceMemory(nn.Module):
    """
    Valence Memory: Accumulates experiential value judgments.
    
    This module maintains a valence vector v_t that represents the agent's
    learned preferences based on past experiences with coherence changes.
    """
    
    def __init__(
        self,
        valence_dim: int = 32,
        decay_rate: float = 0.1,
        init_valence: float = 1.0
    ):
        """
        Args:
            valence_dim: Dimension of valence vector
            decay_rate: Decay rate β for exponential forgetting (0 = no learning, 1 = no memory)
            init_valence: Initial value for valence (neutral = 1.0)
        """
        super().__init__()
        
        self.valence_dim = valence_dim
        self.decay_rate = decay_rate
        self.init_valence = init_valence
        
        # Learnable projection from coherence change to valence update
        # This allows the model to learn which kinds of coherence changes matter
        self.coherence_to_valence = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, valence_dim)
        )
    
    def initial_valence(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize valence vector to neutral values.
        
        Args:
            batch_size: Batch size
            device: Device
        
        Returns:
            valence: (B, valence_dim) initialized to init_valence
        """
        return torch.full(
            (batch_size, self.valence_dim),
            self.init_valence,
            device=device,
            dtype=torch.float32
        )
    
    def compute_coherence_change(
        self,
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute coherence change (improvement is positive).
        
        Args:
            coherence_prev: Previous coherence signal (B, 1)
            coherence_curr: Current coherence signal (B, 1)
        
        Returns:
            delta_coherence: (B, 1) coherence change (positive = improvement)
        """
        # Improvement = coherence decreased (smaller distance is better)
        delta_coherence = coherence_prev - coherence_curr
        return delta_coherence
    
    def update(
        self,
        valence_prev: torch.Tensor,
        delta_coherence: torch.Tensor,
        attention_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update valence based on coherence change and attention.
        
        Args:
            valence_prev: Previous valence (B, valence_dim)
            delta_coherence: Coherence change (B, 1)
            attention_weight: Attention weight (B, 1), optional (defaults to 1.0)
        
        Returns:
            valence_new: Updated valence (B, valence_dim)
        """
        batch_size = valence_prev.size(0)
        device = valence_prev.device
        
        # Default attention weight to 1.0 if not provided
        if attention_weight is None:
            attention_weight = torch.ones(batch_size, 1, device=device)
        
        # Project coherence change to valence update
        # This allows the model to learn a rich representation of "good/bad experiences"
        valence_update = self.coherence_to_valence(delta_coherence)  # (B, valence_dim)
        
        # Weight by attention (only experiences we paid attention to matter)
        valence_update = valence_update * attention_weight  # (B, valence_dim)
        
        # Exponential moving average update
        valence_new = (1 - self.decay_rate) * valence_prev + self.decay_rate * valence_update
        
        return valence_new
    
    def forward(
        self,
        valence_prev: torch.Tensor,
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor,
        attention_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for valence update.
        
        Args:
            valence_prev: Previous valence (B, valence_dim)
            coherence_prev: Previous coherence signal (B, 1)
            coherence_curr: Current coherence signal (B, 1)
            attention_weight: Attention weight (B, 1), optional
        
        Returns:
            results: Dictionary with:
                - 'valence': (B, valence_dim) updated valence
                - 'delta_coherence': (B, 1) coherence change
        """
        # Compute coherence change
        delta_coherence = self.compute_coherence_change(coherence_prev, coherence_curr)
        
        # Update valence
        valence_new = self.update(valence_prev, delta_coherence, attention_weight)
        
        return {
            'valence': valence_new,
            'delta_coherence': delta_coherence
        }


class SpatialValenceMemory(nn.Module):
    """
    Spatial Valence Memory: Per-point valence accumulation.
    
    This is an extension that maintains valence at the spatial (per-point) level,
    not just at the sample level. This allows finer-grained value judgments.
    
    NOTE: This is more complex and may not be necessary initially.
    Start with batch-level ValenceMemory first.
    """
    
    def __init__(
        self,
        valence_dim: int = 32,
        decay_rate: float = 0.1,
        init_valence: float = 1.0
    ):
        super().__init__()
        
        self.valence_dim = valence_dim
        self.decay_rate = decay_rate
        self.init_valence = init_valence
        
        # Projection from spatial coherence change to valence update
        self.coherence_to_valence = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, valence_dim)
        )
        
        # Aggregation: per-point valence -> per-sample valence
        self.spatial_aggregator = nn.Sequential(
            nn.Linear(valence_dim, 128),
            nn.ReLU(),
            nn.Linear(128, valence_dim)
        )
    
    def initial_valence(self, num_points: int, device: torch.device) -> torch.Tensor:
        """Initialize spatial valence."""
        return torch.full(
            (num_points, self.valence_dim),
            self.init_valence,
            device=device,
            dtype=torch.float32
        )
    
    def update(
        self,
        valence_prev: torch.Tensor,
        delta_coherence_spatial: torch.Tensor,
        priority_normalized: torch.Tensor
    ) -> torch.Tensor:
        """
        Update spatial valence.
        
        Args:
            valence_prev: Previous valence (N, valence_dim)
            delta_coherence_spatial: Spatial coherence change (N, 1)
            priority_normalized: Normalized priority (attention weight) (N,)
        
        Returns:
            valence_new: Updated valence (N, valence_dim)
        """
        # Ensure delta_coherence_spatial has correct shape
        if delta_coherence_spatial.dim() == 1:
            delta_coherence_spatial = delta_coherence_spatial.unsqueeze(-1)  # (N, 1)
        
        # Project coherence change to valence update
        valence_update = self.coherence_to_valence(delta_coherence_spatial)  # (N, valence_dim)
        
        # Weight by priority (attention)
        valence_update = valence_update * priority_normalized.unsqueeze(-1)  # (N, valence_dim)
        
        # Exponential moving average
        valence_new = (1 - self.decay_rate) * valence_prev + self.decay_rate * valence_update
        
        return valence_new
    
    def aggregate_to_sample(
        self,
        valence_spatial: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate spatial valence to sample-level valence.
        
        Args:
            valence_spatial: Spatial valence (N, valence_dim)
            batch: Batch assignment (N,)
        
        Returns:
            valence_sample: Sample-level valence (B, valence_dim)
        """
        batch_size = batch.max().item() + 1
        device = valence_spatial.device
        
        valence_sample = torch.zeros(batch_size, self.valence_dim, device=device)
        
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() > 0:
                # Average pooling
                valence_sample[b] = valence_spatial[mask].mean(dim=0)
        
        # Apply aggregation network
        valence_sample = self.spatial_aggregator(valence_sample)
        
        return valence_sample


if __name__ == '__main__':
    # Test Valence Memory
    print("Testing Valence Memory...")
    
    batch_size = 4
    valence_dim = 32
    
    # Create module
    valence_memory = ValenceMemory(
        valence_dim=valence_dim,
        decay_rate=0.1,
        init_valence=1.0
    )
    
    print(f"\nValence Memory created")
    print(f"  Valence dim: {valence_memory.valence_dim}")
    print(f"  Decay rate: {valence_memory.decay_rate}")
    
    # Initialize valence
    device = torch.device('cpu')
    valence = valence_memory.initial_valence(batch_size, device)
    print(f"\nInitial valence shape: {valence.shape}")
    print(f"Initial valence mean: {valence.mean().item():.4f}")
    
    # Simulate a sequence of experiences
    print("\nSimulating experiences...")
    for t in range(5):
        # Random coherence signals
        coherence_prev = torch.rand(batch_size, 1) * 0.5
        coherence_curr = torch.rand(batch_size, 1) * 0.5
        attention_weight = torch.rand(batch_size, 1)
        
        # Update valence
        results = valence_memory(valence, coherence_prev, coherence_curr, attention_weight)
        valence = results['valence']
        delta_coherence = results['delta_coherence']
        
        print(f"\nStep {t}:")
        print(f"  Δcoherence mean: {delta_coherence.mean().item():+.4f}")
        print(f"  Attention mean: {attention_weight.mean().item():.4f}")
        print(f"  Valence mean: {valence.mean().item():.4f}")
        print(f"  Valence std: {valence.std().item():.4f}")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Valence Memory implements the third axis of Purpose Space P:")
    print("1. Framework (axis) is DESIGNED: decay rate, update rule")
    print("2. Content (values) EMERGES from experience: coherence changes")
    print("3. Attention-weighted: only attended experiences matter")
    print("4. Exponential decay: old experiences fade, allowing adaptation")
    print("="*60)
    
    # Test Spatial Valence Memory
    print("\n\nTesting Spatial Valence Memory...")
    
    num_points = 512
    batch_size = 2
    
    spatial_valence_memory = SpatialValenceMemory(
        valence_dim=valence_dim,
        decay_rate=0.1,
        init_valence=1.0
    )
    
    # Initialize spatial valence
    valence_spatial = spatial_valence_memory.initial_valence(batch_size * num_points, device)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
    
    print(f"\nSpatial valence shape: {valence_spatial.shape}")
    
    # Update
    delta_coherence_spatial = torch.randn(batch_size * num_points, 1) * 0.1
    priority_normalized = F.softmax(torch.randn(batch_size * num_points), dim=0)
    
    valence_spatial_new = spatial_valence_memory.update(
        valence_spatial, delta_coherence_spatial, priority_normalized
    )
    
    print(f"Updated spatial valence shape: {valence_spatial_new.shape}")
    print(f"Spatial valence mean: {valence_spatial_new.mean().item():.4f}")
    
    # Aggregate to sample level
    valence_sample = spatial_valence_memory.aggregate_to_sample(valence_spatial_new, batch)
    print(f"\nAggregated sample valence shape: {valence_sample.shape}")
    print(f"Sample valence mean: {valence_sample.mean().item():.4f}")
    
    print("\nSpatial Valence Memory test passed!")
