"""
Conditional Adjunction Model v3: F_C ⊣ G_C with Agent C v3 (Purpose Space P)

This integrates Agent Layer C v3 (with Valence Memory) with Functors F and G,
creating a conditional adjunction where the relationship between Shape and Action is
parameterized by the agent's internal state C, including Purpose Space P.

Key difference from v2:
- v2: Priority = coherence × uncertainty
- v3: Priority = coherence × uncertainty × valence (experiential value)

The context vector from C v3 includes valence, modulating F and G through FiLM.
This allows the agent to "preserve itself" through accumulated value judgments.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG
from src.models.agent_layer_v3 import AgentLayerC_v3
from src.models.conditional_adjunction import ConditionalFunctorF, ConditionalFunctorG


class ConditionalAdjunctionModelV3(nn.Module):
    """
    Full Conditional Adjunction Model v3: F_C ⊣ G_C with Agent Layer C v3 (Purpose Space P).
    
    This integrates:
    - Agent Layer C v3 (RSSM + Priority-based Attention + Valence Memory)
    - Conditional Functor F_C (FiLM-modulated)
    - Conditional Functor G_C (FiLM-modulated)
    - Spatial Coherence Signal computation
    - Purpose Space P (experiential value accumulation)
    """
    
    def __init__(
        self,
        num_affordances: int = 5,
        num_points: int = 512,
        f_hidden_dim: int = 64,
        g_hidden_dim: int = 128,
        agent_hidden_dim: int = 256,
        agent_latent_dim: int = 64,
        context_dim: int = 128,
        valence_dim: int = 32,
        valence_decay: float = 0.1,
        uncertainty_type: str = 'entropy',
        attention_temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_affordances = num_affordances
        self.num_points = num_points
        self.context_dim = context_dim
        
        # Create base F and G
        base_f = FunctorF(
            num_affordances=num_affordances,
            hidden_dim=f_hidden_dim
        )
        
        base_g = FunctorG(
            num_affordances=num_affordances,
            num_points=num_points,
            hidden_dim=g_hidden_dim
        )
        
        # Reuse ConditionalFunctorF and ConditionalFunctorG from v1
        self.F = ConditionalFunctorF(base_f, context_dim)
        self.G = ConditionalFunctorG(base_g, context_dim)
        
        # Agent Layer C v3 (with purpose space P)
        self.agent_c = AgentLayerC_v3(
            obs_dim=f_hidden_dim,
            action_dim=num_affordances,
            hidden_dim=agent_hidden_dim,
            latent_dim=agent_latent_dim,
            context_dim=context_dim,
            valence_dim=valence_dim,
            valence_decay=valence_decay,
            uncertainty_type=uncertainty_type,
            attention_temperature=attention_temperature
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize agent state."""
        return self.agent_c.initial_state(batch_size, device)
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        agent_state: Optional[Dict[str, torch.Tensor]] = None,
        coherence_signal_prev: Optional[torch.Tensor] = None,
        coherence_spatial_prev: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the conditional adjunction v3.
        
        Args:
            pos: Point cloud (N, 3) or (B*N, 3)
            batch: Batch assignment (N,) if unbatched
            agent_state: Previous agent state (optional)
            coherence_signal_prev: Previous scalar coherence signal (B, 1) (optional)
            coherence_spatial_prev: Previous spatial coherence signal (N,) (optional)
        
        Returns:
            results: Dictionary with all outputs including priority info
        """
        device = pos.device
        
        # Determine batch size and create batch assignment if needed
        if batch is None:
            if pos.dim() == 2:
                batch_size = 1
                batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)
            else:
                batch_size = pos.size(0)
        else:
            batch_size = batch.max().item() + 1
        
        N = pos.size(0) if pos.dim() == 2 else pos.size(0) * pos.size(1)
        
        # Initialize agent state if not provided
        if agent_state is None:
            agent_state = self.initial_state(batch_size, device)
        
        # Initialize coherence signals if not provided
        if coherence_signal_prev is None:
            coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
        
        if coherence_spatial_prev is None:
            coherence_spatial_prev = torch.zeros(N, device=device)
        
        # Step 1: Get context from Agent C v3
        # Agent C v3 receives spatial coherence + valence-aware priority
        dummy_action = torch.zeros(batch_size, self.num_affordances, device=device)
        
        agent_state_new, context, agent_info = self.agent_c(
            prev_state=agent_state,
            action=dummy_action,
            coherence_signal_scalar=coherence_signal_prev,
            coherence_signal_spatial=coherence_spatial_prev,
            batch=batch,
            obs=None  # Observations will be provided in the training loop
        )
        
        # Step 2: Apply F_C (Shape → Action), modulated by context
        affordances = self.F(pos, batch, context)
        
        # Step 3: Apply G_C (Action → Shape), modulated by context
        # Average affordances per batch for G
        if batch is not None and pos.dim() == 2:
            affordances_batched = torch.zeros(batch_size, self.num_affordances, device=device)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.sum() > 0:
                    affordances_batched[b] = affordances[mask].mean(dim=0)
        else:
            affordances_batched = affordances.mean(dim=1) if affordances.dim() == 3 else affordances
        
        reconstructed = self.G(affordances_batched, self.num_points, context)
        
        # Step 4: Compute coherence signals (both scalar and spatial)
        coherence_signal, coherence_spatial = self._compute_coherence_signal(
            pos, reconstructed, batch
        )
        
        return {
            'affordances': affordances,
            'reconstructed': reconstructed,
            'coherence_signal': coherence_signal,         # (B, 1) scalar
            'coherence_spatial': coherence_spatial,       # (N,) per-point
            'agent_state': agent_state_new,
            'context': context,
            'rssm_info': agent_info,
            'priority': agent_info.get('priority', None),
            'priority_normalized': agent_info.get('priority_normalized', None),
            'uncertainty': agent_info.get('uncertainty', None)
        }
    
    def _compute_coherence_signal(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both scalar and spatial coherence signals.
        
        Returns:
            coherence_scalar: (B, 1) average Chamfer distance per sample
            coherence_spatial: (N,) per-point distance to nearest reconstructed point
        """
        # Handle batching of original points
        if original.dim() == 2:
            if batch is None:
                original_batched = original.unsqueeze(0)
                batch = torch.zeros(original.size(0), dtype=torch.long, device=original.device)
            else:
                batch_size = batch.max().item() + 1
                original_batched = []
                for b in range(batch_size):
                    original_batched.append(original[batch == b])
                max_len = max(p.size(0) for p in original_batched)
                original_batched = torch.stack([
                    torch.cat([p, torch.zeros(max_len - p.size(0), 3, device=p.device)], dim=0)
                    for p in original_batched
                ])
        else:
            original_batched = original
        
        B = original_batched.size(0)
        
        # Pairwise distances
        dist = torch.cdist(original_batched, reconstructed)  # (B, N1, N2)
        
        # Scalar coherence: Chamfer distance
        dist_forward = dist.min(dim=2)[0].mean(dim=1)  # (B,)
        dist_backward = dist.min(dim=1)[0].mean(dim=1)  # (B,)
        coherence_scalar = ((dist_forward + dist_backward) / 2).unsqueeze(-1)  # (B, 1)
        
        # Spatial coherence: per-point distance to nearest reconstructed point
        # For each original point, find the distance to its nearest reconstructed point
        per_point_dist = dist.min(dim=2)[0]  # (B, N1)
        
        # Flatten back to (N,) matching the batch assignment
        coherence_spatial_list = []
        for b in range(B):
            mask = (batch == b)
            n_points = mask.sum()
            coherence_spatial_list.append(per_point_dist[b, :n_points])
        
        coherence_spatial = torch.cat(coherence_spatial_list, dim=0)  # (N,)
        
        return coherence_scalar, coherence_spatial


if __name__ == '__main__':
    # Test Conditional Adjunction Model v3
    print("Testing Conditional Adjunction Model v3...")
    
    model = ConditionalAdjunctionModelV3(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64,
        uncertainty_type='entropy',
        attention_temperature=1.0
    )
    
    print(f"Model created")
    print(f"  F_C: {sum(p.numel() for p in model.F.parameters())} parameters")
    print(f"  G_C: {sum(p.numel() for p in model.G.parameters())} parameters")
    print(f"  Agent C v3: {sum(p.numel() for p in model.agent_c.parameters())} parameters")
    print(f"  Total: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    num_points = 256
    pos = torch.randn(batch_size * num_points, 3)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
    
    print(f"\nInput:")
    print(f"  pos shape: {pos.shape}")
    print(f"  batch shape: {batch.shape}")
    
    results = model(pos, batch)
    
    print(f"\nOutput:")
    print(f"  Affordances shape: {results['affordances'].shape}")
    print(f"  Reconstructed shape: {results['reconstructed'].shape}")
    print(f"  Coherence signal (scalar): {results['coherence_signal']}")
    print(f"  Coherence signal (spatial) shape: {results['coherence_spatial'].shape}")
    print(f"  Context shape: {results['context'].shape}")
    print(f"  Priority shape: {results['priority'].shape if results['priority'] is not None else 'None'}")
    print(f"  Uncertainty: {results['uncertainty']}")
    
    # Test sequential processing (simulating online learning)
    print("\nTesting sequential processing...")
    agent_state = model.initial_state(1, torch.device('cpu'))
    coherence_prev = torch.zeros(1, 1)
    coherence_spatial_prev = torch.zeros(num_points)
    
    pos_single = torch.randn(num_points, 3)
    batch_single = torch.zeros(num_points, dtype=torch.long)
    
    for t in range(3):
        results = model(
            pos_single, batch_single,
            agent_state=agent_state,
            coherence_signal_prev=coherence_prev,
            coherence_spatial_prev=coherence_spatial_prev
        )
        
        agent_state = {k: v.detach() for k, v in results['agent_state'].items()}
        coherence_prev = results['coherence_signal'].detach()
        coherence_spatial_prev = results['coherence_spatial'].detach()
        
        print(f"  Step {t}: Coherence={coherence_prev.item():.4f}, "
              f"Priority mean={results['priority'].mean().item():.4f}")
    
    print("\nConditional Adjunction Model v3 test passed!")
