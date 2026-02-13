"""
Conditional Adjunction Model: F_C ⊣ G_C

This integrates Agent Layer C with Functors F and G, creating a **conditional adjunction**
where the relationship between Shape and Action is parameterized by the agent's internal state C.

Key difference from Phase 1:
- Phase 1: F ⊣ G (fixed adjunction, no agent state)
- Phase 2: F_C ⊣ G_C (conditional adjunction, parameterized by C)

The context vector from C modulates the behavior of F and G through Feature-wise Linear Modulation (FiLM).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG
from src.models.agent_layer import AgentLayerC


class ConditionalFunctorF(nn.Module):
    """
    Conditional Functor F_C: Shape → Action, parameterized by context C.
    
    Uses Feature-wise Linear Modulation (FiLM) to condition F on the context vector.
    """
    
    def __init__(
        self,
        base_f: FunctorF,
        context_dim: int = 128
    ):
        super().__init__()
        
        self.base_f = base_f
        self.context_dim = context_dim
        
        # FiLM layers: context -> (scale, shift) for affordance output
        num_affordances = base_f.num_affordances
        
        self.film_net = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, num_affordances * 2)  # scale and shift
        )
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context conditioning.
        
        Args:
            pos: Point cloud positions (N, 3) or (B, N, 3)
            batch: Batch assignment (N,) if unbatched
            context: Context vector from Agent C (B, context_dim)
        
        Returns:
            affordances: Predicted affordances (N, num_affordances) or (B, N, num_affordances)
        """
        # Simplified implementation: apply base F, then modulate output
        # In a full implementation, we would modulate intermediate layers
        
        affordances = self.base_f(pos, batch)
        
        if context is not None:
            # Generate scale and shift from context
            film_params = self.film_net(context)
            scale, shift = torch.chunk(film_params, 2, dim=-1)  # (B, num_affordances) each
            
            # Determine batch assignment
            if batch is None:
                batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
            
            # Expand scale and shift to match affordances
            scale_expanded = scale[batch]  # (N, num_affordances)
            shift_expanded = shift[batch]  # (N, num_affordances)
            affordances = scale_expanded * affordances + shift_expanded
        
        return affordances


class ConditionalFunctorG(nn.Module):
    """
    Conditional Functor G_C: Action → Shape, parameterized by context C.
    """
    
    def __init__(
        self,
        base_g: FunctorG,
        context_dim: int = 128
    ):
        super().__init__()
        
        self.base_g = base_g
        self.context_dim = context_dim
        
        # FiLM for G
        hidden_dim = base_g.hidden_dim
        
        self.film_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
    
    def forward(
        self,
        affordances: torch.Tensor,
        num_points: int = 512,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context conditioning.
        
        Args:
            affordances: Affordance vectors (B, num_affordances) or (N, num_affordances)
            num_points: Number of points to generate
            context: Context vector (B, context_dim)
        
        Returns:
            reconstructed: Reconstructed point cloud (B, num_points, 3)
        """
        # Simplified implementation: apply base G, then modulate output
        # Note: base_g expects (affordances, batch), but we're passing batched affordances
        # So we pass None for batch
        reconstructed = self.base_g(affordances, batch=None)
        
        if context is not None:
            # Generate scale and shift from context
            film_params = self.film_net(context)
            scale, shift = torch.chunk(film_params, 2, dim=-1)  # (B, hidden_dim)
            
            # Modulate the point cloud
            # scale and shift are (B, hidden_dim), we need to project to 3D
            # For simplicity, we'll apply a uniform scaling and shift
            batch_size = reconstructed.size(0)
            
            # Project to 3D transformation
            scale_3d = scale[:, :3]  # (B, 3)
            shift_3d = shift[:, :3]  # (B, 3)
            
            # Apply to point cloud
            reconstructed = scale_3d.unsqueeze(1) * reconstructed + shift_3d.unsqueeze(1)
        
        return reconstructed


class ConditionalAdjunctionModel(nn.Module):
    """
    Full Conditional Adjunction Model: F_C ⊣ G_C with Agent Layer C.
    
    This integrates:
    - Agent Layer C (RSSM)
    - Conditional Functor F_C
    - Conditional Functor G_C
    - Coherence Signal computation
    """
    
    def __init__(
        self,
        num_affordances: int = 5,
        num_points: int = 512,
        f_hidden_dim: int = 64,
        g_hidden_dim: int = 128,
        agent_hidden_dim: int = 256,
        agent_latent_dim: int = 64,
        context_dim: int = 128
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
        
        # Create conditional versions
        self.F = ConditionalFunctorF(base_f, context_dim)
        self.G = ConditionalFunctorG(base_g, context_dim)
        
        # Create Agent Layer C
        self.agent_c = AgentLayerC(
            obs_dim=f_hidden_dim,  # Observation is encoded shape features
            action_dim=num_affordances,
            hidden_dim=agent_hidden_dim,
            latent_dim=agent_latent_dim,
            context_dim=context_dim
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize agent state."""
        return self.agent_c.initial_state(batch_size, device)
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        agent_state: Optional[Dict[str, torch.Tensor]] = None,
        coherence_signal_prev: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the conditional adjunction.
        
        Args:
            pos: Point cloud (N, 3) or (B, N, 3)
            batch: Batch assignment (N,) if unbatched
            agent_state: Previous agent state (optional, for sequential processing)
            coherence_signal_prev: Previous coherence signal (optional)
        
        Returns:
            results: Dictionary with all outputs
        """
        device = pos.device
        
        # Determine batch size
        if batch is None:
            if pos.dim() == 2:
                batch_size = 1
                batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)
            else:
                batch_size = pos.size(0)
        else:
            batch_size = batch.max().item() + 1
        
        # Initialize agent state if not provided
        if agent_state is None:
            agent_state = self.initial_state(batch_size, device)
        
        # Initialize coherence signal if not provided
        if coherence_signal_prev is None:
            coherence_signal_prev = torch.zeros(batch_size, 1, device=device)
        
        # Step 1: Get context from Agent C (using previous coherence signal)
        # For the first pass, we use a dummy action
        dummy_action = torch.zeros(batch_size, self.num_affordances, device=device)
        agent_state_new, context, rssm_info = self.agent_c(
            agent_state,
            dummy_action,
            coherence_signal_prev,
            obs=None  # We'll update this in the training loop
        )
        
        # Step 2: Apply F_C (Shape → Action)
        affordances = self.F(pos, batch, context)
        
        # Step 3: Apply G_C (Action → Shape)
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
        
        # Step 4: Compute coherence signal
        coherence_signal = self._compute_coherence_signal(pos, reconstructed, batch)
        
        return {
            'affordances': affordances,
            'reconstructed': reconstructed,
            'coherence_signal': coherence_signal,
            'agent_state': agent_state_new,
            'context': context,
            'rssm_info': rssm_info
        }
    
    def _compute_coherence_signal(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between original and reconstructed point clouds.
        
        This is the same as in the base AdjunctionModel.
        """
        # Simplified Chamfer distance
        # reconstructed is already (B, N, 3)
        # original needs to be batched
        
        if original.dim() == 2:
            if batch is None:
                original = original.unsqueeze(0)
            else:
                batch_size = batch.max().item() + 1
                original_batched = []
                for b in range(batch_size):
                    original_batched.append(original[batch == b])
                max_len = max(p.size(0) for p in original_batched)
                original = torch.stack([
                    torch.cat([p, torch.zeros(max_len - p.size(0), 3, device=p.device)], dim=0)
                    for p in original_batched
                ])
        
        B = original.size(0)
        N1 = original.size(1)
        N2 = reconstructed.size(1)
        
        # Pairwise distances
        dist = torch.cdist(original, reconstructed)  # (B, N1, N2)
        
        # Chamfer distance
        dist_forward = dist.min(dim=2)[0].mean(dim=1)  # (B,)
        dist_backward = dist.min(dim=1)[0].mean(dim=1)  # (B,)
        
        chamfer = (dist_forward + dist_backward) / 2
        
        return chamfer.unsqueeze(-1)  # (B, 1)


if __name__ == '__main__':
    # Test Conditional Adjunction Model
    print("Testing Conditional Adjunction Model...")
    
    model = ConditionalAdjunctionModel(
        num_affordances=5,
        num_points=256,
        f_hidden_dim=32,
        g_hidden_dim=64,
        agent_hidden_dim=128,
        agent_latent_dim=32,
        context_dim=64
    )
    
    print(f"Model created")
    print(f"  F_C: {sum(p.numel() for p in model.F.parameters())} parameters")
    print(f"  G_C: {sum(p.numel() for p in model.G.parameters())} parameters")
    print(f"  Agent C: {sum(p.numel() for p in model.agent_c.parameters())} parameters")
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
    print(f"  Coherence signal: {results['coherence_signal']}")
    print(f"  Context shape: {results['context'].shape}")
    
    print("\nConditional Adjunction Model test passed!")
