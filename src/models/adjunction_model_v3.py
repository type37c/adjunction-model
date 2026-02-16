"""
Adjunction Model V3: F_C ⊣ G_C with Agent C V3 (Emergent Valence)

This version uses AgentCV3 instead of AgentC, implementing the "emergent valence"
approach where coherence, uncertainty, and valence are provided as inputs to the
agent, but the agent learns how to use them on its own.

Key differences from AdjunctionModel:
- Uses AgentCV3 (no Priority computation module)
- Valence is fed directly to RSSM
- Agent learns attention mechanism through training

This is for Condition 2 of the Phase 2.5 Valence Role Experiment.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.functor_f import FunctorF
from src.models.functor_g import FunctorG
from src.models.agent_c_v3 import AgentCV3
from src.models.conditional_adjunction import ConditionalFunctorF, ConditionalFunctorG


class AdjunctionModelV3(nn.Module):
    """
    Full Adjunction Model V3: F_C ⊣ G_C with Agent C V3 (Emergent Valence).
    
    This integrates:
    - Agent Layer C V3 (RSSM + Emergent Valence, no Priority module)
    - Conditional Functor F_C (FiLM-modulated)
    - Conditional Functor G_C (FiLM-modulated)
    - Spatial Coherence Signal computation
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
        valence_learning_rate: float = 0.1
    ):
        """
        Args:
            num_affordances: Number of affordance dimensions
            num_points: Expected number of points per shape
            f_hidden_dim: Hidden dimension for Functor F
            g_hidden_dim: Hidden dimension for Functor G
            agent_hidden_dim: Hidden dimension for Agent C's RSSM
            agent_latent_dim: Latent dimension for Agent C's RSSM
            context_dim: Dimension of context vector for F and G
            valence_dim: Dimension of valence vector
            valence_decay: Decay rate for valence memory
            valence_learning_rate: Learning rate for valence updates
        """
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
        
        # Conditional functors (FiLM-modulated)
        self.F = ConditionalFunctorF(base_f, context_dim)
        self.G = ConditionalFunctorG(base_g, context_dim)
        
        # Agent Layer C V3 (emergent valence usage)
        self.agent_c = AgentCV3(
            obs_dim=f_hidden_dim,
            action_dim=num_affordances,
            hidden_dim=agent_hidden_dim,
            latent_dim=agent_latent_dim,
            context_dim=context_dim,
            valence_dim=valence_dim,
            valence_decay=valence_decay,
            valence_learning_rate=valence_learning_rate
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize agent state."""
        return self.agent_c.initial_state(batch_size, device)
    
    def _compute_coherence_chamfer(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Chamfer Distance for coherence signal.
        
        Args:
            original: (N, 3) original point cloud
            reconstructed: (N, 3) reconstructed point cloud
            batch: (N,) batch assignment
        
        Returns:
            coherence_scalar: (B, 1) average Chamfer distance per sample
            coherence_spatial: (N,) per-point distance to nearest reconstructed point
        """
        device = original.device
        batch_size = batch.max().item() + 1
        
        coherence_scalar = torch.zeros(batch_size, 1, device=device)
        coherence_spatial = torch.zeros(original.size(0), device=device)
        
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() == 0:
                continue
            
            orig_b = original[mask]  # (N_b, 3)
            recon_b = reconstructed[mask]  # (N_b, 3)
            
            # Pairwise distances
            dist = torch.cdist(orig_b.unsqueeze(0), recon_b.unsqueeze(0)).squeeze(0)  # (N_b, N_b)
            
            # Forward: original -> reconstructed
            dist_forward, _ = dist.min(dim=1)  # (N_b,)
            
            # Backward: reconstructed -> original
            dist_backward, _ = dist.min(dim=0)  # (N_b,)
            
            # Chamfer distance (scalar)
            coherence_scalar[b] = (dist_forward.mean() + dist_backward.mean()) / 2
            
            # Spatial coherence (per-point)
            coherence_spatial[mask] = dist_forward
        
        return coherence_scalar, coherence_spatial
    
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
            results: Dictionary with all outputs
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
        
        # Step 1: Get context from Agent C V3
        dummy_action = torch.zeros(batch_size, self.num_affordances, device=device)
        
        # Ensure coherence_signal_prev is 2D: (B,) -> (B, 1)
        if coherence_signal_prev.dim() == 1:
            coherence_signal_prev = coherence_signal_prev.unsqueeze(-1)
        
        # Update agent state and get context
        # AgentCV3 takes coherence and uncertainty as observations
        new_agent_state, context, agent_info = self.agent_c(
            agent_state,
            dummy_action,
            coherence_signal_prev,
            coherence_spatial_prev,
            batch,
            obs=None  # Will use coherence as obs
        )
        
        # Step 2: Apply F (Shape → Affordance) with context
        affordances, f_features = self.F(pos, batch, context)
        
        # Step 3: Apply G (Affordance → Shape) with context
        reconstructed_pos = self.G(affordances, batch, context)
        
        # Step 4: Compute coherence signals using Chamfer Distance
        # Unit η: Chamfer(pos, G(F(pos)))
        coherence_signal, coherence_spatial = self._compute_coherence_chamfer(
            pos, reconstructed_pos, batch
        )
        
        # Step 5: Compute counit ε: ||affordances - F(G(affordances))||
        reconstructed_affordances, _ = self.F(reconstructed_pos, batch, context)
        counit_spatial = torch.norm(affordances - reconstructed_affordances, dim=-1)  # (N,)
        
        # Aggregate to scalar per sample
        counit_signal = torch.zeros(batch_size, 1, device=device)
        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() > 0:
                counit_signal[b] = counit_spatial[mask].mean()
        
        # Return all results
        return {
            'affordances': affordances,
            'reconstructed_pos': reconstructed_pos,
            'context': context,
            'agent_state': new_agent_state,
            'coherence_signal': coherence_signal,
            'coherence_spatial': coherence_spatial,
            'counit_signal': counit_signal,
            'counit_spatial': counit_spatial,
            'agent_info': agent_info,
            'f_features': f_features
        }


if __name__ == '__main__':
    # Test AdjunctionModelV3
    print("Testing AdjunctionModelV3...")
    
    batch_size = 4
    num_points = 256
    num_affordances = 5
    
    # Create model
    model = AdjunctionModelV3(
        num_affordances=num_affordances,
        num_points=num_points,
        f_hidden_dim=64,
        g_hidden_dim=128,
        agent_hidden_dim=256,
        agent_latent_dim=64,
        context_dim=128,
        valence_dim=32
    )
    
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy input
    pos = torch.randn(batch_size * num_points, 3, device=device)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_points).to(device)
    
    print(f"\nInput:")
    print(f"  pos shape: {pos.shape}")
    print(f"  batch shape: {batch.shape}")
    print(f"  batch_size: {batch_size}")
    
    # Initialize agent state
    agent_state = model.initial_state(batch_size, device)
    print(f"\nInitial agent state keys: {agent_state.keys()}")
    
    # Forward pass
    results = model(pos, batch, agent_state)
    
    print(f"\nForward pass results:")
    print(f"  affordances shape: {results['affordances'].shape}")
    print(f"  reconstructed_pos shape: {results['reconstructed_pos'].shape}")
    print(f"  context shape: {results['context'].shape}")
    print(f"  coherence_signal shape: {results['coherence_signal'].shape}")
    print(f"  coherence_signal values: {results['coherence_signal'].squeeze()}")
    print(f"  counit_signal shape: {results['counit_signal'].shape}")
    print(f"  counit_signal values: {results['counit_signal'].squeeze()}")
    
    # Test multiple steps
    print(f"\nTesting 3 steps...")
    state = model.initial_state(batch_size, device)
    for step in range(3):
        results = model(pos, batch, state, results['coherence_signal'], results['coherence_spatial'])
        state = results['agent_state']
        print(f"  Step {step}: η mean={results['coherence_signal'].mean():.4f}, "
              f"ε mean={results['counit_signal'].mean():.4f}")
    
    print("\nAdjunctionModelV3 test passed!")
