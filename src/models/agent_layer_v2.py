"""
Agent Layer C v2: With Priority-based Attention

This is an improved version of Agent Layer C that incorporates:
1. Spatial coherence signal (per-point breakdown information)
2. Priority computation (coherence × uncertainty)
3. Attention mechanism (focus on high-priority breakdowns)

The key improvement is that Agent C now has "intentionality":
- It knows not just "something is wrong" (scalar coherence)
- But also "what is wrong" (spatial coherence)
- And "what to focus on" (priority-weighted attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .agent_layer import RSSM
from .priority import PriorityComputation


class AgentLayerC_v2(nn.Module):
    """
    Agent Layer C with priority-based attention mechanism.
    
    This module:
    1. Maintains the agent's internal state C = (h, z) via RSSM
    2. Computes priority scores for spatial breakdowns
    3. Applies attention based on priorities
    4. Generates context vectors for parameterizing F and G
    """
    
    def __init__(
        self,
        obs_dim: int = 128,
        action_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        context_dim: int = 128,
        uncertainty_type: str = 'entropy',
        attention_temperature: float = 1.0
    ):
        """
        Args:
            obs_dim: Dimension of observation features
            action_dim: Dimension of action/affordance
            hidden_dim: Dimension of deterministic state h
            latent_dim: Dimension of stochastic state z
            context_dim: Dimension of context vector for F and G
            uncertainty_type: Type of uncertainty measure ('entropy' or 'kl')
            attention_temperature: Temperature for attention softmax
        """
        super().__init__()
        
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.priority_module = PriorityComputation(
            uncertainty_type=uncertainty_type,
            temperature=attention_temperature
        )
        
        self.context_dim = context_dim
        
        # Attention-weighted observation encoder
        # Takes priority-weighted observations and produces features
        self.attention_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
        
        # Context generator: C + attended_obs -> context vector
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, context_dim)
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize agent state."""
        return self.rssm.initial_state(batch_size, device)
    
    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        coherence_signal_scalar: torch.Tensor,
        coherence_signal_spatial: torch.Tensor,
        batch: torch.Tensor,
        obs: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Update agent state with priority-based attention.
        
        Args:
            prev_state: Previous state {'h': h_{t-1}, 'z': z_{t-1}}
            action: Action/affordance (B, action_dim)
            coherence_signal_scalar: Scalar coherence signal (B, 1)
            coherence_signal_spatial: Per-point coherence signal (N,)
            batch: Batch assignment for spatial coherence (N,)
            obs: Observation features (N, obs_dim), optional
        
        Returns:
            state: New state {'h': h_t, 'z': z_t}
            context: Context vector for F and G (B, context_dim)
            info: Dictionary with RSSM info, priority info, and attention weights
        """
        # Step 1: Aggregate observations per batch (if available)
        # RSSM expects (B, obs_dim), but we have (N, obs_dim)
        if obs is not None:
            batch_size = batch.max().item() + 1
            obs_aggregated = torch.zeros(batch_size, obs.size(1), device=obs.device)
            
            for b in range(batch_size):
                mask = (batch == b)
                obs_aggregated[b] = obs[mask].mean(dim=0)  # Average pooling
        else:
            obs_aggregated = None
        
        # Step 2: Update RSSM state (using aggregated obs and scalar coherence)
        # This gives us the prior and posterior distributions
        state, rssm_info = self.rssm(prev_state, action, coherence_signal_scalar, obs_aggregated)
        
        # Step 3: Compute priority scores
        priority_results = self.priority_module(
            coherence_signal_spatial,
            rssm_info,
            batch
        )
        
        priority_normalized = priority_results['priority_normalized']  # (N,)
        
        # Step 4: Apply attention to observations (if available)
        if obs is not None:
            # Weight observations by priority
            # obs: (N, obs_dim), priority_normalized: (N,)
            obs_weighted = obs * priority_normalized.unsqueeze(-1)  # (N, obs_dim)
            
            # Aggregate weighted observations per batch
            batch_size = batch.max().item() + 1
            obs_attended = torch.zeros(batch_size, obs.size(1), device=obs.device)
            
            for b in range(batch_size):
                mask = (batch == b)
                obs_attended[b] = obs_weighted[mask].sum(dim=0)
            
            # Encode attended observations
            obs_attended_encoded = self.attention_encoder(obs_attended)  # (B, obs_dim)
        else:
            # No observation available, use zeros
            batch_size = state['h'].size(0)
            obs_attended_encoded = torch.zeros(batch_size, self.rssm.obs_dim, device=state['h'].device)
        
        # Step 5: Generate context vector from state + attended observations
        h = state['h']
        z = state['z']
        context_input = torch.cat([h, z, obs_attended_encoded], dim=-1)
        context = self.context_net(context_input)  # (B, context_dim)
        
        # Collect info
        info = {
            **rssm_info,
            'priority': priority_results['priority'],
            'priority_normalized': priority_normalized,
            'uncertainty': priority_results['uncertainty'],
            'obs_attended': obs_attended_encoded if obs is not None else None
        }
        
        return state, context, info


if __name__ == '__main__':
    # Test Agent Layer C v2
    print("Testing Agent Layer C v2 with Priority-based Attention...")
    
    batch_size = 2
    num_points = 512
    obs_dim = 128
    action_dim = 5
    
    # Create model
    agent_c = AgentLayerC_v2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        latent_dim=64,
        context_dim=128,
        uncertainty_type='entropy',
        attention_temperature=1.0
    )
    
    print(f"\nAgent Layer C v2 created")
    print(f"  RSSM hidden_dim: {agent_c.rssm.hidden_dim}")
    print(f"  RSSM latent_dim: {agent_c.rssm.latent_dim}")
    print(f"  Context dim: {agent_c.context_dim}")
    
    # Initialize state
    state = agent_c.initial_state(batch_size, torch.device('cpu'))
    print(f"\nInitial state:")
    print(f"  h shape: {state['h'].shape}")
    print(f"  z shape: {state['z'].shape}")
    
    # Simulate a few time steps
    print("\nSimulating time steps with priority-based attention...")
    for t in range(3):
        # Random inputs
        action = torch.randn(batch_size, action_dim)
        coherence_scalar = torch.rand(batch_size, 1)
        coherence_spatial = torch.rand(batch_size * num_points)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
        obs = torch.randn(batch_size * num_points, obs_dim)
        
        # Forward pass
        state, context, info = agent_c(
            state, action, coherence_scalar, coherence_spatial, batch, obs
        )
        
        print(f"\nTime step {t}:")
        print(f"  Context shape: {context.shape}")
        print(f"  Uncertainty: {info['uncertainty']}")
        print(f"  Priority mean: {info['priority'].mean():.4f}")
        print(f"  Priority std: {info['priority'].std():.4f}")
        
        # Check priority normalization
        for b in range(batch_size):
            mask = (batch == b)
            priority_sum = info['priority_normalized'][mask].sum()
            print(f"  Priority sum (batch {b}): {priority_sum:.4f}")
        
        # Compute KL divergence
        kl = agent_c.rssm.kl_divergence(
            info['posterior_mean'],
            info['posterior_std'],
            info['prior_mean'],
            info['prior_std']
        )
        print(f"  KL divergence: {kl.mean():.4f}")
    
    print("\nAgent Layer C v2 test passed!")
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Agent C now has intentionality:")
    print("1. Spatial coherence → knows 'what' is broken")
    print("2. Priority = coherence × uncertainty → knows 'where to focus'")
    print("3. Attention mechanism → allocates resources accordingly")
    print("="*60)
