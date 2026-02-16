"""
Agent Layer C V3: Emergent Valence (Condition 2)

This is a simplified Agent C that removes the Priority computation module
and allows the agent to learn how to use coherence, uncertainty, and valence
on its own.

Key differences from Agent C:
- NO Priority = coherence × uncertainty × valence calculation
- coherence and uncertainty are fed as observations to RSSM
- valence is fed as part of the RSSM state
- The agent learns its own attention mechanism

This implements the "emergent" approach where we provide the axes
(coherence, uncertainty, valence) but let the agent discover how to use them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .agent_layer import RSSM
from .valence_v3 import ValenceMemoryV3


class AgentCV3(nn.Module):
    """
    Agent Layer C V3 with emergent valence usage.
    
    This module:
    1. Maintains the agent's internal state C = (h, z) via RSSM
    2. Accumulates valence based on (action, Δη) associations
    3. Feeds coherence, uncertainty, and valence directly to RSSM
    4. Learns attention mechanism through training (not prescribed)
    5. Generates context vectors for parameterizing F and G
    """
    
    def __init__(
        self,
        obs_dim: int = 128,
        action_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        context_dim: int = 128,
        valence_dim: int = 32,
        valence_decay: float = 0.1,
        valence_learning_rate: float = 0.1
    ):
        """
        Args:
            obs_dim: Dimension of observation features
            action_dim: Dimension of action/affordance
            hidden_dim: Dimension of deterministic state h
            latent_dim: Dimension of stochastic state z
            context_dim: Dimension of context vector for F and G
            valence_dim: Dimension of valence vector
            valence_decay: Decay rate for valence memory
            valence_learning_rate: Learning rate for valence updates
        """
        super().__init__()
        
        # RSSM with extended observation space
        # obs will include: [coherence_scalar, uncertainty, original_obs]
        self.extended_obs_dim = obs_dim + 2  # +2 for coherence and uncertainty
        
        self.rssm = RSSM(
            obs_dim=self.extended_obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.valence_memory = ValenceMemoryV3(
            valence_dim=valence_dim,
            decay_rate=valence_decay,
            learning_rate=valence_learning_rate
        )
        
        self.context_dim = context_dim
        self.valence_dim = valence_dim
        self.obs_dim = obs_dim
        
        # Context generator: [h, z, valence] -> context vector
        # Note: We include valence in context so it can influence F and G
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + valence_dim, 512),
            nn.ReLU(),
            nn.Linear(512, context_dim)
        )
        
        # Compute uncertainty from RSSM state
        # This is a simple measure: we'll use the entropy of the posterior
        # (already computed in RSSM forward pass)
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize agent state including valence.
        
        Returns:
            state: Dictionary with:
                - 'h': (B, hidden_dim) deterministic state
                - 'z': (B, latent_dim) stochastic state
                - 'valence': (B, valence_dim) valence vector
                - 'coherence_prev': (B, 1) previous coherence (for valence update)
        """
        rssm_state = self.rssm.initial_state(batch_size, device)
        valence = self.valence_memory.get_initial_valence(batch_size, device)
        coherence_prev = torch.zeros(batch_size, 1, device=device)
        
        return {
            **rssm_state,
            'valence': valence,
            'coherence_prev': coherence_prev
        }
    
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
        Update agent state with emergent attention mechanism.
        
        Args:
            prev_state: Previous agent state
            action: Action/affordance (B, action_dim)
            coherence_signal_scalar: Scalar coherence per sample (B, 1)
            coherence_signal_spatial: Spatial coherence per point (N,)
            batch: Batch assignment for spatial coherence (N,)
            obs: Optional observation features (B, obs_dim)
        
        Returns:
            new_state: Updated agent state
            context: Context vector for F and G (B, context_dim)
            info: Dictionary with intermediate values for analysis
        """
        batch_size = action.size(0)
        device = action.device
        
        # Extract previous valence and coherence
        valence_prev = prev_state['valence']  # (B, valence_dim)
        coherence_prev = prev_state['coherence_prev']  # (B, 1)
        
        # 1. Compute uncertainty from previous RSSM state
        # Use entropy of the posterior distribution
        h_prev = prev_state['h']
        z_prev = prev_state['z']
        
        # For uncertainty, we use the norm of z as a simple proxy
        # (In a full implementation, we'd use the std from RSSM)
        uncertainty = z_prev.norm(dim=-1, keepdim=True)  # (B, 1)
        
        # 2. Construct extended observation
        # [coherence_scalar, uncertainty, original_obs]
        if obs is None:
            obs = torch.zeros(batch_size, self.obs_dim, device=device)
        
        extended_obs = torch.cat([
            coherence_signal_scalar,  # (B, 1)
            uncertainty,  # (B, 1)
            obs  # (B, obs_dim)
        ], dim=-1)  # (B, obs_dim + 2)
        
        # 3. Update RSSM state
        new_state_rssm, rssm_info = self.rssm(
            prev_state,
            action,
            coherence_signal_scalar,  # Still pass coherence as separate arg for compatibility
            extended_obs  # Extended observation
        )
        
        # 4. Update Valence Memory based on Δη
        valence_results = self.valence_memory(
            valence_prev,
            coherence_prev,
            coherence_signal_scalar
        )
        
        valence_new = valence_results['valence']
        
        # 5. Generate context vector
        # Combine h, z, and valence
        context_input = torch.cat([
            new_state_rssm['h'],
            new_state_rssm['z'],
            valence_new
        ], dim=-1)
        
        context = self.context_net(context_input)  # (B, context_dim)
        
        # 6. Assemble new state
        new_state = {
            **new_state_rssm,
            'valence': valence_new,
            'coherence_prev': coherence_signal_scalar
        }
        
        # 7. Assemble info dict
        info = {
            **rssm_info,
            'valence': valence_new,
            'delta_eta': valence_results['delta_eta'],
            'uncertainty': uncertainty.squeeze(-1),  # (B,)
            'coherence_scalar': coherence_signal_scalar.squeeze(-1)  # (B,)
        }
        
        return new_state, context, info


if __name__ == '__main__':
    # Test AgentCV3
    print("Testing AgentCV3...")
    
    batch_size = 4
    num_points = 512
    obs_dim = 128
    action_dim = 5
    
    # Create agent
    agent = AgentCV3(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        latent_dim=64,
        context_dim=128,
        valence_dim=32
    )
    
    device = torch.device('cpu')
    
    # Initialize state
    state = agent.initial_state(batch_size, device)
    print(f"\nInitial state keys: {state.keys()}")
    print(f"  h shape: {state['h'].shape}")
    print(f"  z shape: {state['z'].shape}")
    print(f"  valence shape: {state['valence'].shape}")
    
    # Create dummy inputs
    action = torch.randn(batch_size, action_dim, device=device)
    coherence_scalar = torch.rand(batch_size, 1, device=device) * 0.5 + 0.3
    coherence_spatial = torch.rand(batch_size * num_points, device=device)
    batch_assign = torch.repeat_interleave(torch.arange(batch_size), num_points)
    obs = torch.randn(batch_size, obs_dim, device=device)
    
    # Forward pass
    new_state, context, info = agent(
        state, action, coherence_scalar, coherence_spatial, batch_assign, obs
    )
    
    print(f"\nForward pass results:")
    print(f"  Context shape: {context.shape}")
    print(f"  New valence shape: {new_state['valence'].shape}")
    print(f"  Δη: {info['delta_eta']}")
    print(f"  Uncertainty: {info['uncertainty']}")
    
    # Test multiple steps
    print(f"\nTesting 5 steps...")
    state = agent.initial_state(batch_size, device)
    for step in range(5):
        coherence_scalar = torch.rand(batch_size, 1, device=device) * 0.5 + 0.3
        state, context, info = agent(
            state, action, coherence_scalar, coherence_spatial, batch_assign, obs
        )
        print(f"  Step {step}: Δη mean={info['delta_eta'].mean():.4f}, "
              f"valence norm mean={state['valence'].norm(dim=-1).mean():.4f}")
    
    print("\nAgentCV3 test passed!")
