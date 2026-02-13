"""
Agent Layer C: Recurrent State-Space Model (RSSM)

This implements the agent's internal state C, which parameterizes the adjunction F_C ⊣ G_C.
The design is inspired by DreamerV3's RSSM, but simplified for the current scope.

The agent state C consists of:
- h_t: Deterministic recurrent state (captures temporal dependencies)
- z_t: Stochastic latent state (captures uncertainty and exploration)

The state C is updated based on:
1. Previous state C_{t-1}
2. Current observation (shape features from F)
3. Coherence signal (triggers mode transitions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class RSSM(nn.Module):
    """
    Recurrent State-Space Model for Agent Layer C.
    
    This is a simplified version of DreamerV3's RSSM, adapted for the adjunction model.
    
    State update:
        h_t = GRU(h_{t-1}, [z_{t-1}, action_features, coherence_signal])
        z_t ~ p(z_t | h_t)
    """
    
    def __init__(
        self,
        obs_dim: int = 128,           # Dimension of observation features (from F)
        action_dim: int = 5,          # Dimension of action/affordance
        hidden_dim: int = 256,        # Dimension of h (deterministic state)
        latent_dim: int = 64,         # Dimension of z (stochastic state)
        coherence_dim: int = 1        # Dimension of coherence signal
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.coherence_dim = coherence_dim
        
        # GRU for deterministic state h
        # Input: [z_{t-1}, action_features, coherence_signal]
        gru_input_dim = latent_dim + action_dim + coherence_dim
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)
        
        # Posterior: p(z_t | h_t, obs_t)
        # Used during training when observations are available
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and log_std
        )
        
        # Prior: p(z_t | h_t)
        # Used during inference when observations are not available
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and log_std
        )
        
        # Observation encoder (processes raw observations into features)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize the agent state C = (h, z).
        
        Args:
            batch_size: Batch size
            device: Device
        
        Returns:
            state: Dictionary with 'h' and 'z'
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return {'h': h, 'z': z}
    
    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        coherence_signal: torch.Tensor,
        obs: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Update the agent state C.
        
        Args:
            prev_state: Previous state {'h': h_{t-1}, 'z': z_{t-1}}
            action: Action/affordance at time t (B, action_dim)
            coherence_signal: Coherence signal at time t (B, 1)
            obs: Observation at time t (B, obs_dim), optional
        
        Returns:
            state: New state {'h': h_t, 'z': z_t}
            info: Dictionary with prior/posterior distributions
        """
        h_prev = prev_state['h']
        z_prev = prev_state['z']
        
        # Update deterministic state h
        # Ensure coherence_signal is 2D: (B,) -> (B, 1)
        if coherence_signal.dim() == 1:
            coherence_signal = coherence_signal.unsqueeze(-1)
        gru_input = torch.cat([z_prev, action, coherence_signal], dim=-1)
        h = self.gru(gru_input, h_prev)
        
        # Compute prior p(z | h)
        prior_params = self.prior_net(h)
        prior_mean, prior_log_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_log_std) + 1e-5
        
        # If observation is available, compute posterior p(z | h, obs)
        if obs is not None:
            # Debug: print shapes
            # print(f"[RSSM Debug] obs shape: {obs.shape}, h shape: {h.shape}")
            obs_encoded = self.obs_encoder(obs)
            # print(f"[RSSM Debug] obs_encoded shape: {obs_encoded.shape}")
            # Ensure obs_encoded has same batch dimension as h
            if obs_encoded.shape[0] != h.shape[0]:
                raise RuntimeError(f"Batch size mismatch: h={h.shape[0]}, obs_encoded={obs_encoded.shape[0]}")
            posterior_params = self.posterior_net(torch.cat([h, obs_encoded], dim=-1))
            posterior_mean, posterior_log_std = torch.chunk(posterior_params, 2, dim=-1)
            posterior_std = F.softplus(posterior_log_std) + 1e-5
            
            # Sample from posterior during training
            z = self._sample(posterior_mean, posterior_std)
        else:
            # Sample from prior during inference
            posterior_mean = prior_mean
            posterior_std = prior_std
            z = self._sample(prior_mean, prior_std)
        
        state = {'h': h, 'z': z}
        
        info = {
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std
        }
        
        return state, info
    
    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Sample from a Gaussian distribution using the reparameterization trick.
        
        Args:
            mean: Mean of the distribution (B, latent_dim)
            std: Standard deviation (B, latent_dim)
        
        Returns:
            sample: Sampled latent variable (B, latent_dim)
        """
        eps = torch.randn_like(mean)
        return mean + std * eps
    
    def kl_divergence(
        self,
        posterior_mean: torch.Tensor,
        posterior_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        KL(q(z|h,obs) || p(z|h)) = 0.5 * sum(log(σ_p^2/σ_q^2) + (σ_q^2 + (μ_q - μ_p)^2)/σ_p^2 - 1)
        
        Args:
            posterior_mean: Mean of posterior (B, latent_dim)
            posterior_std: Std of posterior (B, latent_dim)
            prior_mean: Mean of prior (B, latent_dim)
            prior_std: Std of prior (B, latent_dim)
        
        Returns:
            kl: KL divergence (B,)
        """
        kl = 0.5 * (
            2 * torch.log(prior_std / posterior_std) +
            (posterior_std.pow(2) + (posterior_mean - prior_mean).pow(2)) / prior_std.pow(2) - 1
        )
        return kl.sum(dim=-1)


class AgentLayerC(nn.Module):
    """
    Complete Agent Layer C with RSSM and context generation.
    
    This module:
    1. Maintains the agent's internal state C = (h, z)
    2. Generates context vectors for parameterizing F and G
    """
    
    def __init__(
        self,
        obs_dim: int = 128,
        action_dim: int = 5,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        context_dim: int = 128  # Dimension of context vector for F and G
    ):
        super().__init__()
        
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.context_dim = context_dim
        
        # Context generator: C -> context vector
        # The context vector will be used to modulate F and G
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, context_dim)
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize agent state."""
        return self.rssm.initial_state(batch_size, device)
    
    def forward(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        coherence_signal: torch.Tensor,
        obs: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Update agent state and generate context vector.
        
        Args:
            prev_state: Previous state
            action: Action/affordance
            coherence_signal: Coherence signal
            obs: Observation (optional)
        
        Returns:
            state: New state
            context: Context vector for F and G (B, context_dim)
            info: RSSM info (prior/posterior distributions)
        """
        # Update RSSM state
        state, info = self.rssm(prev_state, action, coherence_signal, obs)
        
        # Generate context vector
        h = state['h']
        z = state['z']
        context = self.context_net(torch.cat([h, z], dim=-1))
        
        return state, context, info


if __name__ == '__main__':
    # Test Agent Layer C
    print("Testing Agent Layer C...")
    
    batch_size = 4
    obs_dim = 128
    action_dim = 5
    
    # Create model
    agent_c = AgentLayerC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        latent_dim=64,
        context_dim=128
    )
    
    print(f"Agent Layer C created")
    print(f"  RSSM hidden_dim: {agent_c.rssm.hidden_dim}")
    print(f"  RSSM latent_dim: {agent_c.rssm.latent_dim}")
    print(f"  Context dim: {agent_c.context_dim}")
    
    # Initialize state
    state = agent_c.initial_state(batch_size, torch.device('cpu'))
    print(f"\nInitial state:")
    print(f"  h shape: {state['h'].shape}")
    print(f"  z shape: {state['z'].shape}")
    
    # Simulate a few time steps
    print("\nSimulating time steps...")
    for t in range(3):
        # Random inputs
        action = torch.randn(batch_size, action_dim)
        coherence_signal = torch.randn(batch_size, 1)
        obs = torch.randn(batch_size, obs_dim)
        
        # Forward pass
        state, context, info = agent_c(state, action, coherence_signal, obs)
        
        print(f"\nTime step {t}:")
        print(f"  Context shape: {context.shape}")
        print(f"  Prior mean range: [{info['prior_mean'].min():.3f}, {info['prior_mean'].max():.3f}]")
        print(f"  Posterior mean range: [{info['posterior_mean'].min():.3f}, {info['posterior_mean'].max():.3f}]")
        
        # Compute KL divergence
        kl = agent_c.rssm.kl_divergence(
            info['posterior_mean'],
            info['posterior_std'],
            info['prior_mean'],
            info['prior_std']
        )
        print(f"  KL divergence: {kl.mean():.4f}")
    
    print("\nAgent Layer C test passed!")
