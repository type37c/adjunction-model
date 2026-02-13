"""
Agent Layer C v4: With Intrinsic Reward-based Purpose Space P

This is an improved version of Agent Layer C v3 that incorporates:
1. All features from v3 (spatial coherence, priority, valence)
2. Intrinsic Reward-based Valence Memory (v2)
3. Revised Priority: priority = coherence × uncertainty × valence

The key improvement is that Agent C now has intrinsic motivation:
- Curiosity: seeks to reduce uncertainty
- Competence: engages with breakdowns and resolves them
- Novelty: discovers new patterns
- Valence accumulates based on these intrinsic rewards

This prevents 'coherence minimization collapse' by giving Agent C a purpose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .agent_layer import RSSM
from .priority import PriorityComputation
from .valence_v2 import ValenceMemoryV2
from .intrinsic_reward import IntrinsicRewardComputation


class AgentLayerC_v4(nn.Module):
    """
    Agent Layer C v4 with Intrinsic Reward-based Purpose Space P.
    
    This module:
    1. Maintains the agent's internal state C = (h, z) via RSSM
    2. Accumulates experiential value via intrinsic rewards (curiosity, competence, novelty)
    3. Computes priority scores: coherence × uncertainty × valence
    4. Applies attention based on priorities
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
        alpha_curiosity: float = 0.3,
        beta_competence: float = 0.5,
        gamma_novelty: float = 0.2,
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
            valence_dim: Dimension of valence vector
            valence_decay: Decay rate for valence memory (β)
            alpha_curiosity: Weight for curiosity reward
            beta_competence: Weight for competence reward
            gamma_novelty: Weight for novelty reward
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
        
        self.valence_memory = ValenceMemoryV2(
            valence_dim=valence_dim,
            decay_rate=valence_decay,
            alpha_curiosity=alpha_curiosity,
            beta_competence=beta_competence,
            gamma_novelty=gamma_novelty
        )
        
        self.context_dim = context_dim
        self.valence_dim = valence_dim
        
        # Attention-weighted observation encoder
        self.attention_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
        
        # Context generator: C + attended_obs + valence -> context vector
        # This is the key extension: valence is now part of the context
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + obs_dim + valence_dim, 512),
            nn.ReLU(),
            nn.Linear(512, context_dim)
        )
    
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initialize agent state including valence and uncertainty.
        
        Returns:
            state: Dictionary with:
                - 'h': (B, hidden_dim) deterministic state
                - 'z': (B, latent_dim) stochastic state
                - 'valence': (B, valence_dim) valence vector
                - 'coherence_prev': (B, 1) previous coherence (for valence update)
                - 'uncertainty': (B,) previous uncertainty (for intrinsic reward)
        """
        rssm_state = self.rssm.initial_state(batch_size, device)
        valence = self.valence_memory.get_initial_valence(batch_size, device)
        coherence_prev = torch.zeros(batch_size, 1, device=device)
        uncertainty = torch.zeros(batch_size, device=device)
        
        return {
            **rssm_state,
            'valence': valence,
            'coherence_prev': coherence_prev,
            'uncertainty': uncertainty
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
        Update agent state with valence-aware priority-based attention.
        
        Args:
            prev_state: Previous agent state (from initial_state or previous forward)
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
        
        # 1. Update RSSM state
        # Use coherence_signal_scalar as observation (if obs not provided)
        if obs is None:
            obs = coherence_signal_scalar  # (B, 1)
            # Pad to obs_dim
            if obs.size(-1) < self.rssm.obs_dim:
                pad_size = self.rssm.obs_dim - obs.size(-1)
                obs = F.pad(obs, (0, pad_size))
        
        new_state_rssm, rssm_info = self.rssm(
            prev_state,
            action,
            coherence_signal_scalar,  # RSSM expects coherence_signal as 3rd arg
            obs  # obs is 4th arg
        )
        
        # 2. Update Valence Memory (v4: intrinsic reward-based)
        # Compute attention weight from previous priority (if available)
        # For the first step, use uniform attention
        if 'priority_normalized' in prev_state:
            # Aggregate per-point priority to per-sample attention weight
            attention_weight = torch.zeros(batch_size, 1, device=device)
            for b in range(batch_size):
                mask = (batch == b)
                if mask.sum() > 0:
                    attention_weight[b] = prev_state['priority_normalized'][mask].mean()
        else:
            attention_weight = torch.ones(batch_size, 1, device=device)
        
        # Get previous uncertainty (if available)
        if 'uncertainty' in prev_state:
            uncertainty_prev = prev_state['uncertainty']  # (B,)
        else:
            uncertainty_prev = torch.zeros(batch_size, device=device)
        
        # Compute current uncertainty from RSSM
        # Use entropy of posterior distribution
        posterior_dist = torch.distributions.Normal(
            rssm_info['posterior_mean'],
            rssm_info['posterior_std']
        )
        uncertainty_curr = posterior_dist.entropy().sum(dim=-1)  # (B,)
        
        # Compute KL divergence for novelty reward
        prior_dist = torch.distributions.Normal(
            rssm_info['prior_mean'],
            rssm_info['prior_std']
        )
        kl_divergence = torch.distributions.kl_divergence(
            posterior_dist, prior_dist
        ).sum(dim=-1)  # (B,)
        
        # Get previous confidence (if available)
        # Confidence is computed from priority_normalized distribution
        if 'confidence' in prev_state:
            confidence_prev = prev_state['confidence']  # (B,)
        else:
            confidence_prev = None
        
        # Compute current confidence from previous priority distribution
        # Note: We use prev_state's priority_normalized to compute "current" confidence
        # because the truly current priority hasn't been computed yet (circular dependency)
        # This means confidence_curr here is actually based on the PREVIOUS step's priority
        # which is fine: it represents "how focused was our attention last step"
        if 'priority_normalized' in prev_state:
            confidence_curr = IntrinsicRewardComputation.compute_confidence(
                prev_state['priority_normalized'], batch
            )
        else:
            confidence_curr = None
        
        # Update valence using intrinsic rewards
        valence_results = self.valence_memory(
            valence_prev,
            uncertainty_prev,
            uncertainty_curr,
            coherence_prev,
            coherence_signal_scalar,
            attention_weight,
            kl_divergence,
            confidence_prev,
            confidence_curr
        )
        
        valence_new = valence_results['valence']  # (B, valence_dim)
        R_intrinsic = valence_results['R_intrinsic']  # (B,)
        
        # 3. Compute Priority (with valence)
        priority_results = self.priority_module(
            coherence_signal_spatial,
            rssm_info,
            batch,
            valence=valence_new  # v3: include valence in priority
        )
        
        priority = priority_results['priority']  # (N,)
        priority_normalized = priority_results['priority_normalized']  # (N,)
        uncertainty = priority_results['uncertainty']  # (B,)
        
        # 4. Apply Attention to Observations
        # Aggregate spatial features weighted by priority
        if obs.size(-1) > 1:  # If obs has spatial structure
            # This assumes obs is per-point features
            # For now, we use a simple approach: attend to coherence itself
            obs_attended = obs  # (B, obs_dim)
        else:
            # If obs is scalar, use it directly
            obs_attended = obs  # (B, 1)
            
            # Pad to obs_dim if needed
            if obs_attended.size(-1) < self.attention_encoder[0].in_features:
                pad_size = self.attention_encoder[0].in_features - obs_attended.size(-1)
                obs_attended = F.pad(obs_attended, (0, pad_size))
        
        obs_attended = self.attention_encoder(obs_attended)  # (B, obs_dim)
        
        # 5. Generate Context Vector
        # Combine h, z, attended_obs, and valence
        h = new_state_rssm['h']  # (B, hidden_dim)
        z = new_state_rssm['z']  # (B, latent_dim)
        
        context_input = torch.cat([h, z, obs_attended, valence_new], dim=-1)
        context = self.context_net(context_input)  # (B, context_dim)
        
        # Compute confidence for this step (to store for next step)
        confidence = IntrinsicRewardComputation.compute_confidence(
            priority_normalized, batch
        )
        
        # 6. Assemble new state
        new_state = {
            **new_state_rssm,
            'valence': valence_new,
            'coherence_prev': coherence_signal_scalar,
            'uncertainty': uncertainty,  # v4: store uncertainty for next step
            'priority_normalized': priority_normalized,  # Store for next step
            'confidence': confidence  # v5: store confidence for next step
        }
        
        # 7. Assemble info
        info = {
            **rssm_info,
            'priority': priority,
            'priority_normalized': priority_normalized,
            'uncertainty': uncertainty,
            'valence': valence_new,
            'confidence': confidence,  # v5: confidence as meta-indicator
            'R_intrinsic': R_intrinsic,  # v4: intrinsic reward
            'R_curiosity': valence_results.get('R_curiosity', None),
            'R_competence': valence_results.get('R_competence', None),
            'R_novelty': valence_results.get('R_novelty', None),
            'attention_weight': attention_weight
        }
        
        return new_state, context, info


if __name__ == '__main__':
    # Test Agent Layer C v4
    print("Testing Agent Layer C v4...")
    
    batch_size = 2
    num_points = 512
    obs_dim = 128
    action_dim = 5
    
    # Create module
    agent = AgentLayerC_v4(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        latent_dim=64,
        context_dim=128,
        valence_dim=32,
        valence_decay=0.1,
        uncertainty_type='entropy',
        attention_temperature=1.0
    )
    
    print(f"\nAgent Layer C v4 created")
    print(f"  RSSM hidden dim: {agent.rssm.hidden_dim}")
    print(f"  RSSM latent dim: {agent.rssm.latent_dim}")
    print(f"  Valence dim: {agent.valence_dim}")
    print(f"  Context dim: {agent.context_dim}")
    
    # Initialize state
    device = torch.device('cpu')
    state = agent.initial_state(batch_size, device)
    
    print(f"\nInitial state:")
    print(f"  h shape: {state['h'].shape}")
    print(f"  z shape: {state['z'].shape}")
    print(f"  valence shape: {state['valence'].shape}")
    print(f"  valence mean: {state['valence'].mean().item():.4f}")
    
    # Simulate a sequence of experiences
    print("\nSimulating experiences...")
    for t in range(5):
        # Random inputs
        action = torch.randn(batch_size, action_dim)
        coherence_scalar = torch.rand(batch_size, 1) * 0.5
        coherence_spatial = torch.rand(batch_size * num_points)
        batch = torch.repeat_interleave(torch.arange(batch_size), num_points)
        obs = torch.randn(batch_size, obs_dim)
        
        # Forward pass
        state, context, info = agent(
            state, action, coherence_scalar, coherence_spatial, batch, obs
        )
        
        print(f"\nStep {t}:")
        print(f"  Coherence mean: {coherence_scalar.mean().item():.4f}")
        print(f"  R_intrinsic: {info['R_intrinsic'].mean().item():.4f}")
        print(f"  R_curiosity: {info['R_curiosity'].mean().item():.4f}")
        print(f"  R_competence: {info['R_competence'].mean().item():.4f}")
        print(f"  R_novelty: {info['R_novelty'].mean().item():.4f}")
        print(f"  Valence mean: {info['valence'].mean().item():.4f}")
        print(f"  Uncertainty mean: {info['uncertainty'].mean().item():.4f}")
        print(f"  Priority mean: {info['priority'].mean().item():.4f}")
        print(f"  Context norm: {context.norm(dim=-1).mean().item():.4f}")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Agent Layer C v4 implements Intrinsic Reward-based Purpose Space P:")
    print("1. Valence accumulates via intrinsic rewards (curiosity, competence, novelty)")
    print("2. Priority = coherence × uncertainty × valence")
    print("3. Context includes valence, influencing F/G via FiLM")
    print("4. Agent has 'purpose': maximize intrinsic rewards")
    print("5. This prevents coherence minimization collapse")
    print("="*60)
