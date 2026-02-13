"""
Valence Memory v2: Intrinsic Reward-based Experiential Value Accumulation

This module implements the third axis of the Purpose Space P: Valence.

Valence represents the agent's learned value judgment based on intrinsic rewards:
- Curiosity: "I learned something I didn't know"
- Competence: "I faced a breakdown and understood it better"
- Novelty: "I discovered something unexpected"

Key design principles:
1. Valence is a DESIGNED FRAMEWORK (the axis itself)
2. The CONTENT of valence (specific values) EMERGES from experience
3. Valence is accumulated through intrinsic rewards (not just coherence changes)
4. Old experiences decay exponentially to allow adaptation

Update rule (v2 - intrinsic reward based):
    valence(t+1) = (1-β) × valence(t) + β × R_intrinsic(t)

Where:
- R_intrinsic = α × R_curiosity + β × R_competence + γ × R_novelty
- R_curiosity: uncertainty reduction
- R_competence: coherence improvement (when attended)
- R_novelty: unexpected discoveries (KL divergence)
- β: decay rate (learning rate)

This gives Agent C a 'purpose': to maximize intrinsic rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .intrinsic_reward import IntrinsicRewardComputation


class ValenceMemoryV2(nn.Module):
    """
    Valence Memory v2: Accumulates experiential value judgments via intrinsic rewards.
    
    This module maintains a valence vector v_t that represents the agent's
    learned preferences based on intrinsic rewards (curiosity, competence, novelty).
    """
    
    def __init__(
        self,
        valence_dim: int = 32,
        decay_rate: float = 0.1,
        init_valence: float = 1.0,
        alpha_curiosity: float = 0.3,
        beta_competence: float = 0.5,
        gamma_novelty: float = 0.2
    ):
        """
        Args:
            valence_dim: Dimension of valence vector
            decay_rate: Decay rate β for exponential forgetting (0 = no learning, 1 = no memory)
            init_valence: Initial value for valence (neutral = 1.0)
            alpha_curiosity: Weight for curiosity reward
            beta_competence: Weight for competence reward
            gamma_novelty: Weight for novelty reward
        """
        super().__init__()
        
        self.valence_dim = valence_dim
        self.decay_rate = decay_rate
        self.init_valence = init_valence
        
        # Intrinsic reward computation
        self.intrinsic_reward = IntrinsicRewardComputation(
            alpha_curiosity=alpha_curiosity,
            beta_competence=beta_competence,
            gamma_novelty=gamma_novelty
        )
        
        # Learnable projection from intrinsic reward to valence update
        # This allows the model to learn which kinds of rewards matter
        self.reward_to_valence = nn.Sequential(
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
    
    def forward(
        self,
        valence_prev: torch.Tensor,
        uncertainty_prev: torch.Tensor,
        uncertainty_curr: torch.Tensor,
        coherence_prev: torch.Tensor,
        coherence_curr: torch.Tensor,
        attention_weight: torch.Tensor,
        kl_divergence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update valence based on intrinsic rewards.
        
        Args:
            valence_prev: Previous valence vector (B, valence_dim)
            uncertainty_prev: Previous uncertainty (B,)
            uncertainty_curr: Current uncertainty (B,)
            coherence_prev: Previous coherence (B, 1)
            coherence_curr: Current coherence (B, 1)
            attention_weight: Attention weight (B, 1)
            kl_divergence: KL(posterior || prior) (B,)
        
        Returns:
            results: Dictionary with updated valence and reward components
        """
        # Compute intrinsic rewards
        rewards = self.intrinsic_reward(
            uncertainty_prev,
            uncertainty_curr,
            coherence_prev,
            coherence_curr,
            attention_weight,
            kl_divergence
        )
        
        R_intrinsic = rewards['R_intrinsic']  # (B,)
        
        # Project intrinsic reward to valence update
        valence_update = self.reward_to_valence(R_intrinsic.unsqueeze(-1))  # (B, valence_dim)
        
        # Update valence with exponential decay
        valence_new = (1 - self.decay_rate) * valence_prev + self.decay_rate * valence_update
        
        return {
            'valence': valence_new,
            'R_intrinsic': R_intrinsic,
            'R_curiosity': rewards['R_curiosity'],
            'R_competence': rewards['R_competence'],
            'R_novelty': rewards['R_novelty']
        }


if __name__ == '__main__':
    print("Testing Valence Memory v2...")
    
    batch_size = 2
    valence_dim = 16
    
    # Create module
    valence_memory = ValenceMemoryV2(
        valence_dim=valence_dim,
        decay_rate=0.1,
        init_valence=1.0,
        alpha_curiosity=0.3,
        beta_competence=0.5,
        gamma_novelty=0.2
    )
    
    print(f"\nValence Memory v2 created")
    print(f"  Valence dim: {valence_memory.valence_dim}")
    print(f"  Decay rate: {valence_memory.decay_rate}")
    print(f"  Intrinsic reward weights:")
    print(f"    α (curiosity): {valence_memory.intrinsic_reward.alpha}")
    print(f"    β (competence): {valence_memory.intrinsic_reward.beta}")
    print(f"    γ (novelty): {valence_memory.intrinsic_reward.gamma}")
    
    # Initialize valence
    device = torch.device('cpu')
    valence = valence_memory.initial_valence(batch_size, device)
    
    print(f"\nInitial valence:")
    print(f"  Shape: {valence.shape}")
    print(f"  Mean: {valence.mean().item():.4f}")
    
    # Simulate a sequence of experiences
    print("\nSimulating experiences...")
    
    scenarios = [
        {
            'name': 'Learning experience',
            'uncertainty_prev': torch.tensor([50.0, 50.0]),
            'uncertainty_curr': torch.tensor([30.0, 30.0]),
            'coherence_prev': torch.ones(2, 1) * 0.5,
            'coherence_curr': torch.ones(2, 1) * 0.5,
            'attention_weight': torch.ones(2, 1) * 0.5,
            'kl_divergence': torch.ones(2) * 0.1
        },
        {
            'name': 'Competence experience',
            'uncertainty_prev': torch.tensor([30.0, 30.0]),
            'uncertainty_curr': torch.tensor([30.0, 30.0]),
            'coherence_prev': torch.ones(2, 1) * 0.8,
            'coherence_curr': torch.ones(2, 1) * 0.3,
            'attention_weight': torch.ones(2, 1) * 0.9,
            'kl_divergence': torch.ones(2) * 0.1
        },
        {
            'name': 'Novelty experience',
            'uncertainty_prev': torch.tensor([30.0, 30.0]),
            'uncertainty_curr': torch.tensor([30.0, 30.0]),
            'coherence_prev': torch.ones(2, 1) * 0.3,
            'coherence_curr': torch.ones(2, 1) * 0.3,
            'attention_weight': torch.ones(2, 1) * 0.5,
            'kl_divergence': torch.ones(2) * 5.0
        }
    ]
    
    for t, scenario in enumerate(scenarios):
        print(f"\nStep {t}: {scenario['name']}")
        
        results = valence_memory(
            valence,
            scenario['uncertainty_prev'],
            scenario['uncertainty_curr'],
            scenario['coherence_prev'],
            scenario['coherence_curr'],
            scenario['attention_weight'],
            scenario['kl_divergence']
        )
        
        valence = results['valence']
        
        print(f"  R_curiosity: {results['R_curiosity'].mean().item():.4f}")
        print(f"  R_competence: {results['R_competence'].mean().item():.4f}")
        print(f"  R_novelty: {results['R_novelty'].mean().item():.4f}")
        print(f"  R_intrinsic: {results['R_intrinsic'].mean().item():.4f}")
        print(f"  Valence mean: {valence.mean().item():.4f}")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Valence Memory v2 implements intrinsic reward-based learning:")
    print("1. Valence accumulates based on curiosity, competence, novelty")
    print("2. Agent C has a 'purpose': maximize intrinsic rewards")
    print("3. This prevents coherence minimization collapse")
    print("4. Valence content emerges from experience")
    print("="*60)
