"""
Value Function for Agent C: Estimating Future Intrinsic Rewards

This module implements a value function V(state) that estimates the expected
cumulative intrinsic reward from a given agent state.

Theoretical foundation:
- Agent C learns "what is valuable" not "what is correct"
- Value = expected future intrinsic rewards (curiosity + competence + novelty)
- This aligns with the "suspension structure" philosophy: value emerges from experience

Training method: Temporal Difference (TD) learning
- V(s_t) ← V(s_t) + α × [R_t + γ × V(s_{t+1}) - V(s_t)]
- Where:
  - R_t: intrinsic reward at time t
  - γ: discount factor
  - α: learning rate

Integration with Agent C:
- Value function takes Agent C's internal state (h, z, valence) as input
- Agent C is trained to maximize V(state) via gradient ascent
- This creates a feedback loop:
  - Agent C → state → V(state) → gradient → update Agent C
  - Agent C learns to produce states with high expected future rewards

Difference from policy gradient:
- Policy gradient: learn probability of actions
- Value-based: learn expected future rewards
- For Agent C (deterministic FiLM), value-based is more natural
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ValueFunction(nn.Module):
    """
    Value function V(state) for Agent C.
    
    Estimates expected cumulative intrinsic reward from a given state.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        valence_dim: int = 32,
        value_hidden_dim: int = 256
    ):
        """
        Args:
            hidden_dim: Dimension of RSSM hidden state h
            latent_dim: Dimension of RSSM latent state z
            valence_dim: Dimension of valence vector
            value_hidden_dim: Hidden dimension for value network
        """
        super().__init__()
        
        # Value network: state → value
        state_dim = hidden_dim + latent_dim + valence_dim
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1)
        )
    
    def forward(
        self,
        agent_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute value V(state).
        
        Args:
            agent_state: Dictionary with:
                - 'h': (B, hidden_dim) deterministic state
                - 'z': (B, latent_dim) stochastic state
                - 'valence': (B, valence_dim) valence vector
        
        Returns:
            value: (B, 1) estimated future intrinsic reward
        """
        h = agent_state['h']
        z = agent_state['z']
        valence = agent_state['valence']
        
        # Concatenate state components
        state = torch.cat([h, z, valence], dim=-1)  # (B, state_dim)
        
        # Compute value
        value = self.value_net(state)  # (B, 1)
        
        return value


class TDLearner:
    """
    Temporal Difference (TD) learner for value function.
    
    Updates value function using TD error:
    TD_error = R_t + γ × V(s_{t+1}) - V(s_t)
    """
    
    def __init__(
        self,
        value_function: ValueFunction,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99
    ):
        """
        Args:
            value_function: Value function to train
            optimizer: Optimizer for value function
            gamma: Discount factor
        """
        self.value_function = value_function
        self.optimizer = optimizer
        self.gamma = gamma
    
    def compute_td_error(
        self,
        state_t: Dict[str, torch.Tensor],
        reward_t: torch.Tensor,
        state_t1: Dict[str, torch.Tensor],
        done: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD error.
        
        Args:
            state_t: State at time t
            reward_t: Intrinsic reward at time t (B,)
            state_t1: State at time t+1
            done: (B,) whether episode is done
        
        Returns:
            td_error: (B, 1) TD error
        """
        # Compute V(s_t)
        value_t = self.value_function(state_t)  # (B, 1)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            value_t1 = self.value_function(state_t1)  # (B, 1)
        
        # TD target: R_t + γ × V(s_{t+1}) × (1 - done)
        td_target = reward_t.unsqueeze(-1) + self.gamma * value_t1 * (1 - done.unsqueeze(-1))
        
        # TD error
        td_error = td_target - value_t
        
        return td_error
    
    def update(
        self,
        state_t: Dict[str, torch.Tensor],
        reward_t: torch.Tensor,
        state_t1: Dict[str, torch.Tensor],
        done: torch.Tensor
    ) -> float:
        """
        Update value function using TD learning.
        
        Returns:
            loss: TD loss value
        """
        # Compute TD error
        td_error = self.compute_td_error(state_t, reward_t, state_t1, done)
        
        # Loss: mean squared TD error
        loss = (td_error ** 2).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return loss.item()


if __name__ == '__main__':
    print("Testing Value Function and TD Learner...")
    
    batch_size = 4
    hidden_dim = 256
    latent_dim = 64
    valence_dim = 32
    
    # Create value function
    value_fn = ValueFunction(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        valence_dim=valence_dim,
        value_hidden_dim=256
    )
    
    print(f"\nValue Function created")
    print(f"  State dim: {hidden_dim + latent_dim + valence_dim}")
    print(f"  Output: scalar value")
    
    # Create TD learner
    optimizer = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
    td_learner = TDLearner(value_fn, optimizer, gamma=0.99)
    
    print(f"\nTD Learner created")
    print(f"  Gamma: {td_learner.gamma}")
    
    # Simulate a trajectory
    print("\nSimulating trajectory...")
    
    device = torch.device('cpu')
    
    # Initial state
    state_t = {
        'h': torch.randn(batch_size, hidden_dim),
        'z': torch.randn(batch_size, latent_dim),
        'valence': torch.ones(batch_size, valence_dim)
    }
    
    # Simulate 10 steps
    for t in range(10):
        # Next state (random for testing)
        state_t1 = {
            'h': torch.randn(batch_size, hidden_dim),
            'z': torch.randn(batch_size, latent_dim),
            'valence': state_t['valence'] + torch.randn(batch_size, valence_dim) * 0.1
        }
        
        # Intrinsic reward (random for testing)
        reward_t = torch.randn(batch_size) * 0.5 + 0.3
        
        # Done flag (last step)
        done = torch.zeros(batch_size)
        if t == 9:
            done = torch.ones(batch_size)
        
        # Update value function
        loss = td_learner.update(state_t, reward_t, state_t1, done)
        
        # Compute current value
        with torch.no_grad():
            value_t = value_fn(state_t)
        
        print(f"  Step {t}: reward={reward_t.mean().item():.4f}, "
              f"value={value_t.mean().item():.4f}, loss={loss:.4f}")
        
        state_t = state_t1
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Value function implements Agent C's learning of 'what is valuable':")
    print("1. V(state) estimates expected future intrinsic rewards")
    print("2. TD learning updates V based on experienced rewards")
    print("3. Agent C can be trained to maximize V(state)")
    print("4. This creates purposeful behavior: seek valuable states")
    print("5. Aligns with suspension structure: value emerges from experience")
    print("="*60)
