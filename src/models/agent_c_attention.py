"""
Agent C with Attention Selection for Phase 2

This module implements Agent C with the ability to select which part of an object
to focus on, enabling decomposition of unknown composite objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


class AttentionAgent(nn.Module):
    """
    Agent C with attention selection capability.
    
    The agent learns to select which segment of an object to focus on
    by maximizing the intrinsic reward: r(t) = α · (η_whole(t) - η_part(t))
    """
    
    def __init__(
        self,
        state_dim: int = 512,  # Dimension of state representation
        num_segments: int = 8,  # Maximum number of segments per object
        hidden_dim: int = 256,
        eta_history_len: int = 10  # Length of η trajectory history
    ):
        """
        Args:
            state_dim: Dimension of state representation (from F/G)
            num_segments: Maximum number of segments an object can have
            hidden_dim: Hidden layer dimension
            eta_history_len: Number of past η values to remember
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_segments = num_segments
        self.eta_history_len = eta_history_len
        
        # State encoder: encodes current observation and η history
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + eta_history_len * 2, hidden_dim),  # *2 for η_whole and η_part
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Q-network: estimates Q(s, a) for each attention action
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_segments)  # Q-value for each segment
        )
        
        # Initialize η history
        self.eta_whole_history = []
        self.eta_part_history = []
    
    def forward(
        self,
        state: torch.Tensor,
        eta_whole: float,
        eta_part: float
    ) -> torch.Tensor:
        """
        Forward pass to compute Q-values for attention selection.
        
        Args:
            state: State representation from F/G, shape (batch_size, state_dim)
            eta_whole: Reconstruction error for whole object
            eta_part: Reconstruction error for currently focused part
            
        Returns:
            Q-values for each segment, shape (batch_size, num_segments)
        """
        batch_size = state.shape[0]
        
        # Update η history
        self.eta_whole_history.append(eta_whole)
        self.eta_part_history.append(eta_part)
        
        # Keep only recent history
        if len(self.eta_whole_history) > self.eta_history_len:
            self.eta_whole_history = self.eta_whole_history[-self.eta_history_len:]
            self.eta_part_history = self.eta_part_history[-self.eta_history_len:]
        
        # Pad history if not enough
        eta_whole_hist = self.eta_whole_history + [0.0] * (self.eta_history_len - len(self.eta_whole_history))
        eta_part_hist = self.eta_part_history + [0.0] * (self.eta_history_len - len(self.eta_part_history))
        
        # Create η history tensor
        eta_history = torch.FloatTensor(eta_whole_hist + eta_part_hist).unsqueeze(0).repeat(batch_size, 1)
        if state.is_cuda:
            eta_history = eta_history.cuda()
        
        # Concatenate state and η history
        combined_state = torch.cat([state, eta_history], dim=1)
        
        # Encode state
        encoded_state = self.state_encoder(combined_state)
        
        # Compute Q-values
        q_values = self.q_network(encoded_state)
        
        return q_values
    
    def select_action(
        self,
        state: torch.Tensor,
        eta_whole: float,
        eta_part: float,
        epsilon: float = 0.1,
        available_segments: List[int] = None
    ) -> int:
        """
        Select an attention action using ε-greedy policy.
        
        Args:
            state: State representation
            eta_whole: Reconstruction error for whole object
            eta_part: Reconstruction error for currently focused part
            epsilon: Exploration rate
            available_segments: List of available segment indices (None = all available)
            
        Returns:
            Selected segment index
        """
        if available_segments is None:
            available_segments = list(range(self.num_segments))
        
        # ε-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.choice(available_segments)
        
        # Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state, eta_whole, eta_part)
            
            # Mask unavailable segments
            mask = torch.full_like(q_values, float('-inf'))
            mask[0, available_segments] = 0
            masked_q_values = q_values + mask
            
            action = masked_q_values.argmax(dim=1).item()
        
        return action
    
    def reset_history(self):
        """Reset η history (call at the start of each episode)."""
        self.eta_whole_history = []
        self.eta_part_history = []


class DQNTrainer:
    """
    Deep Q-Network trainer for attention selection.
    """
    
    def __init__(
        self,
        agent: AttentionAgent,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        target_update_freq: int = 100,
        alpha: float = 1.0  # Intrinsic reward scaling factor
    ):
        """
        Args:
            agent: AttentionAgent to train
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            target_update_freq: Frequency of target network updates
            alpha: Scaling factor for intrinsic reward r(t) = α · (η_whole - η_part)
        """
        self.agent = agent
        self.target_agent = AttentionAgent(
            state_dim=agent.state_dim,
            num_segments=agent.num_segments,
            hidden_dim=256,
            eta_history_len=agent.eta_history_len
        )
        self.target_agent.load_state_dict(agent.state_dict())
        self.target_agent.eval()
        
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.alpha = alpha
        self.update_count = 0
    
    def compute_intrinsic_reward(self, eta_whole: float, eta_part: float) -> float:
        """
        Compute intrinsic reward based on comparative η.
        
        r(t) = α · (η_whole(t) - η_part(t))
        
        This reward encourages the agent to find parts where η_part < η_whole,
        i.e., parts that are more "known" than the whole object.
        """
        return self.alpha * (eta_whole - eta_part)
    
    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        eta_whole: float,
        eta_part: float,
        next_eta_whole: float,
        next_eta_part: float
    ) -> float:
        """
        Update agent using DQN algorithm.
        
        Returns:
            TD loss value
        """
        # Compute Q(s, a)
        q_values = self.agent(state, eta_whole, eta_part)
        q_value = q_values[0, action]
        
        # Compute target Q-value
        with torch.no_grad():
            if done:
                target_q_value = reward
            else:
                next_q_values = self.target_agent(next_state, next_eta_whole, next_eta_part)
                max_next_q = next_q_values.max(dim=1)[0]
                target_q_value = reward + self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)
        
        # Update agent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
        
        return loss.item()


if __name__ == "__main__":
    # Test AttentionAgent
    agent = AttentionAgent(state_dim=512, num_segments=8)
    
    # Dummy state
    state = torch.randn(1, 512)
    eta_whole = 0.5
    eta_part = 0.3
    
    # Forward pass
    q_values = agent(state, eta_whole, eta_part)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    
    # Action selection
    action = agent.select_action(state, eta_whole, eta_part, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test DQN trainer
    trainer = DQNTrainer(agent, alpha=1.0)
    reward = trainer.compute_intrinsic_reward(eta_whole, eta_part)
    print(f"Intrinsic reward: {reward}")
