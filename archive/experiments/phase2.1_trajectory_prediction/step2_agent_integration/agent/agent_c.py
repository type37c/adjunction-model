"""
Agent C: LSTM-based Actor-Critic

Implements the agent that uses F/G affordance features or baseline features
to learn reaching behavior.
"""

import torch
import torch.nn as nn
import numpy as np


class AgentC(nn.Module):
    """
    Agent C: LSTM-based Actor-Critic
    
    Takes observation (robot state + affordance features or baseline features)
    and outputs action distribution and value estimate.
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128, use_lstm=True):
        """
        Initialize Agent C
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            use_lstm: Whether to use LSTM (vs simple MLP)
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        
        if use_lstm:
            # LSTM for temporal dependencies
            self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        else:
            # Simple MLP
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, hidden=None):
        """
        Forward pass
        
        Args:
            obs: Observation tensor (B, T, obs_dim) or (B, obs_dim)
            hidden: LSTM hidden state (optional)
        
        Returns:
            action_mean: Mean of action distribution (B, T, action_dim) or (B, action_dim)
            action_logstd: Log std of action distribution (action_dim,)
            value: Value estimate (B, T, 1) or (B, 1)
            hidden: New LSTM hidden state (if use_lstm)
        """
        if self.use_lstm:
            # Ensure obs is 3D (B, T, obs_dim)
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # (B, obs_dim) -> (B, 1, obs_dim)
            
            # LSTM forward
            lstm_out, hidden = self.lstm(obs, hidden)  # (B, T, hidden_dim)
            features = lstm_out
        else:
            # MLP forward
            features = self.encoder(obs)  # (B, hidden_dim)
            if features.dim() == 2:
                features = features.unsqueeze(1)  # (B, hidden_dim) -> (B, 1, hidden_dim)
        
        # Actor
        action_mean = self.actor_mean(features)  # (B, T, action_dim)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Critic
        value = self.critic(features)  # (B, T, 1)
        
        return action_mean, action_logstd, value, hidden
    
    def get_action(self, obs, hidden=None, deterministic=False):
        """
        Sample action from policy
        
        Args:
            obs: Observation tensor (B, obs_dim) or (obs_dim,)
            hidden: LSTM hidden state (optional)
            deterministic: Whether to use deterministic action (mean)
        
        Returns:
            action: Sampled action (B, action_dim) or (action_dim,)
            log_prob: Log probability of action
            value: Value estimate
            hidden: New LSTM hidden state
        """
        # Ensure obs is batched
        single_obs = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)
            single_obs = True
        
        # Forward pass
        action_mean, action_logstd, value, hidden = self.forward(obs, hidden)
        
        # Remove time dimension if present
        if action_mean.dim() == 3:
            action_mean = action_mean.squeeze(1)  # (B, 1, action_dim) -> (B, action_dim)
            action_logstd = action_logstd.squeeze(1)
            value = value.squeeze(1)  # (B, 1, 1) -> (B, 1)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian distribution
            action_std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        # Compute log probability
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # (B, 1)
        
        # Remove batch dimension if single observation
        if single_obs:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(self, obs, actions, hidden=None):
        """
        Evaluate actions (for PPO update)
        
        Args:
            obs: Observation tensor (B, T, obs_dim)
            actions: Action tensor (B, T, action_dim)
            hidden: LSTM hidden state (optional)
        
        Returns:
            log_probs: Log probabilities of actions (B, T)
            values: Value estimates (B, T)
            entropy: Entropy of action distribution (B, T)
        """
        # Forward pass
        action_mean, action_logstd, value, _ = self.forward(obs, hidden)
        
        # Compute log probabilities
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # (B, T)
        
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)  # (B, T)
        
        # Remove last dimension from value
        value = value.squeeze(-1)  # (B, T, 1) -> (B, T)
        
        return log_probs, value, entropy


class AgentCWithFG(nn.Module):
    """
    Agent C with F/G affordance features
    
    Combines F/G affordance features with robot state.
    """
    
    def __init__(self, robot_state_dim, affordance_dim, action_dim, hidden_dim=128):
        """
        Initialize Agent C with F/G
        
        Args:
            robot_state_dim: Robot state dimension (joint angles + velocities)
            affordance_dim: Affordance feature dimension from F/G
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.robot_state_dim = robot_state_dim
        self.affordance_dim = affordance_dim
        self.action_dim = action_dim
        
        # Combined observation dimension
        obs_dim = robot_state_dim + affordance_dim
        
        # Agent C
        self.agent_c = AgentC(obs_dim, action_dim, hidden_dim, use_lstm=True)
    
    def forward(self, robot_state, affordance_features, hidden=None):
        """
        Forward pass
        
        Args:
            robot_state: Robot state tensor (B, T, robot_state_dim) or (B, robot_state_dim)
            affordance_features: Affordance features from F/G (B, T, affordance_dim) or (B, affordance_dim)
            hidden: LSTM hidden state (optional)
        
        Returns:
            action_mean, action_logstd, value, hidden
        """
        # Concatenate robot state and affordance features
        obs = torch.cat([robot_state, affordance_features], dim=-1)
        
        return self.agent_c(obs, hidden)
    
    def get_action(self, robot_state, affordance_features, hidden=None, deterministic=False):
        """Sample action"""
        obs = torch.cat([robot_state, affordance_features], dim=-1)
        return self.agent_c.get_action(obs, hidden, deterministic)
    
    def evaluate_actions(self, robot_state, affordance_features, actions, hidden=None):
        """Evaluate actions"""
        obs = torch.cat([robot_state, affordance_features], dim=-1)
        return self.agent_c.evaluate_actions(obs, actions, hidden)


class AgentCBaseline(nn.Module):
    """
    Agent C baseline (without F/G features)
    
    Uses robot state + object position directly.
    """
    
    def __init__(self, robot_state_dim, object_pos_dim, action_dim, hidden_dim=128):
        """
        Initialize Agent C baseline
        
        Args:
            robot_state_dim: Robot state dimension (joint angles + velocities)
            object_pos_dim: Object position dimension (3)
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.robot_state_dim = robot_state_dim
        self.object_pos_dim = object_pos_dim
        self.action_dim = action_dim
        
        # Combined observation dimension
        obs_dim = robot_state_dim + object_pos_dim
        
        # Agent C
        self.agent_c = AgentC(obs_dim, action_dim, hidden_dim, use_lstm=True)
    
    def forward(self, robot_state, object_position, hidden=None):
        """
        Forward pass
        
        Args:
            robot_state: Robot state tensor (B, T, robot_state_dim) or (B, robot_state_dim)
            object_position: Object position tensor (B, T, object_pos_dim) or (B, object_pos_dim)
            hidden: LSTM hidden state (optional)
        
        Returns:
            action_mean, action_logstd, value, hidden
        """
        # Concatenate robot state and object position
        obs = torch.cat([robot_state, object_position], dim=-1)
        
        return self.agent_c(obs, hidden)
    
    def get_action(self, robot_state, object_position, hidden=None, deterministic=False):
        """Sample action"""
        obs = torch.cat([robot_state, object_position], dim=-1)
        return self.agent_c.get_action(obs, hidden, deterministic)
    
    def evaluate_actions(self, robot_state, object_position, actions, hidden=None):
        """Evaluate actions"""
        obs = torch.cat([robot_state, object_position], dim=-1)
        return self.agent_c.evaluate_actions(obs, actions, hidden)
