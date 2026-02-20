"""
Proposal Agent: Generate and Filter Action Candidates

This module implements Agent C with proposal generation mechanism.

Theory:
Instead of directly outputting actions, Agent C generates multiple candidates
and filters them using the adjunction structure:
1. Generate N action proposals (stochastic sampling from policy)
2. Filter by ε: Keep only "meaningful" actions (low ε)
3. Select by η: Choose action with lowest η for current shape

This allows the agent to explore the action space while respecting the
coherence structure defined by F/G.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')


class ProposalAgent(nn.Module):
    """
    Agent C with proposal generation and filtering.
    
    Generates multiple action candidates and filters them using η and ε.
    """
    
    def __init__(
        self,
        observation_dim: int = 16,
        action_dim: int = 3,
        hidden_dim: int = 128,
        num_proposals: int = 10,
        epsilon_threshold: float = 0.1
    ):
        """
        Args:
            observation_dim: Observation dimension (affordance dim)
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_proposals: Number of action proposals to generate
            epsilon_threshold: Threshold for ε filtering
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.epsilon_threshold = epsilon_threshold
        
        # Policy network: outputs mean and log_std for action distribution
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        # Value network (for PPO or other RL algorithms)
        self.value_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: output action distribution parameters.
        
        Args:
            observation: Observation (B, observation_dim)
        
        Returns:
            mean: Action mean (B, action_dim)
            std: Action std (B, action_dim)
        """
        output = self.policy_net(observation)
        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim:]
        std = torch.exp(log_std).clamp(min=1e-3, max=1.0)
        
        return mean, std
    
    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get value estimate."""
        return self.value_net(observation)
    
    def generate_proposals(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Generate multiple action proposals.
        
        Args:
            observation: Observation (B, observation_dim)
            deterministic: If True, return mean action repeated
        
        Returns:
            proposals: Action proposals (B, num_proposals, action_dim)
        """
        mean, std = self.forward(observation)
        B = observation.shape[0]
        
        if deterministic:
            # Return mean action repeated
            proposals = mean.unsqueeze(1).repeat(1, self.num_proposals, 1)
        else:
            # Sample from Gaussian distribution
            proposals = torch.zeros(B, self.num_proposals, self.action_dim, device=observation.device)
            for i in range(self.num_proposals):
                noise = torch.randn_like(mean)
                proposals[:, i] = mean + std * noise
        
        return proposals
    
    def filter_by_epsilon(
        self,
        proposals: torch.Tensor,
        fg_model: nn.Module
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Filter proposals by ε (action reconstruction error).
        
        Keep only actions with ε < threshold (meaningful actions).
        
        Args:
            proposals: Action proposals (B, num_proposals, action_dim)
            fg_model: Bidirectional F/G model
        
        Returns:
            filtered_proposals: List of filtered proposals per batch
            epsilons: List of ε values per batch
            num_valid: List of number of valid proposals per batch
        """
        B, N, D = proposals.shape
        
        filtered_proposals = []
        epsilons_list = []
        num_valid_list = []
        
        for b in range(B):
            batch_proposals = proposals[b]  # (N, D)
            
            # Compute ε for all proposals
            with torch.no_grad():
                epsilons = fg_model.compute_epsilon(batch_proposals)  # (N,)
            
            # Filter by threshold
            valid_mask = epsilons < self.epsilon_threshold
            valid_proposals = batch_proposals[valid_mask]
            valid_epsilons = epsilons[valid_mask]
            
            filtered_proposals.append(valid_proposals)
            epsilons_list.append(valid_epsilons)
            num_valid_list.append(valid_mask.sum().item())
        
        return filtered_proposals, epsilons_list, num_valid_list
    
    def select_by_eta(
        self,
        filtered_proposals: List[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
        fg_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best action from filtered proposals using η.
        
        For each batch, choose the action with lowest η (best shape-action coherence).
        
        Args:
            filtered_proposals: List of filtered proposals per batch
            pos: Point cloud positions (N, 3)
            batch: Batch indices (N,)
            fg_model: Bidirectional F/G model
        
        Returns:
            best_actions: Best action per batch (B, action_dim)
            best_etas: η values for best actions (B,)
        """
        B = len(filtered_proposals)
        device = pos.device
        
        best_actions = torch.zeros(B, self.action_dim, device=device)
        best_etas = torch.zeros(B, device=device)
        
        for b in range(B):
            proposals = filtered_proposals[b]
            
            if len(proposals) == 0:
                # No valid proposals, return random action
                best_actions[b] = torch.randn(self.action_dim, device=device)
                best_etas[b] = float('inf')
                continue
            
            # Get shape for this batch
            mask = (batch == b)
            pos_b = pos[mask]
            
            # Compute η for each proposal
            etas = []
            for proposal in proposals:
                with torch.no_grad():
                    eta = fg_model.compute_eta(pos_b.unsqueeze(0))
                    etas.append(eta.item() if eta.numel() == 1 else eta.mean().item())
            
            # Select action with lowest η
            best_idx = torch.argmin(torch.tensor(etas))
            best_actions[b] = proposals[best_idx]
            best_etas[b] = etas[best_idx]
        
        return best_actions, best_etas
    
    def select_action(
        self,
        observation: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        fg_model: nn.Module,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full pipeline: generate → filter by ε → select by η.
        
        Args:
            observation: Observation (B, observation_dim)
            pos: Point cloud positions (N, 3)
            batch: Batch indices (N,)
            fg_model: Bidirectional F/G model
            deterministic: If True, use deterministic policy
        
        Returns:
            actions: Selected actions (B, action_dim)
            info: Dictionary with selection statistics
        """
        # Generate proposals
        proposals = self.generate_proposals(observation, deterministic)
        
        # Filter by ε
        filtered_proposals, epsilons, num_valid = self.filter_by_epsilon(proposals, fg_model)
        
        # Select by η
        actions, etas = self.select_by_eta(filtered_proposals, pos, batch, fg_model)
        
        # Statistics
        info = {
            'num_proposals': self.num_proposals,
            'num_valid': num_valid,
            'avg_valid_ratio': sum(num_valid) / (len(num_valid) * self.num_proposals),
            'avg_eta': etas.mean().item(),
            'avg_epsilon': torch.cat([e for e in epsilons if len(e) > 0]).mean().item() if any(len(e) > 0 for e in epsilons) else float('inf')
        }
        
        return actions, info
    
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple action selection without F/G filtering (for baseline).
        For discrete actions, we output logits and sample from categorical.
        
        Args:
            observation: Observation (B, observation_dim)
            deterministic: If True, return argmax action
        
        Returns:
            actions: Actions (B,) - discrete action indices
            log_probs: Log probabilities (B,)
        """
        # Use policy network to get action logits
        logits = self.policy_net(observation)[:, :self.action_dim]  # (B, action_dim)
        
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        
        # Compute log probability
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
