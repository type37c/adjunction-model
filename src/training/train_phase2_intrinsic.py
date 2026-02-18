"""
Phase 2 Training: Intrinsic Reward Baseline

This implements the second stage of the learning framework:
- F/G are frozen (loaded from Phase 1 checkpoint)
- Agent C learns via value-based RL with intrinsic rewards only
- No external objectives (no affordance loss)

Training cycle (per epoch):
1. Trajectory Collection: Agent interacts with environment
2. Value Function Update: TD learning to predict future rewards
3. Agent C Update: Policy gradient to maximize value
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.adjunction_model import AdjunctionModel
from src.models.value_function import ValueFunction
from src.models.intrinsic_reward import IntrinsicRewardModule


class Phase2IntrinsicTrainer:
    """
    Trainer for Phase 2: Intrinsic Reward Baseline.
    
    F/G are frozen. Agent C learns via value-based RL.
    """
    
    def __init__(
        self,
        model: AdjunctionModel,
        value_function: ValueFunction,
        device: torch.device = torch.device('cpu'),
        lr_agent: float = 1e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        competence_weight: float = 1.0,
        novelty_weight: float = 0.1
    ):
        """
        Args:
            model: AdjunctionModel instance (F/G will be frozen)
            value_function: ValueFunction instance
            device: Device to train on
            lr_agent: Learning rate for Agent C
            lr_value: Learning rate for Value Function
            gamma: Discount factor
            competence_weight: Weight for competence reward
            novelty_weight: Weight for novelty reward
        """
        self.model = model.to(device)
        self.value_function = value_function.to(device)
        self.device = device
        self.gamma = gamma
        
        # Freeze F and G
        for param in self.model.F.parameters():
            param.requires_grad = False
        for param in self.model.G.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.agent_optimizer = optim.Adam(
            self.model.agent_c.parameters(),
            lr=lr_agent
        )
        self.value_optimizer = optim.Adam(
            self.value_function.parameters(),
            lr=lr_value
        )
        
        # Intrinsic reward module
        self.reward_module = IntrinsicRewardModule(
            competence_weight=competence_weight,
            novelty_weight=novelty_weight
        )
        
        print("Phase2IntrinsicTrainer initialized:")
        print(f"  F/G frozen: True")
        print(f"  Agent C parameters: {sum(p.numel() for p in self.model.agent_c.parameters()):,}")
        print(f"  Value Function parameters: {sum(p.numel() for p in self.value_function.parameters()):,}")
        print(f"  Intrinsic rewards: Competence ({competence_weight}) + Novelty ({novelty_weight})")
    
    def collect_trajectories(
        self,
        dataloader: DataLoader,
        num_episodes: int = 10
    ) -> List[Dict]:
        """
        Phase 1: Collect trajectories by interacting with environment.
        
        Args:
            dataloader: Data loader for environment samples
            num_episodes: Number of episodes to collect
        
        Returns:
            trajectories: List of trajectory dictionaries
        """
        self.model.eval()
        self.value_function.eval()
        
        trajectories = []
        
        for episode in range(num_episodes):
            for batch_data in dataloader:
                points = batch_data['points'].to(self.device)
                B, N, _ = points.shape
                
                # Flatten to graph format
                pos = points.reshape(B * N, 3)
                batch = torch.repeat_interleave(torch.arange(B, device=self.device), N)
                
                # Initialize agent state
                agent_state = self.model.agent_c.initial_state(B, self.device)
                
                # Initialize coherence signal
                eta_prev = torch.zeros(B, 1, device=self.device)
                coherence_spatial_prev = torch.zeros(B * N, device=self.device)
                
                # Forward pass
                with torch.no_grad():
                    results = self.model(pos, batch, agent_state, eta_prev, coherence_spatial_prev)
                
                eta_curr = results['coherence_signal']
                agent_state_next = results['agent_state']
                
                # Compute state representation for value function
                state = torch.cat([agent_state['h'], agent_state['z']], dim=-1)
                state_next = torch.cat([agent_state_next['h'], agent_state_next['z']], dim=-1)
                
                # Compute intrinsic reward
                rewards = self.reward_module.compute_intrinsic_reward(
                    eta_prev, eta_curr, state
                )
                
                # Store trajectory
                trajectories.append({
                    'state': state.detach(),
                    'state_next': state_next.detach(),
                    'reward': rewards['total'].detach(),
                    'eta_prev': eta_prev.detach(),
                    'eta_curr': eta_curr.detach(),
                    'reward_competence': rewards['competence'].detach(),
                    'reward_novelty': rewards['novelty'].detach()
                })
                
                # Only collect one batch per episode for now
                break
        
        return trajectories
    
    def update_value_function(
        self,
        trajectories: List[Dict]
    ) -> float:
        """
        Phase 2: Update value function using TD learning.
        
        Args:
            trajectories: List of trajectory dictionaries
        
        Returns:
            td_error: Average TD error
        """
        self.value_function.train()
        
        total_td_error = 0.0
        
        for traj in trajectories:
            state = traj['state']
            state_next = traj['state_next']
            reward = traj['reward']
            
            # Compute value estimates
            v_curr = self.value_function(state)
            v_next = self.value_function(state_next)
            
            # TD target: r + γ * V(s')
            td_target = reward + self.gamma * v_next.detach()
            
            # TD error
            td_error = (td_target - v_curr) ** 2
            loss = td_error.mean()
            
            # Backward pass
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
            
            total_td_error += td_error.mean().item()
        
        return total_td_error / len(trajectories)
    
    def update_agent(
        self,
        dataloader: DataLoader
    ) -> float:
        """
        Phase 3: Update Agent C to maximize value.
        
        Args:
            dataloader: Data loader
        
        Returns:
            policy_loss: Average policy loss
        """
        self.model.train()
        self.value_function.eval()
        
        total_policy_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            points = batch_data['points'].to(self.device)
            B, N, _ = points.shape
            
            # Flatten to graph format
            pos = points.reshape(B * N, 3)
            batch = torch.repeat_interleave(torch.arange(B, device=self.device), N)
            
            # Initialize agent state
            agent_state = self.model.agent_c.initial_state(B, self.device)
            
            # Initialize coherence signal
            eta_prev = torch.zeros(B, 1, device=self.device)
            coherence_spatial_prev = torch.zeros(B * N, device=self.device)
            
            # Forward pass
            results = self.model(pos, batch, agent_state, eta_prev, coherence_spatial_prev)
            agent_state_next = results['agent_state']
            
            # Compute state representation
            state_next = torch.cat([agent_state_next['h'], agent_state_next['z']], dim=-1)
            
            # Compute value
            value = self.value_function(state_next)
            
            # Policy loss: maximize value (minimize -value)
            policy_loss = -value.mean()
            
            # Backward pass
            self.agent_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.agent_c.parameters(), max_norm=1.0)
            self.agent_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            num_batches += 1
        
        return total_policy_loss / num_batches
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch (3-phase cycle).
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            num_episodes: Number of episodes for trajectory collection
        
        Returns:
            metrics: Dictionary of training metrics
        """
        print(f"\nEpoch {epoch}")
        print("-" * 60)
        
        # Phase 1: Collect trajectories
        print("Phase 1: Collecting trajectories...")
        trajectories = self.collect_trajectories(dataloader, num_episodes)
        
        # Compute trajectory statistics
        avg_reward = torch.stack([t['reward'] for t in trajectories]).mean().item()
        avg_competence = torch.stack([t['reward_competence'] for t in trajectories]).mean().item()
        avg_novelty = torch.stack([t['reward_novelty'] for t in trajectories]).mean().item()
        avg_eta = torch.stack([t['eta_curr'] for t in trajectories]).mean().item()
        
        print(f"  Collected {len(trajectories)} trajectories")
        print(f"  Avg reward: {avg_reward:.4f} (Comp: {avg_competence:.4f}, Nov: {avg_novelty:.4f})")
        print(f"  Avg η: {avg_eta:.4f}")
        
        # Phase 2: Update value function
        print("Phase 2: Updating value function...")
        td_error = self.update_value_function(trajectories)
        print(f"  TD error: {td_error:.4f}")
        
        # Phase 3: Update Agent C
        print("Phase 3: Updating Agent C...")
        policy_loss = self.update_agent(dataloader)
        print(f"  Policy loss: {policy_loss:.4f}")
        
        metrics = {
            'reward_total': avg_reward,
            'reward_competence': avg_competence,
            'reward_novelty': avg_novelty,
            'eta': avg_eta,
            'td_error': td_error,
            'policy_loss': policy_loss
        }
        
        return metrics
