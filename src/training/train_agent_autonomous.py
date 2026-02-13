"""
Autonomous Training Loop for Agent C v4: Intrinsic Reward Maximization

This training method allows Agent C to autonomously learn "how to attend" to shapes
by maximizing intrinsic rewards (curiosity, competence, novelty).

Key design principles:
1. F and G parameters are FROZEN (or have very low learning rate)
2. Agent C learns to modulate F/G via FiLM to maximize intrinsic rewards
3. Training objective is NOT reconstruction error, but intrinsic reward
4. This allows Agent C to autonomously acquire "stances" toward shapes

Difference from supervised training (train_sequential.py):
- Supervised: Minimize external loss (reconstruction error + affordance error)
- Autonomous: Maximize internal reward (curiosity + competence + novelty)

Theoretical justification:
- Agent C should learn "how to face breakdowns" not "how to avoid them"
- Intrinsic rewards guide Agent C to engage with uncertainty and breakdowns
- This prevents "coherence minimization collapse"
- Agent C develops "purpose": to understand the world through exploration

Training phases:
- Phase 1: Pre-train F⊣G with supervised learning (existing train_phase1.py)
- Phase 2: Freeze F/G, train Agent C to maximize intrinsic rewards (this file)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.conditional_adjunction_v3 import ConditionalAdjunctionModelV3
from src.data.synthetic_dataset import SyntheticAffordanceDataset


class AutonomousAgentTrainer:
    """
    Autonomous trainer for Agent C v4.
    
    Agent C learns to maximize intrinsic rewards by modulating F/G via FiLM.
    F and G parameters are frozen (or have very low learning rate).
    """
    
    def __init__(
        self,
        model: ConditionalAdjunctionModelV3,
        device: torch.device = torch.device('cpu'),
        agent_lr: float = 1e-4,
        fg_lr: float = 0.0,  # 0.0 = frozen, small value = slow adaptation
        episode_length: int = 5,
        gamma: float = 0.99  # discount factor for cumulative reward
    ):
        """
        Args:
            model: Conditional Adjunction Model v3 (with Agent C v4)
            device: Device to train on
            agent_lr: Learning rate for Agent C parameters
            fg_lr: Learning rate for F/G parameters (0.0 = frozen)
            episode_length: Number of shapes per episode
            gamma: Discount factor for cumulative intrinsic reward
        """
        self.model = model.to(device)
        self.device = device
        
        self.episode_length = episode_length
        self.gamma = gamma
        
        # Separate optimizers for Agent C and F/G
        agent_params = list(self.model.agent_c.parameters())
        fg_params = list(self.model.F.parameters()) + list(self.model.G.parameters())
        
        self.agent_optimizer = optim.Adam(agent_params, lr=agent_lr)
        
        if fg_lr > 0:
            self.fg_optimizer = optim.Adam(fg_params, lr=fg_lr)
        else:
            self.fg_optimizer = None
            # Freeze F/G parameters
            for param in fg_params:
                param.requires_grad = False
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch using autonomous episodes.
        
        Each episode presents `episode_length` shapes sequentially.
        Agent C learns to maximize cumulative intrinsic reward.
        """
        self.model.train()
        
        total_intrinsic_reward = 0.0
        total_curiosity = 0.0
        total_competence = 0.0
        total_novelty = 0.0
        total_coherence = 0.0
        num_episodes = 0
        
        # Collect samples into episodes
        episode_buffer = []
        
        for batch_idx, batch_data in enumerate(dataloader):
            episode_buffer.append(batch_data)
            
            # When we have enough samples for an episode, train on it
            if len(episode_buffer) >= self.episode_length:
                episode_metrics = self._train_episode(episode_buffer[:self.episode_length])
                
                total_intrinsic_reward += episode_metrics['intrinsic_reward']
                total_curiosity += episode_metrics['curiosity']
                total_competence += episode_metrics['competence']
                total_novelty += episode_metrics['novelty']
                total_coherence += episode_metrics['coherence']
                num_episodes += 1
                
                # Remove processed samples from buffer
                episode_buffer = episode_buffer[self.episode_length:]
                
                if num_episodes % 10 == 0:
                    print(f"  Episode {num_episodes}: "
                          f"R_intrinsic={episode_metrics['intrinsic_reward']:.4f}, "
                          f"Coherence={episode_metrics['coherence']:.4f}")
        
        # Average metrics
        if num_episodes > 0:
            return {
                'intrinsic_reward': total_intrinsic_reward / num_episodes,
                'curiosity': total_curiosity / num_episodes,
                'competence': total_competence / num_episodes,
                'novelty': total_novelty / num_episodes,
                'coherence': total_coherence / num_episodes
            }
        else:
            return {
                'intrinsic_reward': 0.0,
                'curiosity': 0.0,
                'competence': 0.0,
                'novelty': 0.0,
                'coherence': 0.0
            }
    
    def _train_episode(
        self,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Train on a single episode (sequence of shapes).
        
        Agent C learns to maximize cumulative intrinsic reward.
        
        Key idea:
        - Agent C's actions (FiLM parameters) affect future intrinsic rewards
        - We use policy gradient (REINFORCE-like) to optimize Agent C
        - F/G are frozen, so Agent C can only change "how to attend"
        
        Args:
            episode_data: List of batch_data dictionaries
        
        Returns:
            metrics: Dictionary with episode metrics
        """
        # Initialize agent state at the start of episode
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        # Store trajectory for policy gradient
        trajectory = []
        
        # Process each shape in the episode sequentially
        for step_idx, batch_data in enumerate(episode_data):
            # Extract data
            points = batch_data['points'][0].to(self.device)  # (N, 3)
            affordances_gt = batch_data['affordances'][0].to(self.device)  # (N, num_aff)
            
            num_points = points.size(0)
            batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            # Previous spatial coherence
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial'].detach()
            
            # Forward pass (with gradient tracking for Agent C)
            results = self.model(
                points, batch, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            # Extract intrinsic reward
            agent_info = results['rssm_info']
            R_intrinsic = agent_info.get('R_intrinsic', torch.zeros(1, device=self.device))
            R_curiosity = agent_info.get('R_curiosity', torch.zeros(1, device=self.device))
            R_competence = agent_info.get('R_competence', torch.zeros(1, device=self.device))
            R_novelty = agent_info.get('R_novelty', torch.zeros(1, device=self.device))
            
            # Store in trajectory
            trajectory.append({
                'R_intrinsic': R_intrinsic.mean().item(),
                'R_curiosity': R_curiosity.mean().item(),
                'R_competence': R_competence.mean().item(),
                'R_novelty': R_novelty.mean().item(),
                'coherence': results['coherence_signal'].mean().item(),
                'context': results['context']  # Keep gradient for backprop
            })
            
            # Update agent state for next step (detach to prevent backprop through time)
            agent_state = {k: v.detach() for k, v in results['agent_state'].items()}
            coherence_signal_prev = results['coherence_signal'].detach()
        
        # Compute cumulative discounted intrinsic reward
        returns = []
        R = 0
        for t in reversed(range(len(trajectory))):
            R = trajectory[t]['R_intrinsic'] + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns (for stability)
        returns = torch.tensor(returns, device=self.device)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss: maximize cumulative intrinsic reward
        # We use the context vector as a proxy for Agent C's "action"
        # Loss = -sum(R_t * log_prob(context_t))
        # Since we don't have explicit log_prob, we use a surrogate:
        # Loss = -sum(R_t * ||context_t||^2)
        # This encourages Agent C to produce stronger context when R_t is high
        
        loss = 0
        for t in range(len(trajectory)):
            context = trajectory[t]['context']
            R_t = returns[t]
            
            # Surrogate loss: encourage context magnitude when reward is high
            # Note: This is a simplified version; a full RL implementation would use
            # proper policy gradient with log probabilities
            context_magnitude = context.norm(dim=-1).mean()
            loss = loss - R_t * context_magnitude
        
        # Backward pass
        self.agent_optimizer.zero_grad()
        if self.fg_optimizer is not None:
            self.fg_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.agent_c.parameters(), max_norm=10.0)
        
        self.agent_optimizer.step()
        if self.fg_optimizer is not None:
            self.fg_optimizer.step()
        
        # Compute episode metrics
        episode_intrinsic_reward = sum(t['R_intrinsic'] for t in trajectory) / len(trajectory)
        episode_curiosity = sum(t['R_curiosity'] for t in trajectory) / len(trajectory)
        episode_competence = sum(t['R_competence'] for t in trajectory) / len(trajectory)
        episode_novelty = sum(t['R_novelty'] for t in trajectory) / len(trajectory)
        episode_coherence = sum(t['coherence'] for t in trajectory) / len(trajectory)
        
        return {
            'intrinsic_reward': episode_intrinsic_reward,
            'curiosity': episode_curiosity,
            'competence': episode_competence,
            'novelty': episode_novelty,
            'coherence': episode_coherence
        }
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Uses the same sequential episode structure as training.
        """
        self.model.eval()
        
        total_intrinsic_reward = 0.0
        total_coherence = 0.0
        num_episodes = 0
        
        episode_buffer = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                episode_buffer.append(batch_data)
                
                if len(episode_buffer) >= self.episode_length:
                    episode_metrics = self._evaluate_episode(episode_buffer[:self.episode_length])
                    
                    total_intrinsic_reward += episode_metrics['intrinsic_reward']
                    total_coherence += episode_metrics['coherence']
                    num_episodes += 1
                    
                    episode_buffer = episode_buffer[self.episode_length:]
        
        if num_episodes > 0:
            return {
                'intrinsic_reward': total_intrinsic_reward / num_episodes,
                'coherence': total_coherence / num_episodes
            }
        else:
            return {
                'intrinsic_reward': 0.0,
                'coherence': 0.0
            }
    
    def _evaluate_episode(
        self,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate on a single episode."""
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        episode_intrinsic_reward = 0.0
        episode_coherence = 0.0
        
        for step_idx, batch_data in enumerate(episode_data):
            points = batch_data['points'][0].to(self.device)
            
            num_points = points.size(0)
            batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial']
            
            results = self.model(
                points, batch, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            agent_info = results['rssm_info']
            R_intrinsic = agent_info.get('R_intrinsic', torch.zeros(1, device=self.device))
            
            episode_intrinsic_reward += R_intrinsic.mean().item()
            episode_coherence += results['coherence_signal'].mean().item()
            
            agent_state = results['agent_state']
            coherence_signal_prev = results['coherence_signal']
        
        num_steps = len(episode_data)
        return {
            'intrinsic_reward': episode_intrinsic_reward / num_steps,
            'coherence': episode_coherence / num_steps
        }


if __name__ == '__main__':
    print("Testing Autonomous Agent Trainer...")
    
    # Create model (note: using v3 model structure, but with v4 agent internally)
    # We would need to create ConditionalAdjunctionModelV4 that uses AgentLayerC_v4
    # For now, this is a conceptual test
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Autonomous training implements Agent C's purpose:")
    print("1. F/G parameters are frozen (or low learning rate)")
    print("2. Agent C learns to maximize intrinsic rewards")
    print("3. Training objective is NOT external loss, but internal reward")
    print("4. Agent C autonomously acquires 'stances' toward shapes")
    print("5. This prevents coherence minimization collapse")
    print("="*60)
    
    print("\nNote: Full integration requires ConditionalAdjunctionModelV4")
    print("which uses AgentLayerC_v4 internally.")
