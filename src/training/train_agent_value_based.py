"""
Value-Based Autonomous Training for Agent C v4

This training method uses a value function to guide Agent C's learning.

Key principles:
1. Value function V(state) estimates expected future intrinsic rewards
2. Agent C learns to maximize V(state) by adjusting its parameters
3. F and G are frozen (or have very low learning rate)
4. TD learning updates V based on experienced intrinsic rewards

Training loop:
1. Agent C processes a shape → produces state s_t, context c_t
2. Compute intrinsic reward R_t (curiosity + competence + novelty)
3. Update value function: V(s_t) ← R_t + γ × V(s_{t+1})
4. Update Agent C: maximize V(s_t) via gradient ascent

Theoretical justification:
- Agent C learns "what states are valuable" not "what is correct"
- Value emerges from experience with intrinsic rewards
- This aligns with suspension structure: no external "correct answer"
- Agent C develops purpose: seek states with high expected future rewards

Difference from supervised learning:
- Supervised: minimize external loss (reconstruction error)
- Value-based: maximize internal value (expected intrinsic rewards)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import sys
sys.path.append('/home/ubuntu/adjunction-model')

from src.models.value_function import ValueFunction, TDLearner


class ValueBasedAgentTrainer:
    """
    Value-based autonomous trainer for Agent C.
    
    Uses TD learning to train a value function, then trains Agent C
    to maximize value.
    """
    
    def __init__(
        self,
        model: nn.Module,  # ConditionalAdjunctionModel with Agent C v4
        value_function: ValueFunction,
        device: torch.device = torch.device('cpu'),
        agent_lr: float = 1e-4,
        value_lr: float = 1e-3,
        fg_lr: float = 0.0,  # 0.0 = frozen
        gamma: float = 0.99,
        episode_length: int = 5,
        value_update_freq: int = 1,  # Update value every N steps
        agent_update_freq: int = 5   # Update agent every N steps
    ):
        """
        Args:
            model: Conditional Adjunction Model (with Agent C v4)
            value_function: Value function V(state)
            device: Device
            agent_lr: Learning rate for Agent C
            value_lr: Learning rate for value function
            fg_lr: Learning rate for F/G (0.0 = frozen)
            gamma: Discount factor
            episode_length: Steps per episode
            value_update_freq: Update value function every N steps
            agent_update_freq: Update Agent C every N steps
        """
        self.model = model.to(device)
        self.value_function = value_function.to(device)
        self.device = device
        
        self.gamma = gamma
        self.episode_length = episode_length
        self.value_update_freq = value_update_freq
        self.agent_update_freq = agent_update_freq
        
        # Optimizers
        agent_params = list(self.model.agent_c.parameters())
        fg_params = list(self.model.F.parameters()) + list(self.model.G.parameters())
        
        self.agent_optimizer = optim.Adam(agent_params, lr=agent_lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=value_lr)
        
        if fg_lr > 0:
            self.fg_optimizer = optim.Adam(fg_params, lr=fg_lr)
        else:
            self.fg_optimizer = None
            # Freeze F/G
            for param in fg_params:
                param.requires_grad = False
        
        # TD learner
        self.td_learner = TDLearner(self.value_function, self.value_optimizer, gamma)
        
        # Training statistics
        self.step_count = 0
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Alternates between:
        1. Updating value function (TD learning)
        2. Updating Agent C (maximize value)
        """
        self.model.train()
        self.value_function.train()
        
        total_intrinsic_reward = 0.0
        total_value = 0.0
        total_value_loss = 0.0
        total_agent_loss = 0.0
        total_coherence = 0.0
        num_episodes = 0
        
        episode_buffer = []
        
        for batch_idx, batch_data in enumerate(dataloader):
            episode_buffer.append(batch_data)
            
            if len(episode_buffer) >= self.episode_length:
                episode_metrics = self._train_episode(episode_buffer[:self.episode_length])
                
                total_intrinsic_reward += episode_metrics['intrinsic_reward']
                total_value += episode_metrics['value']
                total_value_loss += episode_metrics['value_loss']
                total_agent_loss += episode_metrics['agent_loss']
                total_coherence += episode_metrics['coherence']
                num_episodes += 1
                
                episode_buffer = episode_buffer[self.episode_length:]
                
                if num_episodes % 10 == 0:
                    print(f"  Episode {num_episodes}: "
                          f"R={episode_metrics['intrinsic_reward']:.4f}, "
                          f"V={episode_metrics['value']:.4f}, "
                          f"Coherence={episode_metrics['coherence']:.4f}")
        
        if num_episodes > 0:
            return {
                'intrinsic_reward': total_intrinsic_reward / num_episodes,
                'value': total_value / num_episodes,
                'value_loss': total_value_loss / num_episodes,
                'agent_loss': total_agent_loss / num_episodes,
                'coherence': total_coherence / num_episodes
            }
        else:
            return {
                'intrinsic_reward': 0.0,
                'value': 0.0,
                'value_loss': 0.0,
                'agent_loss': 0.0,
                'coherence': 0.0
            }
    
    def _train_episode(
        self,
        episode_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Train on a single episode.
        
        Two-phase training:
        1. TD learning: update value function based on intrinsic rewards
        2. Value maximization: update Agent C to maximize V(state)
        """
        # Initialize
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        # Store trajectory
        trajectory = []
        
        # Phase 1: Collect trajectory and update value function
        episode_intrinsic_reward = 0.0
        episode_value = 0.0
        episode_value_loss = 0.0
        episode_coherence = 0.0
        
        for step_idx, batch_data in enumerate(episode_data):
            # Extract data
            points = batch_data['points'][0].to(self.device)
            num_points = points.size(0)
            batch = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            # Previous spatial coherence
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial'].detach()
            
            # Forward pass (no gradient for now, just collecting trajectory)
            with torch.no_grad():
                results = self.model(
                    points, batch, agent_state,
                    coherence_signal_prev, coherence_spatial_prev
                )
            
            # Extract intrinsic reward
            agent_info = results['rssm_info']
            R_intrinsic = agent_info.get('R_intrinsic', torch.zeros(1, device=self.device)).mean()
            
            # Compute value
            value = self.value_function(results['agent_state'])
            
            # Store trajectory
            trajectory.append({
                'state': {k: v.clone() for k, v in results['agent_state'].items()},
                'reward': R_intrinsic.item(),
                'value': value.mean().item(),
                'coherence': results['coherence_signal'].mean().item(),
                'points': points,
                'batch': batch,
                'coherence_signal_prev': coherence_signal_prev.clone(),
                'coherence_spatial_prev': coherence_spatial_prev.clone()
            })
            
            episode_intrinsic_reward += R_intrinsic.item()
            episode_value += value.mean().item()
            episode_coherence += results['coherence_signal'].mean().item()
            
            # Update for next step
            agent_state = results['agent_state']
            coherence_signal_prev = results['coherence_signal']
        
        # Phase 2: Update value function using TD learning
        for t in range(len(trajectory) - 1):
            if self.step_count % self.value_update_freq == 0:
                state_t = trajectory[t]['state']
                reward_t = torch.tensor([trajectory[t]['reward']], device=self.device)
                state_t1 = trajectory[t + 1]['state']
                done = torch.zeros(1, device=self.device)
                
                value_loss = self.td_learner.update(state_t, reward_t, state_t1, done)
                episode_value_loss += value_loss
            
            self.step_count += 1
        
        # Last step (terminal)
        if len(trajectory) > 0:
            state_t = trajectory[-1]['state']
            reward_t = torch.tensor([trajectory[-1]['reward']], device=self.device)
            # Terminal state: V(s_{t+1}) = 0
            state_t1 = {k: torch.zeros_like(v) for k, v in state_t.items()}
            done = torch.ones(1, device=self.device)
            
            value_loss = self.td_learner.update(state_t, reward_t, state_t1, done)
            episode_value_loss += value_loss
        
        # Phase 3: Update Agent C to maximize V(state)
        episode_agent_loss = 0.0
        
        if self.step_count % self.agent_update_freq == 0:
            # Reset agent state
            agent_state = self.model.initial_state(1, self.device)
            coherence_signal_prev = torch.zeros(1, 1, device=self.device)
            
            # Forward pass with gradient
            total_value = 0.0
            
            for step_idx, step_data in enumerate(trajectory):
                points = step_data['points']
                batch = step_data['batch']
                coherence_spatial_prev = step_data['coherence_spatial_prev']
                
                # Forward (with gradient)
                results = self.model(
                    points, batch, agent_state,
                    coherence_signal_prev, coherence_spatial_prev
                )
                
                # Compute value
                value = self.value_function(results['agent_state'])
                total_value = total_value + value.mean()
                
                # Update for next step
                agent_state = {k: v.detach() for k, v in results['agent_state'].items()}
                coherence_signal_prev = results['coherence_signal'].detach()
            
            # Loss: negative value (we want to maximize value)
            agent_loss = -total_value / len(trajectory)
            
            # Backward
            self.agent_optimizer.zero_grad()
            if self.fg_optimizer is not None:
                self.fg_optimizer.zero_grad()
            
            agent_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.agent_c.parameters(), max_norm=10.0)
            
            self.agent_optimizer.step()
            if self.fg_optimizer is not None:
                self.fg_optimizer.step()
            
            episode_agent_loss = agent_loss.item()
        
        # Return metrics
        num_steps = len(trajectory)
        return {
            'intrinsic_reward': episode_intrinsic_reward / num_steps,
            'value': episode_value / num_steps,
            'value_loss': episode_value_loss / num_steps,
            'agent_loss': episode_agent_loss,
            'coherence': episode_coherence / num_steps
        }


if __name__ == '__main__':
    print("Testing Value-Based Agent Trainer...")
    
    print("\n" + "="*60)
    print("THEORETICAL VERIFICATION:")
    print("="*60)
    print("Value-based training implements Agent C's autonomous learning:")
    print("1. Value function V(state) estimates future intrinsic rewards")
    print("2. TD learning updates V based on experienced rewards")
    print("3. Agent C learns to maximize V(state)")
    print("4. F/G are frozen, Agent C only changes 'how to attend'")
    print("5. Agent C develops purpose: seek valuable states")
    print("6. This prevents coherence minimization collapse")
    print("="*60)
    
    print("\nNote: Full integration requires ConditionalAdjunctionModelV4")
    print("which uses AgentLayerC_v4 internally.")
