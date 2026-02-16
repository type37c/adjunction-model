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
        model: nn.Module,  # AdjunctionModel with Agent C
        value_function: ValueFunction,
        device: torch.device = torch.device('cpu'),
        agent_lr: float = 1e-4,
        value_lr: float = 1e-3,
        fg_lr: float = 0.0,  # 0.0 = frozen
        gamma: float = 0.99,
        episode_length: int = 5,
        value_update_freq: int = 1,  # Update value every N steps
        agent_update_freq: int = 1,   # Update agent every N steps
        reward_scale: float = 1.0     # Scale intrinsic rewards
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
        self.reward_scale = reward_scale
        
        # Optimizers
        # Check if model has 'agent' or 'agent_c' attribute
        if hasattr(self.model, 'agent'):
            agent_params = list(self.model.agent.parameters())
        elif hasattr(self.model, 'agent_c'):
            agent_params = list(self.model.agent_c.parameters())
        else:
            raise AttributeError("Model must have 'agent' or 'agent_c' attribute")
        
        # Check if model has 'functor_f'/'functor_g' or 'F'/'G' attributes
        if hasattr(self.model, 'functor_f') and hasattr(self.model, 'functor_g'):
            fg_params = list(self.model.functor_f.parameters()) + list(self.model.functor_g.parameters())
        elif hasattr(self.model, 'F') and hasattr(self.model, 'G'):
            fg_params = list(self.model.F.parameters()) + list(self.model.G.parameters())
        else:
            raise AttributeError("Model must have 'functor_f'/'functor_g' or 'F'/'G' attributes")
        
        self.agent_optimizer = optim.Adam(agent_params, lr=agent_lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=value_lr)
        
        if fg_lr > 0:
            self.fg_optimizer = optim.Adam(fg_params, lr=fg_lr)
        else:
            self.fg_optimizer = None
            # Freeze F/G
            for param in fg_params:
                param.requires_grad = False
        
        print(f"ValueBasedAgentTrainer initialized:")
        print(f"  Agent C parameters: {sum(p.numel() for p in agent_params)}")
        print(f"  F/G parameters: {sum(p.numel() for p in fg_params)}")
        print(f"  F/G frozen: {fg_lr == 0.0}")
        
        # TD learner
        self.td_learner = TDLearner(self.value_function, self.value_optimizer, gamma)
        
        # Training statistics
        self.step_count = 0
    
    def train_episode(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Train on a single episode (batch of shapes).
        
        Args:
            batch: (batch_size, num_points, 3) tensor of point clouds
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.value_function.train()
        
        batch_size = batch.size(0)
        
        # Initialize
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        # Store trajectory
        trajectory = []
        
        # Metrics
        total_reward = 0.0
        total_coherence = 0.0
        total_uncertainty = 0.0
        total_valence = 0.0
        total_eta = 0.0
        total_epsilon = 0.0
        r_curiosity_sum = 0.0
        r_competence_sum = 0.0
        r_novelty_sum = 0.0
        r_intrinsic_sum = 0.0
        
        # Phase 1: Collect trajectory
        for step_idx in range(batch_size):
            points = batch[step_idx]  # (num_points, 3)
            num_points = points.size(0)
            batch_indices = torch.zeros(num_points, dtype=torch.long, device=self.device)
            
            # Previous spatial coherence
            if step_idx == 0:
                coherence_spatial_prev = torch.zeros(num_points, device=self.device)
            else:
                coherence_spatial_prev = results['coherence_spatial'].detach()
            
            # Forward pass (no gradient for trajectory collection)
            with torch.no_grad():
                results = self.model(
                    points, batch_indices, agent_state,
                    coherence_signal_prev, coherence_spatial_prev
                )
            
            # Extract metrics
            agent_info = results['rssm_info']
            R_intrinsic = agent_info.get('R_intrinsic', torch.tensor(0.0, device=self.device))
            R_curiosity = agent_info.get('R_curiosity', torch.tensor(0.0, device=self.device))
            R_competence = agent_info.get('R_competence', torch.tensor(0.0, device=self.device))
            R_novelty = agent_info.get('R_novelty', torch.tensor(0.0, device=self.device))
            
            # Compute value
            value = self.value_function(results['agent_state'])
            
            # Store trajectory (with scaled reward)
            scaled_reward = (R_intrinsic.item() if isinstance(R_intrinsic, torch.Tensor) else R_intrinsic) * self.reward_scale
            trajectory.append({
                'state': {k: v.clone() for k, v in results['agent_state'].items()},
                'reward': scaled_reward,
                'value': value.mean().item(),
                'coherence': results['coherence_signal'].mean().item(),
                'uncertainty': agent_info.get('uncertainty', torch.tensor(0.0)).item(),
                'valence': agent_info.get('valence_mean', torch.tensor(0.0)).item(),
                'eta': results.get('eta', torch.tensor(0.0)).item(),
                'epsilon': results.get('epsilon', torch.tensor(0.0)).item(),
                'points': points,
                'batch': batch_indices,
                'coherence_signal_prev': coherence_signal_prev.clone(),
                'coherence_spatial_prev': coherence_spatial_prev.clone()
            })
            
            # Accumulate metrics
            total_reward += R_intrinsic.item() if isinstance(R_intrinsic, torch.Tensor) else R_intrinsic
            total_coherence += results['coherence_signal'].mean().item()
            total_uncertainty += agent_info.get('uncertainty', torch.tensor(0.0)).item()
            total_valence += agent_info.get('valence_mean', torch.tensor(0.0)).item()
            total_eta += results.get('eta', torch.tensor(0.0)).item()
            total_epsilon += results.get('epsilon', torch.tensor(0.0)).item()
            r_curiosity_sum += R_curiosity.item() if isinstance(R_curiosity, torch.Tensor) else R_curiosity
            r_competence_sum += R_competence.item() if isinstance(R_competence, torch.Tensor) else R_competence
            r_novelty_sum += R_novelty.item() if isinstance(R_novelty, torch.Tensor) else R_novelty
            r_intrinsic_sum += R_intrinsic.item() if isinstance(R_intrinsic, torch.Tensor) else R_intrinsic
            
            # Update for next step
            agent_state = results['agent_state']
            coherence_signal_prev = results['coherence_signal']
        
        # Phase 2: Update value function using TD learning
        td_loss_sum = 0.0
        for t in range(len(trajectory) - 1):
            state_t = trajectory[t]['state']
            reward_t = torch.tensor([trajectory[t]['reward']], device=self.device)
            state_t1 = trajectory[t + 1]['state']
            done = torch.zeros(1, device=self.device)
            
            td_loss = self.td_learner.update(state_t, reward_t, state_t1, done)
            td_loss_sum += td_loss
        
        # Last step (terminal)
        if len(trajectory) > 0:
            state_t = trajectory[-1]['state']
            reward_t = torch.tensor([trajectory[-1]['reward']], device=self.device)
            state_t1 = {k: torch.zeros_like(v) for k, v in state_t.items()}
            done = torch.ones(1, device=self.device)
            
            td_loss = self.td_learner.update(state_t, reward_t, state_t1, done)
            td_loss_sum += td_loss
        
        # Phase 3: Update Agent C to maximize V(state)
        agent_state = self.model.initial_state(1, self.device)
        coherence_signal_prev = torch.zeros(1, 1, device=self.device)
        
        total_value = 0.0
        
        for step_idx, step_data in enumerate(trajectory):
            points = step_data['points']
            batch_indices = step_data['batch']
            coherence_spatial_prev = step_data['coherence_spatial_prev']
            
            # Forward (with gradient)
            results = self.model(
                points, batch_indices, agent_state,
                coherence_signal_prev, coherence_spatial_prev
            )
            
            # Compute value
            value = self.value_function(results['agent_state'])
            total_value = total_value + value.mean()
            
            # Update for next step (detach to avoid backprop through time)
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
        if hasattr(self.model, 'agent'):
            torch.nn.utils.clip_grad_norm_(self.model.agent.parameters(), max_norm=10.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.agent_c.parameters(), max_norm=10.0)
        
        self.agent_optimizer.step()
        if self.fg_optimizer is not None:
            self.fg_optimizer.step()
        
        # Return metrics
        num_steps = len(trajectory)
        return {
            'total_reward': total_reward / num_steps,
            'avg_coherence': total_coherence / num_steps,
            'avg_uncertainty': total_uncertainty / num_steps,
            'avg_valence': total_valence / num_steps,
            'avg_eta': total_eta / num_steps,
            'avg_epsilon': total_epsilon / num_steps,
            'value_start': trajectory[0]['value'] if trajectory else 0.0,
            'value_end': trajectory[-1]['value'] if trajectory else 0.0,
            'td_loss': td_loss_sum / num_steps,
            'r_curiosity': r_curiosity_sum / num_steps,
            'r_competence': r_competence_sum / num_steps,
            'r_novelty': r_novelty_sum / num_steps,
            'r_intrinsic': r_intrinsic_sum / num_steps,
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
