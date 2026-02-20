"""
Proximal Policy Optimization (PPO) implementation

Simple PPO for training Agent C.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class PPOTrainer:
    """
    PPO Trainer
    
    Implements PPO algorithm for training Agent C.
    """
    
    def __init__(
        self,
        agent,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64
    ):
        """
        Initialize PPO trainer
        
        Args:
            agent: Agent C model
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for PPO update
        """
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        # Storage for rollouts
        self.rollouts = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
    
    def store_transition(self, obs, action, reward, done, log_prob, value):
        """
        Store transition
        
        Args:
            obs: Observation
            action: Action
            reward: Reward
            done: Done flag
            log_prob: Log probability of action
            value: Value estimate
        """
        self.rollouts['observations'].append(obs)
        self.rollouts['actions'].append(action)
        self.rollouts['rewards'].append(reward)
        self.rollouts['dones'].append(done)
        self.rollouts['log_probs'].append(log_prob)
        self.rollouts['values'].append(value)
    
    def compute_returns_and_advantages(self):
        """
        Compute returns and advantages using GAE
        
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.rollouts['rewards'])
        dones = np.array(self.rollouts['dones'])
        values = np.array(self.rollouts['values'])
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # Compute returns
        returns = advantages + values
        
        return returns, advantages
    
    def update(self):
        """
        Update agent using PPO
        
        Returns:
            dict: Training statistics
        """
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        observations = torch.FloatTensor(np.array(self.rollouts['observations']))
        actions = torch.FloatTensor(np.array(self.rollouts['actions']))
        old_log_probs = torch.FloatTensor(np.array(self.rollouts['log_probs']))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_obs.unsqueeze(1),  # Add time dimension
                    batch_actions.unsqueeze(1)
                )
                
                # Remove time dimension
                log_probs = log_probs.squeeze(1)
                values = values.squeeze(1)
                entropy = entropy.squeeze(1)
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Clear rollouts
        for key in self.rollouts:
            self.rollouts[key] = []
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize rollout buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, done, log_prob, value):
        """
        Add transition to buffer
        
        Args:
            obs: Observation
            action: Action
            reward: Reward
            done: Done flag
            log_prob: Log probability of action
            value: Value estimate
        """
        self.buffer.append((obs, action, reward, done, log_prob, value))
    
    def sample(self, batch_size):
        """
        Sample batch from buffer
        
        Args:
            batch_size: Batch size
        
        Returns:
            Batch of transitions
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs, actions, rewards, dones, log_probs, values = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(dones),
            np.array(log_probs),
            np.array(values)
        )
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        """Get buffer size"""
        return len(self.buffer)
