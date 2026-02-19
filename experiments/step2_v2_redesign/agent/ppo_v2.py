"""
PPO Trainer v2: Improved PPO with dual reward support

改善点:
1. Running mean/std による報酬正規化
2. 二重報酬構造（外発的 + 内発的Δη）のサポート
3. 適切なGAE計算
4. 学習率スケジューリング
5. Early stopping（KL divergence監視）

哲学的根拠:
- 価値関数分析v3: r(t) = α_ext·r_ext(t) + α_int·Δη(t)
- 「報酬は軌跡の瞬間的な傾き」
- 「TD誤差は予想した軌跡と実際の軌跡のずれ」
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class RunningMeanStd:
    """Running mean and standard deviation"""
    
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    """Rollout buffer for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        # Optional F/G features
        self.affordances = []
        self.etas = []
        self.delta_etas = []
    
    def add(self, state, action, reward, done, log_prob, value,
            affordance=None, eta=None, delta_eta=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        if affordance is not None:
            self.affordances.append(affordance)
        if eta is not None:
            self.etas.append(eta)
        if delta_eta is not None:
            self.delta_etas.append(delta_eta)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.affordances.clear()
        self.etas.clear()
        self.delta_etas.clear()
    
    def __len__(self):
        return len(self.states)


class PPOTrainer_v2:
    """
    PPO Trainer v2
    
    Features:
    - Dual reward: r(t) = α_ext·r_ext(t) + α_int·Δη(t)
    - Reward normalization
    - GAE advantage estimation
    - Learning rate scheduling
    - KL divergence monitoring
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
        ppo_epochs=10,
        mini_batch_size=64,
        target_kl=0.015,
        normalize_rewards=True,
        alpha_intrinsic=0.1,
    ):
        """
        Args:
            agent: Agent C model (AgentC_v2 or AgentCWithFG)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Mini-batch size
            target_kl: Target KL divergence for early stopping
            normalize_rewards: Whether to normalize rewards
            alpha_intrinsic: Weight for intrinsic reward (Δη)
        """
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
        self.normalize_rewards = normalize_rewards
        self.alpha_intrinsic = alpha_intrinsic
        
        # Only optimize trainable parameters
        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=lr, eps=1e-5)
        
        # Learning rate scheduler
        self.lr_scheduler = None
        
        # Reward normalization
        self.reward_rms = RunningMeanStd()
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def compute_combined_rewards(self):
        """
        Compute combined rewards: r(t) = r_ext(t) + α_int·Δη(t)
        
        哲学的根拠:
        - 外発的報酬: タスク成功のための距離改善
        - 内発的報酬: Δη = ηの変化率（握りの改善）
        - 「目的は設計されるのではなく、価値ある軌跡のパターンとして創発する」
        """
        rewards = np.array(self.buffer.rewards)
        
        # Add intrinsic reward if available
        if len(self.buffer.delta_etas) > 0:
            delta_etas = np.array(self.buffer.delta_etas)
            # Negative Δη = η decreased = grip improved = positive reward
            intrinsic_rewards = -delta_etas * self.alpha_intrinsic
            rewards = rewards + intrinsic_rewards
        
        return rewards
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation
        
        哲学的根拠:
        - TD誤差 δ = r(t) + γ·V(s(t+1)) - V(s(t))
        - 「予想した軌跡と実際の軌跡のずれ」
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return returns, advantages
    
    def update(self):
        """
        PPO update
        
        Returns:
            dict: Training statistics
        """
        if len(self.buffer) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0}
        
        # Compute combined rewards
        rewards = self.compute_combined_rewards()
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        
        # Normalize rewards
        if self.normalize_rewards:
            self.reward_rms.update(rewards)
            rewards_normalized = self.reward_rms.normalize(rewards)
        else:
            rewards_normalized = rewards
        
        # Compute GAE
        returns, advantages = self.compute_gae(rewards_normalized, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        device = next(self.agent.parameters()).device
        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions_t = torch.FloatTensor(np.array(self.buffer.actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(self.buffer.log_probs)).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        
        # Optional F/G features
        affordances_t = None
        etas_t = None
        if len(self.buffer.affordances) > 0:
            affordances_t = torch.FloatTensor(np.array(self.buffer.affordances)).to(device)
        if len(self.buffer.etas) > 0:
            etas_arr = np.array(self.buffer.etas)
            if etas_arr.ndim == 1:
                etas_arr = etas_arr.reshape(-1, 1)
            etas_t = torch.FloatTensor(etas_arr).to(device)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        num_updates = 0
        early_stop = False
        
        n_samples = len(states_t)
        
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_samples)
                batch_idx = indices[start:end]
                
                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                
                batch_affordances = affordances_t[batch_idx] if affordances_t is not None else None
                batch_etas = etas_t[batch_idx] if etas_t is not None else None
                
                # Evaluate actions
                # Determine which evaluate method to use
                if hasattr(self.agent, 'evaluate_actions'):
                    if batch_affordances is not None:
                        log_probs, values_pred, entropy = self.agent.evaluate_actions(
                            batch_states, batch_actions, batch_affordances, batch_etas
                        )
                    else:
                        log_probs, values_pred, entropy = self.agent.evaluate_actions(
                            batch_states, batch_actions
                        )
                else:
                    log_probs, values_pred, entropy = self.agent.agent_c.evaluate_actions(
                        batch_states, batch_actions
                    )
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.agent.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track KL divergence
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += abs(approx_kl)
                num_updates += 1
                
                # Early stopping on KL divergence
                if abs(approx_kl) > 1.5 * self.target_kl:
                    early_stop = True
                    break
        
        # Clear buffer
        self.buffer.clear()
        
        if num_updates == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl': 0}
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl': total_kl / num_updates,
            'num_updates': num_updates,
            'early_stop': early_stop,
        }
