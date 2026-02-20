"""
Agent C v2: MLP-based Actor-Critic with η Integration

再設計のポイント:
1. MLP（LSTMではなく）: Reachingタスクはマルコフ的
2. 正規化された入力: LayerNormで安定化
3. 分離されたActor/Critic: 共有表現 + 独立ヘッド
4. F/G特徴量の統合: アフォーダンス表現をstate vectorと結合
5. η（再構成誤差）の統合: 内発的報酬信号として利用

哲学的根拠:
- 価値関数分析v3: 「F/Gという川床の上をAgent Cという水が流れると、
  はじめて随伴が成立し、価値関数が評価すべき意味のある軌跡が生まれる」
- Agent Cの行動選択は、F/Gの出力を「行動の文脈で解釈」すること
- 報酬 r(t) = α_ext·r_ext(t) + α_int·Δη(t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AgentC_v2(nn.Module):
    """
    Agent C v2: MLP Actor-Critic
    
    Architecture:
        Shared encoder → Actor head (policy)
                       → Critic head (value function)
    
    State input: state_vector (16) + optional affordance_features (32)
    Action output: Δθ (3) - joint angle deltas in [-1, 1]
    """
    
    def __init__(
        self,
        state_dim=16,
        action_dim=3,
        hidden_dim=256,
        affordance_dim=0,
        eta_dim=0,
    ):
        """
        Args:
            state_dim: State vector dimension (16 for reaching task)
            action_dim: Action dimension (3 for 3-DOF arm)
            hidden_dim: Hidden layer dimension
            affordance_dim: F/G affordance feature dimension (0 = baseline)
            eta_dim: η (reconstruction error) feature dimension (0 = no η)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.affordance_dim = affordance_dim
        self.eta_dim = eta_dim
        
        total_input_dim = state_dim + affordance_dim + eta_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(total_input_dim)
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy: mean + log_std)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization (PPO best practice)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Actor output layer: small initialization
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        
        # Critic output layer: unit gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, state, affordance=None, eta=None):
        """
        Forward pass
        
        Args:
            state: State vector (B, state_dim)
            affordance: Affordance features from F/G (B, affordance_dim), optional
            eta: η features (B, eta_dim), optional
        
        Returns:
            action_mean: Mean of action distribution (B, action_dim)
            action_logstd: Log std of action distribution (action_dim,)
            value: Value estimate (B, 1)
        """
        # Build input
        inputs = [state]
        if affordance is not None and self.affordance_dim > 0:
            inputs.append(affordance)
        if eta is not None and self.eta_dim > 0:
            inputs.append(eta)
        
        x = torch.cat(inputs, dim=-1)
        x = self.input_norm(x)
        
        # Shared encoding
        features = self.shared_encoder(x)
        
        # Actor
        action_mean = self.actor(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, state, affordance=None, eta=None, deterministic=False):
        """
        Sample action from policy
        
        Args:
            state: State vector (B, state_dim) or (state_dim,)
            affordance: Affordance features (optional)
            eta: η features (optional)
            deterministic: Whether to use mean action
        
        Returns:
            action: Sampled action (B, action_dim)
            log_prob: Log probability (B, 1)
            value: Value estimate (B, 1)
        """
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
            if affordance is not None:
                affordance = affordance.unsqueeze(0)
            if eta is not None:
                eta = eta.unsqueeze(0)
        
        action_mean, action_logstd, value = self.forward(state, affordance, eta)
        
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # Clamp to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, actions, affordance=None, eta=None):
        """
        Evaluate actions (for PPO update)
        
        Args:
            state: State vector (B, state_dim)
            actions: Actions (B, action_dim)
            affordance: Affordance features (optional)
            eta: η features (optional)
        
        Returns:
            log_probs: Log probabilities (B,)
            values: Value estimates (B,)
            entropy: Entropy (B,)
        """
        action_mean, action_logstd, value = self.forward(state, affordance, eta)
        
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = value.squeeze(-1)
        
        return log_probs, value, entropy


class FGFeatureExtractor(nn.Module):
    """
    F/G Feature Extractor for Agent C
    
    Phase 1.5で訓練されたF/Gを使って、点群からアフォーダンス特徴量とη（再構成誤差）を抽出する。
    F/Gのパラメータは凍結される（川床は変わらない）。
    
    哲学的根拠:
    - 「F/Gの出力は潜在表現である。それがアフォーダンスになるのは、
      Agent Cがそれを行動の文脈で解釈し、選択したときである。」
    - ηは「握り（grip）」の指標: 低いη = 世界をよく把握している
    """
    
    def __init__(self, functor_f, functor_g, freeze=True):
        """
        Args:
            functor_f: Trained FunctorF_v2
            functor_g: Trained FunctorG
            freeze: Whether to freeze F/G parameters
        """
        super().__init__()
        
        self.functor_f = functor_f
        self.functor_g = functor_g
        
        if freeze:
            for param in self.functor_f.parameters():
                param.requires_grad = False
            for param in self.functor_g.parameters():
                param.requires_grad = False
    
    def forward(self, point_cloud, goal=None):
        """
        Extract affordance features and η from point cloud
        
        Args:
            point_cloud: (B, N, 3) point cloud
            goal: (B, goal_dim) goal vector (optional)
        
        Returns:
            affordance_summary: (B, affordance_dim) - global affordance features
            eta: (B, 1) - reconstruction error (η)
            eta_per_point: (B, N) - per-point reconstruction error
        """
        B, N, D = point_cloud.shape
        
        # F: point cloud → affordance representation
        affordances = self.functor_f(point_cloud, goal=goal)  # (B, N, affordance_dim)
        
        # G: affordance → reconstructed point cloud
        aff_flat = affordances.reshape(-1, affordances.shape[-1])  # (B*N, affordance_dim)
        recon_flat = self.functor_g(aff_flat)  # (B*N, 3)
        recon = recon_flat.reshape(B, N, D)  # (B, N, 3)
        
        # η: reconstruction error (per-point MSE)
        eta_per_point = ((recon - point_cloud) ** 2).sum(dim=-1)  # (B, N)
        
        # Global η: mean over all points
        eta = eta_per_point.mean(dim=-1, keepdim=True)  # (B, 1)
        
        # Affordance summary: mean pooling
        affordance_summary = affordances.mean(dim=1)  # (B, affordance_dim)
        
        return affordance_summary, eta, eta_per_point


class AgentCWithFG(nn.Module):
    """
    Agent C with F/G integration
    
    Combines:
    1. State vector (robot state + object position + relative vector)
    2. F/G affordance features (from point cloud)
    3. η (reconstruction error) as grip indicator
    
    二重報酬構造:
    - 外発的報酬: 距離改善量（タスク成功のため）
    - 内発的報酬: Δη（ηの変化率 = 握りの改善）
    """
    
    def __init__(
        self,
        state_dim=16,
        action_dim=3,
        hidden_dim=256,
        affordance_dim=32,
        functor_f=None,
        functor_g=None,
        freeze_fg=True,
        use_eta=True,
    ):
        """
        Args:
            state_dim: State vector dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for Agent C
            affordance_dim: F/G affordance dimension
            functor_f: Trained FunctorF_v2
            functor_g: Trained FunctorG
            freeze_fg: Whether to freeze F/G
            use_eta: Whether to include η in state
        """
        super().__init__()
        
        self.use_eta = use_eta
        self.affordance_dim = affordance_dim
        eta_dim = 1 if use_eta else 0
        
        # F/G feature extractor
        if functor_f is not None and functor_g is not None:
            self.fg_extractor = FGFeatureExtractor(functor_f, functor_g, freeze=freeze_fg)
        else:
            self.fg_extractor = None
        
        # Agent C
        self.agent_c = AgentC_v2(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            affordance_dim=affordance_dim,
            eta_dim=eta_dim,
        )
        
        # η tracking for intrinsic reward
        self.prev_eta = None
    
    def reset_eta_tracking(self):
        """Reset η tracking at episode start"""
        self.prev_eta = None
    
    def forward(self, state, point_cloud=None, goal=None):
        """
        Forward pass
        
        Args:
            state: State vector (B, state_dim)
            point_cloud: Point cloud (B, N, 3), optional
            goal: Goal vector (B, goal_dim), optional
        
        Returns:
            action_mean, action_logstd, value, eta, delta_eta
        """
        affordance = None
        eta = None
        delta_eta = None
        
        if self.fg_extractor is not None and point_cloud is not None:
            affordance, eta, _ = self.fg_extractor(point_cloud, goal=goal)
            
            # Compute Δη (intrinsic reward signal)
            if self.prev_eta is not None:
                delta_eta = eta - self.prev_eta  # Negative = improvement
            self.prev_eta = eta.detach()
        
        eta_input = eta if self.use_eta else None
        action_mean, action_logstd, value = self.agent_c.forward(state, affordance, eta_input)
        
        return action_mean, action_logstd, value, eta, delta_eta
    
    def get_action(self, state, point_cloud=None, goal=None, deterministic=False):
        """
        Sample action
        
        Returns:
            action, log_prob, value, eta, delta_eta, affordance
        """
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
            if point_cloud is not None:
                point_cloud = point_cloud.unsqueeze(0)
            if goal is not None:
                goal = goal.unsqueeze(0)
        
        affordance = None
        eta = None
        delta_eta = None
        
        if self.fg_extractor is not None and point_cloud is not None:
            with torch.no_grad():
                affordance, eta, _ = self.fg_extractor(point_cloud, goal=goal)
            
            if self.prev_eta is not None:
                delta_eta = eta - self.prev_eta
            self.prev_eta = eta.detach()
        
        eta_input = eta if self.use_eta else None
        
        # Call forward directly to avoid double single-detection
        action_mean, action_logstd, value = self.agent_c.forward(state, affordance, eta_input)
        
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
            if eta is not None:
                eta = eta.squeeze(0)
            if delta_eta is not None:
                delta_eta = delta_eta.squeeze(0)
        
        if single and affordance is not None:
            affordance = affordance.squeeze(0)
        
        return action, log_prob, value, eta, delta_eta, affordance
    
    def evaluate_actions(self, states, actions, affordances=None, etas=None):
        """Evaluate actions for PPO update"""
        return self.agent_c.evaluate_actions(states, actions, affordances, etas)
