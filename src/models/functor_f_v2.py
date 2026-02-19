"""
FunctorF_v2: 改修版のFunctor F

改修内容:
1. 近傍構造の導入: k近傍を用いて局所的な幾何学構造を捉える
2. 目的条件付け: 目的ベクトルgによるFiLM変調

哲学的根拠:
- ハイデガーの「道具連関」: 道具は孤立せず、他の道具との参照連関の中にある
- ハイデガーの「〜のために」: 目的がアフォーダンスの現れ方を構成する
"""

import torch
import torch.nn as nn


class FunctorF_v2(nn.Module):
    """
    改修版のFunctor F: 物理空間からアフォーダンス空間への写像
    
    主な改修点:
    1. 近傍構造の導入 (k-NN based local feature extraction)
    2. 目的条件付け (FiLM modulation with goal vector)
    
    Args:
        input_dim (int): 入力点群の次元 (デフォルト: 3)
        affordance_dim (int): アフォーダンス表現の次元 (デフォルト: 32)
        goal_dim (int): 目的ベクトルの次元 (デフォルト: 16)
        k (int): 近傍点の数 (デフォルト: 16)
        hidden_dim (int): 隠れ層の次元 (デフォルト: 128)
    """
    
    def __init__(
        self,
        input_dim=3,
        affordance_dim=32,
        goal_dim=16,
        k=16,
        hidden_dim=128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.affordance_dim = affordance_dim
        self.goal_dim = goal_dim
        self.k = k
        self.hidden_dim = hidden_dim
        
        # 局所特徴抽出: 点の座標 + 近傍との相対位置
        # Local feature extraction: point coordinates + relative positions to neighbors
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        
        # 目的による条件付け (FiLM変調)
        # Goal conditioning via FiLM modulation
        self.goal_to_gamma = nn.Linear(goal_dim, hidden_dim)
        self.goal_to_beta = nn.Linear(goal_dim, hidden_dim)
        
        # アフォーダンス表現への射影
        # Projection to affordance representation
        self.to_affordance = nn.Sequential(
            nn.Linear(hidden_dim, affordance_dim),
            nn.ReLU()
        )
    
    def forward(self, pos, goal=None):
        """
        Forward pass
        
        Args:
            pos (torch.Tensor): 点群座標 (B, N, input_dim)
            goal (torch.Tensor, optional): 目的ベクトル (B, goal_dim)
                Noneの場合は無条件（Phase 1互換モード）
        
        Returns:
            torch.Tensor: アフォーダンス表現 (B, N, affordance_dim)
        """
        B, N, D = pos.shape
        
        # 近傍探索 (k-nearest neighbors)
        # k-NN search
        dists = torch.cdist(pos, pos)  # (B, N, N)
        _, idx = dists.topk(self.k, dim=-1, largest=False)  # (B, N, k)
        
        # 近傍との相対位置を計算
        # Compute relative positions to neighbors
        # idx: (B, N, k) → (B, N, k, 1) → (B, N, k, D)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        
        # pos: (B, N, D) → (B, N, 1, D) → (B, N, N, D)
        pos_expanded = pos.unsqueeze(2).expand(-1, -1, N, -1)
        
        # 近傍点を収集: (B, N, k, D)
        neighbors = torch.gather(pos_expanded, 2, idx_expanded)
        
        # 相対位置: neighbors - pos
        relative = neighbors - pos.unsqueeze(2)  # (B, N, k, D)
        
        # 局所文脈: 近傍との相対位置の平均
        # Local context: average of relative positions to neighbors
        local_context = relative.mean(dim=2)  # (B, N, D)
        
        # 局所入力: 点の座標 + 局所文脈
        # Local input: point coordinates + local context
        local_input = torch.cat([pos, local_context], dim=-1)  # (B, N, 2*D)
        
        # 局所特徴の抽出
        # Extract local features
        features = self.local_encoder(local_input)  # (B, N, hidden_dim)
        
        # 目的による条件付け (FiLM変調)
        # Goal conditioning via FiLM modulation
        if goal is not None:
            gamma = self.goal_to_gamma(goal).unsqueeze(1)  # (B, 1, hidden_dim)
            beta = self.goal_to_beta(goal).unsqueeze(1)    # (B, 1, hidden_dim)
            features = gamma * features + beta
        
        # アフォーダンス表現への射影
        # Project to affordance representation
        affordances = self.to_affordance(features)  # (B, N, affordance_dim)
        
        return affordances
    
    def load_phase1_weights(self, phase1_checkpoint_path):
        """
        Phase 1のチェックポイントから重みをロードする（転移学習）
        
        注意: Phase 1のFunctorFは点ごとの独立処理なので、
        local_encoderの最初の層の重みの一部のみをロードできる。
        
        Args:
            phase1_checkpoint_path (str): Phase 1のチェックポイントのパス
        """
        checkpoint = torch.load(phase1_checkpoint_path, map_location='cpu')
        
        # Phase 1のモデルからFunctorFの重みを抽出
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # FunctorFの重みを抽出（キーに'functor_f'が含まれるもの）
        functor_f_state = {
            k.replace('functor_f.', ''): v 
            for k, v in state_dict.items() 
            if 'functor_f' in k
        }
        
        # 部分的にロード（互換性のある層のみ）
        # 注意: Phase 1のFunctorFは入力が(B, N, 3)で、
        # v2は(B, N, 6)なので、最初の層の重みは部分的にしか使えない
        print("Loading Phase 1 weights (partial transfer)...")
        print(f"Phase 1 FunctorF keys: {list(functor_f_state.keys())}")
        
        # 互換性チェック: 後続の層の重みをロード可能か確認
        # ここでは簡易的に、形状が一致する層のみをロード
        current_state = self.state_dict()
        loaded_keys = []
        
        for key, value in functor_f_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                current_state[key] = value
                loaded_keys.append(key)
        
        self.load_state_dict(current_state)
        print(f"Loaded {len(loaded_keys)} layers from Phase 1: {loaded_keys}")
        print(f"Randomly initialized: {set(current_state.keys()) - set(loaded_keys)}")


# テスト用のコード
if __name__ == "__main__":
    # FunctorF_v2のインスタンス化
    model = FunctorF_v2(
        input_dim=3,
        affordance_dim=32,
        goal_dim=16,
        k=16,
        hidden_dim=128
    )
    
    # ダミーデータでテスト
    batch_size = 4
    num_points = 512
    pos = torch.randn(batch_size, num_points, 3)
    goal = torch.randn(batch_size, 16)
    
    # Forward pass (目的あり)
    affordances_with_goal = model(pos, goal)
    print(f"Input shape: {pos.shape}")
    print(f"Goal shape: {goal.shape}")
    print(f"Output shape (with goal): {affordances_with_goal.shape}")
    
    # Forward pass (目的なし、Phase 1互換モード)
    affordances_without_goal = model(pos, goal=None)
    print(f"Output shape (without goal): {affordances_without_goal.shape}")
    
    # パラメータ数の確認
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
