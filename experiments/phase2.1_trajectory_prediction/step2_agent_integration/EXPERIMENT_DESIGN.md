# Step 2: Agent Cとの統合 - 実験設計

**作成日**: 2026年2月19日  
**目的**: Phase 1.5のF/Gのアフォーダンス特徴が、Agent Cのタスク達成に有用かを検証する

## 実験の問い

**「F/Gのアフォーダンス特徴を使うAgent C」は、「使わないAgent C」よりも速く学習するか？**

もし差が出れば、F/Gが提供するアフォーダンス特徴が実際に有用であることの証拠となる。

## 実験設計

### タスク: Object Reaching

最もシンプルなタスクから開始する。

- **目標**: ロボットアームの先端を、テーブル上のオブジェクトに近づける
- **報酬**: `-distance(end_effector, object)`（距離が近いほど高い報酬）
- **エピソード終了条件**: 
  - 距離が0.05m以下になったら成功（報酬+10）
  - 200ステップ経過したら終了
- **オブジェクト**: box, cup, bowl（ランダムに配置）

### なぜこのタスクか？

1. **単純だが意味がある**: Pickingよりも複雑（複数ステップが必要）だが、Container Fillingほど複雑ではない
2. **アフォーダンスが関与する**: オブジェクトの位置・形状を理解する必要がある
3. **学習曲線の差が出やすい**: 適度な難易度で、F/G特徴の有無による差が観察しやすい

### 2つのバリアント

#### Variant A: F/G特徴あり

```
点群 → FunctorF_v2(goal=reach) → アフォーダンス特徴(N, 32)
                                        ↓
                                   平均プーリング
                                        ↓
                                  グローバル特徴(32,)
                                        ↓
                        ロボット状態(joint angles, velocities)
                                        ↓
                                    結合(32+N_joints*2,)
                                        ↓
                                Agent C (LSTM + Actor-Critic)
                                        ↓
                                    行動(joint torques)
```

#### Variant B: F/G特徴なし（ベースライン）

```
ロボット状態(joint angles, velocities, object_position)
                    ↓
        Agent C (LSTM + Actor-Critic)
                    ↓
            行動(joint torques)
```

**重要**: Variant Bには、オブジェクトの位置を直接与える。これにより、F/G特徴の優位性が「視覚情報の有無」ではなく、**「アフォーダンス特徴の質」**に起因することを確認できる。

### Agent Cのアーキテクチャ

```python
class AgentC(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, hidden=None):
        lstm_out, hidden = self.lstm(obs, hidden)
        action_logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return action_logits, value, hidden
```

### 訓練設定

- **アルゴリズム**: PPO（Proximal Policy Optimization）
- **エピソード数**: 1000エピソード
- **バッチサイズ**: 32
- **学習率**: 3e-4
- **割引率（γ）**: 0.99
- **GAE λ**: 0.95
- **エポック数（PPO）**: 4
- **クリッピング（ε）**: 0.2

### 評価指標

1. **学習曲線**: エピソード報酬の移動平均（窓サイズ=50）
2. **成功率**: 最後100エピソードでの成功率
3. **収束速度**: 報酬が閾値（例: -5.0）を超えるまでのエピソード数
4. **最終性能**: 最後100エピソードの平均報酬

## 予測されるシナリオ

### シナリオA: F/G特徴が有用

- Variant Aの学習曲線がVariant Bよりも速く上昇する
- 収束速度が2倍以上速い
- 最終性能も高い

**解釈**: F/Gのアフォーダンス特徴が、オブジェクトの位置・形状を効果的に表現しており、Agent Cの学習を加速している。

### シナリオB: 差がない

- 両者の学習曲線がほぼ同じ
- 収束速度も最終性能も同等

**解釈**: F/Gのアフォーダンス特徴は、ロボット状態+オブジェクト位置と比較して、追加の情報を提供していない。Phase 1.5のF/Gは「順モデル」を学習したが、それはReachingタスクには不要な情報である。

### シナリオC: F/G特徴が妨害

- Variant Aの学習曲線がVariant Bよりも遅い
- 最終性能も低い

**解釈**: F/Gのアフォーダンス特徴にノイズが多く、Agent Cの学習を妨害している。Phase 1.5の訓練データ（push/pull/lift/topple）とReachingタスクの間にミスマッチがある。

## 実装の詳細

### ディレクトリ構造

```
step2_agent_integration/
├── EXPERIMENT_DESIGN.md
├── env/
│   ├── __init__.py
│   ├── reaching_env.py          # PyBullet Reaching環境
│   └── robot_arm.py              # ロボットアームの定義
├── agent/
│   ├── __init__.py
│   ├── agent_c.py                # Agent C（LSTM + Actor-Critic）
│   └── ppo.py                    # PPO実装
├── train_variant_a.py            # F/G特徴ありの訓練
├── train_variant_b.py            # F/G特徴なしの訓練
├── evaluate.py                   # 評価スクリプト
└── results/
    ├── variant_a_learning_curve.png
    ├── variant_b_learning_curve.png
    └── comparison.png
```

### 推定所要時間

- 環境実装: 2-3時間
- Agent C実装: 2-3時間
- 訓練（各バリアント）: 2-3時間
- 評価・分析: 1-2時間

**総所要時間**: 約1日

## 次のステップ

1. PyBullet Reaching環境を実装
2. Agent Cを実装
3. 2つのバリアントを訓練
4. 学習曲線を比較して結果を分析
