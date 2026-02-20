# Dynamic F/G実験: 最終報告

## 概要

**目的:** 静的F/Gの失敗（Step 2 v2）を受けて、時系列点群から学習した**動的F/G**を用いてAgent Cの学習を改善する実験。

**仮説:** 表現空間（F/G特徴量）とタスク空間（Reachingタスク）のミスマッチが原因であれば、動的な到達可能性を表現するF/Gを訓練すれば改善されるはず。

**結果:** ❌ **失敗** - Dynamic F/GもAgent Cの学習を改善せず、むしろ悪化させた。

## ディレクトリ構造

```
dynamic_fg/
├── README.md                          # 本ファイル
├── ANALYSIS.md                        # 詳細な結果分析と理論的考察
├── models/
│   ├── dynamic_fg.py                  # Dynamic F/Gモデル（TemporalPointNet + FunctorG）
│   └── checkpoints_online/
│       └── dynamic_fg_online.pth      # 訓練済みDynamic F/Gモデル
├── scripts/
│   ├── train_agent_c_with_online_fg.py  # オンラインF/G訓練スクリプト
│   └── train_agent_c_v3.py            # Agent C v3訓練スクリプト
├── results_online/
│   ├── online_stats.json              # F/G訓練統計
│   └── online_training_curves.png     # F/G訓練曲線
├── results_agent_c_v3/
│   ├── agent_c_v3_stats.json          # Agent C v3統計
│   └── agent_c_v3_training_curves.png # Agent C v3訓練曲線
├── training_online.log                # F/G訓練ログ
├── training_baseline_v3.log           # ベースライン訓練ログ
└── training_with_fg_v3.log            # F/G-Enhanced訓練ログ
```

## 実験設計

### Phase 1: 動的F/Gの訓練

**データ収集:**
- ランダムポリシーで1000エピソードの軌跡を収集
- 各ステップで時系列点群ペア（t, t+1）を記録

**モデルアーキテクチャ:**
```python
DynamicFGModel:
  TemporalPointNet(pc_t, pc_t1) → affordance (32次元)
  FunctorG(affordance, action) → ee_pos_pred (3次元)
  η = ||ee_pos_pred - ee_pos_true||
```

**訓練:**
- 損失: MSE(予測end-effector位置, 実際のend-effector位置)
- 最適化: Adam, lr=0.001
- バッチサイズ: 32
- 更新頻度: 10エピソードごと

**結果:**
- F/G Loss: 0.123 → 0.047（収束）
- 訓練時間: 40分（1000エピソード）

### Phase 2: Agent C v3の訓練

**ベースライン:**
- 入力: 状態ベクトル（16次元）
- ネットワーク: MLP Actor-Critic（hidden_dim=256）
- アルゴリズム: PPO

**F/G-Enhanced:**
- 入力: 状態ベクトル（16次元） + affordance（32次元） + η（1次元） = 49次元
- ネットワーク: MLP Actor-Critic（hidden_dim=256）
- アルゴリズム: PPO
- F/Gは凍結（更新なし）

**訓練設定:**
- エピソード数: 1500
- 学習率: 3e-4
- PPO epochs: 10
- Mini-batch size: 64
- 環境: PyBullet Reachingタスク（位置制御、3-DOF プレーナーアーム）

## 結果

### 定量的比較

| 指標 | ベースライン | F/G-Enhanced | 変化 |
|------|-------------|--------------|------|
| 最終平均報酬 | -4.26 | -4.78 | -12% ❌ |
| 最終成功率 | 0% | 0% | 0% |
| 最終平均距離 | 1.238m | 2.046m | +65% ❌ |
| 最高性能（エピソード800） | 報酬27.63, 成功率25% | 報酬-3.44, 成功率0% | 大幅悪化 ❌ |
| 訓練時間 | 5.5分 | 65分 | 12倍 ❌ |

### 主要な観察

1. **F/G-Enhancedは全く学習しなかった:**
   - 成功率は全エピソードで0%
   - 報酬は-3～-5の範囲で停滞
   - 距離は増加傾向（悪化）

2. **ηの発散:**
   - 初期: 3.3
   - 最終: 14.7（4.5倍に増加）
   - F/Gの予測精度が低いか、分布シフトが発生

3. **ベースラインも不安定:**
   - エピソード800で最高性能（成功率25%）
   - その後、性能崩壊（成功率0%）
   - 環境設計の問題を示唆

4. **計算コストの増大:**
   - F/G-Enhancedは12倍遅い（点群処理のオーバーヘッド）

## 根本原因の分析

詳細は `ANALYSIS.md` を参照。要約：

1. **ηの発散:** F/Gの予測精度が低い、または分布シフトが発生
2. **高次元化の弊害:** 状態次元が16→49に増加し、サンプル効率が低下
3. **タスクミスマッチ（根本的問題）:** F/Gは把持可能性を表現するが、Reachingタスクには不要
4. **環境設計の問題:** ベースラインも失敗しており、報酬関数や探索戦略に問題

## 理論的含意

**随伴の成立条件（改訂版）:**

F/GとAgent Cの随伴が成立し、学習を改善するためには、以下の条件が**すべて**満たされる必要がある：

1. ✅ 表現空間の整合性（部分的改善）
2. ❌ 情報の有用性（把持情報は不要）
3. ❌ 予測精度（ηが発散）
4. ❌ 分布の一致（ランダム vs 学習済み）
5. ❌ 次元の適切性（16→49は過剰）

**結論:** Dynamic F/Gは条件1を部分的に改善したが、他の条件が満たされていないため、学習を改善しなかった。

## 次のステップ

### 推奨: F/Gの放棄（このタスクでは）

Reachingタスクは、F/Gの能力を検証するには**単純すぎる**。状態ベクトル（16次元）で十分に表現可能であり、F/Gの複雑さは不要。

**F/Gが有効な可能性のあるタスク:**
- 把持（Grasping）
- 組み立て（Assembly）
- 道具使用（Tool Use）
- 複雑な形状操作（Complex Manipulation）

### 代替案: 環境とベースラインの改善

F/Gを諦める前に、まず**環境とベースラインを改善**すべき：

1. **報酬関数の改善:**
   - Sparse reward + potential-based shaping
   - 距離改善量だけでなく、方向性も考慮

2. **探索戦略の追加:**
   - ε-greedy
   - Entropy bonus
   - Curiosity-driven exploration

3. **ハイパーパラメータ最適化:**
   - 学習率、clip_epsilon、entropy_coef
   - Grid search or Bayesian optimization

4. **カリキュラム学習:**
   - 近い目標から遠い目標へ
   - 成功閾値を徐々に厳しくする

5. **エピソード数の増加:**
   - 1500 → 5000エピソード

## 使用方法

### 1. Dynamic F/Gの訓練

```bash
cd /home/ubuntu/adjunction-model/experiments/dynamic_fg
python3 scripts/train_agent_c_with_online_fg.py
```

出力:
- `models/checkpoints_online/dynamic_fg_online.pth`
- `results_online/online_stats.json`
- `results_online/online_training_curves.png`

### 2. Agent C v3の訓練（ベースライン）

```bash
python3 scripts/train_agent_c_v3.py --episodes 1500 --hidden-dim 256 --lr 3e-4
```

### 3. Agent C v3の訓練（F/G-Enhanced）

```bash
python3 scripts/train_agent_c_v3.py --episodes 1500 --use-fg \
    --fg-checkpoint models/checkpoints_online/dynamic_fg_online.pth \
    --affordance-dim 32 --hidden-dim 256 --lr 3e-4
```

## 依存関係

- PyBullet
- PyTorch
- NumPy
- Matplotlib
- tqdm

Step 2 v2の環境とエージェントを再利用：
- `experiments/step2_v2_redesign/env/`
- `experiments/step2_v2_redesign/agent/`

## 関連実験

- **Step 2 v2 (静的F/G):** `experiments/step2_v2_redesign/`
  - 結果: F/Gは学習を阻害（報酬2.69 → -6.22）
  - 原因: 表現空間とタスク空間のミスマッチ

- **Phase 1.5 (F/G訓練):** `experiments/phase1.5_fg_retraining/`
  - 結果: F/Gは物理的相互作用に応答（η成功）
  - ただし、静的形状ベース

## 参考文献

- 価値関数分析v3: `notes/value_function_analysis_v3.md`
- 随伴理論: `notes/adjunction_theory.md`

## ライセンス

MIT License
