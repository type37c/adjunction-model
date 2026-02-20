# Step 2 v2: Agent C再設計実験

## 概要

本実験では、Agent Cを再設計し、F/G特徴量（affordance + η）の統合が強化学習の性能を向上させるかを検証した。Phase 1.5で訓練されたFunctor F/Gを用いて、PyBulletのReachingタスクにおけるAgent Cの学習を評価した。

## 実験設定

### 環境（`env/`）

**RobotArm (`robot_arm.py`):**
- 3自由度プレーナーアーム（XZ平面）
- Y軸周りの回転関節
- 位置制御（トルク制御から改善）
- リンク長: [0.3, 0.3, 0.3]

**ReachingEnv (`reaching_env.py`):**
- 目標: ロボットのend-effectorを目標物体に近づける
- 状態空間（16次元）: end-effector位置(3), 物体位置(3), 相対ベクトル(3), 正規化距離(1), 関節角度(3), 関節速度(3)
- 行動空間（3次元）: 関節角度の変化量 Δθ ∈ [-1, 1]^3
- 報酬関数: r(t) = 10·Δd + 100·success - 0.1·step - 0.01·||a||²
  - Δd: 距離改善量
  - success: 成功ボーナス（距離 < 0.1m）
  - step: ステップペナルティ
  - ||a||²: 行動の大きさペナルティ
- 最大ステップ数: 100
- 点群: 128点（F/G統合版のみ）

### Agent C v2 (`agent/`)

**AgentC_v2 (`agent_c_v2.py`):**
- アーキテクチャ: MLP Actor-Critic（LSTMなし）
- 入力: 状態ベクトル（16次元）または 状態 + affordance(32) + η(1) = 49次元
- 共有エンコーダ: [49→256→256]
- Actor: [256→128→3] + Tanh（[-1, 1]に制限）
- Critic: [256→128→1]
- パラメータ数: 145,001

**FGFeatureExtractor (`agent_c_v2.py`):**
- FunctorF_v2: 点群(N, 3) → affordance(32)
- FunctorG: affordance(32) → goal(3)
- η: 再構成誤差 ||goal - obj_pos||²

**PPO v2 (`ppo_v2.py`):**
- 最適化: Adam (lr=3e-4)
- エポック数: 10
- ミニバッチサイズ: 64
- クリップ係数: 0.2
- GAE (γ=0.99, λ=0.95)
- 報酬正規化: あり
- 内発的報酬（F/G版のみ）: r_int(t) = α_int·(-Δη(t)), α_int=0.1

### 訓練設定

**共通:**
- エピソード数: 1500
- 更新間隔: 10エピソード
- シード: 42

**Baseline:**
- 状態: 16次元ベクトルのみ
- 訓練時間: 5.1分

**F/G-Enhanced:**
- 状態: 16次元 + affordance(32) + η(1) = 49次元
- 内発的報酬: α_int = 0.1
- 訓練時間: 67.0分

## 実験結果

| メトリック | Baseline | F/G-Enhanced | 差分 |
|-----------|----------|--------------|------|
| 最終平均報酬 | **2.69** | -6.22 | -8.91 |
| 成功率 | **3%** | 0% | -3% |
| 最終平均距離 | **0.4534** | 1.2705 | +0.817 |
| 総成功数 | **61/1500** | 21/1500 | -40 |
| ベスト平均報酬 | 18.87 | 18.83 | -0.04 |
| 訓練時間 | 5.1分 | 67.0分 | +61.9分 |

**結論:** F/G特徴量の統合は、予想に反してAgent Cの学習を**阻害**した。

## ファイル構成

```
step2_v2_redesign/
├── README.md              # 本ファイル
├── ANALYSIS.md            # 詳細な結果分析と仮説
├── env/
│   ├── robot_arm.py       # 3-DOFロボットアーム
│   ├── reaching_env.py    # Reaching環境
│   └── __init__.py
├── agent/
│   ├── agent_c_v2.py      # Agent C v2（MLP Actor-Critic + F/G統合）
│   ├── ppo_v2.py          # PPO v2トレーナー
│   └── __init__.py
├── train.py               # 訓練スクリプト
├── test_env.py            # 環境テストスクリプト
├── run_experiment.py      # 比較実験ランナー
├── checkpoints_baseline/  # ベースラインチェックポイント
├── checkpoints_with_fg/   # F/G-enhancedチェックポイント
└── results/
    ├── baseline_stats.json
    ├── fg_enhanced_stats.json
    ├── baseline_training_curves.png
    ├── fg_enhanced_training_curves.png
    └── comparison_plot.png
```

## 使用方法

### 環境テスト
```bash
python test_env.py
```

### ベースライン訓練
```bash
python train.py --mode baseline --episodes 1500 --update-interval 10 \
    --checkpoint-dir checkpoints_baseline --results-dir results --seed 42
```

### F/G-Enhanced訓練
```bash
python train.py --mode fg_enhanced --episodes 1500 --update-interval 10 \
    --checkpoint-dir checkpoints_with_fg --results-dir results \
    --fg-checkpoint ../phase1.5_fg_retraining/checkpoints/phase1.5_best.pth --seed 42
```

### 比較実験（両方を順次実行）
```bash
python run_experiment.py
```

## 依存関係

- Python 3.11
- PyTorch 2.6+
- PyBullet
- NumPy
- Matplotlib
- tqdm

## 主要な発見

本実験から得られた重要な知見は以下の通りである。

**環境再設計の成功:** 位置制御への変更、改善された報酬関数、豊かな状態表現により、ベースラインAgent Cは以前のStep 2（報酬-84930、成功率0%）から劇的に改善した（報酬2.69、成功率3%）。これは環境設計の重要性を示している。

**F/G統合の失敗:** F/G特徴量の追加は学習を阻害した。初期学習（0-200エピソード）では急速な改善が見られたが、中期以降（200-1500エピソード）で性能が崩壊し、回復しなかった。

**表現空間のミスマッチ:** F/Gは静的な点群（物体の形状）で訓練されたが、Reachingタスクは動的な相対位置が重要である。この表現空間とタスク空間のミスマッチが失敗の主要因と考えられる。

**ηの無効性:** η（再構成誤差）は3.38-3.40の範囲でほぼ一定であり、Δηは学習信号として機能しなかった。ηは静的な形状理解の指標であり、動的な行動には無関係である。

**計算コストの問題:** F/G版はベースラインの13倍遅い（67分 vs 5分）。点群処理がボトルネックとなり、実用性に問題がある。

詳細な分析と仮説については`ANALYSIS.md`を参照されたい。

## 理論的含意

価値関数分析v3では「F/Gという川床の上をAgent Cという水が流れると、はじめて随伴が成立する」と予測されたが、本実験では随伴は成立しなかった。これは理論的枠組みの限界ではなく、F/Gの表現空間とAgent Cのタスク空間が整合していなかったためと考えられる。

**修正された理解:** 随伴が成立するには、F/Gの表現空間とAgent Cのタスク空間が整合している必要がある。静的な形状理解（F/G）と動的な行動制御（Reaching）のミスマッチが、本実験の失敗の根本原因である。

## 次のステップ

本実験の失敗から、以下の改善方向が示唆される。

**短期的改善:** affordanceのみを使用（ηを除外）、内発的報酬の削除、ハイパーパラメータ調整（mini_batch_size増加、learning rate減少）、affordanceの次元削減（PCA/Autoencoder）。

**長期的方向性:** タスク適合型F/Gの再訓練（動的点群、Reaching特化）、階層的強化学習（F/Gを高レベル目標設定に使用）、End-to-end学習（点群→行動）。

## 参照

- Phase 1.5: F/G Retraining (`../phase1.5_fg_retraining/`)
- 価値関数分析v3: `/home/ubuntu/adjunction-model/docs/value_function_analysis_v3.md`
- 元のStep 2: `../phase2.1_trajectory_prediction/step2_agent_integration/`
