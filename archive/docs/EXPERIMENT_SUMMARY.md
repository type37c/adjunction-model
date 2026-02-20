# 実験結果総括 / Experiment Summary

**日付 / Date:** 2026-02-19

## 概要 / Overview

本ドキュメントは、Agent Cの再設計と動的F/Gの実験結果を総括する。Phase 2.1の失敗を受けて、環境とエージェントを根本から再設計し、複数の実験を実施した。

This document summarizes the redesign of Agent C and the experimental results of dynamic F/G. Following the failure of Phase 2.1, we fundamentally redesigned the environment and agent, conducting multiple experiments.

---

## 実験1: Step 2 v2 - Agent Cの再設計 / Experiment 1: Step 2 v2 - Agent C Redesign

**ディレクトリ / Directory:** `experiments/step2_v2_redesign/`

### 動機 / Motivation

Phase 2.1のStep 2は壊滅的に失敗した（報酬-84930、成功率0%）。根本原因は以下の通り：

Phase 2.1 Step 2 failed catastrophically (reward -84930, success rate 0%). Root causes:
- トルク制御が学習困難すぎる / Torque control too difficult to learn
- 報酬関数が不適切（単純な距離） / Inappropriate reward function (simple distance)
- 状態表現が貧弱（end-effector位置なし） / Poor state representation (no end-effector position)

### 再設計の要点 / Key Redesign Points

#### 環境 / Environment
1. **位置制御への変更** / Position control
   - トルク制御 → 位置制御（関節角度を直接指定）
   - Torque control → Position control (directly specify joint angles)

2. **改善された報酬関数** / Improved reward function
   ```python
   reward = distance_improvement * 10.0  # 距離改善量ベース
   if distance < 0.1:  # 成功
       reward += 50.0
   ```

3. **豊かな状態表現** / Rich state representation
   - 関節角度、関節速度、end-effector位置、目標位置、相対ベクトル
   - Joint angles, velocities, EE position, target position, relative vector

#### Agent C v2
1. **MLPアーキテクチャ** / MLP architecture
   - LSTMを削除（Reachingはマルコフ的タスク）
   - Removed LSTM (Reaching is a Markovian task)

2. **F/G統合** / F/G integration
   - Affordance (32次元) + η (1次元) を状態に追加
   - Added affordance (32-dim) + η (1-dim) to state

3. **二重報酬** / Dual reward
   - 外発的報酬（タスク達成）+ 内発的報酬（Δη）
   - Extrinsic (task) + intrinsic (Δη) rewards

### 結果 / Results

| 指標 / Metric | ベースライン / Baseline | F/G-Enhanced | 改善 / Improvement |
|--------------|------------------------|--------------|-------------------|
| 最終報酬 / Final Reward | **2.69** | -6.22 | ❌ 悪化 / Worse |
| 成功率 / Success Rate | **3%** | 0% | ❌ 悪化 / Worse |
| 最終距離 / Final Distance | **0.45m** | 1.27m | ❌ 悪化 / Worse |
| 訓練時間 / Training Time | 5分 / 5min | 67分 / 67min | 13倍遅い / 13x slower |

### 分析 / Analysis

**ベースラインの成功:**
- 環境の再設計により、Phase 2.1の壊滅的失敗（-84930）から大幅改善（2.69）
- Environment redesign drastically improved from catastrophic failure (-84930) to (2.69)

**F/G統合の失敗:**
- F/G特徴量が学習を阻害
- F/G features hindered learning
- 根本原因：**表現空間とタスク空間のミスマッチ**
- Root cause: **Mismatch between representation space and task space**
  - F/Gは静的形状のaffordanceを学習
  - F/G learned affordance of static shapes
  - Reachingは動的な運動タスク
  - Reaching is a dynamic motion task

**詳細:** `experiments/step2_v2_redesign/ANALYSIS.md`

---

## 実験2: Dynamic F/G - 動的表現の学習 / Experiment 2: Dynamic F/G - Learning Dynamic Representations

**ディレクトリ / Directory:** `experiments/dynamic_fg/`

### 動機 / Motivation

Step 2 v2の失敗から、仮説を立てた：「F/Gの表現空間（静的形状）とタスク空間（動的運動）が整合していない」。解決策として、F/Gを時系列点群で訓練し、動的な到達可能性を表現させる。

From Step 2 v2 failure, we hypothesized: "F/G's representation space (static shapes) mismatches task space (dynamic motion)". Solution: train F/G on temporal point clouds to represent dynamic reachability.

### 設計 / Design

#### Dynamic F/G
```python
class TemporalFunctorF:
    # 時系列点群 (T, N, 3) → affordance
    # Temporal point cloud (T, N, 3) → affordance
    
class TemporalFunctorG:
    # affordance + action → 次状態（EE位置、距離）
    # affordance + action → next state (EE position, distance)
```

#### オンライン学習 / Online Learning
- データセット生成が遅すぎる（2.2秒/軌跡）
- Dataset generation too slow (2.2s/trajectory)
- 代わりに、ランダムポリシーで環境と相互作用しながらF/Gを訓練
- Instead, train F/G while interacting with environment using random policy

### 結果 / Results

#### Dynamic F/G訓練 / Dynamic F/G Training
- **F/G Loss:** 0.123 → 0.068（収束）
- **F/G Loss:** 0.123 → 0.068 (converged)
- 1000エピソード、約80分
- 1000 episodes, ~80 min

#### Agent C v3訓練 / Agent C v3 Training

| 指標 / Metric | ベースライン / Baseline | Dynamic F/G | 改善 / Improvement |
|--------------|------------------------|-------------|-------------------|
| 最高報酬 / Peak Reward | **27.63** (ep 800) | -3.44 | ❌ 大幅悪化 / Much worse |
| 最高成功率 / Peak Success | **25%** (ep 800) | 0% | ❌ 失敗 / Failed |
| 最終報酬 / Final Reward | -4.26 | **-4.78** | ❌ 悪化 / Worse |
| 最終距離 / Final Distance | 1.24m | **2.05m** | ❌ 悪化 / Worse |
| ηの推移 / η Trajectory | 3.3 → 安定 / stable | 3.3 → **14.7** | ❌ 発散 / Diverged |

### 分析 / Analysis

**Dynamic F/Gも失敗:**
- 静的F/Gと同様に、学習を改善しなかった
- Like static F/G, failed to improve learning
- むしろ悪化させた（ηの発散、学習の不安定化）
- Actually worsened (η divergence, learning instability)

**根本的問題の再考:**
1. **ηの発散** / η divergence
   - 予測誤差が累積（3.3 → 14.7）
   - Prediction error accumulated (3.3 → 14.7)
   
2. **高次元化の弊害** / Curse of dimensionality
   - 状態次元が16 → 49（3倍）に増加
   - State dimension increased 16 → 49 (3x)
   - 学習が困難に
   - Learning became difficult
   
3. **タスクミスマッチ** / Task mismatch
   - Reachingタスクは単純すぎる
   - Reaching task too simple
   - F/Gの能力を検証できない
   - Cannot validate F/G's capabilities

**詳細:** `experiments/dynamic_fg/ANALYSIS.md`

---

## 重要な洞察 / Key Insights

### 1. 随伴の成立条件 / Conditions for Adjunction

随伴が成立するには、表現空間の整合性だけでは不十分。以下が必要：

Alignment of representation spaces alone is insufficient for adjunction. Required:

1. **情報の有用性** / Information utility
   - Affordanceがタスクに関連する情報を含む
   - Affordance contains task-relevant information

2. **予測精度** / Prediction accuracy
   - ηが安定し、発散しない
   - η remains stable, does not diverge

3. **分布の一致** / Distribution alignment
   - 訓練データとテストデータの分布が整合
   - Training and test data distributions align

4. **次元の適切性** / Appropriate dimensionality
   - 高次元すぎると学習困難
   - Too high-dimensional makes learning difficult

### 2. タスクの複雑性 / Task Complexity

Reachingタスクは単純すぎて、F/Gの能力を検証できない。F/Gは以下のような複雑なタスクで検証すべき：

Reaching task too simple to validate F/G. F/G should be tested on complex tasks like:

- **把持 / Grasping:** 形状に応じた把持点の推論
- **組み立て / Assembly:** 部品間の幾何学的制約
- **道具使用 / Tool use:** 道具のaffordanceの理解

### 3. 知能 vs 知性 / Intelligence vs Wisdom

現在のモデルは「知性寄り」：

Current model leans toward "wisdom":

- **知能 / Intelligence:** 与えられたタスクを効率的に解く
- **知能 / Intelligence:** Efficiently solve given tasks
- **知性 / Wisdom:** 何が問題かを自分で見出す
- **知性 / Wisdom:** Identify problems autonomously

ηの改善を駆動力として、自分で「何を見るか」を選ぶ。しかし、タスクが与えられないと、目的地に着かない。

Driven by η improvement, autonomously chooses "what to observe". However, without given tasks, cannot reach destination.

**解決策:** 命令解釈機構の追加

**Solution:** Add command interpretation mechanism

---

## 次のステップ / Next Steps

詳細は `TODO.md` を参照。

See `TODO.md` for details.

### 短期 / Short-term
1. **言語層の導入** / Introduce language layer
   - CLIP text encoderを使用
   - Use CLIP text encoder
   - 命令 → goal embedding → affordance
   - Command → goal embedding → affordance

2. **Multi-task環境** / Multi-task environment
   - Reaching, Grasping, Pushingの3タスク
   - Three tasks: Reaching, Grasping, Pushing

### 中期 / Mid-term
1. **Goal-conditioned F/G** / Goal-conditioned F/G
   - 言語条件付きaffordance
   - Language-conditioned affordance

2. **複雑なタスクでの検証** / Validation on complex tasks
   - 把持、組み立て、道具使用
   - Grasping, assembly, tool use

### 長期 / Long-term
1. **η-grounded goal vectors**
   - Goal vectorがηを通じて意味を獲得
   - Goal vectors acquire meaning through η

2. **階層的な目的分解** / Hierarchical goal decomposition
   - 抽象的命令 → 具体的なηプロファイル
   - Abstract commands → concrete η profiles

---

## 結論 / Conclusion

今日の実験は、F/G統合の難しさを明らかにした。しかし、重要な洞察も得られた：

Today's experiments revealed the difficulty of F/G integration. However, we gained important insights:

1. **環境の再設計は成功** / Environment redesign succeeded
   - 位置制御、改善された報酬、豊かな状態表現
   - Position control, improved reward, rich state representation

2. **F/G統合は失敗** / F/G integration failed
   - 静的F/Gも動的F/Gも学習を改善せず
   - Neither static nor dynamic F/G improved learning

3. **根本原因を特定** / Identified root cause
   - タスクミスマッチ、次元の呪い、ηの発散
   - Task mismatch, curse of dimensionality, η divergence

4. **解決策を提案** / Proposed solutions
   - 言語層の導入、複雑なタスク、η-grounding
   - Language layer, complex tasks, η-grounding

次のフェーズでは、言語層を導入し、F/Gに「何をすべきか」を伝える機構を実装する。

In the next phase, we will introduce a language layer and implement a mechanism to tell F/G "what to do".

---

**作成者 / Author:** Manus AI Agent  
**レビュー / Review:** Pending
