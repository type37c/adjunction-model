# TODO List: Prototype for Theoretical Validation

This document outlines the development tasks for building a prototype to validate the core theoretical claims of the Physical-Semantic Adjunction Model. The primary goal is to demonstrate the model's ability to address the symbol grounding problem, generalize to unknown objects, and exhibit creative problem-solving under constraint.

**原則**: 保留構造は設計しない、創発する条件を設計する。タスクの完璧な実装を待つより、不完全な実装で何が創発するかを観察し続ける方がこのモデルの思想に忠実である。

---

## 最優先: Agent Cの訓練方法の根本的見直し

2月13日の実験により、以下が判明した。

- OnlineLearnerがF/Gの重みを直接更新すると、Agent Cの役割を奪う（Conditional Adjunctionの原則に反する）
- OnlineLearnerを無効化すると、Agent Cの内部状態は変化するが、FiLMを通じたF/Gへの影響が微弱
- 原因: 訓練時にAgent Cのcontextがほぼ一定（1ステップのみ）で、逐次的な経験を積んでいない

### タスク

-   [ ] **逐次的訓練ループの設計**: Agent Cが複数の形状を逐次的に経験し、内部状態が蓄積される中でF/Gの性能が変化することを学習する訓練方法を設計する。ただし、「Coherenceを下げる方向に最適化する」のではなく、「経験を蓄積する」設計とする。
-   [ ] **FiLMの有効性検証**: 逐次的訓練後、Agent Cのcontext変化がFiLMを通じてF/Gの出力を有意に変えるかを検証する。

---

## 理論的課題: 目的空間Pの形式化

2月13日の議論により、不採用とされていた「目的空間P」が再浮上した。ただし、当初の「外部から目的を注入する空間」ではなく、以下のように再定義される。

- Pは Agent Cの内部状態の中で、経験から創発する構造
- 余白が先にあり、外部からの刺激を受け取り、Pを形成していく
- Pの枠組み（価値判断の軸）は設計する。内容は創発させる
- 言語は Pを経由して接地される（随伴の階層化ではない）

### タスク

-   [ ] **目的空間Pの理論的記述**: Agent Cの内部状態（RSSM の z, h）の中でPがどう創発するかを形式化する
-   [ ] **research_note_ja.md の改訂**: セクション3.5（目的空間は不要→必要に修正）とセクション5.5（言語は随伴の階層化→Pを経由した3層構造に修正）を更新する
-   [ ] **価値判断の軸の設計**: 「良いものを良いと感じる力」の実装。Coherence（破綻＝注意すべき）、Uncertainty（不確実＝探索すべき）に加え、他に必要な軸があるかを検討する

---

## Phase 0: Foundational Implementation (Replicating Knowns)

This phase focuses on establishing the basic architecture and ensuring that the core components (F, G, C) can be trained and function as expected on known data.

### Core Components

-   [x] **Environment Setup**: PyTorch, PyTorch Geometric をインストール済み
-   [x] **Adjoint Layer (F)**: GNN-based encoder 実装済み (`src/models/functor_f.py`)
-   [x] **Adjoint Layer (G)**: Conditional decoder 実装済み (`src/models/functor_g.py`)
-   [x] **Agent Layer (C) v1**: RSSM構造 実装済み (`src/models/agent_layer.py`)
-   [x] **Agent Layer (C) v2**: Priority原理 + 空間的Coherence Signal 実装済み (`src/models/agent_layer_v2.py`, `src/models/priority.py`)
-   [x] **Conditional Adjunction v1**: FiLM conditioning 実装済み (`src/models/conditional_adjunction.py`)
-   [x] **Conditional Adjunction v2**: Agent C v2統合版 実装済み (`src/models/conditional_adjunction_v2.py`)
-   [ ] **Action Selection**: EFEの実装。**注意**: EFEは「何を達成すべきか」という目的を暗黙に前提とするため、保留構造の原則と矛盾する可能性がある。慎重に設計すること。
-   [ ] **Loss Function**: 複合損失関数の実装。**注意**: 「Coherenceを最小化する」損失は保留構造を消失させる可能性がある。
-   [ ] **Training Loop**: 上記「最優先」の逐次的訓練ループとして再設計する。

## Phase 1: Theoretical Validation - Setting A (Zero-Shot Affordance)

-   [x] **test_coherence_signal.py**: 未知形状に対するCoherence Signal上昇を確認（+105.7%）
-   [x] **test_online_adaptation.py**: オンライン学習による適応能力向上を確認（-47.0%）

## Phase 2: Theoretical Validation - Setting B (Suspension Structure Emergence)

### 保留構造の4要件の検証状況

| 要件 | 状況 | 実験 | 備考 |
|:---|:---|:---|:---|
| 差異への感受性 | **強い証拠** | test_coherence_signal.py | +105.7% |
| 時間的持続性 | **強い証拠** | test_memory_recall.py | 安定した再現 |
| 創造的再構成 | **強い証拠** | test_online_adaptation.py | -47.0% |
| 志向性 | **進行中** | test_prioritization_v2.py | v2で原始的な志向性の兆候（Priority逆転、KL>0）を確認。FiLMの有効性が課題。 |
| 飽和 | **部分的** | test_saturation.py | 100回の曝露では不十分 |

### タスク

-   [ ] **志向性の再検証**: 逐次的訓練ループ完成後に test_prioritization_v2.py を再実行
-   [ ] **飽和の再検証**: 曝露回数を増やして再実行

## Phase 3: Theoretical Validation - Setting C (Symbol Grounding with Language)

**2月13日の議論により、このPhaseの設計は根本的に見直す必要がある。**

言語は随伴の要件を満たさない可能性がある（多対多の写像、身体からの切り離し）。言語接地は、随伴の階層化ではなく、目的空間Pを経由した3層構造として再設計する。

| 層 | 内容 | 身体との関係 |
|:---|:---|:---|
| Layer 1: 身体的随伴（F ⊣ G） | 形状⇄アフォーダンス | 身体に縛られている |
| Layer 2: 目的空間（P） | 経験から創発する「何をしたいか」の表現 | 身体から部分的に切り離されている |
| Layer 3: 言語（L） | Pの構造に名前をつけたもの | 身体なしでも操作可能 |

-   [ ] **目的空間Pの実装**: 上記「理論的課題」を先に完了する必要がある
-   [ ] **言語入力の設計**: 言語を「Pを書き換える命令」ではなく「Pを揺さぶる刺激」として設計する
-   [ ] **評価方法の再設計**: 従来のCLIP的アラインメントではなく、Pを経由した接地の評価方法を設計する

## Undecided / Needs Further Discussion

-   [ ] **Suspension Structure Emergence Verification**: Based on the new principle ("we don't design the suspension structure, we design the conditions for its emergence"), this task is redefined. The goal is to design and verify the conditions that force the suspension structure to emerge.
    -   [ ] **Condition 1 (Non-vanishing Coherence Signal)**: Design a learning environment or loss function where the coherence signal is never permanently zero, forcing the agent to continuously adapt.
    -   [ ] **Condition 2 (Breakdown is not Fatal)**: Ensure the agent's architecture can withstand and recover from coherence breakdowns without catastrophic failure (e.g., through robust state management in Layer C).
    -   [ ] **Condition 3 (Free Movement of Abstraction)**: Implement a mechanism that allows the agent to shift its level of abstraction (λ) in response to coherence signals, enabling it to re-frame problems at different granularities.

## References

[1] Kim, H., et al. (2024). *Zero-Shot Learning for the Primitives of 3D Affordance in General Objects*. arXiv:2401.12978.
[2] Sundermeyer, M., et al. (2021). *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*. arXiv:2103.14243.
[3] Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. arXiv:1903.02428.
[4] Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models*. arXiv:2301.04104.
[5] Çatal, O., et al. (2020). *Learning Generative State Space Models for Active Inference*. arXiv:2006.06520.
[6] Andries, J., et al. (2020). *Automatic Generation of Object Shapes With Desired Affordances*. Frontiers in Neurorobotics, 14, 22. doi: 10.3389/fnbot.2020.00022
