# Adjunction Model: Physical-Semantic Adjunction for Embodied AI

**プロジェクト状況:** フル実装への突貫工事を開始（2026年2月20日）

---

## 現在の状況

### 実験の失敗と学び

2つの主要な実験（Step 2 v2、Dynamic F/G）が失敗に終わりました。しかし、外部エージェントとの議論を通じて、**より根本的な問題**が明らかになりました：

1. **保留構造の未実装**: 初期実験ノートの核心概念が実装されていなかった
2. **Counit εの未実装**: 随伴構造（unit η + counit ε）が不完全だった
3. **F/Gの凍結**: 「川床が削られる」プロセスが存在しなかった
4. **タスクミスマッチ**: Reachingタスクは形状理解が不要で、F/Gの能力を検証できなかった

詳細は以下のドキュメントを参照：
- `docs/initial_note_vs_current_gap_analysis.md`: 初期実験ノートと現状のギャップ分析
- `EXPERIMENT_SUMMARY.md`: 実験失敗のサマリー
- `THEORETICAL_DISCUSSIONS.md`: 理論的議論

---

## 新しい方向性: 初期ノートへの回帰

初期実験ノート（`docs/initial_experiment_note.md`）の理論的枠組みに**回帰**し、理論が示す方向性に沿って実装を進めます。

### 核心概念

1. **保留構造（Suspension Structure）**: 知性の本質を「未確定性を保持し、新しい意味を創発する能力」と定義
2. **Coherence Signal (η + ε)**: 随伴のunit ηとcounit εによる整合性の測定
3. **Coherence Breakdown**: ηの急激な増大が保留構造を起動させるトリガー
4. **川床の削れ**: F/Gが世界と相互作用した結果によって動的に更新される

---

## フル実装計画: 4週間の突貫工事

MVP方式ではなく、すべての要素を並行して実装し、徐々に完成度を高める方針です。

詳細は `docs/full_implementation_blitz_plan.md` を参照。

### Week 1: 双方向再構成 + 脱出部屋環境の骨格

**目標:**
- εが計算できる
- 脱出部屋環境でエージェントが動ける（ランダム行動でも可）

**実装:**
- [x] `src/models/bidirectional_fg.py`: 双方向再構成の実装
- [x] `experiments/week1_bidirectional/train_bidirectional_fg.py`: 訓練スクリプト
- [ ] `src/envs/escape_room_env.py`: 脱出部屋環境の骨格
- [ ] Week 1の実験実行

**成功基準:**
- [ ] εが訓練中に収束する（0.1以下）
- [ ] ηとεの両方が低い行動が存在する
- [ ] 脱出部屋環境でランダム行動が実行できる

---

### Week 2: 保留構造 + Agent Cの提案生成メカニズム

**目標:**
- ηが閾値を超えたら行動が止まる
- Agent Cが複数の行動候補を生成し、εとηでフィルタリングできる

**実装:**
- [ ] `src/models/suspension_structure.py`: 保留構造の実装
- [ ] `src/training/suspension_training_loop.py`: 保留を含む訓練ループ
- [ ] `src/models/proposal_agent.py`: 提案生成Agent C
- [ ] Week 2の実験実行

**成功基準:**
- [ ] ηが閾値を超えたら行動が止まる
- [ ] 保留バッファにデータが溜まる
- [ ] Agent Cが複数の行動候補を生成できる

---

### Week 3: F/Gの動的更新 + 内的ループと外的ループの統合

**目標:**
- 保留中にF/Gが更新される（「川床が削られる」）
- 内的ループ（保留中の提案生成）と外的ループ（実際の行動）が統合される

**実装:**
- [ ] `src/training/dynamic_fg_update.py`: F/Gの動的更新
- [ ] `src/training/dual_loop_training.py`: 二重ループの訓練
- [ ] Week 3の実験実行

**成功基準:**
- [ ] 保留中にF/Gが更新され、ηが低下する
- [ ] 内的ループで保留が解除される
- [ ] 外的ループの予測誤差に基づいてF/Gが更新される

---

### Week 4: 全体の統合とPhase 0-2の実験実行

**目標:**
- すべての要素を統合し、脱出部屋実験を完遂する
- 初期ノートのPhase 0-2を検証する

**実装:**
- [ ] `src/envs/escape_room_env.py`: 完成版の脱出部屋環境
- [ ] `experiments/phase0_basic_learning/`: Phase 0の実験
- [ ] `experiments/phase1_unknown_objects/`: Phase 1の実験
- [ ] `experiments/phase2_creative_solution/`: Phase 2の実験

**成功基準:**
- [ ] Phase 0: F/Gが既知の形状を学習できる
- [ ] Phase 1: 未知の形状に対してaffordanceを推論できる
- [ ] Phase 2: 制約下で保留構造が起動し、創造的解決が生まれる

---

## ディレクトリ構造

```
adjunction-model/
├── src/
│   ├── models/
│   │   ├── bidirectional_fg.py          # Week 1 ✅
│   │   ├── suspension_structure.py      # Week 2
│   │   ├── proposal_agent.py            # Week 2
│   │   └── ...
│   ├── envs/
│   │   ├── escape_room_env.py           # Week 1-4
│   │   └── ...
│   ├── training/
│   │   ├── suspension_training_loop.py  # Week 2
│   │   ├── dynamic_fg_update.py         # Week 3
│   │   ├── dual_loop_training.py        # Week 3
│   │   └── ...
│   └── utils/
│       └── ...
├── experiments/
│   ├── week1_bidirectional/             # Week 1 (進行中)
│   │   ├── train_bidirectional_fg.py    # ✅
│   │   └── ...
│   ├── week1_escape_room/               # Week 1
│   ├── week2_suspension/                # Week 2
│   ├── week2_proposal/                  # Week 2
│   ├── week3_dynamic_fg/                # Week 3
│   ├── week3_dual_loop/                 # Week 3
│   ├── phase0_basic_learning/           # Week 4
│   ├── phase1_unknown_objects/          # Week 4
│   └── phase2_creative_solution/        # Week 4
├── docs/
│   ├── initial_experiment_note.md       # 初期実験ノート
│   ├── initial_note_vs_current_gap_analysis.md  # ギャップ分析
│   ├── full_implementation_blitz_plan.md        # フル実装計画
│   └── ...
├── EXPERIMENT_SUMMARY.md                # 実験失敗のサマリー
├── THEORETICAL_DISCUSSIONS.md           # 理論的議論
├── TODO.md                              # タスクリスト
└── README.md                            # このファイル
```

---

## 次のステップ

### 即座に実行すべきこと（Week 1）

1. **双方向再構成の訓練**
   ```bash
   cd /home/ubuntu/adjunction-model
   python experiments/week1_bidirectional/train_bidirectional_fg.py
   ```

2. **脱出部屋環境の骨格を作成**
   - [ ] `src/envs/escape_room_env.py`を作成
   - [ ] PyBulletの基本的なセットアップ
   - [ ] 簡易的な点群取得

3. **Week 1の結果を確認**
   - [ ] εが収束したか
   - [ ] ηとεの両方が低い行動が存在するか

---

## 重要なドキュメント

### 理論

- **初期実験ノート**: `docs/initial_experiment_note.md`
  - 保留構造、Coherence Signal、Phase 0-2の実験計画
  
- **理論的議論**: `THEORETICAL_DISCUSSIONS.md`
  - 随伴構造、Active Inferenceとの関係、実装への示唆

### 実験結果

- **実験サマリー**: `EXPERIMENT_SUMMARY.md`
  - Step 2 v2、Dynamic F/Gの失敗の詳細

- **ギャップ分析**: `docs/initial_note_vs_current_gap_analysis.md`
  - 初期ノートと現状のギャップ、失敗の原因

### 実装計画

- **フル実装計画**: `docs/full_implementation_blitz_plan.md`
  - Week 1-4の詳細な実装計画、タイムライン、成功基準

---

## 理論的背景

### 保留構造の5つの要件

| 要件 | 内容 | 実装における所在 |
|-----|------|---------------|
| **志向性** | 何かに向かう指向性を持つこと | エージェント状態 C |
| **差異への感受性** | 違いを検出できること | Coherence signal (η + ε) |
| **時間的持続** | 状態を一定期間保持できること | 動的ループ C(t) → C(t+1) |
| **自己と非自己の区別** | 自分の行動と環境の変化を区別できること | 随伴の非対称性 |
| **記憶** | 過去の経験を保持し参照できること | エージェント状態 C |

### Coherence Signal

- **η (unit)**: Shape → F → G → Shape'  
  形状の再構成誤差。低いηは「この形状は掴める」ことを意味する。

- **ε (counit)**: Action → F_inv → G_inv → Action'  
  行動の再構成誤差。低いεは「この行動は意味がある」ことを意味する。

- **η + ε**: 両方が低い行動は「この形状に対して意味があり、かつ整合的な行動」

---

## 連絡先・質問

プロジェクトに関する質問や議論は、GitHubのIssuesまたはDiscussionsで行ってください。

---

**最終更新:** 2026年2月20日  
**ステータス:** Week 1進行中（双方向再構成の実装完了、訓練準備中）
