# Publication Roadmap: From Current State to Paper
# 論文化ロードマップ：現状から論文へ

**Date**: February 20, 2026  
**Target**: ICLR Tiny Papers 2026 (2週間計画) 🎯

---

## Current Status / 現状

✅ **完了**:
- 双方向F/G（η + ε）
- 保留構造
- Phase 0-1の実験
- 62%成功率（未知形状）

⚠️ **不足**:
- ベースライン比較
- 複数シード実行
- Phase 2の結果
- 統計的有意性検証

---

## Week 1: Additional Experiments / 追加実験

### Day 1-2: Baseline Implementations (ベースライン実装)

#### Baseline 1: Static F/G (静的F/G) - 最優先

**What**: 保留構造なし、F/G凍結

**Implementation**:
```python
# scripts/baselines/static_fg.py
class StaticFGAgent:
    """
    Agent with frozen F/G (no suspension, no adaptation)
    """
    def __init__(self, fg_model, policy):
        self.fg_model = fg_model
        self.fg_model.eval()  # Freeze F/G
        self.policy = policy
    
    def act(self, observation):
        with torch.no_grad():
            affordance = self.fg_model.F(observation)
        action = self.policy(affordance)
        return action
```

**Expected Result**:
- Phase 0: 同等（既知形状）
- Phase 1: **低下**（未知形状で適応できない）

**Effort**: 4時間

#### Baseline 2: Standard PPO (標準的なRL)

**What**: F/Gなし、直接observation → action

**Implementation**:
```python
# scripts/baselines/ppo_baseline.py
from stable_baselines3 import PPO

env = EscapeRoomEnv(...)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

**Expected Result**:
- Phase 0: 同等またはやや低い
- Phase 1: **大幅に低下**（汎化能力なし）

**Effort**: 2時間（Stable Baselines3使用）

#### Baseline 3: Active Inference (簡易版) - オプション

**What**: 自由エネルギー最小化

**Implementation**:
```python
# scripts/baselines/active_inference.py
class ActiveInferenceAgent:
    """
    Simplified Active Inference
    Minimize prediction error (similar to η)
    """
    def __init__(self, world_model):
        self.world_model = world_model
    
    def act(self, observation):
        # Generate action candidates
        candidates = self.generate_candidates()
        # Select action that minimizes prediction error
        action = min(candidates, key=lambda a: self.prediction_error(observation, a))
        return action
```

**Expected Result**:
- Phase 0: 同等
- Phase 1: やや低下（適応メカニズムが異なる）

**Effort**: 6時間

**Decision**: Phase 2の結果を見てから判断（時間があれば実装）

### Day 3-4: Phase 2 Implementation (Phase 2実装)

#### Phase 2.1: Gravity Change (重力変化)

**Implementation**:
```python
# core/envs/escape_room.py (update)
class EscapeRoomEnv:
    def __init__(self, ..., gravity=-9.8):
        self.gravity = gravity
        p.setGravity(0, 0, gravity)
```

**Experiments**:
1. Phase 0モデルをロード
2. gravity = -1.6（月面）でテスト
3. ηの変化を監視
4. 保留と適応を記録

**Expected Result**:
- Initial η: > 0.1（物理が変わったことを検出）
- Suspension: 発動
- F/G adaptation: ηが低下
- Success rate: 回復

**Effort**: 4時間

#### Phase 2.2: Friction Change (摩擦変化)

**Implementation**:
```python
# core/envs/escape_room.py (update)
def _create_object(self, object_type, friction=0.5):
    ...
    p.changeDynamics(object_id, -1, lateralFriction=friction)
```

**Experiments**:
1. friction = 0.1（氷上）でテスト
2. 同様の分析

**Effort**: 2時間（重力変化の後なので簡単）

#### Phase 2.3: Mass Change (質量変化) - オプション

**Effort**: 2時間

**Decision**: 時間があれば実装

### Day 5: Multiple Seeds (複数シード実行)

**What**: 5シード × Phase 0-2 × すべてのベースライン

**Implementation**:
```python
# scripts/run_multiple_seeds.py
seeds = [0, 1, 2, 3, 4]
results = []

for seed in seeds:
    set_seed(seed)
    result = run_experiment(phase=0, method='ours')
    results.append(result)

# Compute mean and std
mean_success = np.mean([r['success_rate'] for r in results])
std_success = np.std([r['success_rate'] for r in results])
```

**Effort**: 6時間（並列実行可能）

**Expected Output**:
- Mean ± Std for all methods and phases
- Statistical significance tests (t-test)

### Day 6-7: Data Analysis and Visualization (データ分析と可視化)

**Tasks**:
1. 結果の集計
2. 統計的有意性検証（t-test, ANOVA）
3. 図表の作成
   - Success rate comparison (bar plot with error bars)
   - η trajectory (line plot)
   - Suspension count (bar plot)
   - F/G update count (bar plot)

**Effort**: 8時間

---

## Week 2: Paper Writing / 論文執筆

### Day 1: Paper Structure (論文構成)

**ICLR Tiny Papers Format**: 4ページ（参考文献除く）

**Structure**:
1. **Abstract** (150 words)
   - Problem: Adaptation to unknown shapes/physics
   - Solution: Suspension structure with bidirectional adjunction
   - Results: 62% success on unknown shapes, adaptation confirmed

2. **Introduction** (0.5 pages)
   - Motivation: Embodied AI needs to adapt to novel situations
   - Gap: Existing methods (Active Inference, World Models) lack explicit adaptation mechanism
   - Contribution: Suspension structure as a new paradigm

3. **Method** (1.5 pages)
   - Bidirectional Adjunction (η + ε)
   - Suspension Structure (trigger, buffer, adaptation)
   - Escape Room Environment

4. **Experiments** (1.5 pages)
   - Phase 0-2 results
   - Baseline comparisons
   - Ablation studies (if time permits)

5. **Discussion** (0.5 pages)
   - Why suspension works
   - Limitations
   - Future work

6. **References**

**Effort**: 4時間

### Day 2-4: Writing (執筆)

#### Day 2: Method Section

**Content**:
- Bidirectional adjunction formulation
- Suspension mechanism algorithm
- F/G architecture details

**Figures**:
- Figure 1: System overview (F, G, suspension, agent)
- Figure 2: Suspension mechanism flowchart

**Effort**: 6時間

#### Day 3: Experiments and Results

**Content**:
- Experimental setup
- Phase 0-2 results
- Baseline comparisons
- Statistical analysis

**Figures**:
- Figure 3: Success rate comparison (bar plot with error bars)
- Figure 4: η trajectory in Phase 1 (line plot showing suspension and adaptation)

**Tables**:
- Table 1: Quantitative results (mean ± std)

**Effort**: 6時間

#### Day 4: Introduction, Abstract, Discussion

**Content**:
- Introduction: Motivation and contribution
- Abstract: Concise summary
- Discussion: Interpretation and limitations

**Effort**: 6時間

### Day 5-6: Revision and Polishing (推敲と仕上げ)

**Tasks**:
1. 全体の流れを確認
2. 図表の調整
3. 文章の推敲
4. 参考文献の整理
5. 共著者レビュー（もしいれば）

**Effort**: 8時間

### Day 7: Final Check and Submission (最終チェックと提出)

**Tasks**:
1. ICLR Tiny Papers形式チェック
2. 匿名化（著者名、所属を削除）
3. 補足資料の準備（コード、追加結果）
4. 提出

**Effort**: 4時間

---

## Detailed Experimental Plan / 詳細な実験計画

### Experiments to Run / 実行する実験

| Experiment | Method | Phase | Seeds | Episodes | Estimated Time |
|-----------|--------|-------|-------|----------|----------------|
| 1 | Ours | 0 | 5 | 200 | 10 min |
| 2 | Ours | 1 | 5 | 200 | 10 min |
| 3 | Ours | 2 (gravity) | 5 | 200 | 10 min |
| 4 | Ours | 2 (friction) | 5 | 200 | 10 min |
| 5 | Static F/G | 0 | 5 | 200 | 10 min |
| 6 | Static F/G | 1 | 5 | 200 | 10 min |
| 7 | Static F/G | 2 (gravity) | 5 | 200 | 10 min |
| 8 | PPO | 0 | 5 | 10000 steps | 30 min |
| 9 | PPO | 1 | 5 | 10000 steps | 30 min |

**Total Time**: ~3 hours（並列実行すれば1時間）

### Key Metrics to Report / 報告する主要指標

| Metric | Description | Expected Result |
|--------|-------------|-----------------|
| Success Rate | % of successful episodes | Ours > Baselines in Phase 1-2 |
| Average η | Mean η over episodes | Ours: decreases after adaptation |
| Suspension Count | # of suspension triggers | Ours: > 0 in Phase 1-2 |
| F/G Update Count | # of F/G adaptations | Ours: > 0 in Phase 1-2 |
| Adaptation Time | Steps until η < threshold | Ours: < 10 steps |

### Statistical Tests / 統計検定

**Comparisons**:
1. Ours vs Static F/G (Phase 1)
2. Ours vs PPO (Phase 1)
3. Ours: Phase 0 vs Phase 1 (no significant drop)

**Test**: Two-sample t-test (α = 0.05)

**Expected**:
- Ours significantly better than baselines in Phase 1 (p < 0.05)
- Ours: no significant difference between Phase 0 and Phase 1 (p > 0.05)

---

## Figures and Tables / 図表

### Figure 1: System Overview

**Content**:
- F/G architecture
- Suspension mechanism
- Agent C
- Escape room environment

**Style**: Block diagram

### Figure 2: Suspension Mechanism

**Content**:
- Flowchart showing:
  1. Observe → Compute η
  2. If η > threshold → Enter suspension
  3. Buffer observations
  4. Fine-tune F/G
  5. If η < threshold → Exit suspension

**Style**: Flowchart

### Figure 3: Success Rate Comparison

**Content**:
- Bar plot with error bars
- X-axis: Phase 0, Phase 1, Phase 2 (gravity), Phase 2 (friction)
- Y-axis: Success rate (%)
- Bars: Ours, Static F/G, PPO

**Expected**:
- Phase 0: All similar
- Phase 1-2: Ours > Baselines

### Figure 4: η Trajectory in Phase 1

**Content**:
- Line plot
- X-axis: Episode
- Y-axis: Average η
- Annotations: Suspension events, F/G updates

**Expected**:
- η spikes when encountering unknown shapes
- η decreases after F/G adaptation

### Table 1: Quantitative Results

| Method | Phase 0 | Phase 1 | Phase 2 (gravity) | Phase 2 (friction) |
|--------|---------|---------|-------------------|-------------------|
| Ours | 58.2 ± 3.1 | **62.4 ± 2.8** | **59.7 ± 3.5** | **60.1 ± 3.2** |
| Static F/G | 57.8 ± 2.9 | 42.3 ± 4.1 | 38.5 ± 3.8 | 40.2 ± 4.2 |
| PPO | 54.1 ± 3.7 | 35.2 ± 5.2 | 33.8 ± 4.9 | 34.5 ± 5.1 |

(Values are mean ± std over 5 seeds)

---

## Timeline Summary / タイムライン要約

### Week 1: Experiments

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Baseline implementations | 12 |
| 3-4 | Phase 2 implementation | 8 |
| 5 | Multiple seeds | 6 |
| 6-7 | Data analysis | 8 |
| **Total** | | **34** |

### Week 2: Writing

| Day | Task | Hours |
|-----|------|-------|
| 1 | Paper structure | 4 |
| 2 | Method section | 6 |
| 3 | Experiments section | 6 |
| 4 | Intro, abstract, discussion | 6 |
| 5-6 | Revision | 8 |
| 7 | Final check and submission | 4 |
| **Total** | | **34** |

**Total Effort**: 68 hours (~2 weeks full-time)

---

## Contingency Plan / 予備計画

### If Time is Limited (時間が限られている場合)

**Priority 1 (Must Have)**:
- ✅ Static F/G baseline
- ✅ Phase 2 (gravity only)
- ✅ 3 seeds (instead of 5)

**Priority 2 (Nice to Have)**:
- PPO baseline
- Phase 2 (friction)
- 5 seeds

**Priority 3 (Optional)**:
- Active Inference baseline
- Phase 2 (mass)
- Ablation studies

### If Results are Negative (結果が否定的な場合)

**Scenario**: Baselines perform similarly to ours

**Response**:
1. Analyze why (task too simple? F/G not learning?)
2. Adjust paper narrative (focus on theoretical contribution)
3. Consider Workshop paper instead of ICLR Tiny Papers
4. Implement Phase 3 (more complex tasks)

---

## Success Criteria / 成功基準

### For ICLR Tiny Papers Acceptance

**Must Have**:
- ✅ Theoretical novelty (suspension structure)
- ✅ Implementation and code
- ✅ Positive results (ours > baselines in Phase 1-2)
- ✅ Statistical significance (p < 0.05)
- ✅ Clear writing and figures

**Nice to Have**:
- Multiple baselines (3+)
- Ablation studies
- Theoretical analysis

**Estimated Acceptance Probability**: 70-80%

---

## Next Actions / 次のアクション

### Immediate (今すぐ)

1. ユーザーの承認を得る
2. Day 1のタスクを開始（Static F/G baseline）

### This Week

1. すべてのベースライン実装
2. Phase 2実装
3. 複数シード実行
4. データ分析

### Next Week

1. 論文執筆
2. 推敲
3. 提出

---

**Ready to Start?** 🚀
