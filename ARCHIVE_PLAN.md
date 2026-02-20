# Archive Plan - Protect Today's Success
# アーカイブ計画 - 今日の成功実験を保護

**Date**: February 20, 2026

---

## 🔒 Protected Files (保護対象 - アーカイブしない)

### Today's Successful Experiments (今日成功した実験)

#### Core Models (コアモデル)
- ✅ `core/models/bidirectional_fg.py` - 双方向F/G（今日実装）
- ✅ `core/models/suspension.py` - 保留構造（今日実装）
- ✅ `core/models/proposal_agent.py` - 提案Agent（今日実装）
- ✅ `core/envs/escape_room.py` - 脱出部屋環境（今日実装）

#### Scripts (スクリプト)
- ✅ `scripts/train_bidirectional_fg.py` - F/G訓練（今日実行成功）
- ✅ `scripts/run_phases.py` - Phase 0-1実行（今日実行成功）

#### Results (結果)
- ✅ `results/phase0/` - Phase 0結果（今日生成）
  - `best_bidirectional_fg.pt`
  - `final_bidirectional_fg.pt`
  - `phase0_agent.pt`
  - `training_curves.png`
  - `eta_vs_epsilon.png`
  - `phase_0_results.png`
- ✅ `results/phase1/` - Phase 1結果（今日生成）
  - `phase1_models.pt`
  - `phase_1_results.png`

#### Documentation (ドキュメント)
- ✅ `FINAL_REPORT.md` - 最終レポート（今日作成）
- ✅ `README.md` - 更新済みREADME（今日更新）
- ✅ `IMPLEMENTATION_STRATEGY.md` - 実装戦略（今日作成）
- ✅ `REVISED_PHASE2_BODILY_CONSTRAINTS.md` - Phase 2設計（今日作成）
- ✅ `ESCAPE_ROOM_IMPROVEMENTS.md` - 環境改善案（今日作成）
- ✅ `PUBLICATION_ASSESSMENT.md` - 論文価値評価（今日作成）
- ✅ `PUBLICATION_ROADMAP.md` - 論文化計画（今日作成）

---

## 📦 Archive Targets (アーカイブ対象)

### Old Experiments (古い実験)

#### 1. experiments/phase1_basic_adjunction/
- **Status**: 古い実験（Phase 1の初期版）
- **Action**: `archive/experiments/phase1_basic_adjunction/` に移動

#### 2. experiments/phase1.5_fg_retraining/
- **Status**: 古い実験（Phase 1.5）
- **Action**: `archive/experiments/phase1.5_fg_retraining/` に移動

#### 3. experiments/phase2.1_trajectory_prediction/
- **Status**: 古い実験（Phase 2.1の初期版、軌道予測）
- **Action**: `archive/experiments/phase2.1_trajectory_prediction/` に移動

#### 4. experiments/step2_v2_redesign/
- **Status**: 失敗した実験（Step 2 v2）
- **Action**: `archive/experiments/step2_v2_redesign/` に移動
- **Note**: ANALYSIS.mdに失敗の分析あり（重要）

#### 5. experiments/dynamic_fg/
- **Status**: 失敗した実験（Dynamic F/G）
- **Action**: `archive/experiments/dynamic_fg/` に移動
- **Note**: ANALYSIS.mdに失敗の分析あり（重要）

#### 6. experiments/week1_bidirectional/
- **Status**: 今日の実験の前身（スクリプトが `scripts/` に移動済み）
- **Action**: `archive/experiments/week1_bidirectional/` に移動

#### 7. experiments/week1_escape_room/
- **Status**: 今日の実験の前身（環境が `core/envs/` に移動済み）
- **Action**: `archive/experiments/week1_escape_room/` に移動

### Old Models (古いモデル)

#### 1. src/models/
- **Keep**: 
  - `adjunction_model.py` - 基礎モデル（依存関係あり）
  - `functor_f.py` - F（依存関係あり）
  - `functor_g.py` - G（依存関係あり）
- **Archive**:
  - `functor_f_v2.py` → `archive/src/models/`
  - `agent_c.py` → `archive/src/models/`
  - `agent_c_attention.py` → `archive/src/models/`
  - `intrinsic_reward.py` → `archive/src/models/`
  - `value_function.py` → `archive/src/models/`
  - `bidirectional_fg.py` → `archive/src/models/` (新版は `core/models/` にある)

#### 2. src/training/
- **Archive**:
  - `train_phase1_basic.py` → `archive/src/training/`
  - `train_phase2_intrinsic.py` → `archive/src/training/`

#### 3. src/data/
- **Archive**:
  - `composite_dataset_old.py` → `archive/src/data/`

### Old Documentation (古いドキュメント)

#### 1. research/
- **Keep**: すべて保持（理論的背景として重要）
- **Action**: そのまま

#### 2. docs/
- **Keep**: すべて保持
- **Action**: そのまま

#### 3. Root-level docs
- **Archive**:
  - `EXPERIMENT_SUMMARY.md` → `archive/docs/`
  - `TODO.md` → `archive/docs/`
  - `THEORETICAL_DISCUSSIONS.md` → `archive/docs/`

---

## 📁 Target Directory Structure (目標ディレクトリ構造)

```
adjunction-model/
├── core/                             # 今日実装したコアコンポーネント ✅
│   ├── models/
│   │   ├── bidirectional_fg.py       # 双方向F/G ✅
│   │   ├── suspension.py             # 保留構造 ✅
│   │   └── proposal_agent.py         # 提案Agent ✅
│   └── envs/
│       └── escape_room.py            # 脱出部屋環境 ✅
├── scripts/                          # 今日実行したスクリプト ✅
│   ├── train_bidirectional_fg.py     # F/G訓練 ✅
│   └── run_phases.py                 # Phase 0-1実行 ✅
├── results/                          # 今日の結果 ✅
│   ├── phase0/                       # Phase 0結果 ✅
│   └── phase1/                       # Phase 1結果 ✅
├── src/                              # 基礎コンポーネント（依存関係あり）
│   ├── models/
│   │   ├── adjunction_model.py       # 基礎モデル（保持）
│   │   ├── functor_f.py              # F（保持）
│   │   └── functor_g.py              # G（保持）
│   ├── data/
│   │   ├── synthetic_dataset.py      # データセット（保持）
│   │   └── composite_dataset.py      # データセット（保持）
│   └── envs/                         # 環境（保持）
├── research/                         # 理論的背景（すべて保持）
├── docs/                             # ドキュメント（すべて保持）
├── archive/                          # アーカイブ（新規作成）
│   ├── experiments/                  # 古い実験
│   │   ├── phase1_basic_adjunction/
│   │   ├── phase1.5_fg_retraining/
│   │   ├── phase2.1_trajectory_prediction/
│   │   ├── step2_v2_redesign/        # 失敗実験（分析あり）
│   │   ├── dynamic_fg/               # 失敗実験（分析あり）
│   │   ├── week1_bidirectional/
│   │   └── week1_escape_room/
│   ├── src/
│   │   ├── models/                   # 古いモデル
│   │   ├── training/                 # 古い訓練スクリプト
│   │   └── data/                     # 古いデータセット
│   └── docs/                         # 古いドキュメント
├── FINAL_REPORT.md                   # 最終レポート ✅
├── README.md                         # 更新済みREADME ✅
├── REVISED_PHASE2_BODILY_CONSTRAINTS.md  # Phase 2設計 ✅
├── ESCAPE_ROOM_IMPROVEMENTS.md       # 環境改善案 ✅
├── PUBLICATION_ASSESSMENT.md         # 論文価値評価 ✅
└── PUBLICATION_ROADMAP.md            # 論文化計画 ✅
```

---

## 🔄 Migration Steps (移行手順)

### Step 1: Create Archive Directory
```bash
mkdir -p archive/experiments
mkdir -p archive/src/models
mkdir -p archive/src/training
mkdir -p archive/src/data
mkdir -p archive/docs
```

### Step 2: Move Old Experiments
```bash
mv experiments/phase1_basic_adjunction archive/experiments/
mv experiments/phase1.5_fg_retraining archive/experiments/
mv experiments/phase2.1_trajectory_prediction archive/experiments/
mv experiments/step2_v2_redesign archive/experiments/
mv experiments/dynamic_fg archive/experiments/
mv experiments/week1_bidirectional archive/experiments/
mv experiments/week1_escape_room archive/experiments/
```

### Step 3: Move Old Models
```bash
mv src/models/functor_f_v2.py archive/src/models/
mv src/models/agent_c.py archive/src/models/
mv src/models/agent_c_attention.py archive/src/models/
mv src/models/intrinsic_reward.py archive/src/models/
mv src/models/value_function.py archive/src/models/
mv src/models/bidirectional_fg.py archive/src/models/
```

### Step 4: Move Old Training Scripts
```bash
mv src/training/train_phase1_basic.py archive/src/training/
mv src/training/train_phase2_intrinsic.py archive/src/training/
```

### Step 5: Move Old Data
```bash
mv src/data/composite_dataset_old.py archive/src/data/
```

### Step 6: Move Old Docs
```bash
mv EXPERIMENT_SUMMARY.md archive/docs/
mv TODO.md archive/docs/
mv THEORETICAL_DISCUSSIONS.md archive/docs/
```

### Step 7: Clean Up Empty Directories
```bash
# Remove experiments/archived if empty
rmdir experiments/archived 2>/dev/null || true

# Remove empty __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

---

## ✅ Verification (検証)

### After Migration, Verify:

1. **Today's experiments still work**:
   ```bash
   python scripts/train_bidirectional_fg.py --epochs 1
   python scripts/run_phases.py --phase 0 --episodes 1
   ```

2. **Results are intact**:
   ```bash
   ls -lh results/phase0/
   ls -lh results/phase1/
   ```

3. **Core models are accessible**:
   ```bash
   python -c "from core.models.bidirectional_fg import BidirectionalFG; print('OK')"
   python -c "from core.models.suspension import SuspensionStructure; print('OK')"
   python -c "from core.models.proposal_agent import ProposalAgent; print('OK')"
   ```

4. **Dependencies are intact**:
   ```bash
   python -c "from src.models.functor_f import FunctorF; print('OK')"
   python -c "from src.models.functor_g import FunctorG; print('OK')"
   python -c "from src.data.synthetic_dataset import SyntheticAffordanceDataset; print('OK')"
   ```

---

## 📝 Archive README (アーカイブREADME)

Create `archive/README.md`:

```markdown
# Archive - Old Experiments and Models

This directory contains archived experiments, models, and scripts from before February 20, 2026.

## Why Archived?

On February 20, 2026, we successfully implemented the core suspension structure and bidirectional F/G model, completing Phase 0-1 experiments with positive results. To maintain a clean repository structure, older experiments and models were archived here.

## Contents

### experiments/
- **phase1_basic_adjunction/**: Initial Phase 1 experiment
- **phase1.5_fg_retraining/**: Phase 1.5 experiment
- **phase2.1_trajectory_prediction/**: Early Phase 2.1 (trajectory prediction approach)
- **step2_v2_redesign/**: Failed experiment (see ANALYSIS.md for insights)
- **dynamic_fg/**: Failed experiment (see ANALYSIS.md for insights)
- **week1_bidirectional/**: Predecessor of current `scripts/train_bidirectional_fg.py`
- **week1_escape_room/**: Predecessor of current `core/envs/escape_room.py`

### src/
- **models/**: Old model implementations (replaced by `core/models/`)
- **training/**: Old training scripts (replaced by `scripts/`)
- **data/**: Old dataset implementations

### docs/
- **EXPERIMENT_SUMMARY.md**: Summary of experiments before Feb 20
- **TODO.md**: Old TODO list
- **THEORETICAL_DISCUSSIONS.md**: Theoretical discussions

## Important Notes

- **Failed experiments** (step2_v2_redesign, dynamic_fg) contain valuable ANALYSIS.md files explaining why they failed
- These archives may be useful for understanding the project's evolution
- Do not delete this directory without backing up to external storage
```

---

## Summary

- ✅ **Protected**: Today's successful experiments (core/, scripts/, results/)
- 📦 **Archived**: Old experiments and models
- 🔒 **Preserved**: Dependencies (src/models/adjunction_model.py, functor_f.py, functor_g.py)
- 📚 **Kept**: Research notes and documentation (research/, docs/)

**Ready to execute the archive plan?**
