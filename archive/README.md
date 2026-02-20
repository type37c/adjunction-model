# Archive - Old Experiments and Models

This directory contains archived experiments, models, and scripts from before February 20, 2026.

## Why Archived?

On February 20, 2026, we successfully implemented the core suspension structure and bidirectional F/G model, completing Phase 0-1 experiments with positive results:

- **Phase 0 (Known shapes)**: 58% success rate
- **Phase 1 (Unknown shapes)**: 62% success rate
- **Bidirectional F/G**: 78.9% of actions are coherent (η < 0.1 and ε < 0.1)
- **Suspension structure**: Successfully triggered 11 times in Phase 1
- **F/G adaptation**: 2 updates executed, η decreased after adaptation

To maintain a clean repository structure for the next phase (bodily constraints experiments), older experiments and models were archived here.

---

## Contents

### experiments/

#### phase1_basic_adjunction/
- **Description**: Initial Phase 1 experiment
- **Status**: Completed, superseded by current implementation
- **Key files**: `run.py`, checkpoints

#### phase1.5_fg_retraining/
- **Description**: Phase 1.5 experiment (F/G retraining)
- **Status**: Completed, superseded by current implementation

#### phase2.1_trajectory_prediction/
- **Description**: Early Phase 2.1 (trajectory prediction approach)
- **Status**: Abandoned, different approach taken
- **Subdirectories**:
  - `step1_eta_validation/`: η validation experiments
  - `step2_agent_integration/`: Agent integration experiments

#### step2_v2_redesign/ ⚠️
- **Description**: Failed experiment (Step 2 v2 redesign)
- **Status**: Failed
- **Important**: See `ANALYSIS.md` for detailed failure analysis
- **Key findings**: 
  - Environment redesign succeeded
  - F/G integration failed (reward -6.22, success rate 0%)
  - Baseline: reward 2.69, success rate 3%

#### dynamic_fg/ ⚠️
- **Description**: Failed experiment (Dynamic F/G with temporal point clouds)
- **Status**: Failed
- **Important**: See `ANALYSIS.md` for detailed failure analysis
- **Key findings**:
  - Worse than baseline (0% vs 25% success rate)
  - η diverged (3.3 → 14.7)
  - Hypothesis: Task mismatch (Reaching is too simple for F/G)

#### week1_bidirectional/
- **Description**: Predecessor of current `scripts/train_bidirectional_fg.py`
- **Status**: Superseded by current implementation

#### week1_escape_room/
- **Description**: Predecessor of current `core/envs/escape_room.py`
- **Status**: Superseded by current implementation

---

### src/

#### models/
- **functor_f_v2.py**: Old version of Functor F
- **agent_c.py**: Old Agent C implementation
- **agent_c_attention.py**: Agent C with attention mechanism
- **intrinsic_reward.py**: Intrinsic reward module
- **value_function.py**: Value function module
- **bidirectional_fg.py**: Old bidirectional F/G (replaced by `core/models/bidirectional_fg.py`)

#### training/
- **train_phase1_basic.py**: Old Phase 1 training script
- **train_phase2_intrinsic.py**: Old Phase 2 training script (intrinsic motivation)

#### data/
- **composite_dataset_old.py**: Old composite dataset implementation

---

### docs/

#### EXPERIMENT_SUMMARY.md
- **Description**: Summary of all experiments before Feb 20, 2026
- **Contents**: Phase 1, Phase 1.5, Phase 2.1, Step 2 v2, Dynamic F/G

#### TODO.md
- **Description**: Old TODO list before Feb 20, 2026
- **Contents**: Next steps, priorities, open questions

#### THEORETICAL_DISCUSSIONS.md
- **Description**: Theoretical discussions and design decisions
- **Contents**: Adjunction theory, coherence signals, suspension structure

---

## Important Notes

### Failed Experiments

The **failed experiments** (step2_v2_redesign, dynamic_fg) contain valuable ANALYSIS.md files explaining why they failed:

1. **step2_v2_redesign/ANALYSIS.md**:
   - Environment redesign succeeded
   - F/G integration failed
   - Root cause: Information usefulness, not just coherence

2. **dynamic_fg/ANALYSIS.md**:
   - Dynamic F/G made things worse
   - η diverged instead of converging
   - Root cause: Task mismatch (Reaching is too simple)

These analyses led to the current understanding:
- **Suspension structure** is needed (not just F/G)
- **Bodily constraints** are the right test (not physical constraints)
- **Escape room** is a better environment (not Reaching)

### Preservation

These archives may be useful for:
- Understanding the project's evolution
- Learning from past mistakes
- Referencing old implementations
- Writing the paper (failure analysis section)

**Do not delete this directory without backing up to external storage.**

---

## Current Implementation (Not Archived)

For the current, working implementation, see:

- **Core models**: `core/models/`
- **Scripts**: `scripts/`
- **Results**: `results/phase0/`, `results/phase1/`
- **Documentation**: `FINAL_REPORT.md`, `README.md`

---

**Archive created**: February 20, 2026  
**Last successful experiment**: Phase 0-1 (62% success rate on unknown shapes)  
**Next phase**: Phase 2 (bodily constraints - right hand injury)
