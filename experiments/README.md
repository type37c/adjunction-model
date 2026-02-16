# Experiments Directory

This directory contains all experimental implementations for the Adjunction Model project.

## Experiment Organization Principles

Each experiment is organized into its own directory with a clear name that reflects:
1. **Driving mechanism**: Loss function-driven OR intrinsic reward-driven
2. **F/G state**: Frozen (fixed weights) OR learnable (updated during training)
3. **Purpose**: What hypothesis is being tested

## Current Experiments

### 1. `slack_affordance_loss/` - Loss Function-Driven Slack Preservation

**Formerly**: Phase 2 Slack

**Driving mechanism**: Affordance Loss (external objective)  
**F/G state**: Learnable  
**Status**: ✅ Completed (2026-02-14)

**Key findings**:
- η preserved: 0.502-0.520 (mean: 0.5153)
- ε dynamic: 0.096-1.044 (mean: 0.8238)
- Strong η-ε correlation: r = 0.835 (p < 0.001)
- Affordance Loss reduced by 96.78%

**Conclusion**: Slack can be preserved with external objectives, but this is an extension of existing AI paradigms.

---

### 2. `intrinsic_reward_baseline/` - Pure Intrinsic Motivation (2/13 Reproduction)

**Driving mechanism**: Intrinsic rewards only (Competence + Novelty)  
**F/G state**: Frozen  
**Status**: 🚧 In development

**Goal**: Reproduce the 2/13 experiment results in a clean codebase without legacy dependencies.

**Key design principles** (from 2/13 archive):
- **Competence reward = Attention to breakdown**: `R_competence = coherence_curr × attention × 100`
- **No Affordance Loss**: No external objective
- **Curiosity disabled**: α = 0.0 (due to sign issues in 2/13)
- **Prevent coherence collapse**: Coherence is an observation, not a loss to minimize

**Expected results** (from 2/13):
- Intrinsic reward: +1584%
- Value function: +51000%
- Valence growth: 0.58 → 0.667 (accelerating)
- Coherence stable: Agent does not flee from breakdown
- Competence reward: Primary driving force (59.5%)

---

### 3. `intrinsic_reward_curiosity/` - Intrinsic Motivation with Curiosity v6

**Driving mechanism**: Intrinsic rewards (Competence + Novelty + Curiosity)  
**F/G state**: Frozen  
**Status**: 📋 Planned

**Goal**: Test the redesigned Curiosity reward (案G) on top of the baseline.

**Curiosity v6 definition**:
```python
R_curiosity = η(t-1) - η(t)  # Reduction in perceptual slack
```

**Rationale**: η-ε correlation (r=0.835) suggests η-only curiosity is sufficient.

---

### 4. `temporal_suspension/` - Temporal Suspension Structure (Condition B)

**Driving mechanism**: TBD  
**F/G state**: TBD  
**Status**: 📋 Planned

**Goal**: Test whether suspension structures emerge over time with a single target.

---

### 5. `purpose_emergent/` - Purpose-Emergent Active Assembly (Condition A/C)

**Driving mechanism**: Purpose loss (`min_shape CD(assembled, reference)`)  
**F/G state**: Learnable  
**Status**: ✅ Implemented, pending extended experiments

**Goal**: Test whether purpose (directional intent) emerges from slack without explicit target assignment.

**Key question**: Does the agent spontaneously choose and commit to specific shapes?

---

## Creating a New Experiment

1. Copy the template from `_templates/`
2. Create a new directory with a descriptive name
3. Fill in `config.yaml` with experiment parameters
4. Implement `run.py` and `analyze.py`
5. Write `README.md` following the template format
6. Run the experiment and document results

## Experiment README Template

See `_templates/README.md.template` for the required structure.

---

**Last Updated**: 2026-02-16  
**Maintainer**: Adjunction Model Development Team
