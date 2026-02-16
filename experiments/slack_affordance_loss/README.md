# Slack Affordance Loss Experiment

**Formerly**: Phase 2 Slack

## Driving Mechanism
- [x] Loss function-driven (Affordance Loss)
- [ ] Intrinsic reward-driven

## F/G State
- [ ] Frozen
- [x] Learnable (updated during training)

## Goal

To test whether slack (η and ε) can be preserved while training F⊣G and Agent C simultaneously, using an external objective (Affordance Loss) as the driving force.

## Hypothesis

By removing the reconstruction loss (`L_recon`) and using only Affordance Loss (`L_aff`) as the primary objective, the model will preserve slack (η and ε) while still learning to predict affordances accurately.

## Design

### Model Configuration
- **F/G**: Learnable (trained from scratch)
- **Agent C**: Full RSSM with Priority, Valence, and intrinsic motivation modules
- **Priority module**: Implemented but not used in loss function

### Training Configuration
- **Driving mode**: loss_function
- **Loss functions**:
  - **Affordance Loss**: Enabled (primary driver)
  - **KL Loss**: Enabled (RSSM regularization)
  - **Coherence Loss**: Enabled (`-log(η)` to prevent collapse to zero)
  - **Reconstruction Loss**: **Disabled** (this is the key change)
- **Intrinsic rewards**:
  - Competence, Novelty, Curiosity: Computed but not used in loss
  - `alpha_curiosity = 0.0` (disabled)

### Hyperparameters
- Epochs: 50
- Batch size: 8
- Learning rate: 0.001
- Num shapes: 100
- Num points: 256

## Implementation

### Files
- `run.py`: Training script (formerly `phase2_slack_experiment.py`)
- `analyze.py`: Analysis script (formerly `analyze_phase2_slack.py`)
- `results/`: Experiment results
  - `KEY_FINDINGS.md`: Summary of key findings
  - `COMPREHENSIVE_ANALYSIS_REPORT.md`: Detailed analysis
  - `metrics.json`: Recorded metrics
  - `correlation_analysis.json`: η-ε correlation analysis
  - `detailed_analysis.png`: Visualization

### Dependencies
- `src/training/train_phase2_slack.py`: Phase2SlackTrainer
- `src/models/adjunction_model.py`: AdjunctionModel
- `src/models/agent_c.py`: AgentLayerC

## Metrics Tracked

- [x] Affordance Loss
- [x] KL divergence
- [x] Coherence Loss
- [x] η (unit/perceptual slack) time evolution
- [x] ε (counit/semantic slack) time evolution
- [x] η-ε correlation
- [x] Valence (mean)
- [x] Priority (mean)

## Results

**Status**: ✅ Completed (2026-02-14)

### Key Findings

1. **Slack Preservation Successful**
   - η preserved: 0.502-0.520 (mean: 0.5153)
   - ε dynamic: 0.096-1.044 (mean: 0.8238)
   - Neither collapsed to zero

2. **Strong η-ε Correlation**
   - Pearson correlation: r = 0.835 (p < 0.001)
   - This suggests controlling η also controls ε
   - Implication: η-only curiosity may be sufficient (Curiosity v6 design)

3. **Task Performance**
   - Affordance Loss reduced by 96.78% (0.031 → 0.001)
   - Learning was successful despite no reconstruction loss

4. **Internal Signals**
   - Valence, Priority, and intrinsic rewards were computed
   - However, they did not drive training (not in loss function)
   - `alpha_curiosity = 0.0` meant valence was not updated

### Visualizations

See `results/detailed_analysis.png` for:
- Affordance Loss trajectory
- η and ε time evolution
- η-ε scatter plot with correlation
- KL divergence
- Valence and Priority trends

### Comparison with Other Experiments

This experiment demonstrates that slack can be preserved with an **external objective** (Affordance Loss). However, this is an extension of existing AI paradigms.

The unique claim of this project is that **intrinsic motivation alone** can drive structure formation without external objectives. That hypothesis is tested in `intrinsic_reward_baseline/`.

## Conclusion

Phase 2 Slack successfully demonstrated that:
1. Slack (η and ε) can be preserved by removing reconstruction loss
2. An external objective (Affordance Loss) is sufficient to drive learning while preserving slack
3. η and ε are strongly correlated, simplifying future curiosity designs

However, this experiment does not test the core hypothesis of the project: **Can structure emerge from intrinsic motivation alone, without external objectives?**

That question is addressed in the `intrinsic_reward_baseline/` experiment.

---

**Status**: ✅ Completed  
**Date Created**: 2026-02-14  
**Date Completed**: 2026-02-14  
**Moved to new structure**: 2026-02-16
