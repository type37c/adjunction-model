# Phase 2.5 Valence Role Experiment

**Date**: 2026-02-16  
**Status**: Implementation Complete, Ready to Run

## Overview

This experiment tests the role of `valence` in the adjunction model by comparing three conditions:

| Condition | Description | Implementation |
|:---|:---|:---|
| **Condition 1: Baseline** | Phase 2 Slack with no valence updates | `alpha_curiosity=0.0` |
| **Condition 2: Emergent** | Valence fed directly to RSSM, agent learns usage | `AgentCV3` (pending integration) |
| **Condition 3: Designed** | Priority = coherence × uncertainty × valence | `alpha_curiosity=1.0` |

## Motivation

The 2026-02-16 discussion revealed that in Phase 2 Slack experiments, `valence` and `priority` were not actually driving the training—only `L_aff` (affordance loss) was. This experiment isolates the effect of `valence` by:

1. **Condition 1 vs 3**: Testing whether valence updates improve performance
2. **Condition 2 vs 3**: Testing whether emergent usage of valence is better than designed usage

## Research Questions

1. Does `valence` improve Slack management (η/ε dynamics)?
2. Does `valence` improve task performance (L_aff convergence)?
3. Is emergent valence usage (Condition 2) better than designed usage (Condition 3)?

## Running the Experiment

### Prerequisites

```bash
cd /home/ubuntu/adjunction-model
# Ensure PyTorch and dependencies are installed
```

### Run All Conditions

```bash
python experiments/run_valence_experiment.py
```

This will:
- Run Condition 1 for 50 epochs
- Run Condition 2 for 50 epochs (currently placeholder)
- Run Condition 3 for 50 epochs
- Save results to `experiments/phase2_valence_experiment/condition_{1,2,3}/`

### Analyze Results

```bash
python experiments/analyze_valence_experiment.py
```

This will:
- Load results from all conditions
- Generate comparison plots
- Generate a summary report

## Expected Outcomes

### If valence is effective:
- Condition 3 should show better L_aff convergence than Condition 1
- η and ε dynamics should be more stable in Condition 3

### If emergent usage is better:
- Condition 2 should outperform Condition 3
- This would validate the "provide axes, let agent discover usage" principle

### If valence has no effect:
- All conditions perform similarly
- This would suggest valence needs redesign or different integration

## Implementation Status

- [x] Condition 1: Implemented and ready
- [x] Condition 2: Implemented and ready (`AdjunctionModelV3` integrated)
- [x] Condition 3: Implemented and ready
- [x] Analysis script: Complete
- [x] Documentation: Complete

## Next Steps

1. **Run experiments**: Execute all conditions (estimated 2-3 hours on CPU)
   ```bash
   python experiments/run_valence_experiment.py
   ```
2. **Analyze results**: Compare conditions and draw conclusions
   ```bash
   python experiments/analyze_valence_experiment.py
   ```
3. **Update theory docs**: Revise `priority_and_valence_reconsidered.md` based on findings

## File Structure

```
phase2_valence_experiment/
├── README.md (this file)
├── condition_1/
│   ├── results.json
│   └── model_final.pt
├── condition_2/
│   ├── results.json
│   └── model_final.pt
├── condition_3/
│   ├── results.json
│   └── model_final.pt
├── comparison_plot.png
└── analysis_report.txt
```

## References

- `NEW_PLAN.md`: Overall development plan
- `docs/theory/priority_and_valence_reconsidered.md`: Theoretical motivation
- `TODO.md`: Task tracking
