# Legacy Code Archive

**Date Archived**: 2026-02-14

This directory contains code that was removed during the "Great Cleanup" refactoring. These files are preserved for historical reference but are no longer part of the active codebase.

## Why These Files Were Removed

The project underwent rapid iterative development, creating multiple versions (v2, v3, v4) of core components. After the successful Phase 2 Slack experiment validated the v4 architecture, older versions became obsolete.

## What's Archived Here

### models/
- `agent_layer_v2.py`, `agent_layer_v3.py`: Earlier iterations of Agent C
- `conditional_adjunction_v2.py`, `conditional_adjunction_v3.py`: Earlier adjunction models
- `intrinsic_reward.py`, `priority.py`, `valence.py`, `valence_v2.py`: Legacy reward components (copies of active `src/models/` files, preserved for v2/v3/v4 internal dependencies)
- `adjunction.py`: Original non-conditional adjunction

### training/
- `train_phase1.py`: Phase 1 pretraining (superseded by unified trainer)
- `train_phase2_v2.py`: Earlier Phase 2 attempt
- `online_learning_v2.py`, `train_agent_autonomous.py`, `train_sequential.py`: Experimental training loops
- `train_agent_value_based.py`: Value-based training experiment

### experiments/
- Various test scripts that were superseded by `phase2_slack_experiment.py`
- `temporal_suspension_experiment.py`: Temporal Suspension (Progressive Revelation) experiment (superseded design)
- `analyze_temporal_suspension.py`: Analysis script for Temporal Suspension experiment
- `full_scale_training.py`: Full-scale training experiment with broken imports (depends on non-existent `src.models.value_function` and `src.training.train_agent_value_based`)

### tests/
- `test_agent_c_v2_integration.py`: Tests for v2 models

### results/
- `training_log.txt`, `training_log_v2.txt`: Legacy training logs
- `full_scale_training/`: Results from full-scale training experiment (broken dependencies)

## Current Active Code

The canonical implementations are:
- **Models**: `agent_layer.py` (formerly v4), `adjunction_model.py` (formerly ConditionalAdjunctionModelV4)
- **Training**: `train_phase2_slack.py`
- **Experiments**: 
  - `phase2_slack_experiment.py`, `analyze_phase2_slack.py` (completed, valuable results)
  - `purpose_emergent_experiment.py`, `analyze_purpose_emergent.py` (latest active research)

## Restoration

If you need to restore any of these files, simply copy them back to their original locations. However, be aware that they may have dependencies on other archived files.
