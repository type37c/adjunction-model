# Curiosity v6: η-based Intrinsic Motivation Design

**Date**: 2026-02-14  
**Based on**: Phase 2 Slack実験結果  
**Status**: Design Complete, Ready for Implementation

---

## Executive Summary

Phase 2 Slack実験の結果に基づき、**Curiosity v6**を設計する。実験データから得られた重要な発見：

1. **ηは保存される**: 0.50-0.52の範囲で安定（0に収束しない）
2. **εは動的に変化**: 0.10-1.04の範囲で大きく変動
3. **η-ε強相関**: r = 0.835（ηを制御すればεも連動）
4. **学習とslackの両立**: Affordance Lossは96.78%減少、同時にη/εは保存

これらの発見に基づき、**ηのみを使用したシンプルなCuriosity定義**を採用する。

---

## Design Rationale

### Why η-only?

#### 理論的根拠

1. **Active Inferenceとの整合性**
   - ηは「予測誤差」に対応
   - η減少 = 環境モデルの改善
   - 既存の理論的枠組みと一致

2. **η-ε強相関（r=0.835）**
   - ηを制御すれば、εも自動的に制御される
   - εを明示的に扱う必要性が低い
   - シンプルさと効果のバランス

3. **ηの安定性**
   - 標準偏差: 0.004（εの0.226と比較して非常に小さい）
   - 予測可能で、報酬信号として安定
   - 学習の収束性が高い

#### 実験的根拠

Phase 2 Slack実験から：
- ηは0に収束せず、保留構造を保存
- ηの変化は学習の進行と相関
- ηの範囲（0.50-0.52）は実用的な報酬スケール

---

## Curiosity v6 Definition

### Mathematical Formulation

```python
# Curiosity Reward (Unit η-based)
R_curiosity(t) = η(t-1) - η(t)

where:
  η(t) = ||Shape - G(F(Shape))||  # Unit of adjunction F⊣G
```

### Interpretation

- **Positive reward**: η減少 → 環境理解が深まった
- **Negative reward**: η増加 → 環境が予測不可能になった
- **Zero reward**: η不変 → 新しい情報なし

### Reward Scaling

実験データから、ηの典型的な変化量は：
- 平均変化: ±0.001-0.005 per epoch
- 最大変化: ±0.01

推奨スケーリング:
```python
R_curiosity_scaled = 100 × (η(t-1) - η(t))
# → 典型的な報酬: ±0.1-0.5
```

---

## Architecture Design

### Agent Layer C v6

```python
class AgentLayerC_v6(nn.Module):
    """
    Agent Layer C v6: η-based Intrinsic Motivation
    
    Key changes from v4:
    - Curiosity based on η only (not η + ε)
    - Simplified intrinsic reward computation
    - Removed ε-based novelty component
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        context_dim: int = 128,
        valence_dim: int = 32,
        valence_decay: float = 0.1,
        alpha_curiosity: float = 1.0,      # Weight for η-based curiosity
        beta_competence: float = 0.5,      # Weight for competence
        uncertainty_type: str = 'entropy',
        attention_temperature: float = 1.0
    ):
        super().__init__()
        
        self.alpha_curiosity = alpha_curiosity
        self.beta_competence = beta_competence
        
        # RSSM (same as v4)
        self.rssm = RSSM(obs_dim, action_dim, hidden_dim, latence_dim)
        
        # Valence (same as v4)
        self.valence = ValenceModule(valence_dim, valence_decay)
        
        # Context generation (same as v4)
        self.context_net = nn.Sequential(
            nn.Linear(latent_dim + valence_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
    
    def compute_intrinsic_reward(
        self,
        coherence_signal_prev: torch.Tensor,  # η(t-1)
        coherence_signal_curr: torch.Tensor,  # η(t)
        affordance_pred: torch.Tensor,
        affordance_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute intrinsic reward based on η change and competence.
        
        Args:
            coherence_signal_prev: Previous η (B, 1)
            coherence_signal_curr: Current η (B, 1)
            affordance_pred: Predicted affordances (B, A)
            affordance_gt: Ground truth affordances (B, A)
        
        Returns:
            Dictionary with reward components
        """
        # Curiosity: η reduction
        R_curiosity = coherence_signal_prev - coherence_signal_curr  # (B, 1)
        
        # Competence: Affordance prediction accuracy
        affordance_error = (affordance_pred - affordance_gt).pow(2).mean(dim=-1, keepdim=True)
        R_competence = -affordance_error  # (B, 1)
        
        # Total intrinsic reward
        R_intrinsic = (
            self.alpha_curiosity * R_curiosity +
            self.beta_competence * R_competence
        )
        
        return {
            'intrinsic_reward': R_intrinsic,
            'curiosity': R_curiosity,
            'competence': R_competence,
        }
```

---

## Comparison with Previous Versions

| Version | Curiosity Definition | Issues | Status |
|:---|:---|:---|:---|
| **v1-v3** | Confidence-based | 再構成損失前提、Phase 2 Slackで破綻 | ❌ Deprecated |
| **v4** | Intrinsic rewards (curiosity + competence + novelty) | Curiosity未定義、実装不完全 | ⚠️ Incomplete |
| **v5** | (Proposed) η + ε reduction | ηとεの非対称性を無視、理論的矛盾 | ❌ Rejected |
| **v6** | **η-based curiosity** | **シンプル、理論的に健全、実験的根拠あり** | ✅ **Recommended** |

---

## Implementation Plan

### Step 1: Create Agent Layer C v6

**File**: `src/models/agent_layer_v6.py`

**Changes from v4**:
1. Remove `gamma_novelty` parameter
2. Simplify `compute_intrinsic_reward()` to use only η
3. Update docstrings

**Estimated time**: 30 minutes

---

### Step 2: Create Conditional Adjunction Model v6

**File**: `src/models/conditional_adjunction_v6.py`

**Changes from v4**:
1. Use `AgentLayerC_v6` instead of `AgentLayerC_v4`
2. Update `forward()` to pass η(t-1) and η(t)
3. Add intrinsic reward computation

**Estimated time**: 30 minutes

---

### Step 3: Create Training Script

**File**: `src/training/train_phase2_curiosity_v6.py`

**Based on**: `train_phase2_slack.py`

**Changes**:
1. Use `ConditionalAdjunctionModelV6`
2. Add intrinsic reward logging
3. Track valence updates

**Estimated time**: 20 minutes

---

### Step 4: Run Experiment

**Configuration**:
- Epochs: 50
- Shapes: 100
- Batch size: 8
- α_curiosity: 1.0
- β_competence: 0.5

**Expected results**:
- Valence increases when η decreases
- Agent learns to reduce η (curiosity-driven)
- Affordance prediction improves

**Estimated time**: 2-3 hours (CPU)

---

## Expected Outcomes

### Success Criteria

1. **Curiosity drives learning**:
   - Valence correlates with η reduction
   - Agent actively seeks to reduce η

2. **Suspension structure preserved**:
   - η does not collapse to 0
   - ε remains observable

3. **Performance improvement**:
   - Affordance loss decreases
   - Better than Phase 2 Slack (no curiosity)

### Metrics to Track

- `η(t)` - Unit value over time
- `ε(t)` - Counit value over time
- `R_curiosity(t)` - Curiosity reward
- `R_competence(t)` - Competence reward
- `R_intrinsic(t)` - Total intrinsic reward
- `Valence(t)` - Valence state
- `Affordance Loss` - Task performance

---

## Alternative: Curiosity v7 (Future Work)

If Curiosity v6 shows limitations, consider **Curiosity v7**:

```python
# Dual-component intrinsic reward
R_understanding = η(t-1) - η(t)      # Understanding (η reduction)
R_exploration = ε(t) - ε(t-1)       # Exploration (ε increase)

R_intrinsic = α × R_understanding + β × R_exploration + γ × R_competence
```

**When to use**:
- If v6 shows insufficient exploration
- If ε dynamics are important for the task
- If η-only is too conservative

**Pros**:
- More flexible
- Captures both η and ε dynamics

**Cons**:
- More complex
- Requires hyperparameter tuning (α, β, γ)

---

## Theoretical Implications

### Suspension Structure and Curiosity

Curiosity v6 establishes a connection between:

1. **Category Theory**: Unit η of adjunction F⊣G
2. **Active Inference**: Prediction error minimization
3. **Intrinsic Motivation**: Curiosity as η reduction

**Key insight**:
> Curiosity can be formalized as the motivation to reduce the unit η of an adjunction, which represents the "slack" or "incompleteness" in the agent's world model.

This provides a **mathematical foundation** for curiosity that goes beyond heuristic definitions.

### Comparison with Existing Theories

| Theory | Curiosity Definition | Connection to v6 |
|:---|:---|:---|
| **Active Inference** | Prediction error reduction | η = prediction error |
| **Information Gain** | Maximize information | η reduction = information gain |
| **Empowerment** | Maximize control | Competence component |
| **Curiosity v6** | **η reduction** | **Unifies above concepts** |

---

## Next Steps

1. **Implement Curiosity v6** (estimated: 1.5 hours)
2. **Run experiment** (estimated: 2-3 hours)
3. **Analyze results** (estimated: 1 hour)
4. **Compare with Phase 2 Slack** (no curiosity baseline)
5. **Document findings** in research notes
6. **Decide**: v6 sufficient, or proceed to v7?

---

## References

- Phase 2 Slack実験結果 (`/home/ubuntu/results/phase2_slack_local/`)
- η-ε相関分析 (`correlation_analysis.json`)
- 重要な発見 (`KEY_FINDINGS.md`)
- 以前の議論: ηとεの非対称性（ユーザー提供）

---

**Design Status**: ✅ Complete  
**Ready for Implementation**: ✅ Yes  
**Recommended Action**: Implement Curiosity v6 and run experiment
