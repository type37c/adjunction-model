# Adjunction Slack Analysis: Unit η and Counit ε as "余白" (Margin)

**Date**: 2026-02-13  
**Context**: Analysis of the theoretical insight that unit η and counit ε should be preserved as "slack" rather than minimized as "error"

---

## 1. Current Implementation: η and ε as Error

### 1.1 Mathematical Definition

In the adjunction F ⊣ G:
- **Unit η**: `Shape → G(F(Shape))`
- **Counit ε**: `F(G(Action)) → Action`

These natural transformations represent the "mismatch" between the composition of functors and the identity.

### 1.2 Current Treatment in Implementation

**Location**: `src/models/conditional_adjunction_v4.py`, `_compute_coherence_signal()`

```python
# Coherence Signal = Chamfer distance between original and reconstructed
coherence_signal = ||original - G(F(original))||
```

**Interpretation**:
- **η is measured as reconstruction error** (Chamfer distance)
- **η is treated as a signal to minimize** (lower coherence = better)
- **ε is not explicitly computed** (no F(G(action)) → action path)

**Training objective** (from ARCHITECTURE.md):
> "The reconstruction error has theoretical significance. Here, `||Shape - G(F(Shape))||` is the **coherence signal** — it measures how well the agent's action-based understanding of a shape captures the shape's actual geometry."

**Current philosophy**: 
- High coherence signal = agent's model is incomplete
- Training drives coherence signal toward zero
- **η is an error to be eliminated**

---

## 2. Theoretical Insight: η and ε as "余白" (Margin/Slack)

### 2.1 The Proposal (from `suspension_and_confidence.md`)

> **現状**：このズレを「エラー」として最小化している  
> **提案**：ηとεを「余白」として保持することが、保留構造の本質？

### 2.2 What Does "余白" Mean?

**余白 (yohaku)** in Japanese means:
1. **Margin**: blank space, room for movement
2. **Slack**: looseness, flexibility, tolerance
3. **Breathing room**: space for uncertainty and adaptation

In the context of adjunction:
- **η ≠ 0**: The shape-to-action-to-shape round trip does not return to the exact original
- **ε ≠ 0**: The action-to-shape-to-action round trip does not return to the exact original action

**Preserving slack means**: 
- **Not forcing η → 0** (not forcing perfect reconstruction)
- **Not forcing ε → 0** (not forcing action uniqueness)
- **Allowing the adjunction to be "loose"** rather than "tight"

### 2.3 Why Preserve Slack?

#### 2.3.1 Categorical Perspective

In category theory, an adjunction F ⊣ G is defined by:
- **Unit η**: `id → G ∘ F`
- **Counit ε**: `F ∘ G → id`
- **Triangle identities**: `(F ∘ η) ; (ε ∘ F) = id_F` and `(η ∘ G) ; (G ∘ ε) = id_G`

**Key insight**: The triangle identities do NOT require η = id or ε = id. They only require that the compositions cancel out.

**Implication**: **η and ε are allowed to be non-trivial**. The adjunction is "optimal" in a universal property sense, not in a "minimal error" sense.

#### 2.3.2 Suspension Structure Perspective

From `suspension_and_confidence.md`:

> 保留が創発するためには、以下の3つの条件が必要である：
> 1. 「確定」と「未確定」の両方の状態が可能であること
> 2. 「確定を壊す」トリガーがあること
> 3. **「未確定のまま保持する」能力があること** ← **ここが欠けている**

**Preserving slack = preserving the ability to remain indefinite**

- **η ≠ 0**: The shape is not fully captured by its affordances → room for reinterpretation
- **ε ≠ 0**: The action is not uniquely determined by its shape → room for multiple possibilities

**Slack is the space where suspension lives.**

#### 2.3.3 Confidence Perspective

From our earlier discussion:
- **Confidence = 1 - entropy(priority_distribution)**
- **High confidence**: choices are narrowed down (η → 0, ε → 0)
- **Low confidence**: choices remain open (η ≠ 0, ε ≠ 0)

**Preserving slack = allowing low confidence states**

---

## 3. Implications for Implementation

### 3.1 Current Problem

**Current training objective**:
```python
loss = L_recon + L_aff
L_recon = ||original - reconstructed||  # Minimize η
```

**This drives η → 0**, which means:
- The adjunction becomes "tight"
- No room for suspension
- Confidence is forced to be high
- **Suspension structure cannot emerge**

### 3.2 Proposed Change: Preserve Slack

#### Option A: Remove Reconstruction Loss Minimization

**Idea**: Do not train F and G to minimize reconstruction error. Instead, train them only on affordance prediction.

```python
# Current
loss = λ_recon * L_recon + λ_aff * L_aff

# Proposed
loss = L_aff  # Only affordance prediction, no reconstruction loss
```

**Effect**:
- η is no longer driven toward zero
- G(F(shape)) will capture only the "functional core" (as intended)
- Slack is preserved naturally

**Problem**: Without reconstruction loss, G may collapse (produce constant output)

#### Option B: Regularize Slack Preservation

**Idea**: Add a term that **penalizes η being too small**, encouraging slack.

```python
# Encourage non-zero slack
slack_penalty = -torch.log(coherence_signal + ε)  # Penalize η → 0

loss = λ_aff * L_aff + λ_slack * slack_penalty
```

**Effect**:
- η is kept away from zero
- Adjunction remains "loose"
- Slack is explicitly preserved

#### Option C: Use η as a Signal, Not a Loss

**Idea**: Do not minimize η. Instead, use η as an **observation** for Agent C.

**Current**:
```python
coherence_signal = ||original - reconstructed||
# Used as input to Agent C, but also minimized in training
```

**Proposed**:
```python
coherence_signal = ||original - reconstructed||
# Used ONLY as input to Agent C, never minimized
# Agent C learns to interpret η, not eliminate it
```

**Effect**:
- η is preserved as information
- Agent C learns "what does this slack mean?"
- Suspension structure can emerge from Agent C's interpretation of η

### 3.3 Implementing ε: The Missing Half

**Current**: ε (counit) is not explicitly computed.

**Proposed**: Compute ε as the "action ambiguity" signal.

```python
# Given an affordance distribution, reconstruct a shape, then re-encode it
shape_reconstructed = G(affordances)
affordances_reconstructed = F(shape_reconstructed)

# Counit ε: how much does F(G(affordances)) differ from affordances?
counit_signal = ||affordances - affordances_reconstructed||
```

**Interpretation**:
- **High ε**: The affordance is ambiguous (multiple shapes could produce it)
- **Low ε**: The affordance is specific (unique shape)

**Use**: Provide `counit_signal` to Agent C as another dimension of uncertainty.

---

## 4. Connection to Curiosity and Confidence

### 4.1 Curiosity as Slack Reduction

**Current Curiosity definition** (v5, confidence-based):
```python
R_curiosity = confidence_curr - confidence_prev
```

**Interpretation**:
- Curiosity rewards **narrowing down choices** (reducing slack)
- This is the "確定への移行" (transition to definiteness)

**With slack preservation**:
- Agent C is rewarded for reducing slack (Curiosity)
- But F/G preserve slack (do not minimize η)
- **This creates a dynamic tension**: Agent C wants to reduce slack, but the world (F/G) maintains slack
- **This tension IS the suspension structure**

### 4.2 Competence as Slack Engagement

**Current Competence definition** (v3):
```python
R_competence = coherence_signal  # Attending to breakdowns
```

**Interpretation**:
- Competence rewards **engaging with slack** (high η)
- This is the "破綻への注目" (attention to breakdowns)

**With slack preservation**:
- High η is not an error, but a **feature** of the world
- Competence rewards Agent C for noticing and engaging with this feature
- **Slack is the site of learning**

---

## 5. Proposed Experiment

### 5.1 Hypothesis

**If we preserve slack (η, ε) instead of minimizing it, suspension structure will emerge more naturally.**

### 5.2 Implementation

1. **Modify training objective**:
   ```python
   # Remove reconstruction loss from F/G training
   loss_FG = L_aff  # Only affordance prediction
   
   # Agent C still receives coherence_signal as observation
   # But F/G do not minimize it
   ```

2. **Add counit computation**:
   ```python
   def forward(self, pos, batch, ...):
       affordances = self.F(pos, batch, context)
       reconstructed = self.G(affordances, num_points, context)
       
       # Unit η
       coherence_signal = ||pos - reconstructed||
       
       # Counit ε
       affordances_reencoded = self.F(reconstructed, batch, context)
       counit_signal = ||affordances - affordances_reencoded||
       
       return {..., 'coherence_signal': coherence_signal, 'counit_signal': counit_signal}
   ```

3. **Provide both signals to Agent C**:
   ```python
   agent_state_new, context, agent_info = self.agent_c(
       ...,
       coherence_signal_scalar=coherence_signal,  # η
       counit_signal_scalar=counit_signal,        # ε (new)
       ...
   )
   ```

### 5.3 Expected Outcome

- **η and ε remain non-zero** (slack is preserved)
- **Agent C learns to navigate the slack** (suspension structure)
- **Confidence varies dynamically** (high when slack is low, low when slack is high)
- **Curiosity drives slack reduction** (learning)
- **Competence drives slack engagement** (attention)

---

## 6. Summary

| Aspect | Current (η as Error) | Proposed (η as Slack) |
|:---|:---|:---|
| **Training objective** | Minimize η (reconstruction loss) | Do not minimize η (only affordance loss) |
| **Interpretation of η** | Error to eliminate | Information to preserve |
| **Agent C's role** | Adapt to reduce η | Interpret and navigate η |
| **Suspension structure** | Cannot emerge (η → 0) | Can emerge (η ≠ 0) |
| **Confidence dynamics** | Forced high (η small) | Varies naturally (η varies) |
| **Counit ε** | Not computed | Computed as action ambiguity |

**Key insight**: **Slack is not a bug, it's a feature.** Preserving η and ε as non-zero "余白" is essential for suspension structure to emerge.

---

## 7. Next Steps

1. **Implement counit ε computation** in `conditional_adjunction_v4.py`
2. **Remove reconstruction loss** from F/G training (or add slack preservation penalty)
3. **Provide both η and ε to Agent C** as observations
4. **Run experiment** and observe:
   - Does η remain non-zero?
   - Does confidence vary more dynamically?
   - Does suspension structure emerge?

---

**References**:
- `docs/suspension_and_confidence.md`: Original insight about η/ε as 余白
- `ARCHITECTURE.md`: Current treatment of η as coherence signal
- `src/models/conditional_adjunction_v4.py`: Current implementation
